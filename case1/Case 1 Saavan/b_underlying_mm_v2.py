from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
import time
from typing import Any

from a_bot_config import BConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


@dataclass(frozen=True)
class BUnderlyingFairV2:
    base_center: float | None
    reference_fair: float | None
    composite_synthetic_fair: float | None
    synthetic_dispersion: float | None
    composite_basis: float | None
    used_synthetic_anchor: bool
    dynamic_half_spread: int | None
    mode_reason: str


class BUnderlyingMMv2:
    """Always-on, inventory-aware market maker for the B underlying.

    This deliberately trades only the stock. Options remain signal inputs until
    we have enough live evidence to justify routing option legs.
    """

    def __init__(self, b_config: BConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.b_config = b_config
        self.risk = risk
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.symbol = b_config.underlying_symbol
        self.order_manager = OrderManager(symbol=self.symbol, risk=risk)
        self.book = BookSnapshot()
        self.inventory = 0
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.last_mode = "OBSERVE_ONLY"
        self.last_fair = BUnderlyingFairV2(
            base_center=None,
            reference_fair=None,
            composite_synthetic_fair=None,
            synthetic_dispersion=None,
            composite_basis=None,
            used_synthetic_anchor=False,
            dynamic_half_spread=None,
            mode_reason="not_started",
        )
        self.last_bid_px: int | None = None
        self.last_ask_px: int | None = None
        self.last_block_reason: str | None = None
        self.healthy_book_since_ms: int | None = None
        self.last_bad_book_fill_ms: int | None = None

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != self.symbol:
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        if self._book_is_healthy():
            if self.healthy_book_since_ms is None:
                self.healthy_book_since_ms = int(now_ms)
        else:
            self.healthy_book_since_ms = None
        return True

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)

    def sync_inventory_from_exchange(self, inventory: int) -> None:
        self.inventory = int(inventory)

    def on_fill(self, order_id: str, qty: int, price: int, *, authoritative_inventory: int | None = None) -> Any | None:
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        if authoritative_inventory is None:
            signed_qty = int(qty) if order.side == "BUY" else -int(qty)
            self.inventory += signed_qty
        else:
            self.inventory = int(authoritative_inventory)
        if self.book.spread is not None and int(self.book.spread) <= 1:
            self.last_bad_book_fill_ms = self._now_ms()
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms()
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> Any | None:
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str) -> Any | None:
        return self.order_manager.handle_rejection(order_id)

    def compute_quotes(self, *, now_ms: int, residual_payload: dict[str, Any] | None) -> QuotePlan:
        signal_id = f"b_mm_v2_{int(now_ms)}"
        self.last_bid_px = None
        self.last_ask_px = None
        self.last_block_reason = None

        if self.book.best_bid is None or self.book.best_ask is None or self.book.spread is None:
            self.last_mode = "OBSERVE_ONLY"
            self.last_fair = BUnderlyingFairV2(
                base_center=None,
                reference_fair=None,
                composite_synthetic_fair=None,
                synthetic_dispersion=None,
                composite_basis=None,
                used_synthetic_anchor=False,
                dynamic_half_spread=None,
                mode_reason="missing_b_book",
            )
            self.last_block_reason = "missing_b_book"
            return QuotePlan("OBSERVE_ONLY", None, None, (), True, "missing_b_book")

        if not self._book_is_healthy():
            return self._bad_book_plan(now_ms=now_ms)
        if self.last_bad_book_fill_ms is not None:
            cooldown_remaining_ms = int(self.b_config.mm_bad_fill_cooldown_ms) - (int(now_ms) - int(self.last_bad_book_fill_ms))
            if cooldown_remaining_ms > 0:
                self.last_mode = "OBSERVE_ONLY"
                self.last_block_reason = "recent_bad_book_fill_cooldown"
                return QuotePlan("OBSERVE_ONLY", None, None, (), True, "recent_bad_book_fill_cooldown")
        healthy_age_ms = 0 if self.healthy_book_since_ms is None else int(now_ms) - int(self.healthy_book_since_ms)
        if healthy_age_ms < int(self.b_config.mm_min_healthy_book_age_ms):
            self.last_mode = "OBSERVE_ONLY"
            self.last_block_reason = "b_book_waiting_for_healthy_age"
            return QuotePlan("OBSERVE_ONLY", None, None, (), True, "b_book_waiting_for_healthy_age")

        fair = self._fair_snapshot(residual_payload)
        self.last_fair = fair
        if fair.reference_fair is None or fair.dynamic_half_spread is None:
            self.last_mode = "OBSERVE_ONLY"
            self.last_block_reason = fair.mode_reason
            return QuotePlan("OBSERVE_ONLY", None, None, (), True, fair.mode_reason)

        allowed_buy, allowed_sell = self._allowed_sizes()
        quote_size = max(0, int(self.b_config.quote_size))
        if quote_size <= 0:
            self.last_mode = "OBSERVE_ONLY"
            self.last_block_reason = "zero_quote_size"
            return QuotePlan("OBSERVE_ONLY", None, None, (), True, "zero_quote_size")

        reduce_only = abs(self.inventory) >= int(self.b_config.passive_reduce_start)
        full_reduce = abs(self.inventory) >= int(self.b_config.passive_reduce_full)
        mode = "REDUCE_ONLY" if reduce_only else "B_UNDERLYING_MM_V2"
        action_class = "reduce_only" if reduce_only else "market_making"
        center = fair.reference_fair - (float(self.inventory) * float(self.b_config.inventory_skew_ticks_per_unit))
        bid_px = int(floor(center - fair.dynamic_half_spread))
        ask_px = int(ceil(center + fair.dynamic_half_spread))
        bid_px, ask_px = self._clamp_quote_prices(bid_px, ask_px, full_reduce=full_reduce)

        bid_qty = min(quote_size, allowed_buy)
        ask_qty = min(quote_size, allowed_sell)
        reduce_bonus = max(0, int(self.b_config.mm_v2_reduce_size_bonus))
        if self.inventory > 0:
            ask_qty = min(max(0, allowed_sell), quote_size + reduce_bonus)
        elif self.inventory < 0:
            bid_qty = min(max(0, allowed_buy), quote_size + reduce_bonus)

        if reduce_only:
            if self.inventory > 0:
                bid_qty = 0
            elif self.inventory < 0:
                ask_qty = 0

        self.last_mode = mode
        self.last_bid_px = bid_px if bid_qty > 0 else None
        self.last_ask_px = ask_px if ask_qty > 0 else None
        reason = fair.mode_reason
        return QuotePlan(
            mode=mode,
            bid=None if bid_qty <= 0 else self._desired("BUY", bid_px, bid_qty, reason, signal_id=signal_id, action_class=action_class),
            ask=None if ask_qty <= 0 else self._desired("SELL", ask_px, ask_qty, reason, signal_id=signal_id, action_class=action_class),
            aggressive_actions=(),
            observe_only=False,
            reason=reason,
        )

    def _book_is_healthy(self) -> bool:
        if self.book.best_bid is None or self.book.best_ask is None or self.book.spread is None:
            return False
        return int(self.book.spread) >= max(2, int(self.b_config.mm_min_valid_spread_ticks))

    def _bad_book_plan(self, *, now_ms: int) -> QuotePlan:
        assert self.book.best_bid is not None and self.book.best_ask is not None
        spread = int(self.book.spread or 0)
        reason = "b_book_crossed_or_locked" if spread <= 0 else "b_book_too_tight"
        base_center = self.book.microprice if self.book.microprice is not None else self.book.mid
        self.last_fair = BUnderlyingFairV2(
            base_center=None if base_center is None else float(base_center),
            reference_fair=None if base_center is None else float(base_center),
            composite_synthetic_fair=None,
            synthetic_dispersion=None,
            composite_basis=None,
            used_synthetic_anchor=False,
            dynamic_half_spread=None,
            mode_reason=reason,
        )
        self.last_block_reason = reason
        self.last_mode = "REDUCE_ONLY"
        if self.b_config.mm_cancel_on_bad_book:
            return QuotePlan("REDUCE_ONLY", None, None, (), True, reason)
        self.last_mode = "OBSERVE_ONLY"
        return QuotePlan("OBSERVE_ONLY", None, None, (), True, reason)

    def trace_state(self, now_ms: int) -> dict[str, Any]:
        allowed_buy, allowed_sell = self._allowed_sizes()
        return {
            "symbol": self.symbol,
            "market_key": "B",
            "mode": self.last_mode,
            "fair_value": None if self.last_fair.reference_fair is None else round(self.last_fair.reference_fair, 4),
            "trusted_multiplier": None,
            "multiplier_confidence": None,
            "latest_earnings": None,
            "inventory": self.inventory,
            "earnings_position": 0,
            "mm_position": self.inventory,
            "earnings_budget": 0,
            "mm_budget": self.b_config.max_position,
            "budget_shift_active": False,
            "unwind_active": False,
            "unwind_aggressive_active": False,
            "buy_exposure": self.order_manager.buy_exposure(),
            "sell_exposure": self.order_manager.sell_exposure(),
            "allowed_buy_size": allowed_buy,
            "allowed_sell_size": allowed_sell,
            "position_cap": self.b_config.max_position,
            "overlay_exposures": {},
            "live_orders": self.order_manager.live_orders_snapshot(),
            "book": {
                "best_bid_px": None if self.book.best_bid is None else self.book.best_bid.px,
                "best_bid_qty": None if self.book.best_bid is None else self.book.best_bid.qty,
                "best_ask_px": None if self.book.best_ask is None else self.book.best_ask.px,
                "best_ask_qty": None if self.book.best_ask is None else self.book.best_ask.qty,
                "spread": self.book.spread,
                "mid": self.book.mid,
                "microprice": self.book.microprice,
                "top_of_book_imbalance": self.book.top_of_book_imbalance,
                "bid_levels": [{"px": level.px, "qty": level.qty} for level in self.book.bid_levels],
                "ask_levels": [{"px": level.px, "qty": level.qty} for level in self.book.ask_levels],
            },
            "last_trade_px": self.last_trade_px,
            "last_trade_qty": self.last_trade_qty,
            "last_trade_ms": self.last_trade_ms,
            "exchange_tick": None,
            "news_caution_active": False,
            "news_caution_until_ms": 0,
            "news_caution_remaining_ms": 0,
            "current_earnings_signal_id": None,
            "current_news_signal_id": None,
            "composite_synthetic_fair": self.last_fair.composite_synthetic_fair,
            "synthetic_dispersion": self.last_fair.synthetic_dispersion,
            "composite_basis": self.last_fair.composite_basis,
            "block_reason": self.last_block_reason,
            "b_mm_v2_base_center": self.last_fair.base_center,
            "b_mm_v2_used_synthetic_anchor": self.last_fair.used_synthetic_anchor,
            "b_mm_v2_dynamic_half_spread": self.last_fair.dynamic_half_spread,
            "b_mm_v2_bid_px": self.last_bid_px,
            "b_mm_v2_ask_px": self.last_ask_px,
            "b_mm_v2_healthy_book_since_ms": self.healthy_book_since_ms,
        }

    def _fair_snapshot(self, residual_payload: dict[str, Any] | None) -> BUnderlyingFairV2:
        base_center = self.book.microprice if self.book.microprice is not None else self.book.mid
        if base_center is None or self.book.spread is None:
            return BUnderlyingFairV2(None, None, None, None, None, False, None, "missing_b_center")

        composite_synthetic_fair = None
        synthetic_dispersion = None
        composite_basis = None
        if residual_payload is not None:
            if residual_payload.get("composite_synthetic_fair") is not None:
                composite_synthetic_fair = float(residual_payload["composite_synthetic_fair"])
            if residual_payload.get("synthetic_dispersion") is not None:
                synthetic_dispersion = float(residual_payload["synthetic_dispersion"])
            else:
                synthetic_by_strike = residual_payload.get("synthetic_forward_by_strike") or {}
                values = [float(value) for value in synthetic_by_strike.values()]
                if values:
                    synthetic_dispersion = max(values) - min(values)
            if residual_payload.get("composite_basis") is not None:
                composite_basis = float(residual_payload["composite_basis"])

        max_dispersion = float(self.b_config.max_synthetic_dispersion)
        used_synth = (
            composite_synthetic_fair is not None
            and synthetic_dispersion is not None
            and synthetic_dispersion <= max_dispersion
        )
        if used_synth:
            book_weight = max(0.0, float(self.b_config.mm_v2_book_weight))
            synth_weight = max(0.0, float(self.b_config.mm_v2_synth_weight))
            total_weight = book_weight + synth_weight
            if total_weight <= 0:
                reference_fair = float(base_center)
            else:
                reference_fair = ((book_weight * float(base_center)) + (synth_weight * composite_synthetic_fair)) / total_weight
            reason = "b_underlying_mm_v2_synth_anchored"
        else:
            reference_fair = float(base_center)
            reason = "b_underlying_mm_v2_book_center"

        half_spread = max(
            int(self.b_config.mm_v2_min_half_spread_ticks),
            int(round(float(self.book.spread) / 2.0)) - int(self.b_config.mm_v2_inside_improve_ticks),
        )
        if synthetic_dispersion is not None and synthetic_dispersion > max_dispersion:
            extra = int(ceil((synthetic_dispersion - max_dispersion) * float(self.b_config.mm_v2_dispersion_widen_factor)))
            half_spread += max(0, extra)
            reason = f"{reason}_wide_dispersion"

        return BUnderlyingFairV2(
            base_center=float(base_center),
            reference_fair=float(reference_fair),
            composite_synthetic_fair=composite_synthetic_fair,
            synthetic_dispersion=synthetic_dispersion,
            composite_basis=composite_basis,
            used_synthetic_anchor=used_synth,
            dynamic_half_spread=max(1, int(half_spread)),
            mode_reason=reason,
        )

    def _allowed_sizes(self) -> tuple[int, int]:
        buy_exposure = self.order_manager.buy_exposure()
        sell_exposure = self.order_manager.sell_exposure()
        allowed_buy = max(0, int(self.b_config.max_position) - (self.inventory + buy_exposure))
        allowed_sell = max(0, int(self.b_config.max_position) + self.inventory - sell_exposure)
        return allowed_buy, allowed_sell

    def _clamp_quote_prices(self, bid_px: int, ask_px: int, *, full_reduce: bool) -> tuple[int, int]:
        assert self.book.best_bid is not None and self.book.best_ask is not None
        best_bid = self.book.best_bid.px
        best_ask = self.book.best_ask.px
        improve = max(0, int(self.b_config.mm_v2_inside_improve_ticks))
        bid_cap = min(best_ask - 1, best_bid + improve)
        ask_floor = max(best_bid + 1, best_ask - improve)
        bid_px = min(int(bid_px), bid_cap)
        ask_px = max(int(ask_px), ask_floor)
        if full_reduce:
            if self.inventory > 0:
                ask_px = max(best_bid + 1, min(ask_px, best_ask))
            elif self.inventory < 0:
                bid_px = min(best_ask - 1, max(bid_px, best_bid))
        if bid_px >= ask_px:
            midpoint = (best_bid + best_ask) / 2.0
            bid_px = min(best_ask - 1, int(floor(midpoint)))
            ask_px = max(best_bid + 1, int(ceil(midpoint)))
            if bid_px >= ask_px:
                bid_px = best_bid
                ask_px = best_ask
        return bid_px, ask_px

    def _desired(self, side: str, px: int, qty: int, reason: str, *, signal_id: str, action_class: str) -> DesiredOrder:
        return DesiredOrder(
            side=side,
            px=int(px),
            qty=int(qty),
            overlay="mm",
            aggressive=False,
            reason=reason,
            intent="b_underlying_mm_v2_passive",
            mode_at_submit=self.last_mode,
            evaluation_reason=reason,
            market_key="B",
            strategy_family="b_underlying_mm_v2",
            action_class=action_class,
            pnl_owner="b_underlying_mm_v2",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role="single",
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
