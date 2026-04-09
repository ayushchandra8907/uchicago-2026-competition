from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
import time
from typing import Any

from a_bot_config import BConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


@dataclass(frozen=True)
class BFairSnapshot:
    reference_fair: int | None
    composite_synthetic_fair: float | None
    synthetic_dispersion: float | None
    block_reason: str | None
    reduce_only: bool


class BUnderlyingMMStrategy:
    """Small, underlying-only B market maker anchored to option-implied fair."""

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
        self.last_reference_fair: int | None = None
        self.last_composite_synthetic_fair: float | None = None
        self.last_synthetic_dispersion: float | None = None
        self.last_block_reason: str | None = None
        self.last_mode = "OBSERVE_ONLY"

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != self.symbol:
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        return True

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)

    def sync_inventory_from_exchange(self, inventory: int) -> None:
        self.inventory = int(inventory)

    def on_fill(self, order_id: str, qty: int, price: int) -> Any | None:
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        signed_qty = qty if order.side == "BUY" else -qty
        self.inventory += signed_qty
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms()
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> Any | None:
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str) -> Any | None:
        return self.order_manager.handle_rejection(order_id)

    def compute_quotes(self, *, now_ms: int, residual_payload: dict[str, Any] | None) -> QuotePlan:
        signal_id = f"b_mm_{int(now_ms)}"
        fair_snapshot = self._fair_snapshot(residual_payload)
        self.last_reference_fair = fair_snapshot.reference_fair
        self.last_composite_synthetic_fair = fair_snapshot.composite_synthetic_fair
        self.last_synthetic_dispersion = fair_snapshot.synthetic_dispersion
        self.last_block_reason = fair_snapshot.block_reason

        spread = self.book.spread
        if spread is None:
            self.last_mode = "OBSERVE_ONLY"
            return QuotePlan(
                mode="OBSERVE_ONLY",
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="missing_b_book",
            )

        if spread < self.b_config.min_book_spread:
            self.last_mode = "OBSERVE_ONLY"
            return QuotePlan(
                mode="OBSERVE_ONLY",
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="book_spread_too_tight",
            )

        if fair_snapshot.block_reason is not None and self.inventory == 0:
            self.last_mode = "OBSERVE_ONLY"
            return QuotePlan(
                mode="OBSERVE_ONLY",
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason=fair_snapshot.block_reason,
            )

        anchor = self._reduce_anchor(fair_snapshot.reference_fair)
        if anchor is None:
            self.last_mode = "OBSERVE_ONLY"
            return QuotePlan(
                mode="OBSERVE_ONLY",
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="missing_reference_anchor",
            )

        if fair_snapshot.block_reason is not None and self.inventory != 0:
            self.last_mode = "REDUCE_ONLY"
            return self._reduce_only_plan(anchor, reason=f"reduce_only_{fair_snapshot.block_reason}", signal_id=signal_id)

        self.last_mode = "UNDERLYING_MM"
        return self._market_make_plan(fair_snapshot.reference_fair or round(anchor), reason="b_underlying_mm", signal_id=signal_id)

    def trace_state(self, now_ms: int) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_key": "B",
            "mode": self.last_mode,
            "fair_value": self.last_reference_fair,
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
            "allowed_buy_size": self._allowed_sizes()[0],
            "allowed_sell_size": self._allowed_sizes()[1],
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
            "composite_synthetic_fair": self.last_composite_synthetic_fair,
            "synthetic_dispersion": self.last_synthetic_dispersion,
            "block_reason": self.last_block_reason,
        }

    def _fair_snapshot(self, residual_payload: dict[str, Any] | None) -> BFairSnapshot:
        microprice = self.book.microprice if self.book.microprice is not None else self.book.mid
        if residual_payload is None:
            return BFairSnapshot(
                reference_fair=None,
                composite_synthetic_fair=None,
                synthetic_dispersion=None,
                block_reason="missing_composite_synthetic_fair",
                reduce_only=self.inventory != 0,
            )

        composite_synthetic_fair = residual_payload.get("composite_synthetic_fair")
        synthetic_forward_by_strike = residual_payload.get("synthetic_forward_by_strike") or {}
        synthetic_values = [float(value) for value in synthetic_forward_by_strike.values()]
        synthetic_dispersion = None
        if synthetic_values:
            synthetic_dispersion = max(synthetic_values) - min(synthetic_values)

        if composite_synthetic_fair is None or microprice is None:
            return BFairSnapshot(
                reference_fair=None,
                composite_synthetic_fair=None if composite_synthetic_fair is None else float(composite_synthetic_fair),
                synthetic_dispersion=synthetic_dispersion,
                block_reason="missing_composite_synthetic_fair",
                reduce_only=self.inventory != 0,
            )

        if synthetic_dispersion is not None and synthetic_dispersion > self.b_config.max_synthetic_dispersion:
            return BFairSnapshot(
                reference_fair=None,
                composite_synthetic_fair=float(composite_synthetic_fair),
                synthetic_dispersion=synthetic_dispersion,
                block_reason="synthetic_dispersion_wide",
                reduce_only=self.inventory != 0,
            )

        reference_fair = round((0.5 * float(microprice)) + (0.5 * float(composite_synthetic_fair)))
        return BFairSnapshot(
            reference_fair=reference_fair,
            composite_synthetic_fair=float(composite_synthetic_fair),
            synthetic_dispersion=synthetic_dispersion,
            block_reason=None,
            reduce_only=False,
        )

    def _allowed_sizes(self) -> tuple[int, int]:
        buy_exposure = self.order_manager.buy_exposure()
        sell_exposure = self.order_manager.sell_exposure()
        allowed_buy = max(0, self.b_config.max_position - (self.inventory + buy_exposure))
        allowed_sell = max(0, self.b_config.max_position + self.inventory - sell_exposure)
        return allowed_buy, allowed_sell

    def _reduce_anchor(self, reference_fair: int | None) -> float | None:
        if reference_fair is not None:
            return float(reference_fair)
        if self.book.microprice is not None:
            return float(self.book.microprice)
        if self.book.mid is not None:
            return float(self.book.mid)
        return None

    def _reduce_only_plan(self, anchor: float, *, reason: str, signal_id: str) -> QuotePlan:
        allowed_buy, allowed_sell = self._allowed_sizes()
        quote_size = max(0, int(self.b_config.quote_size))
        half_spread = max(0, int(self.b_config.base_half_spread_ticks))
        if self.inventory > 0 and allowed_sell > 0:
            ask_px = int(ceil(anchor + half_spread))
            if self.book.best_bid is not None:
                ask_px = max(ask_px, self.book.best_bid.px + 1)
            if self.book.best_ask is not None:
                ask_px = max(ask_px, self.book.best_ask.px)
            return QuotePlan(
                mode="REDUCE_ONLY",
                bid=None,
                ask=self._desired("SELL", ask_px, min(quote_size, allowed_sell), "reduce-only ask for B inventory cleanup", signal_id=signal_id),
                aggressive_actions=(),
                observe_only=False,
                reason=reason,
            )
        if self.inventory < 0 and allowed_buy > 0:
            bid_px = int(floor(anchor - half_spread))
            if self.book.best_ask is not None:
                bid_px = min(bid_px, self.book.best_ask.px - 1)
            if self.book.best_bid is not None:
                bid_px = min(bid_px, self.book.best_bid.px)
            return QuotePlan(
                mode="REDUCE_ONLY",
                bid=self._desired("BUY", bid_px, min(quote_size, allowed_buy), "reduce-only bid for B inventory cleanup", signal_id=signal_id),
                ask=None,
                aggressive_actions=(),
                observe_only=False,
                reason=reason,
            )
        return QuotePlan(mode="REDUCE_ONLY", bid=None, ask=None, aggressive_actions=(), observe_only=True, reason=reason)

    def _market_make_plan(self, reference_fair: int, *, reason: str, signal_id: str) -> QuotePlan:
        allowed_buy, allowed_sell = self._allowed_sizes()
        quote_size = max(0, int(self.b_config.quote_size))
        inventory = int(self.inventory)
        if abs(inventory) >= self.b_config.passive_reduce_full:
            return self._reduce_only_plan(float(reference_fair), reason="reduce_only_full_inventory", signal_id=signal_id)

        center = float(reference_fair) - (float(self.b_config.inventory_skew_ticks_per_unit) * float(inventory))
        bid_px = int(floor(center - self.b_config.base_half_spread_ticks))
        ask_px = int(ceil(center + self.b_config.base_half_spread_ticks))
        if self.book.best_ask is not None:
            bid_px = min(bid_px, self.book.best_ask.px - 1)
        if self.book.best_bid is not None:
            ask_px = max(ask_px, self.book.best_bid.px + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_qty = min(quote_size, allowed_buy)
        ask_qty = min(quote_size, allowed_sell)
        if inventory >= self.b_config.passive_reduce_start:
            bid_qty = 0
        elif inventory <= -self.b_config.passive_reduce_start:
            ask_qty = 0

        return QuotePlan(
            mode="UNDERLYING_MM",
            bid=None if bid_qty <= 0 else self._desired("BUY", bid_px, bid_qty, "passive B bid around composite fair", signal_id=signal_id),
            ask=None if ask_qty <= 0 else self._desired("SELL", ask_px, ask_qty, "passive B ask around composite fair", signal_id=signal_id),
            aggressive_actions=(),
            observe_only=False,
            reason=reason,
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def _desired(self, side: str, px: int, qty: int, reason: str, *, signal_id: str) -> DesiredOrder:
        return DesiredOrder(
            side=side,
            px=int(px),
            qty=int(qty),
            overlay="mm",
            aggressive=False,
            reason=reason,
            intent="b_underlying_mm_passive",
            mode_at_submit=self.last_mode,
            evaluation_reason=reason,
            market_key="B",
            strategy_family="b_underlying_mm",
            action_class="market_making",
            pnl_owner="b_underlying_mm",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role="single",
        )
