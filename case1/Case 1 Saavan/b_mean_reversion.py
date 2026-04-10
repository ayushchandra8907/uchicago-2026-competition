from __future__ import annotations

from dataclasses import dataclass
from math import exp
import time
from typing import Any

from a_bot_config import BConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


@dataclass(frozen=True)
class BMeanReversionSignal:
    ema_fast: float | None
    ema_slow: float | None
    mean_reference: float | None
    mean_reference_ema_component: float | None
    mean_reference_synth_component: float | None
    used_synthetic_reference: bool
    sigma: float | None
    z_score: float | None
    deviation_ticks: float | None
    target_inventory: int
    regime_block_reason: str | None


class BMeanReversionStrategy:
    """Risk-capped B underlying mean reversion using EMA and EWMA volatility."""

    def __init__(self, b_config: BConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.config = b_config
        self.risk = risk
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.symbol = b_config.underlying_symbol
        self.order_manager = OrderManager(symbol=self.symbol, risk=risk)
        self.book = BookSnapshot()
        self.inventory = 0
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.ewma_var: float | None = None
        self.last_mid: float | None = None
        self.last_update_ms: int | None = None
        self.healthy_book_since_ms: int | None = None
        self.last_bad_book_ms: int | None = None
        self.last_bad_book_fill_ms: int | None = None
        self.last_entry_ms: int | None = None
        self.position_entry_ms: int | None = None
        self.last_target_inventory = 0
        self.last_z_score: float | None = None
        self.last_sigma: float | None = None
        self.last_deviation_ticks: float | None = None
        self.last_mean_reference: float | None = None
        self.last_mean_reference_ema_component: float | None = None
        self.last_mean_reference_synth_component: float | None = None
        self.last_used_synthetic_reference = False
        self.last_fast_slope: float | None = None
        self.last_extension_ms: int | None = None
        self.last_abs_deviation: float | None = None
        self.last_deviation_sign: int = 0
        self.last_mode = "OBSERVE_ONLY"
        self.last_block_reason: str | None = None
        self.last_risk_off_forced = False
        self.last_entry_style: str | None = None
        self.last_signal = BMeanReversionSignal(None, None, None, None, None, False, None, None, None, 0, "not_started")

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != self.symbol:
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        if self._book_is_healthy():
            if self.healthy_book_since_ms is None:
                self.healthy_book_since_ms = int(now_ms)
        else:
            self.healthy_book_since_ms = None
            if self._book_is_bad():
                self.last_bad_book_ms = int(now_ms)
        if self.book.mid is not None:
            self._update_ema_state(float(self.book.mid), int(now_ms))
        return True

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)

    def sync_inventory_from_exchange(self, inventory: int) -> None:
        previous = self.inventory
        self.inventory = int(inventory)
        self._sync_position_clock(previous_inventory=previous, now_ms=self._now_ms())

    def on_fill(self, order_id: str, qty: int, price: int, *, authoritative_inventory: int | None = None) -> Any | None:
        previous = self.inventory
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        if authoritative_inventory is None:
            signed_qty = int(qty) if order.side == "BUY" else -int(qty)
            self.inventory += signed_qty
        else:
            self.inventory = int(authoritative_inventory)
        now_ms = self._now_ms()
        self._sync_position_clock(previous_inventory=previous, now_ms=now_ms)
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = now_ms
        if self._book_is_bad():
            self.last_bad_book_fill_ms = now_ms
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> Any | None:
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str) -> Any | None:
        return self.order_manager.handle_rejection(order_id)

    def compute_quotes(self, *, now_ms: int, residual_payload: dict[str, Any] | None) -> QuotePlan:
        self.last_block_reason = None
        self.last_risk_off_forced = False
        self.last_entry_style = None
        signal_id = f"b_meanrev_{int(now_ms)}"

        if self.book.best_bid is None or self.book.best_ask is None or self.book.spread is None or self.book.mid is None:
            return self._observe("missing_b_book")

        mean_reference = self._compute_mean_reference(residual_payload)
        z = self._current_z(mean_reference)
        if z is None:
            return self._observe("warming_up_b_meanrev")

        deviation = self._current_deviation(mean_reference)
        target, mode, action_class, reason, aggressive = self._target_and_mode(z, deviation, now_ms, residual_payload)
        self.last_target_inventory = target
        self.last_signal = BMeanReversionSignal(
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            mean_reference=self.last_mean_reference,
            mean_reference_ema_component=self.last_mean_reference_ema_component,
            mean_reference_synth_component=self.last_mean_reference_synth_component,
            used_synthetic_reference=self.last_used_synthetic_reference,
            sigma=self.last_sigma,
            z_score=z,
            deviation_ticks=deviation,
            target_inventory=target,
            regime_block_reason=self.last_block_reason,
        )
        self.last_mode = mode

        delta = target - int(self.inventory)
        if delta == 0:
            return QuotePlan(mode, None, None, (), True, reason)

        side = "BUY" if delta > 0 else "SELL"
        qty = min(abs(delta), max(1, int(self.config.meanrev_quote_size)))
        if qty <= 0:
            return self._observe("zero_b_meanrev_quote_size")

        order = self._desired(
            side=side,
            px=self._execution_price(side, aggressive=aggressive),
            qty=qty,
            reason=reason,
            signal_id=signal_id,
            action_class=action_class,
            aggressive=aggressive,
        )
        if aggressive:
            return QuotePlan(mode, None, None, (order,), False, reason)
        if side == "BUY":
            return QuotePlan(mode, order, None, (), False, reason)
        return QuotePlan(mode, None, order, (), False, reason)

    def trace_state(self, now_ms: int) -> dict[str, Any]:
        allowed_buy, allowed_sell = self._allowed_sizes()
        hold_ms = None if self.position_entry_ms is None or self.inventory == 0 else int(now_ms) - int(self.position_entry_ms)
        return {
            "symbol": self.symbol,
            "market_key": "B",
            "mode": self.last_mode,
            "fair_value": self.ema_slow,
            "inventory": self.inventory,
            "earnings_position": 0,
            "mm_position": self.inventory,
            "earnings_budget": 0,
            "mm_budget": self.config.meanrev_max_position,
            "budget_shift_active": False,
            "unwind_active": self.last_mode in {"B_MEANREV_EXIT", "B_MEANREV_RISK_OFF"},
            "unwind_aggressive_active": self.last_mode == "B_MEANREV_RISK_OFF" and self.last_risk_off_forced,
            "buy_exposure": self.order_manager.buy_exposure(),
            "sell_exposure": self.order_manager.sell_exposure(),
            "allowed_buy_size": allowed_buy,
            "allowed_sell_size": allowed_sell,
            "position_cap": self.config.meanrev_max_position,
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
            "block_reason": self.last_block_reason,
            "b_meanrev_ema_fast": self.ema_fast,
            "b_meanrev_ema_slow": self.ema_slow,
            "b_meanrev_mean_reference": self.last_mean_reference,
            "b_meanrev_mean_reference_ema_component": self.last_mean_reference_ema_component,
            "b_meanrev_mean_reference_synth_component": self.last_mean_reference_synth_component,
            "b_meanrev_used_synthetic_reference": self.last_used_synthetic_reference,
            "b_meanrev_sigma": self.last_sigma,
            "b_meanrev_z": self.last_z_score,
            "b_meanrev_deviation_ticks": self.last_deviation_ticks,
            "b_meanrev_healthy_book_age_ms": (
                None if self.healthy_book_since_ms is None else max(0, int(now_ms) - int(self.healthy_book_since_ms))
            ),
            "b_meanrev_last_bad_book_fill_age_ms": (
                None if self.last_bad_book_fill_ms is None else max(0, int(now_ms) - int(self.last_bad_book_fill_ms))
            ),
            "b_meanrev_turn_confirmed": self._turn_confirmed(self.last_deviation_ticks, int(now_ms)),
            "b_meanrev_target_inventory": self.last_target_inventory,
            "b_meanrev_hold_ms": hold_ms,
            "b_meanrev_regime_block_reason": self.last_block_reason,
            "b_meanrev_risk_off_forced": self.last_risk_off_forced,
            "b_meanrev_entry_style": self.last_entry_style,
        }

    def _target_and_mode(
        self,
        z: float,
        deviation: float | None,
        now_ms: int,
        residual_payload: dict[str, Any] | None,
    ) -> tuple[int, str, str, str, bool]:
        abs_z = abs(z)
        abs_deviation = abs(float(deviation or 0.0))
        book_block = self._book_entry_block(now_ms)
        if book_block is not None:
            self.last_block_reason = book_block
            return int(self.inventory), "OBSERVE_ONLY", "observe_only", book_block, False

        risk_off = (
            abs_z >= float(self.config.meanrev_stop_z)
            or abs_deviation >= float(self.config.meanrev_risk_off_deviation_ticks)
        )
        if risk_off:
            self.last_block_reason = "b_meanrev_stop_z"
            force_exit = abs(int(self.inventory)) >= int(self.config.meanrev_max_position)
            self.last_risk_off_forced = force_exit
            if self.inventory == 0:
                return (
                    0,
                    "B_MEANREV_RISK_OFF",
                    "observe_only",
                    "B mean-reversion stop-z risk-off; no new entry",
                    False,
                )
            if force_exit:
                return (
                    0,
                    "B_MEANREV_RISK_OFF",
                    "mean_reversion_risk_off",
                    "B mean-reversion stop-z risk-off forced reduce at max inventory",
                    bool(self.config.meanrev_aggressive_exit),
                )
            return (
                0,
                "B_MEANREV_RISK_OFF",
                "mean_reversion_risk_off",
                "B mean-reversion stop-z risk-off passive reduce",
                False,
            )

        if self._max_hold_elapsed(now_ms):
            self.last_block_reason = "b_meanrev_max_hold_elapsed"
            return 0, "B_MEANREV_EXIT", "mean_reversion_exit", "B mean-reversion max hold exit", bool(self.config.meanrev_aggressive_exit)

        if abs_deviation <= float(self.config.meanrev_exit_ticks) and self.inventory != 0:
            return 0, "B_MEANREV_EXIT", "mean_reversion_exit", "B mean reverted inside exit band", False

        residual_block = self._residual_entry_block(deviation, residual_payload)
        if residual_block is not None:
            self.last_block_reason = residual_block
            if self.inventory != 0:
                return 0, "B_MEANREV_EXIT", "mean_reversion_exit", f"{residual_block}; reducing B mean-reversion inventory", False
            return 0, "OBSERVE_ONLY", "observe_only", residual_block, False

        if self._recent_bad_fill_blocks_entry(now_ms):
            self.last_block_reason = "b_meanrev_recent_bad_book_fill"
            if self.inventory != 0:
                return 0, "B_MEANREV_EXIT", "mean_reversion_exit", "recent bad-book fill; reducing B inventory", False
            return 0, "OBSERVE_ONLY", "observe_only", "recent bad-book fill cooldown", False

        if abs_deviation < float(self.config.meanrev_entry_ticks):
            if self.inventory != 0:
                return 0, "B_MEANREV_EXIT", "mean_reversion_exit", "B mean-reversion signal below entry; reducing", False
            return 0, "OBSERVE_ONLY", "observe_only", "B mean-reversion waiting for z-score entry", False

        if (
            self.inventory == 0
            and abs_deviation < float(self.config.meanrev_extreme_entry_ticks)
            and not self._turn_confirmed(deviation, now_ms)
        ):
            self.last_block_reason = "b_meanrev_waiting_for_turn_confirmation"
            return 0, "OBSERVE_ONLY", "observe_only", "B mean-reversion waiting for turn confirmation", False

        if self.inventory == 0 and self.last_entry_ms is not None:
            cooldown_remaining = int(self.config.meanrev_cooldown_ms) - (int(now_ms) - int(self.last_entry_ms))
            if cooldown_remaining > 0:
                self.last_block_reason = "b_meanrev_entry_cooldown"
                return 0, "OBSERVE_ONLY", "observe_only", "B mean-reversion entry cooldown", False

        direction = -1 if float(deviation or 0.0) > 0 else 1
        size = (
            int(self.config.meanrev_full_target)
            if abs_deviation >= float(self.config.meanrev_full_entry_ticks)
            else int(self.config.meanrev_base_target)
        )
        target = direction * max(0, min(int(self.config.meanrev_max_position), size))
        if abs_deviation >= float(self.config.meanrev_extreme_entry_ticks):
            aggressive = True
            entry_style = "extreme"
            reason = "B mean-reversion extreme tick-deviation entry"
        elif abs_deviation >= float(self.config.meanrev_full_entry_ticks):
            aggressive = self.inventory == 0
            entry_style = "full"
            reason = "B mean-reversion full tick-deviation entry"
        else:
            aggressive = False
            entry_style = "normal"
            reason = "B mean-reversion tick-deviation entry"
        self.last_entry_style = entry_style
        return target, "B_MEANREV_ENTRY", "mean_reversion_entry", reason, aggressive

    def _book_entry_block(self, now_ms: int) -> str | None:
        if self.book.best_bid is None or self.book.best_ask is None or self.book.spread is None:
            return "missing_b_book"
        if int(self.book.spread) <= 0:
            return "b_meanrev_book_crossed_or_locked"
        if int(self.book.spread) < int(self.config.meanrev_min_spread_ticks):
            return "b_meanrev_book_too_tight"
        if self.healthy_book_since_ms is None:
            return "b_meanrev_waiting_for_healthy_book"
        healthy_age = int(now_ms) - int(self.healthy_book_since_ms)
        if healthy_age < int(self.config.meanrev_min_healthy_book_age_ms):
            return "b_meanrev_waiting_for_healthy_book"
        return None

    def _residual_entry_block(self, deviation: float | None, residual_payload: dict[str, Any] | None) -> str | None:
        if residual_payload is None:
            return None
        dispersion = residual_payload.get("synthetic_dispersion")
        if dispersion is not None and float(dispersion) > float(self.config.max_synthetic_dispersion):
            return "b_meanrev_synthetic_dispersion_wide"
        basis = residual_payload.get("composite_basis")
        if basis is not None:
            basis_value = float(basis)
            deviation_sign = 1 if float(deviation or 0.0) > 0 else (-1 if float(deviation or 0.0) < 0 else 0)
            if deviation_sign != 0 and abs(basis_value) >= float(self.config.basis_strong_threshold_ticks) and (basis_value > 0) == (deviation_sign > 0):
                return "b_meanrev_synthetic_confirms_deviation"
        return None

    def _execution_price(self, side: str, *, aggressive: bool) -> int:
        assert self.book.best_bid is not None and self.book.best_ask is not None
        if aggressive:
            return int(self.book.best_ask.px if side == "BUY" else self.book.best_bid.px)
        if side == "BUY":
            return min(int(self.book.best_ask.px) - 1, int(self.book.best_bid.px) + 1)
        return max(int(self.book.best_bid.px) + 1, int(self.book.best_ask.px) - 1)

    def _desired(
        self,
        *,
        side: str,
        px: int,
        qty: int,
        reason: str,
        signal_id: str,
        action_class: str,
        aggressive: bool,
    ) -> DesiredOrder:
        return DesiredOrder(
            side=side,  # type: ignore[arg-type]
            px=int(px),
            qty=int(qty),
            overlay="mm",
            aggressive=bool(aggressive),
            reason=reason,
            intent="b_mean_reversion",
            mode_at_submit=self.last_mode,
            evaluation_reason=reason,
            market_key="B",
            strategy_family="b_mean_reversion",
            action_class=action_class,
            pnl_owner="b_mean_reversion",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role="single",
        )

    def _observe(self, reason: str) -> QuotePlan:
        self.last_mode = "OBSERVE_ONLY"
        self.last_block_reason = reason
        self.last_target_inventory = 0
        self.last_signal = BMeanReversionSignal(
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            mean_reference=self.last_mean_reference,
            mean_reference_ema_component=self.last_mean_reference_ema_component,
            mean_reference_synth_component=self.last_mean_reference_synth_component,
            used_synthetic_reference=self.last_used_synthetic_reference,
            sigma=self.last_sigma,
            z_score=self.last_z_score,
            deviation_ticks=self.last_deviation_ticks,
            target_inventory=0,
            regime_block_reason=reason,
        )
        return QuotePlan("OBSERVE_ONLY", None, None, (), True, reason)

    def _current_z(self, mean_reference: float | None) -> float | None:
        if self.book.mid is None or mean_reference is None:
            self.last_z_score = None
            return None
        sigma = max(float(self.config.meanrev_sigma_floor), float(self.last_sigma or 0.0))
        if sigma <= 0:
            self.last_z_score = None
            return None
        self.last_sigma = sigma
        self.last_z_score = (float(self.book.mid) - float(mean_reference)) / sigma
        return self.last_z_score

    def _current_deviation(self, mean_reference: float | None) -> float | None:
        if self.book.mid is None or mean_reference is None:
            self.last_deviation_ticks = None
            return None
        self.last_deviation_ticks = float(self.book.mid) - float(mean_reference)
        return self.last_deviation_ticks

    def _compute_mean_reference(self, residual_payload: dict[str, Any] | None) -> float | None:
        self.last_mean_reference = self.ema_slow
        self.last_mean_reference_ema_component = self.ema_slow
        self.last_mean_reference_synth_component = None
        self.last_used_synthetic_reference = False
        if self.ema_slow is None:
            return None
        if residual_payload is None:
            return self.ema_slow
        synthetic = residual_payload.get("composite_synthetic_fair")
        dispersion = residual_payload.get("synthetic_dispersion")
        basis = residual_payload.get("composite_basis")
        if (
            synthetic is not None
            and dispersion is not None
            and float(dispersion) <= float(self.config.max_synthetic_dispersion)
            and abs(float(basis or 0.0)) <= 6.0
        ):
            synth_value = float(synthetic)
            ema_component = 0.65 * float(self.ema_slow)
            synth_component = 0.35 * synth_value
            self.last_mean_reference_ema_component = ema_component
            self.last_mean_reference_synth_component = synth_component
            self.last_mean_reference = ema_component + synth_component
            self.last_used_synthetic_reference = True
        return self.last_mean_reference

    def _update_ema_state(self, mid: float, now_ms: int) -> None:
        if self.last_update_ms is None or self.last_mid is None:
            self.ema_fast = mid
            self.ema_slow = mid
            self.ewma_var = float(self.config.meanrev_sigma_floor) ** 2
            self.last_sigma = float(self.config.meanrev_sigma_floor)
            self.last_mid = mid
            self.last_update_ms = int(now_ms)
            self.last_z_score = 0.0
            self.last_deviation_ticks = 0.0
            self.last_abs_deviation = 0.0
            self.last_deviation_sign = 0
            self.last_extension_ms = int(now_ms)
            return

        elapsed = max(1, int(now_ms) - int(self.last_update_ms))
        fast_alpha = 1.0 - exp(-elapsed / max(1.0, float(self.config.meanrev_ema_fast_ms)))
        slow_alpha = 1.0 - exp(-elapsed / max(1.0, float(self.config.meanrev_ema_slow_ms)))
        vol_alpha = 1.0 - exp(-elapsed / max(1.0, float(self.config.meanrev_vol_ewma_ms)))
        prior_ema_fast = self.ema_fast
        self.ema_fast = mid if self.ema_fast is None else (fast_alpha * mid) + ((1.0 - fast_alpha) * self.ema_fast)
        self.ema_slow = mid if self.ema_slow is None else (slow_alpha * mid) + ((1.0 - slow_alpha) * self.ema_slow)
        self.last_fast_slope = None if prior_ema_fast is None else float(self.ema_fast) - float(prior_ema_fast)
        ret = mid - float(self.last_mid)
        prior_var = float(self.ewma_var if self.ewma_var is not None else float(self.config.meanrev_sigma_floor) ** 2)
        self.ewma_var = max(0.0, (vol_alpha * ret * ret) + ((1.0 - vol_alpha) * prior_var))
        self.last_sigma = self.ewma_var ** 0.5
        self.last_mid = mid
        self.last_update_ms = int(now_ms)
        deviation = None if self.ema_slow is None else float(mid) - float(self.ema_slow)
        if deviation is not None:
            sign = 1 if deviation > 0 else (-1 if deviation < 0 else 0)
            abs_deviation = abs(deviation)
            extending = (
                sign != 0
                and sign == self.last_deviation_sign
                and self.last_abs_deviation is not None
                and abs_deviation > float(self.last_abs_deviation) + 0.25
            )
            if sign != self.last_deviation_sign or extending or self.last_extension_ms is None:
                self.last_extension_ms = int(now_ms)
            self.last_deviation_sign = sign
            self.last_abs_deviation = abs_deviation
            self.last_deviation_ticks = deviation

    def _allowed_sizes(self) -> tuple[int, int]:
        buy_exposure = self.order_manager.buy_exposure()
        sell_exposure = self.order_manager.sell_exposure()
        cap = int(self.config.meanrev_max_position)
        allowed_buy = max(0, cap - (self.inventory + buy_exposure))
        allowed_sell = max(0, cap + self.inventory - sell_exposure)
        return allowed_buy, allowed_sell

    def _max_hold_elapsed(self, now_ms: int) -> bool:
        return (
            self.inventory != 0
            and self.position_entry_ms is not None
            and int(now_ms) - int(self.position_entry_ms) >= int(self.config.meanrev_max_hold_ms)
        )

    def _sync_position_clock(self, *, previous_inventory: int, now_ms: int) -> None:
        if previous_inventory == 0 and self.inventory != 0:
            self.position_entry_ms = int(now_ms)
        elif self.inventory == 0:
            if previous_inventory != 0:
                self.last_entry_ms = int(now_ms)
            self.position_entry_ms = None

    def _book_is_healthy(self) -> bool:
        return (
            self.book.best_bid is not None
            and self.book.best_ask is not None
            and self.book.spread is not None
            and int(self.book.spread) >= int(self.config.meanrev_min_spread_ticks)
        )

    def _book_is_bad(self) -> bool:
        return (
            self.book.best_bid is None
            or self.book.best_ask is None
            or self.book.spread is None
            or int(self.book.spread) <= 1
        )

    def should_force_bad_book_cancel(self) -> bool:
        if not self._book_is_bad():
            return False
        for side in ("BUY", "SELL"):
            order = self.order_manager.live_order(side)  # type: ignore[arg-type]
            if order is not None and not order.cancel_pending:
                return True
        return False

    def _recent_bad_fill_blocks_entry(self, now_ms: int) -> bool:
        return (
            self.last_bad_book_fill_ms is not None
            and int(now_ms) - int(self.last_bad_book_fill_ms) < int(self.config.meanrev_bad_fill_cooldown_ms)
        )

    def _turn_confirmed(self, deviation: float | None, now_ms: int) -> bool:
        if deviation is None:
            return False
        sign = 1 if deviation > 0 else (-1 if deviation < 0 else 0)
        if sign == 0:
            return True
        if self.last_fast_slope is not None and (float(self.last_fast_slope) * sign) <= 0:
            return True
        if self.last_extension_ms is None:
            return False
        return int(now_ms) - int(self.last_extension_ms) >= int(self.config.meanrev_turn_confirm_ms)

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
