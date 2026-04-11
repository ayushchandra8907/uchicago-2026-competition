from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

from a_bot_config import ETFConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, NewsReaction, OrderManager, QuotePlan


@dataclass(frozen=True)
class ETFShockProjection:
    source_market: str
    source_kind: str
    source_signal_id: str | None
    fair_shift_ticks: float
    alpha: float
    source_target_inventory: int | None = None
    source_combo: str = "A_only"
    target_inventory_override: int | None = None
    source_direction: int | None = None


@dataclass(frozen=True)
class ETFASignal:
    signal_id: str
    source_market: str
    source_signal_id: str | None
    source_combo: str
    source_kind: str
    started_ms: int
    alpha: float
    source_fair_shift: float
    a_fair_shift: float
    projected_etf_shift: float
    base_mid: float
    target_fair: float
    target_inventory: int
    source_target_inventory: int | None
    target_from_source_position: int | None
    source_direction: int

    @property
    def target_from_a_position(self) -> int | None:
        return self.target_from_source_position


@dataclass
class ETFEntryDiagnostics:
    order_attempt_count: int = 0
    first_order_attempt_ms: int | None = None
    first_fill_ms: int | None = None
    terminal_reason: str | None = None


class ETFAFollowerStrategy:
    """Directional ETF overlay that follows A shock signals with a tunable damped beta.

    The first-pass strategy is intentionally simple: A remains the alpha source,
    ETF trades only while an A event implies a large enough fair shift, and the
    alpha is exposed/configured so post-run calibration can tell us whether to
    scale ETF exposure up or down.
    """

    def __init__(self, etf_config: ETFConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.config = etf_config
        self.symbol = etf_config.symbol
        self.risk = risk
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.order_manager = OrderManager(symbol=self.symbol, risk=risk)
        self.book = BookSnapshot()
        self.inventory = 0
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.mode = "ETF_OBSERVE_ONLY"
        self.active_signal: ETFASignal | None = None
        self.pending_signal: ETFASignal | None = None
        self._signal_seq = 0
        self.last_block_reason: str | None = None
        self.unwind_reason: str | None = None
        self.last_fill_ms: int | None = None
        self.entry_diagnostics: dict[str, ETFEntryDiagnostics] = {}
        self._top_of_book_change_ms: deque[int] = deque()
        self._last_top_of_book: tuple[int | None, int | None, int | None, int | None] | None = None
        self._stable_book_since_ms: int | None = None
        self._churn_guard_reason: str | None = None

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != self.symbol:
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        self._update_churn_guard(now_ms=int(now_ms))
        return True

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)

    def sync_inventory_from_exchange(self, inventory: int, *, now_ms: int | None = None) -> bool:
        incoming = int(inventory)
        event_ms = self._now_ms() if now_ms is None else int(now_ms)
        if (
            self.last_fill_ms is not None
            and event_ms - int(self.last_fill_ms) <= 250
            and self.inventory != 0
            and incoming != 0
            and (incoming > 0) != (self.inventory > 0)
        ):
            self.last_block_reason = "ignored_stale_etf_position_flip"
            return False
        self.inventory = int(inventory)
        return True

    def on_fill(
        self,
        order_id: str,
        qty: int,
        price: int,
        *,
        authoritative_inventory: int | None = None,
        now_ms: int | None = None,
    ) -> Any | None:
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        if authoritative_inventory is None:
            signed_qty = int(qty) if order.side == "BUY" else -int(qty)
            self.inventory += signed_qty
        else:
            self.inventory = int(authoritative_inventory)
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)
        self.last_fill_ms = self.last_trade_ms
        if self.active_signal is not None:
            diag = self.entry_diagnostics.setdefault(self.active_signal.signal_id, ETFEntryDiagnostics())
            if order.action_class == "etf_shock_take" and diag.first_fill_ms is None:
                diag.first_fill_ms = self.last_fill_ms
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> Any | None:
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str) -> Any | None:
        return self.order_manager.handle_rejection(order_id)

    def on_a_news_reaction(self, reaction: NewsReaction, *, now_ms: int) -> ETFASignal | None:
        if not self.config.enabled or not self.config.trading_enabled or not reaction.relevant:
            return None
        if reaction.earnings_value is not None:
            source_kind = "structured_earnings"
            source_signal_id = None if reaction.tick is None else f"a_eps_tick_{reaction.tick}"
            old_fair = reaction.old_fair_value
            new_fair = reaction.new_fair_value
            alpha = self.config.alpha_from_a_earnings
        else:
            source_kind = "unstructured_news"
            source_signal_id = reaction.active_news_signal_id
            if str(reaction.news_confirmation_state or "").lower() == "pending":
                self.last_block_reason = "a_news_waiting_for_confirmation"
                return None
            old_fair = reaction.base_fair_value if reaction.base_fair_value is not None else reaction.old_fair_value
            new_fair = reaction.news_fair_value if reaction.news_fair_value is not None else reaction.new_fair_value
            alpha = self.config.alpha_from_a_news

        if old_fair is None or new_fair is None:
            self.last_block_reason = "missing_a_fair_shift"
            return None
        a_fair_shift = float(new_fair) - float(old_fair)
        if abs(a_fair_shift) < max(0, int(self.config.min_a_fair_shift_ticks)):
            self.last_block_reason = "a_fair_shift_too_small"
            return None
        source_target = (
            reaction.shock_target_inventory
            if reaction.shock_target_inventory is not None
            else (reaction.news_target_inventory or reaction.pending_news_target_inventory)
        )
        projection = ETFShockProjection(
            source_market="A",
            source_kind=source_kind,
            source_signal_id=source_signal_id,
            fair_shift_ticks=a_fair_shift,
            alpha=self.config.alpha_from_a if alpha is None else alpha,
            source_target_inventory=source_target,
            source_combo="A_only",
            source_direction=reaction.shock_direction or None,
        )
        signal = self.on_shock_projection(projection, now_ms=now_ms)
        if signal is None:
            return None
        return self.activate_signal(signal)

    def activate_signal(self, signal: ETFASignal) -> ETFASignal:
        active = self.active_signal
        if (
            active is not None
            and active.source_direction != signal.source_direction
            and (self.inventory != 0 or self._has_live_orders())
        ):
            active_diag = self.entry_diagnostics.setdefault(active.signal_id, ETFEntryDiagnostics())
            if active_diag.terminal_reason is None:
                active_diag.terminal_reason = "handoff_flatten"
            self.pending_signal = signal
            self.mode = "ETF_HANDOFF_FLATTEN"
            self.last_block_reason = "etf_signal_handoff_pending"
            self.unwind_reason = "handoff_flatten"
            return signal
        self.pending_signal = None
        self.active_signal = signal
        self.mode = "ETF_A_SHOCK"
        self.last_block_reason = None
        self.unwind_reason = None
        return signal

    def on_shock_projection(self, projection: ETFShockProjection, *, now_ms: int) -> ETFASignal | None:
        if not self.config.enabled or not self.config.trading_enabled:
            return None
        preview = self.preview_projection(projection)
        if preview is None:
            return None
        fair_shift = float(preview["fair_shift"])
        chosen_alpha = float(preview["alpha"])
        projected_shift = float(preview["projected_shift"])
        direction = int(preview["direction"])
        target_inventory = int(preview["target_inventory"])
        target_from_source_position = preview["target_from_source_position"]

        self._signal_seq += 1
        signal_id = f"etf_{str(projection.source_market).lower()}_{self._signal_seq}"
        signal = ETFASignal(
            signal_id=signal_id,
            source_market=str(projection.source_market),
            source_signal_id=projection.source_signal_id,
            source_combo=str(projection.source_combo or "A_only"),
            source_kind=str(projection.source_kind),
            started_ms=int(now_ms),
            alpha=chosen_alpha,
            source_fair_shift=fair_shift,
            a_fair_shift=fair_shift if str(projection.source_market).upper() == "A" else 0.0,
            projected_etf_shift=projected_shift,
            base_mid=float(self.book.mid),
            target_fair=float(self.book.mid) + projected_shift,
            target_inventory=target_inventory,
            source_target_inventory=projection.source_target_inventory,
            target_from_source_position=target_from_source_position,
            source_direction=direction,
        )
        self.entry_diagnostics[signal.signal_id] = ETFEntryDiagnostics()
        return signal

    def preview_projection(self, projection: ETFShockProjection) -> dict[str, Any] | None:
        if self.book.mid is None:
            self.last_block_reason = "missing_etf_book_for_signal"
            return None
        fair_shift = float(projection.fair_shift_ticks)
        chosen_alpha = self._bounded_alpha(float(projection.alpha))
        projected_shift = chosen_alpha * fair_shift
        if abs(projected_shift) < max(0, int(self.config.min_projected_edge_ticks)):
            self.last_block_reason = "projected_etf_shift_too_small"
            return None

        direction = int(projection.source_direction or (1 if projected_shift > 0 else -1))
        target_inventory = (
            int(projection.target_inventory_override)
            if projection.target_inventory_override is not None
            else int(round(abs(projected_shift) * float(self.config.target_position_per_etf_tick)))
        )
        target_from_source_position: int | None = None
        if projection.source_target_inventory is not None and int(projection.source_target_inventory) * direction > 0:
            target_from_source_position = int(
                round(abs(int(projection.source_target_inventory)) * float(self.config.target_position_per_a_shock_inventory))
            )
            target_inventory = max(target_inventory, target_from_source_position)
        major_shock = (
            abs(fair_shift) >= int(self.config.major_a_shock_fair_shift_ticks)
            or (
                projection.source_target_inventory is not None
                and abs(int(projection.source_target_inventory)) >= int(self.config.major_a_shock_target_inventory)
            )
        )
        if major_shock:
            target_inventory = max(target_inventory, int(self.config.min_target_position_for_major_a_shock))
        target_inventory = max(1, min(int(self.config.max_position), int(target_inventory))) * direction
        return {
            "fair_shift": fair_shift,
            "alpha": chosen_alpha,
            "projected_shift": projected_shift,
            "direction": direction,
            "target_inventory": int(target_inventory),
            "target_from_source_position": target_from_source_position,
        }

    def compute_quotes(
        self,
        *,
        now_ms: int,
        a_state: dict[str, Any] | None = None,
        c_state: dict[str, Any] | None = None,
    ) -> QuotePlan:
        if self.pending_signal is not None:
            pending_diag = self.entry_diagnostics.setdefault(self.pending_signal.signal_id, ETFEntryDiagnostics())
            pending_elapsed_ms = int(now_ms) - int(self.pending_signal.started_ms)
            if pending_diag.first_fill_ms is None and pending_elapsed_ms >= int(self.config.entry_retry_window_ms):
                pending_diag.terminal_reason = "entry_retry_window_expired"
                self.pending_signal = None
            elif self.inventory == 0 and not self._has_live_orders():
                self.active_signal = self.pending_signal
                self.pending_signal = None
                self.mode = "ETF_A_SHOCK"
                self.last_block_reason = None
                self.unwind_reason = None
            elif self.inventory == 0 and self._has_live_orders():
                self.mode = "ETF_HANDOFF_FLATTEN"
                self.last_block_reason = "handoff_flatten"
                return QuotePlan("ETF_HANDOFF_FLATTEN", None, None, (), False, "waiting_for_etf_order_cancel_before_handoff")
            else:
                self.mode = "ETF_HANDOFF_FLATTEN"
                self.last_block_reason = "handoff_flatten"
                self.unwind_reason = "handoff_flatten"
                return self._unwind_plan(now_ms, reason="handoff_flatten", allow_only_if_flat=False)
        signal = self.active_signal
        if signal is None:
            self.mode = "ETF_OBSERVE_ONLY"
            return QuotePlan(self.mode, None, None, (), True, "waiting_for_a_shock_signal")
        diag = self.entry_diagnostics.setdefault(signal.signal_id, ETFEntryDiagnostics())
        elapsed_ms = int(now_ms) - int(signal.started_ms)
        active_guard_reason = self._active_entry_guard_reason(now_ms)
        if diag.first_fill_ms is None and self.inventory == 0 and elapsed_ms >= int(self.config.entry_retry_window_ms):
            diag.terminal_reason = active_guard_reason or "entry_retry_window_expired"
            self.last_block_reason = diag.terminal_reason
            self.active_signal = None
            self.mode = "ETF_OBSERVE_ONLY"
            return QuotePlan(self.mode, None, None, (), True, diag.terminal_reason)
        if self.book.best_bid is None or self.book.best_ask is None or self.book.mid is None or self.book.spread is None:
            self.mode = "ETF_CHURN_GUARD"
            self.last_block_reason = "missing_etf_book"
            return QuotePlan("ETF_CHURN_GUARD", None, None, (), True, "missing_etf_book")
        if active_guard_reason is not None:
            self.mode = "ETF_CHURN_GUARD"
            self.last_block_reason = active_guard_reason
            return QuotePlan("ETF_CHURN_GUARD", None, None, (), True, active_guard_reason)
        if int(self.book.spread) < max(1, int(self.config.min_book_spread_ticks)):
            self.last_block_reason = "etf_book_too_tight_or_crossed"
            return self._unwind_plan(now_ms, reason="etf_book_too_tight_or_crossed", allow_only_if_flat=False)

        if self.mode == "ETF_A_SHOCK" and self._target_reached(signal):
            diag.terminal_reason = "etf_target_reached"
            self.mode = "ETF_A_HOLD"

        if self.mode in {"ETF_A_SHOCK", "ETF_A_HOLD"}:
            should_unwind, unwind_reason = self._should_unwind(signal, elapsed_ms, a_state, c_state)
            if should_unwind:
                self.mode = "ETF_UNWIND"
                self.unwind_reason = unwind_reason

        if self.mode == "ETF_UNWIND":
            return self._unwind_plan(now_ms, reason=self.unwind_reason or "unwinding_etf_a_follower")

        if self.mode == "ETF_A_HOLD":
            if abs(signal.target_inventory - int(self.inventory)) >= max(1, int(self.config.quote_size)):
                self.mode = "ETF_A_SHOCK"
                return self._shock_plan(signal, now_ms)
            return QuotePlan("ETF_A_HOLD", None, None, (), True, "holding_etf_a_follower_inventory")

        return self._shock_plan(signal, now_ms)

    def trace_state(self, now_ms: int) -> dict[str, Any]:
        signal = self.active_signal
        diag = None if signal is None else self.entry_diagnostics.get(signal.signal_id)
        return {
            "symbol": self.symbol,
            "market_key": "ETF",
            "mode": self.mode,
            "fair_value": None if signal is None else round(signal.target_fair),
            "inventory": self.inventory,
            "earnings_position": 0,
            "news_position": self.inventory,
            "mm_position": 0,
            "buy_exposure": self.order_manager.buy_exposure(),
            "sell_exposure": self.order_manager.sell_exposure(),
            "allowed_buy_size": max(0, int(self.config.max_position) - self.inventory - self.order_manager.buy_exposure()),
            "allowed_sell_size": max(0, int(self.config.max_position) + self.inventory - self.order_manager.sell_exposure()),
            "position_cap": self.config.max_position,
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
            "etf_signal_id": None if signal is None else signal.signal_id,
            "etf_pending_signal_id": None if self.pending_signal is None else self.pending_signal.signal_id,
            "etf_alpha_from_a": None if signal is None else signal.alpha,
            "etf_source_market": None if signal is None else signal.source_market,
            "etf_source_combo": None if signal is None else signal.source_combo,
            "etf_source_signal_id": None if signal is None else signal.source_signal_id,
            "etf_source_signal_kind": None if signal is None else signal.source_kind,
            "etf_source_fair_shift": None if signal is None else signal.source_fair_shift,
            "etf_a_fair_shift": None if signal is None else signal.a_fair_shift,
            "etf_projected_shift": None if signal is None else signal.projected_etf_shift,
            "etf_base_mid": None if signal is None else signal.base_mid,
            "etf_target_fair": None if signal is None else signal.target_fair,
            "etf_target_inventory": None if signal is None else signal.target_inventory,
            "etf_source_target_inventory": None if signal is None else signal.source_target_inventory,
            "etf_target_from_a_position": None if signal is None else signal.target_from_source_position,
            "etf_source_direction": None if signal is None else signal.source_direction,
            "etf_unwind_reason": self.unwind_reason,
            "etf_entry_order_attempt_count": None if diag is None else diag.order_attempt_count,
            "etf_first_order_attempt_latency_ms": (
                None
                if signal is None or diag is None or diag.first_order_attempt_ms is None
                else int(diag.first_order_attempt_ms) - int(signal.started_ms)
            ),
            "etf_first_fill_latency_ms": (
                None
                if signal is None or diag is None or diag.first_fill_ms is None
                else int(diag.first_fill_ms) - int(signal.started_ms)
            ),
            "etf_missed_entry_terminal_reason": None if diag is None else diag.terminal_reason,
            "etf_churn_guard_reason": self._churn_guard_reason,
            "etf_churn_guard_active": self._active_entry_guard_reason(now_ms) is not None,
            "etf_handoff_pending": self.pending_signal is not None,
            "etf_book_change_count_250ms": len(self._top_of_book_change_ms),
            "etf_stable_book_age_ms": (
                None if self._stable_book_since_ms is None else max(0, int(now_ms) - int(self._stable_book_since_ms))
            ),
            "block_reason": self.last_block_reason,
        }

    def signal_payload(self, signal: ETFASignal) -> dict[str, Any]:
        return {
            "signal_id": signal.signal_id,
            "source_market": signal.source_market,
            "source_combo": signal.source_combo,
            "source_signal_id": signal.source_signal_id,
            "source_kind": signal.source_kind,
            "alpha": signal.alpha,
            "source_fair_shift": signal.source_fair_shift,
            "a_fair_shift": signal.a_fair_shift,
            "projected_etf_shift": signal.projected_etf_shift,
            "base_mid": signal.base_mid,
            "target_fair": signal.target_fair,
            "target_inventory": signal.target_inventory,
            "source_target_inventory": signal.source_target_inventory,
            "target_from_a_position": signal.target_from_source_position,
            "source_direction": signal.source_direction,
            "alpha_max": self.config.alpha_max,
            "alpha_step": self.config.alpha_step,
        }

    def _shock_plan(self, signal: ETFASignal, now_ms: int) -> QuotePlan:
        desired_delta = signal.target_inventory - int(self.inventory)
        if desired_delta == 0:
            return QuotePlan("ETF_A_SHOCK", None, None, (), True, "already_at_etf_a_target")
        side = "BUY" if desired_delta > 0 else "SELL"
        assert self.book.best_bid is not None and self.book.best_ask is not None
        edge = signal.target_fair - float(self.book.best_ask.px) if side == "BUY" else float(self.book.best_bid.px) - signal.target_fair
        if edge < max(0, int(self.config.min_projected_edge_ticks)):
            self.last_block_reason = "etf_edge_below_entry_threshold"
            return QuotePlan("ETF_A_SHOCK", None, None, (), True, "etf_edge_below_entry_threshold")
        qty = min(abs(desired_delta), max(1, int(self.config.quote_size)))
        px = self.book.best_ask.px if side == "BUY" else self.book.best_bid.px
        diag = self.entry_diagnostics.setdefault(signal.signal_id, ETFEntryDiagnostics())
        if diag.first_order_attempt_ms is None:
            diag.first_order_attempt_ms = int(now_ms)
        diag.order_attempt_count += 1
        elapsed_ms = int(now_ms) - int(signal.started_ms)
        aggressive = diag.first_fill_ms is None or elapsed_ms <= int(self.config.entry_force_aggressive_ms)
        order = self._desired(
            side,
            px,
            qty,
            reason=f"following A {signal.source_kind} shock with alpha {signal.alpha:.2f}",
            signal_id=signal.signal_id,
            action_class="etf_shock_take",
            aggressive=aggressive,
        )
        return QuotePlan("ETF_A_SHOCK", None, None, (order,), False, order.reason)

    def _unwind_plan(self, now_ms: int, *, reason: str, allow_only_if_flat: bool = True) -> QuotePlan:
        if self.inventory == 0:
            if self.active_signal is not None:
                diag = self.entry_diagnostics.setdefault(self.active_signal.signal_id, ETFEntryDiagnostics())
                diag.terminal_reason = reason
            self.active_signal = None
            self.mode = "ETF_OBSERVE_ONLY"
            return QuotePlan(self.mode, None, None, (), True, "etf_a_follower_flat")
        live_buy = self.order_manager.live_order("BUY")
        live_sell = self.order_manager.live_order("SELL")
        if self.inventory > 0 and live_buy is not None and not live_buy.cancel_pending:
            return QuotePlan("ETF_UNWIND", None, None, (), True, "waiting_for_buy_cancel_before_sell_unwind")
        if self.inventory < 0 and live_sell is not None and not live_sell.cancel_pending:
            return QuotePlan("ETF_UNWIND", None, None, (), True, "waiting_for_sell_cancel_before_buy_unwind")
        if self.book.best_bid is None or self.book.best_ask is None:
            return QuotePlan("ETF_UNWIND", None, None, (), True, "missing_etf_book_for_unwind")
        if allow_only_if_flat and self.book.spread is not None and int(self.book.spread) <= 0:
            return QuotePlan("ETF_UNWIND", None, None, (), True, "waiting_for_uncrossed_etf_unwind_book")
        side = "SELL" if self.inventory > 0 else "BUY"
        px = self.book.best_bid.px if side == "SELL" else self.book.best_ask.px
        qty = min(abs(int(self.inventory)), max(1, int(self.config.quote_size)))
        signal_id = self.active_signal.signal_id if self.active_signal is not None else f"etf_unwind_{int(now_ms)}"
        order = self._desired(side, px, qty, reason=reason, signal_id=signal_id, action_class="etf_shock_unwind")
        self.mode = "ETF_UNWIND"
        return QuotePlan("ETF_UNWIND", None, None, (order,), False, reason)

    def _target_reached(self, signal: ETFASignal) -> bool:
        if signal.target_inventory > 0:
            return self.inventory >= signal.target_inventory
        return self.inventory <= signal.target_inventory

    def _target_price_reached(self, signal: ETFASignal) -> bool:
        if self.book.mid is None:
            return False
        band = max(0, int(self.config.exit_band_ticks))
        if signal.projected_etf_shift > 0:
            return float(self.book.mid) >= signal.target_fair - band
        return float(self.book.mid) <= signal.target_fair + band

    def _should_unwind(
        self,
        signal: ETFASignal,
        elapsed_ms: int,
        a_state: dict[str, Any] | None,
        c_state: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if elapsed_ms >= int(self.config.max_hold_ms):
            return True, "etf_max_hold_elapsed"

        if signal.source_market == "A":
            a_mode = str((a_state or {}).get("mode") or "")
            a_active = a_mode in {"POST_EARNINGS_SHOCK", "POST_NEWS_SHOCK"}
            a_direction = int((a_state or {}).get("shock_direction") or 0)
            if a_active and a_direction and a_direction != signal.source_direction:
                return True, "a_shock_direction_flipped"
            if elapsed_ms < int(self.config.min_hold_ms):
                return False, ""
            if not a_active:
                if self.inventory == 0:
                    return True, "a_signal_invalidated_before_entry"
                return True, "a_shock_lifecycle_inactive_after_min_hold"
            return False, ""

        if signal.source_market == "C":
            c_mode = str((c_state or {}).get("mode") or "")
            c_active = c_mode in {"C_EARNINGS_SHOCK", "C_EARNINGS_UNWIND"}
            c_direction = int((c_state or {}).get("c_shock_target_inventory") or 0)
            if c_direction:
                c_direction = 1 if c_direction > 0 else -1
            if c_active and c_direction and c_direction != signal.source_direction:
                return True, "c_shock_direction_flipped"
            if elapsed_ms < int(self.config.min_hold_ms):
                return False, ""
            if not c_active:
                if self.inventory == 0:
                    return True, "c_signal_invalidated_before_entry"
                return True, "c_shock_lifecycle_inactive_after_min_hold"
            return False, ""

        a_mode = str((a_state or {}).get("mode") or "")
        a_active = a_mode in {"POST_EARNINGS_SHOCK", "POST_NEWS_SHOCK"}
        a_direction = int((a_state or {}).get("shock_direction") or 0)
        c_mode = str((c_state or {}).get("mode") or "")
        c_active = c_mode in {"C_EARNINGS_SHOCK", "C_EARNINGS_UNWIND"}
        c_direction = int((c_state or {}).get("c_shock_target_inventory") or 0)
        c_direction = 1 if c_direction > 0 else -1 if c_direction < 0 else 0
        if a_active and c_active and a_direction and c_direction and a_direction != c_direction:
            return True, "a_c_shock_conflict"
        if elapsed_ms < int(self.config.min_hold_ms):
            return False, ""
        if not a_active and not c_active:
            if self.inventory == 0:
                return True, "source_signal_invalidated_before_entry"
            return True, "source_shock_lifecycle_inactive_after_min_hold"
        return False, ""

    def _bounded_alpha(self, value: float | None) -> float:
        alpha = float(self.config.alpha_from_a if value is None else value)
        return min(float(self.config.alpha_max), max(0.0, alpha))

    def _desired(
        self,
        side: str,
        px: int,
        qty: int,
        reason: str,
        *,
        signal_id: str,
        action_class: str,
        aggressive: bool = False,
    ) -> DesiredOrder:
        return DesiredOrder(
            side=side,  # type: ignore[arg-type]
            px=int(px),
            qty=int(qty),
            overlay="news",
            aggressive=bool(aggressive),
            reason=reason,
            intent="etf_a_follower",
            mode_at_submit=self.mode,
            evaluation_reason=reason,
            market_key="ETF",
            strategy_family="etf_a_follower",
            action_class=action_class,
            pnl_owner="etf_a_follower",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role="single",
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def _update_churn_guard(self, *, now_ms: int) -> None:
        top_of_book = (
            None if self.book.best_bid is None else int(self.book.best_bid.px),
            None if self.book.best_bid is None else int(self.book.best_bid.qty),
            None if self.book.best_ask is None else int(self.book.best_ask.px),
            None if self.book.best_ask is None else int(self.book.best_ask.qty),
        )
        if self._last_top_of_book != top_of_book:
            self._top_of_book_change_ms.append(int(now_ms))
            self._last_top_of_book = top_of_book
        churn_window_ms = max(1, int(self.config.churn_window_ms))
        while self._top_of_book_change_ms and int(now_ms) - int(self._top_of_book_change_ms[0]) > churn_window_ms:
            self._top_of_book_change_ms.popleft()

        valid_book = (
            self.book.best_bid is not None
            and self.book.best_ask is not None
            and self.book.spread is not None
            and int(self.book.spread) > 0
        )
        churn_active = len(self._top_of_book_change_ms) > int(self.config.churn_max_top_of_book_updates)
        if not valid_book:
            self._churn_guard_reason = "crossed_or_locked_etf_book"
            self._stable_book_since_ms = None
            return
        if churn_active:
            self._churn_guard_reason = "etf_quote_churn_guard"
            self._stable_book_since_ms = None
            return
        if self._churn_guard_reason is not None:
            if self._stable_book_since_ms is None:
                self._stable_book_since_ms = int(now_ms)
            elif int(now_ms) - int(self._stable_book_since_ms) >= int(self.config.churn_resume_stable_ms):
                self._churn_guard_reason = None
        else:
            self._stable_book_since_ms = int(now_ms)

    def _active_entry_guard_reason(self, now_ms: int) -> str | None:
        if self._churn_guard_reason is None:
            return None
        if self._stable_book_since_ms is None:
            return self._churn_guard_reason
        stable_age = int(now_ms) - int(self._stable_book_since_ms)
        if stable_age < int(self.config.churn_resume_stable_ms):
            return self._churn_guard_reason
        return None

    def _has_live_orders(self) -> bool:
        return any(
            order is not None and not order.cancel_pending
            for order in (self.order_manager.live_order("BUY"), self.order_manager.live_order("SELL"))
        )
