from __future__ import annotations

from a_bot_config import AConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


class AEarningsTrapOverlay:
    """Small passive overlay that posts into already-stretched A earnings books."""

    def __init__(self, a_config: AConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.a_config = a_config
        self.risk = risk
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.book = BookSnapshot()
        self.order_manager = OrderManager(symbol="A", risk=risk)
        self._attempted_cycle_ids: set[str] = set()

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        del now_ms
        if symbol != "A":
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        return True

    def compute_quotes(self, now_ms: int, a_state: dict[str, object]) -> QuotePlan:
        mode = str(a_state.get("mode") or "")
        if mode != "POST_EARNINGS_SHOCK":
            return self._observe("trap_inactive_non_earnings_mode")

        cycle_id = self._cycle_id(a_state)
        if not cycle_id:
            return self._observe("trap_inactive_missing_cycle")

        if cycle_id in self._attempted_cycle_ids and not self._has_live_cycle(cycle_id):
            return self._observe("trap_already_attempted_this_cycle")

        fair_value = self._int_or_none(a_state.get("fair_value"))
        shock_direction = int(a_state.get("shock_direction") or 0)
        inventory = int(a_state.get("inventory") or 0)
        reference_mid = self._float_or_none(a_state.get("shock_reference_mid"))
        if fair_value is None or reference_mid is None or shock_direction == 0:
            return self._observe("trap_inactive_missing_shock_context")

        fair_shift_ticks = abs(int(round(float(fair_value) - reference_mid)))
        if fair_shift_ticks < int(self.a_config.earnings_trap_min_fair_shift_ticks):
            return self._observe("trap_fair_shift_below_threshold")

        if self._live_order_expired(now_ms):
            return self._observe("trap_order_expired")

        offset_ticks = self._offset_ticks(fair_value)
        signal_id = f"{cycle_id}_trap"
        if self._live_signal_id_mismatch(signal_id):
            return self._observe("trap_waiting_for_cycle_roll_cancel")

        if shock_direction > 0:
            if inventory < int(self.a_config.earnings_trap_min_aligned_inventory):
                return self._observe("trap_waiting_for_aligned_inventory")
            if self.book.best_ask is None:
                return self._observe("trap_waiting_for_best_ask")
            min_trap_px = int(fair_value) + offset_ticks
            if int(self.book.best_ask.px) < min_trap_px:
                return self._observe("trap_waiting_for_stretched_ask")
            qty = min(
                int(self.a_config.earnings_trap_max_qty),
                max(0, int(inventory) - int(self.a_config.earnings_trap_inventory_reserve)),
            )
            if qty <= 0:
                return self._observe("trap_inventory_cushion_exhausted")
            return QuotePlan(
                mode="POST_EARNINGS_SHOCK",
                bid=None,
                ask=DesiredOrder(
                    side="SELL",
                    px=int(self.book.best_ask.px),
                    qty=qty,
                    overlay="earnings",
                    aggressive=False,
                    reason="passive A earnings trap asks into already-stretched upside book",
                    intent="a_earnings_trap",
                    mode_at_submit="POST_EARNINGS_SHOCK",
                    evaluation_reason="earnings trap overlay",
                    market_key="A",
                    strategy_family="a_earnings_trap",
                    action_class="trap_quote",
                    pnl_owner="a_earnings_trap",
                    signal_id=signal_id,
                    trade_group_id=signal_id,
                    leg_role="single",
                ),
                aggressive_actions=(),
                observe_only=False,
                reason="trap_upside_ready",
            )

        if abs(inventory) < int(self.a_config.earnings_trap_min_aligned_inventory):
            return self._observe("trap_waiting_for_aligned_inventory")
        if self.book.best_bid is None:
            return self._observe("trap_waiting_for_best_bid")
        max_trap_px = int(fair_value) - offset_ticks
        if int(self.book.best_bid.px) > max_trap_px:
            return self._observe("trap_waiting_for_stretched_bid")
        qty = min(
            int(self.a_config.earnings_trap_max_qty),
            max(0, abs(int(inventory)) - int(self.a_config.earnings_trap_inventory_reserve)),
        )
        if qty <= 0:
            return self._observe("trap_inventory_cushion_exhausted")
        return QuotePlan(
            mode="POST_EARNINGS_SHOCK",
            bid=DesiredOrder(
                side="BUY",
                px=int(self.book.best_bid.px),
                qty=qty,
                overlay="earnings",
                aggressive=False,
                reason="passive A earnings trap bids into already-stretched downside book",
                intent="a_earnings_trap",
                mode_at_submit="POST_EARNINGS_SHOCK",
                evaluation_reason="earnings trap overlay",
                market_key="A",
                strategy_family="a_earnings_trap",
                action_class="trap_quote",
                pnl_owner="a_earnings_trap",
                signal_id=signal_id,
                trade_group_id=signal_id,
                leg_role="single",
            ),
            ask=None,
            aggressive_actions=(),
            observe_only=False,
            reason="trap_downside_ready",
        )

    def on_order_submitted(self, signal_id: str) -> None:
        cycle_id = self._cycle_id_from_signal(signal_id)
        if cycle_id:
            self._attempted_cycle_ids.add(cycle_id)

    def on_fill(self, order_id: str, qty: int, price: int):
        del price
        return self.order_manager.handle_fill(order_id, qty)

    def on_cancel_response(self, order_id: str, success: bool):
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str):
        return self.order_manager.handle_rejection(order_id)

    def _live_order_expired(self, now_ms: int) -> bool:
        for side in ("BUY", "SELL"):
            order = self.order_manager.live_order(side)
            if order is None or order.cancel_pending:
                continue
            if now_ms - int(order.submitted_ms) >= int(self.a_config.earnings_trap_max_lifetime_ms):
                return True
        return False

    def _has_live_cycle(self, cycle_id: str) -> bool:
        target_signal_id = f"{cycle_id}_trap"
        for side in ("BUY", "SELL"):
            order = self.order_manager.live_order(side)
            if order is None or order.cancel_pending:
                continue
            if str(order.signal_id) == target_signal_id:
                return True
        return False

    def _live_signal_id_mismatch(self, signal_id: str) -> bool:
        for side in ("BUY", "SELL"):
            order = self.order_manager.live_order(side)
            if order is None or order.cancel_pending:
                continue
            if str(order.signal_id) != str(signal_id):
                return True
        return False

    def _offset_ticks(self, fair_value: int) -> int:
        raw = round(abs(float(fair_value)) * float(self.a_config.earnings_trap_offset_fraction))
        lower = int(self.a_config.earnings_trap_offset_min_ticks)
        upper = int(self.a_config.earnings_trap_offset_max_ticks)
        return max(lower, min(upper, int(raw)))

    @staticmethod
    def _observe(reason: str) -> QuotePlan:
        return QuotePlan(
            mode="POST_EARNINGS_SHOCK",
            bid=None,
            ask=None,
            aggressive_actions=(),
            observe_only=True,
            reason=reason,
        )

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cycle_id(a_state: dict[str, object]) -> str | None:
        for key in ("active_earnings_cycle_id", "current_earnings_signal_id"):
            raw = str(a_state.get(key) or "").strip()
            if raw:
                return raw
        return None

    @staticmethod
    def _cycle_id_from_signal(signal_id: str | None) -> str | None:
        if not signal_id:
            return None
        value = str(signal_id)
        return value[:-5] if value.endswith("_trap") else value
