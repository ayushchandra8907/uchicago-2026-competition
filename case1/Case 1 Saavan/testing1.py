from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from a_bot_config import BotConfig, ConfigError, load_bot_config
from a_bot_journal import TradingJournal, select_recovered_pricing_state
from a_bot_strategy import MarketAStrategy
from a_bot_trace import TraceRecorder

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError as exc:
    raise SystemExit(
        "utcxchangelib is not installed. Run `pip install -r requirements.txt` first."
    ) from exc


LOGGER = logging.getLogger("market-a-bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# Quick-start defaults so clicking the IDE Run button works the same way as the
# provided example bot. Environment variables still override these.
DEFAULT_SERVER = "34.197.188.76:3333"
DEFAULT_USERNAME = "uiuc"
DEFAULT_PASSWORD = "mesa-lynx-octopus"

# A's price-to-earnings multiplier changes by round, so we learn it live.
DEFAULT_A_INITIAL_MULTIPLIER: float | None = None
DEFAULT_A_INITIAL_FAIR_VALUE: int | None = None


class MarketABot(XChangeClient):
    """Live A-only bot with fair-value updates, passive quotes, and restart recovery."""

    def __init__(self, config: BotConfig):
        super().__init__(
            config.exchange.host,
            config.exchange.username,
            config.exchange.password,
            symbols=["A"],
        )
        self.config = config
        self.journal = TradingJournal(config.paths.journal_path)
        replay_state = self.journal.load_replay_state()
        (
            recovered_multiplier,
            recovered_multiplier_confidence,
            recovered_fair_value,
            recovered_earnings_value,
        ) = select_recovered_pricing_state(
            replay_state,
            recover_pricing_state=config.market_a.recover_pricing_state,
        )
        self.strategy = MarketAStrategy(
            a_config=config.market_a,
            risk=config.risk,
            restored_orders=replay_state.live_orders,
            recovered_multiplier=recovered_multiplier,
            recovered_multiplier_confidence=recovered_multiplier_confidence,
            recovered_fair_value=recovered_fair_value,
            recovered_earnings_value=recovered_earnings_value,
            book_depth_levels=max(10, config.trace.trace_book_depth_levels),
        )
        self.strategy.set_inventory(replay_state.inventory)
        self.tracer = TraceRecorder.create_if_enabled(config.trace, session_prefix="a_bot_run", symbol="A")
        self._quote_lock = asyncio.Lock()
        self._position_snapshot_seen = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._refresh_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._last_observe_only_reason: str | None = None
        self._last_mode: str | None = None

        if replay_state.live_orders:
            restored_ids = ", ".join(order.order_id for order in replay_state.live_orders)
            LOGGER.warning("Recovered %d local A orders from journal: %s", len(replay_state.live_orders), restored_ids)
        if recovered_multiplier is not None:
            LOGGER.info(
                "Recovered last known A multiplier from journal: %.4f (confidence=%s)",
                recovered_multiplier,
                recovered_multiplier_confidence,
            )
        elif replay_state.multiplier is not None and replay_state.live_orders:
            LOGGER.info(
                "Ignoring recovered A pricing state from journal because A_RECOVER_PRICING_STATE is disabled."
            )
        if recovered_fair_value is not None:
            LOGGER.info("Recovered last known A fair value from journal: %s", recovered_fair_value)
        if config.market_a.initial_multiplier is not None and recovered_multiplier is None:
            LOGGER.info("Seeding A multiplier from config: %.4f", config.market_a.initial_multiplier)
        else:
            LOGGER.info("A valuation will learn a round-specific multiplier from structured earnings.")
        if self.tracer is not None:
            LOGGER.info("Analysis mode enabled; writing trace outputs to %s", self.tracer.run_dir)
        if self.tracer is not None:
            now_ms = self._now_ms()
            self.tracer.record_session_start(
                now_ms=now_ms,
                config_summary={
                    "host": config.exchange.host,
                    "trace_snapshot_interval_ms": config.trace.trace_snapshot_interval_ms,
                    "trace_book_depth_levels": config.trace.trace_book_depth_levels,
                    "trace_markout_windows_ms": list(config.trace.trace_markout_windows_ms),
                    "journal_path": str(config.paths.journal_path),
                    "trace_root": str(config.trace.trace_root),
                    "recover_pricing_state": config.market_a.recover_pricing_state,
                },
                recovered_orders=[self._order_to_dict(order) for order in replay_state.live_orders],
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
            )

    def handle_position_snapshot(self, msg) -> None:
        """Use the exchange snapshot as the anchor for inventory and recovery startup."""
        super().handle_position_snapshot(msg)
        now_ms = self._now_ms()
        a_position = int(self.positions.get("A", 0))
        cash_value = int(self.positions.get("cash", 0))
        self.strategy.set_inventory(a_position)
        self.journal.record_inventory(a_position, cash=cash_value)
        if self.tracer is not None:
            self.tracer.record_inventory_update(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=cash_value,
                trigger="position_snapshot",
            )
        if not self._position_snapshot_seen.is_set():
            self._position_snapshot_seen.set()
            self._recovery_task = asyncio.create_task(self._start_recovery_after_snapshot())

    async def process_message(self, msg) -> None:
        """Mirror exchange position updates into local journaled strategy state."""
        msg_type = msg.WhichOneof("body")
        recovery_event: tuple[str, str, bool, str | None, int | None, int | None] | None = None
        if msg_type == "cancel_response":
            order_id = msg.cancel_response.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                result_type = msg.cancel_response.WhichOneof("result")
                recovery_event = (
                    "cancel_response",
                    order_id,
                    result_type == "ok",
                    None if result_type == "ok" else msg.cancel_response.error,
                    None,
                    None,
                )
        elif msg_type == "order_fill":
            order_id = msg.order_fill.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                recovery_event = (
                    "order_fill",
                    order_id,
                    True,
                    None,
                    int(msg.order_fill.qty),
                    int(msg.order_fill.px),
                )
        elif msg_type == "order_rejected":
            order_id = msg.order_rejected.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                recovery_event = (
                    "order_rejected",
                    order_id,
                    False,
                    msg.order_rejected.reason,
                    None,
                    None,
                )

        await super().process_message(msg)

        if recovery_event is not None:
            event_type, order_id, success, error, qty, price = recovery_event
            if event_type == "cancel_response":
                await self.bot_handle_cancel_response(order_id, success, error)
            elif event_type == "order_fill" and qty is not None and price is not None:
                await self.bot_handle_order_fill(order_id, qty, price)
            elif event_type == "order_rejected" and error is not None:
                await self.bot_handle_order_rejected(order_id, error)

        if msg_type == "position_update" and msg.position_update.symbol == "A":
            now_ms = self._now_ms()
            inventory = int(msg.position_update.value)
            self.strategy.set_inventory(inventory)
            self.journal.record_inventory(inventory, cash=int(self.positions.get("cash", 0)))
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    trigger="position_update",
                )
            await self._evaluate_and_sync("position update")
        elif msg_type == "cash_update":
            now_ms = self._now_ms()
            self.journal.record_inventory(
                int(self.strategy.inventory),
                cash=int(msg.cash_update.value),
            )
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=int(msg.cash_update.value),
                    trigger="cash_update",
                )

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_cancel_response(order_id, success)
        self.journal.record_cancel_response(order_id, success, error)
        if order is not None:
            LOGGER.info("Cancel response for %s success=%s error=%s", order_id, success, error)
        if self.tracer is not None:
            self.tracer.record_cancel_response(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                order_id=order_id,
                success=success,
                error=error,
                side=None if order is None else order.side,
                order=None if order is None else self._order_to_dict(order),
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order cancellation completed")
        await self._evaluate_and_sync("cancel response")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_fill(order_id, qty, price)
        self.journal.record_fill(order_id, qty, price)
        if order is not None:
            LOGGER.info(
                "Fill on %s order %s: %s %s @ %s, estimated inventory=%s",
                order.side,
                order_id,
                qty,
                self.symbol_from_side(order.side),
                price,
                self.strategy.inventory,
            )
        else:
            LOGGER.info("Received fill for unmanaged order %s qty=%s px=%s", order_id, qty, price)
        self.journal.record_inventory(int(self.strategy.inventory), cash=int(self.positions.get("cash", 0)))
        if self.tracer is not None:
            self.tracer.record_fill(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                order=None if order is None else self._order_to_dict(order),
                order_id=order_id,
                qty=qty,
                price=price,
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order filled")
        await self._evaluate_and_sync("fill")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_rejection(order_id)
        self.journal.record_rejection(order_id, reason)
        LOGGER.warning("Order %s rejected: %s", order_id, reason)
        if self.tracer is not None:
            self.tracer.record_rejection(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                order_id=order_id,
                reason=reason,
                order=None if order is None else self._order_to_dict(order),
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order rejected")
        await self._evaluate_and_sync("rejection")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol == "A":
            now_ms = self._now_ms()
            self.strategy.on_market_trade(price, qty, now_ms=now_ms)
            LOGGER.debug("Trade in A at %s for %s", price, qty)
            if self.tracer is not None:
                self.tracer.record_market_trade(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    price=price,
                    qty=qty,
                )

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol != "A":
            return
        now_ms = self._now_ms()
        self.strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
        if self.tracer is not None:
            self.tracer.record_book_update(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                trigger="book_update",
            )
        await self._evaluate_and_sync("book update")

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool) -> None:
        LOGGER.info("Ignoring swap response in A-only bot: %s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict) -> None:
        now_ms = self._now_ms()
        mode_before_news = self.strategy.mode
        reaction = self.strategy.on_news(news_release, now_ms)
        if reaction.note:
            LOGGER.info(reaction.note)
        if self.tracer is not None:
            reaction_dict = self._reaction_to_dict(reaction)
            reaction_dict["mode_before_news"] = mode_before_news
            reaction_dict["mode_after_news"] = self.strategy.mode
            reaction_dict["news_kind"] = news_release.get("kind")
            self.tracer.record_news(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                news_payload=news_release,
                reaction=reaction_dict,
            )
        if not reaction.relevant:
            if reaction.tick is not None:
                await self._evaluate_and_sync("news tick")
            return
        if reaction.fair_value_updated and self.strategy.fair_value is not None:
            self.journal.record_fair_value(
                fair_value=self.strategy.fair_value,
                source=self.strategy.valuation.last_source,
                earnings_value=self.strategy.valuation.last_earnings_value,
            )
            if self.tracer is not None:
                self.tracer.record_valuation_update(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    source="structured_news_provisional_fair",
                    details=self._reaction_to_dict(reaction),
                )
        await self._evaluate_and_sync("news")

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        LOGGER.info("Ignoring market resolution %s winner=%s tick=%s in A-only bot", market_id, winning_symbol, tick)

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        LOGGER.info("Settlement payout user=%s market=%s amount=%s tick=%s", user, market_id, amount, tick)

    async def start(self) -> None:
        self._refresh_task = asyncio.create_task(self._quote_refresh_loop())
        try:
            await self.connect()
        finally:
            self._shutdown.set()
            if self._refresh_task is not None:
                self._refresh_task.cancel()
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.finalize(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    note="Bot shutdown",
                )

    async def _start_recovery_after_snapshot(self) -> None:
        if self.strategy.recovery_active:
            LOGGER.info("Entering startup recovery mode for A; cancelling restored orders before quoting.")
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.record_recovery_state(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    reason="startup recovery began",
                )
            for order in self.strategy.recovery_orders_to_cancel():
                now_ms = self._now_ms()
                tracked_order = self.strategy.order_manager.mark_cancel_requested(order.order_id, now_ms)
                self.journal.record_cancel_requested(order.order_id)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state(now_ms),
                        cash=self._current_cash(),
                        order_id=order.order_id,
                        side=order.side,
                        cancel_reason="recovery cancel on startup",
                        mode_at_cancel=order.mode_at_submit or "recovery_cancel",
                        order=None if tracked_order is None else self._order_to_dict(tracked_order),
                    )
                await self.cancel_order(order.order_id)
                LOGGER.info("Requested cancel for recovered %s order %s @ %s", order.side, order.order_id, order.px)
        else:
            self.strategy.on_recovery_complete()
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.record_recovery_state(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    reason="no recovery work required",
                )
        await self._evaluate_and_sync("startup recovery")

    async def _quote_refresh_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(max(self.config.risk.reprice_cooldown_ms / 1000.0, 0.25))
            await self._evaluate_and_sync("timer")

    async def _evaluate_and_sync(self, reason: str) -> None:
        if not self.connected or not self._position_snapshot_seen.is_set():
            return
        async with self._quote_lock:
            # The strategy only decides what the bot wants to own on each side.
            # The order manager turns that target state into cancel/place actions.
            now_ms = self._now_ms()
            prior_fair = self.strategy.fair_value
            prior_multiplier = self.strategy.trusted_multiplier
            prior_confidence = self.strategy.multiplier_confidence
            plan = self.strategy.compute_quotes(now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )
            for event in self.strategy.drain_learning_events():
                if event.status == "skipped":
                    LOGGER.warning("Skipped A multiplier calibration: %s", event.reason)
                else:
                    LOGGER.info(
                        "A multiplier %s via %s estimate=%.4f settled_mid=%s trusted=%.4f confidence=%s tolerance=%s",
                        event.status,
                        event.method,
                        event.estimate,
                        "n/a" if event.settled_mid is None else f"{event.settled_mid:.2f}",
                        event.trusted_multiplier or 0.0,
                        event.confidence,
                        "n/a" if event.tolerance is None else f"{event.tolerance:.2f}",
                    )
                if event.trusted_multiplier is not None and event.status != "skipped":
                    self.journal.record_multiplier(
                        multiplier=event.trusted_multiplier,
                        confidence=event.confidence,
                        source=event.status,
                        estimate=event.estimate,
                        method=event.method,
                    )
                if self.tracer is not None:
                    self.tracer.record_valuation_update(
                        now_ms=now_ms,
                        state=self._trace_state(now_ms),
                        cash=self._current_cash(),
                        source="multiplier_learning",
                        details={
                            "old_fair_value": prior_fair,
                            "new_fair_value": self.strategy.fair_value,
                            "earnings_value": self.strategy.valuation.last_earnings_value,
                            "status": event.status,
                            "estimate": event.estimate,
                            "trusted_multiplier": event.trusted_multiplier,
                            "confidence": event.confidence,
                            "method": event.method,
                            "settled_mid": event.settled_mid,
                            "reason": event.reason,
                            "tolerance": event.tolerance,
                        },
                    )
            if self.strategy.fair_value is not None and self.strategy.fair_value != prior_fair:
                self.journal.record_fair_value(
                    fair_value=self.strategy.fair_value,
                    source=self.strategy.valuation.last_source,
                    earnings_value=self.strategy.valuation.last_earnings_value,
                )
                if prior_fair is None:
                    LOGGER.info("A fair value initialized at %s after multiplier learning.", self.strategy.fair_value)
                elif self.strategy.trusted_multiplier != prior_multiplier or self.strategy.multiplier_confidence != prior_confidence:
                    LOGGER.info("A fair value refreshed from %s to %s after multiplier update.", prior_fair, self.strategy.fair_value)
            if plan.mode != self._last_mode:
                until_next = self.strategy.ms_until_next_scheduled_earnings(now_ms)
                LOGGER.info(
                    "A mode -> %s fair=%s inventory=%s next_earnings_ms=%s reason=%s",
                    plan.mode,
                    self.strategy.fair_value,
                    self.strategy.inventory,
                    until_next,
                    plan.reason,
                )
                self._last_mode = plan.mode
            if plan.observe_only:
                if plan.reason != self._last_observe_only_reason:
                    LOGGER.info("A bot observe-only: %s", plan.reason)
                    self._last_observe_only_reason = plan.reason
            else:
                self._last_observe_only_reason = None
            actions = self.strategy.order_manager.build_actions(plan, now_ms)

            for cancel in actions.cancels:
                order = self.strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                self.journal.record_cancel_requested(cancel.order_id)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state(now_ms),
                        cash=self._current_cash(),
                        order_id=cancel.order_id,
                        side=cancel.side,
                        cancel_reason=cancel.reason,
                        mode_at_cancel=self.strategy.mode,
                        order=None if order is None else self._order_to_dict(order),
                    )
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling %s order %s because %s", cancel.side, cancel.order_id, cancel.reason)

            for placement in actions.placements:
                side = Side.BUY if placement.side == "BUY" else Side.SELL
                order_id = await self.place_order("A", placement.qty, side, placement.px)
                managed_order = self.strategy.order_manager.note_submitted(
                    order_id=order_id,
                    side=placement.side,
                    px=placement.px,
                    qty=placement.qty,
                    now_ms=now_ms,
                    aggressive=placement.aggressive,
                    intent=placement.intent,
                    mode_at_submit=placement.mode_at_submit,
                    evaluation_reason=placement.evaluation_reason,
                )
                self.journal.record_order_submitted(managed_order)
                if self.tracer is not None:
                    self.tracer.record_order_submitted(
                        now_ms=now_ms,
                        state=self._trace_state(now_ms),
                        cash=self._current_cash(),
                        order=self._order_to_dict(managed_order),
                    )
                LOGGER.info(
                    "Placed %s %s order %s for A: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )
            if self.tracer is not None:
                self.tracer.maybe_record_periodic_snapshot(
                    now_ms=now_ms,
                    state=self._trace_state(now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    @staticmethod
    def symbol_from_side(side: str) -> str:
        return "shares" if side in {"BUY", "SELL"} else side

    def _current_cash(self) -> int | None:
        return int(self.positions.get("cash", 0)) if self._position_snapshot_seen.is_set() else None

    def _trace_state(self, now_ms: int) -> dict:
        return self.strategy.trace_state(now_ms)

    @staticmethod
    def _order_to_dict(order) -> dict:
        return {
            "order_id": order.order_id,
            "side": order.side,
            "px": order.px,
            "qty": order.qty,
            "remaining_qty": order.remaining_qty,
            "submitted_ms": order.submitted_ms,
            "aggressive": order.aggressive,
            "cancel_pending": getattr(order, "cancel_pending", False),
            "restored": getattr(order, "restored", False),
            "intent": getattr(order, "intent", ""),
            "mode_at_submit": getattr(order, "mode_at_submit", ""),
            "evaluation_reason": getattr(order, "evaluation_reason", ""),
        }

    @staticmethod
    def _reaction_to_dict(reaction) -> dict:
        return {
            "relevant": reaction.relevant,
            "fair_value_updated": reaction.fair_value_updated,
            "note": reaction.note,
            "earnings_value": reaction.earnings_value,
            "old_fair_value": reaction.old_fair_value,
            "new_fair_value": reaction.new_fair_value,
            "shock_direction": reaction.shock_direction,
            "shock_threshold": reaction.shock_threshold,
            "tick": reaction.tick,
        }

    def _trace_recovery_transition(self, was_recovery_active: bool, reason: str) -> None:
        if self.tracer is None:
            return
        if was_recovery_active != self.strategy.recovery_active:
            now_ms = self._now_ms()
            self.tracer.record_recovery_state(
                now_ms=now_ms,
                state=self._trace_state(now_ms),
                cash=self._current_cash(),
                reason=reason if self.strategy.recovery_active else f"{reason}; recovery complete",
            )


async def main() -> None:
    base_dir = Path(__file__).resolve().parent
    try:
        config = load_bot_config(
            base_dir,
            default_host=DEFAULT_SERVER,
            default_username=DEFAULT_USERNAME,
            default_password=DEFAULT_PASSWORD,
            default_initial_multiplier=DEFAULT_A_INITIAL_MULTIPLIER,
            default_initial_fair_value=DEFAULT_A_INITIAL_FAIR_VALUE,
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    LOGGER.info("Starting A-only bot against %s", config.exchange.host)
    LOGGER.info("Journal path: %s", config.paths.journal_path)
    bot = MarketABot(config)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down A-only bot.")
