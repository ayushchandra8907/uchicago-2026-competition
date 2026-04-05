from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from a_bot_config import BotConfig, ConfigError, load_bot_config
from a_bot_journal import TradingJournal
from a_bot_strategy import MarketAStrategy

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

# A has a fixed economic P/E of 10 and trades in cents.
DEFAULT_A_PE_RATIO: float | None = 10.0
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
        recovered_fair_value = replay_state.fair_value if replay_state.live_orders else None
        recovered_earnings_value = replay_state.earnings_value if replay_state.live_orders else None
        self.strategy = MarketAStrategy(
            a_config=config.market_a,
            risk=config.risk,
            restored_orders=replay_state.live_orders,
            recovered_fair_value=recovered_fair_value,
            recovered_earnings_value=recovered_earnings_value,
        )
        self.strategy.set_inventory(replay_state.inventory)
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
        if recovered_fair_value is not None:
            LOGGER.info("Recovered last known A fair value from journal: %s", recovered_fair_value)
        LOGGER.info(
            "A valuation uses fixed PE=%s and price_scale=%s",
            config.market_a.pe_ratio,
            config.market_a.price_scale,
        )

    def handle_position_snapshot(self, msg) -> None:
        """Use the exchange snapshot as the anchor for inventory and recovery startup."""
        super().handle_position_snapshot(msg)
        a_position = int(self.positions.get("A", 0))
        cash_value = int(self.positions.get("cash", 0))
        self.strategy.set_inventory(a_position)
        self.journal.record_inventory(a_position, cash=cash_value)
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
            inventory = int(msg.position_update.value)
            self.strategy.set_inventory(inventory)
            self.journal.record_inventory(inventory, cash=int(self.positions.get("cash", 0)))
            await self._evaluate_and_sync("position update")
        elif msg_type == "cash_update":
            self.journal.record_inventory(
                int(self.strategy.inventory),
                cash=int(msg.cash_update.value),
            )

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        order = self.strategy.on_cancel_response(order_id, success)
        self.journal.record_cancel_response(order_id, success, error)
        if order is not None:
            LOGGER.info("Cancel response for %s success=%s error=%s", order_id, success, error)
        await self._evaluate_and_sync("cancel response")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
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
        await self._evaluate_and_sync("fill")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self.strategy.on_rejection(order_id)
        self.journal.record_rejection(order_id, reason)
        LOGGER.warning("Order %s rejected: %s", order_id, reason)
        await self._evaluate_and_sync("rejection")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol == "A":
            LOGGER.debug("Trade in A at %s for %s", price, qty)

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol != "A":
            return
        self.strategy.on_book_update_at(symbol, self.order_books[symbol], self._now_ms())
        await self._evaluate_and_sync("book update")

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool) -> None:
        LOGGER.info("Ignoring swap response in A-only bot: %s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict) -> None:
        reaction = self.strategy.on_news(news_release, self._now_ms())
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
            LOGGER.info(
                "A earnings tick=%s moved fair from %s to %s on earnings=%s; shock_direction=%s threshold=%s",
                reaction.tick,
                reaction.old_fair_value,
                reaction.new_fair_value,
                reaction.earnings_value,
                reaction.shock_direction,
                reaction.shock_threshold,
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

    async def _start_recovery_after_snapshot(self) -> None:
        if self.strategy.recovery_active:
            LOGGER.info("Entering startup recovery mode for A; cancelling restored orders before quoting.")
            for order in self.strategy.recovery_orders_to_cancel():
                self.strategy.order_manager.mark_cancel_requested(order.order_id, self._now_ms())
                self.journal.record_cancel_requested(order.order_id)
                await self.cancel_order(order.order_id)
                LOGGER.info("Requested cancel for recovered %s order %s @ %s", order.side, order.order_id, order.px)
        else:
            self.strategy.on_recovery_complete()
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
            plan = self.strategy.compute_quotes(now_ms=now_ms)
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
                self.strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                self.journal.record_cancel_requested(cancel.order_id)
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling %s order %s because %s", cancel.side, cancel.order_id, reason)

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
                )
                self.journal.record_order_submitted(managed_order)
                LOGGER.info(
                    "Placed %s %s order %s for A: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    @staticmethod
    def symbol_from_side(side: str) -> str:
        return "shares" if side in {"BUY", "SELL"} else side


async def main() -> None:
    base_dir = Path(__file__).resolve().parent
    try:
        config = load_bot_config(
            base_dir,
            default_host=DEFAULT_SERVER,
            default_username=DEFAULT_USERNAME,
            default_password=DEFAULT_PASSWORD,
            default_pe_ratio=DEFAULT_A_PE_RATIO,
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
