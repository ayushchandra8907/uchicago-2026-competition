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

# Leave this as None until you know the correct constant P/E ratio for A.
# With None, the bot will connect successfully but stay observe-only.
DEFAULT_A_PE_RATIO: float | None = None
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
        self.strategy = MarketAStrategy(
            a_config=config.market_a,
            risk=config.risk,
            restored_orders=replay_state.live_orders,
            recovered_fair_value=recovered_fair_value,
            learned_pe_ratio=replay_state.learned_pe_ratio,
            learned_pe_confidence=replay_state.learned_pe_confidence,
        )
        self.strategy.set_inventory(replay_state.inventory)
        self._quote_lock = asyncio.Lock()
        self._position_snapshot_seen = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._refresh_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._last_observe_only_reason: str | None = None

        if replay_state.live_orders:
            restored_ids = ", ".join(order.order_id for order in replay_state.live_orders)
            LOGGER.warning("Recovered %d local A orders from journal: %s", len(replay_state.live_orders), restored_ids)
        if recovered_fair_value is not None:
            LOGGER.info("Recovered last known A fair value from journal: %s", recovered_fair_value)
        if replay_state.learned_pe_ratio is not None:
            LOGGER.info(
                "Loaded trusted A P/E estimate %.4f with confidence %s",
                replay_state.learned_pe_ratio,
                replay_state.learned_pe_confidence,
            )
        elif config.market_a.pe_ratio is None:
            LOGGER.warning("No configured A P/E estimate yet; the bot will learn it from A earnings before trading.")

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
        await super().process_message(msg)
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
            return
        if reaction.started_pe_calibration:
            LOGGER.info(
                "Started A P/E calibration from earnings=%s; waiting %sms to settle and %sms to sample.",
                reaction.earnings_value,
                self.config.market_a.pe_learning_delay_ms,
                self.config.market_a.pe_learning_sample_window_ms,
            )
        if reaction.fair_value_updated and self.strategy.fair_value is not None:
            self.journal.record_fair_value(
                fair_value=self.strategy.fair_value,
                source=self.strategy.valuation.last_source,
                earnings_value=self.strategy.valuation.last_earnings_value,
            )
            LOGGER.info(
                "Updated A fair value to %s from earnings=%s using PE=%s",
                self.strategy.fair_value,
                self.strategy.valuation.last_earnings_value,
                self.strategy.effective_pe_ratio,
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
                await self.cancel_order(order.order_id)
                self.strategy.order_manager.mark_cancel_requested(order.order_id, self._now_ms())
                self.journal.record_cancel_requested(order.order_id)
                LOGGER.info("Requested cancel for recovered %s order %s @ %s", order.side, order.order_id, order.px)
        else:
            self.strategy.on_recovery_complete()
        await self._evaluate_and_sync("startup recovery")

    async def _quote_refresh_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(max(self.config.risk.reprice_cooldown_ms / 1000.0, 0.25))
            now_ms = self._now_ms()
            if self.strategy.order_manager.has_stale_quote(now_ms):
                await self._evaluate_and_sync("stale quote")

    async def _evaluate_and_sync(self, reason: str) -> None:
        if not self.connected or not self._position_snapshot_seen.is_set():
            return
        async with self._quote_lock:
            learned_pe_update = self.strategy.maybe_finalize_pe_learning(self._now_ms())
            if learned_pe_update is not None:
                self.journal.record_learned_pe(
                    effective_pe=learned_pe_update.trusted_pe,
                    confidence=learned_pe_update.trusted_confidence,
                    implied_pe=learned_pe_update.implied_pe,
                    reference_price=learned_pe_update.reference_price,
                    earnings_value=self.strategy.valuation.last_earnings_value or 0.0,
                    sample_count=learned_pe_update.sample_count,
                )
                if self.strategy.fair_value is not None:
                    self.journal.record_fair_value(
                        fair_value=self.strategy.fair_value,
                        source=self.strategy.valuation.last_source,
                        earnings_value=self.strategy.valuation.last_earnings_value,
                    )
                if learned_pe_update.status in {"bootstrap_started", "bootstrap_confirmed", "bootstrap_reset", "trusted_confirmed"}:
                    LOGGER.info(
                        "Updated trusted A P/E from earnings: status=%s implied=%.4f reference_mid=%.2f trusted_pe=%.4f confidence=%s trading_ready=%s",
                        learned_pe_update.status,
                        learned_pe_update.implied_pe,
                        learned_pe_update.reference_price,
                        learned_pe_update.trusted_pe,
                        learned_pe_update.trusted_confidence,
                        learned_pe_update.trading_ready,
                    )
                elif learned_pe_update.status in {"candidate_started", "candidate_confirmed", "candidate_restarted"}:
                    LOGGER.info(
                        "Observed candidate A P/E: status=%s implied=%.4f trusted_pe=%.4f candidate_pe=%.4f candidate_confidence=%s; continuing to trade trusted P/E.",
                        learned_pe_update.status,
                        learned_pe_update.implied_pe,
                        learned_pe_update.trusted_pe,
                        learned_pe_update.candidate_pe or learned_pe_update.trusted_pe,
                        learned_pe_update.candidate_confidence,
                    )
                elif learned_pe_update.status == "candidate_promoted":
                    LOGGER.info(
                        "Promoted candidate A P/E to trusted: implied=%.4f new_trusted_pe=%.4f confidence=%s trading_ready=%s",
                        learned_pe_update.implied_pe,
                        learned_pe_update.trusted_pe,
                        learned_pe_update.trusted_confidence,
                        learned_pe_update.trading_ready,
                    )
            # The strategy only decides what the bot wants to own on each side.
            # The order manager turns that target state into cancel/place actions.
            plan = self.strategy.compute_quotes()
            if plan.observe_only:
                if plan.reason != self._last_observe_only_reason:
                    LOGGER.info("A bot observe-only: %s", plan.reason)
                    self._last_observe_only_reason = plan.reason
            else:
                self._last_observe_only_reason = None
            actions = self.strategy.order_manager.build_actions(plan, self._now_ms())

            for cancel in actions.cancels:
                self.strategy.order_manager.mark_cancel_requested(cancel.order_id, self._now_ms())
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
                    now_ms=self._now_ms(),
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
    if not config.trading_enabled and config.trading_disabled_reason:
        LOGGER.warning(config.trading_disabled_reason)
    bot = MarketABot(config)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down A-only bot.")
