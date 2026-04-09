from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

try:
    from testing1 import MarketABot
except (ModuleNotFoundError, SystemExit):
    MarketABot = None


class _FakeCall:
    def __init__(self) -> None:
        self.cancel_count = 0

    def cancel(self) -> bool:
        self.cancel_count += 1
        return True


@unittest.skipIf(MarketABot is None, "runtime exchange dependencies are unavailable in this test environment")
class RuntimeShutdownTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(
        self,
        *,
        startup_assume_fresh_round: bool = True,
        auto_stop_after_round_complete: bool = True,
        auto_stop_on_followup_position_snapshot: bool = False,
        auto_stop_on_market_resolved: bool = False,
        assumed_round_duration_ms: int = 900_000,
        round_completion_grace_ms: int = 5_000,
    ) -> MarketABot:
        bot = MarketABot.__new__(MarketABot)
        bot.config = SimpleNamespace(
            auto_stop_after_round_complete=auto_stop_after_round_complete,
            auto_stop_on_followup_position_snapshot=auto_stop_on_followup_position_snapshot,
            auto_stop_on_market_resolved=auto_stop_on_market_resolved,
            assumed_round_duration_ms=assumed_round_duration_ms,
            round_completion_grace_ms=round_completion_grace_ms,
            market_a=SimpleNamespace(startup_assume_fresh_round=startup_assume_fresh_round),
        )
        bot._round_end_task = None
        bot._position_snapshot_count = 0
        bot._auto_stop_requested = False
        bot._shutdown_note = "Bot shutdown"
        bot._shutdown = asyncio.Event()
        bot.call = _FakeCall()
        bot.tracer = None
        return bot

    async def test_followup_position_snapshot_requests_clean_auto_stop(self) -> None:
        bot = self.make_bot(startup_assume_fresh_round=False, auto_stop_on_followup_position_snapshot=True)

        self.assertFalse(bot._on_position_snapshot_received(100))
        self.assertTrue(bot._on_position_snapshot_received(200))
        self.assertTrue(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "round_complete_auto_stop:followup_position_snapshot")
        self.assertEqual(bot.call.cancel_count, 1)

    async def test_followup_position_snapshot_is_ignored_by_default(self) -> None:
        bot = self.make_bot(startup_assume_fresh_round=False, auto_stop_on_followup_position_snapshot=False)

        self.assertFalse(bot._on_position_snapshot_received(100))
        self.assertFalse(bot._on_position_snapshot_received(200))
        self.assertFalse(bot._shutdown.is_set())
        self.assertEqual(bot.call.cancel_count, 0)

    async def test_round_completion_watch_requests_clean_auto_stop_after_timeout(self) -> None:
        bot = self.make_bot(assumed_round_duration_ms=1, round_completion_grace_ms=1)

        await bot._round_completion_watch_loop()

        self.assertTrue(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "round_complete_auto_stop:wall_clock_round_timeout")
        self.assertEqual(bot.call.cancel_count, 1)

    async def test_market_resolution_is_observed_without_stopping_by_default(self) -> None:
        bot = self.make_bot()

        await bot.bot_handle_market_resolved("market-1", "A", 4350)

        self.assertFalse(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "Bot shutdown")
        self.assertEqual(bot.call.cancel_count, 0)

    async def test_market_resolution_can_be_opted_into_clean_auto_stop(self) -> None:
        bot = self.make_bot(auto_stop_on_market_resolved=True)

        await bot.bot_handle_market_resolved("market-1", "A", 4350)

        self.assertTrue(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "round_complete_auto_stop:market_resolved")
        self.assertEqual(bot.call.cancel_count, 1)

    async def test_settlement_payout_requests_clean_auto_stop(self) -> None:
        bot = self.make_bot()

        await bot.bot_handle_settlement_payout("user", "market-1", 100, 4350)

        self.assertTrue(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "round_complete_auto_stop:settlement_payout")
        self.assertEqual(bot.call.cancel_count, 1)
