from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

try:
    from a_bot_config import load_bot_config
    from b_mean_reversion import BMeanReversionStrategy
    from b_underlying_mm_v2 import BUnderlyingMMv2
    from testing1 import MarketABot
except (ModuleNotFoundError, SystemExit):
    MarketABot = None


class _FakeCall:
    def __init__(self) -> None:
        self.cancel_count = 0

    def cancel(self) -> bool:
        self.cancel_count += 1
        return True


class _FakeBObserver:
    symbols = {"B", "B_C_1000"}

    def __init__(self) -> None:
        self.book_updates: list[str] = []
        self.trades: list[str] = []

    def on_book_update(self, symbol, book, *, now_ms=None) -> None:
        self.book_updates.append(symbol)

    def on_market_trade(self, symbol, price, qty, *, now_ms=None) -> None:
        self.trades.append(symbol)


class _FakeBStrategy:
    def __init__(self) -> None:
        self.book_updates: list[str] = []
        self.trades: list[str] = []

    def on_book_update_at(self, symbol, book, now_ms) -> bool:
        self.book_updates.append(symbol)
        return True

    def on_market_trade(self, price, qty, *, now_ms=None) -> None:
        self.trades.append("B")


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
        bot.b_option_strategy = None
        bot.etf_strategy = None
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

    async def test_b_option_updates_refresh_observer_without_repricing_live_quotes(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        bot.config = SimpleNamespace(
            market_b=SimpleNamespace(enabled=True, underlying_symbol="B"),
        )
        bot.b_observer = _FakeBObserver()
        bot.b_strategy = _FakeBStrategy()
        bot.b_option_strategy = None
        bot.etf_strategy = None
        bot.order_books = {"B": object(), "B_C_1000": object()}
        bot.tracer = None
        bot._now_ms = lambda: 1_000
        eval_reasons: list[str] = []

        async def fake_eval(reason: str, *, force: bool = False) -> None:
            eval_reasons.append(reason)

        bot._evaluate_and_sync_b = fake_eval

        await bot.bot_handle_book_update("B_C_1000")
        await bot.bot_handle_trade_msg("B_C_1000", 10, 1)
        self.assertEqual(eval_reasons, [])
        self.assertEqual(bot.b_observer.book_updates, ["B_C_1000"])
        self.assertEqual(bot.b_observer.trades, ["B_C_1000"])
        self.assertEqual(bot.b_strategy.book_updates, [])
        self.assertEqual(bot.b_strategy.trades, [])

        await bot.bot_handle_book_update("B")
        await bot.bot_handle_trade_msg("B", 1000, 1)
        self.assertEqual(eval_reasons, ["book update:B", "market trade:B"])

    async def test_etf_signal_forces_immediate_etf_evaluation_before_news_return(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        reaction = SimpleNamespace(
            note=None,
            relevant=False,
            tick=123,
            fair_value_updated=False,
        )
        bot.strategy = SimpleNamespace(
            mode="IDLE",
            on_news=lambda news_release, now_ms: reaction,
        )
        bot.etf_strategy = SimpleNamespace(
            on_a_news_reaction=lambda reaction, now_ms: SimpleNamespace(signal_id="etf_a_1"),
        )
        bot.tracer = None
        bot._now_ms = lambda: 1_000
        calls: list[str] = []

        async def fake_a_eval(reason: str, *, force: bool = False) -> None:
            calls.append(f"a:{reason}:{force}")

        async def fake_etf_eval(reason: str, *, force: bool = False) -> None:
            calls.append(f"etf:{reason}:{force}")

        bot._evaluate_and_sync = fake_a_eval
        bot._evaluate_and_sync_etf = fake_etf_eval

        await bot.bot_handle_news({"kind": "unstructured"})

        self.assertEqual(calls, ["etf:A shock signal:True", "a:news tick:False"])

    async def test_meanrev_enabled_selects_mean_reversion_underlying_strategy(self) -> None:
        env_keys = {
            "UTC_HOST": "practice.uchicago.exchange:3333",
            "UTC_USERNAME": "user",
            "UTC_PASSWORD": "pass",
            "TRACE_ENABLED": "0",
            "B_MEANREV_ENABLED": "1",
            "B_MM_V2_ENABLED": "1",
        }
        old_values = {key: os.environ.get(key) for key in env_keys}
        try:
            os.environ.update(env_keys)
            with tempfile.TemporaryDirectory() as temp_dir:
                config = load_bot_config(Path(temp_dir))
                bot = MarketABot(config)
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        self.assertIsInstance(bot.b_strategy, BMeanReversionStrategy)

    async def test_meanrev_disabled_falls_back_to_mm_v2(self) -> None:
        env_keys = {
            "UTC_HOST": "practice.uchicago.exchange:3333",
            "UTC_USERNAME": "user",
            "UTC_PASSWORD": "pass",
            "TRACE_ENABLED": "0",
            "B_MEANREV_ENABLED": "0",
            "B_MM_V2_ENABLED": "1",
        }
        old_values = {key: os.environ.get(key) for key in env_keys}
        try:
            os.environ.update(env_keys)
            with tempfile.TemporaryDirectory() as temp_dir:
                config = load_bot_config(Path(temp_dir))
                bot = MarketABot(config)
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        self.assertIsInstance(bot.b_strategy, BUnderlyingMMv2)
