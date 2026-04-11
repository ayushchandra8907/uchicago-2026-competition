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
    from a_bot_config import AConfig, RiskConfig, load_bot_config
    from a_earnings_trap_overlay import AEarningsTrapOverlay
    from b_mean_reversion import BMeanReversionStrategy
    from b_underlying_mm_v2 import BUnderlyingMMv2
    from etf_a_follower import ETFShockProjection
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

    async def test_settlement_payout_is_observed_without_stopping(self) -> None:
        bot = self.make_bot()

        await bot.bot_handle_settlement_payout("user", "market-1", 100, 4350)

        self.assertFalse(bot._shutdown.is_set())
        self.assertEqual(bot._shutdown_note, "Bot shutdown")
        self.assertEqual(bot.call.cancel_count, 0)

    async def test_b_option_updates_refresh_observer_without_repricing_live_quotes(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        bot.config = SimpleNamespace(
            market_b=SimpleNamespace(enabled=True, underlying_symbol="B"),
        )
        bot.b_observer = _FakeBObserver()
        bot.b_strategy = _FakeBStrategy()
        bot.b_option_strategy = None
        bot.etf_strategy = None
        bot.c_strategy = None
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
        bot.c_strategy = None
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

    async def test_c_enabled_constructs_c_strategy(self) -> None:
        env_keys = {
            "UTC_HOST": "practice.uchicago.exchange:3333",
            "UTC_USERNAME": "user",
            "UTC_PASSWORD": "pass",
            "TRACE_ENABLED": "0",
            "C_ENABLED": "1",
            "C_TRADING_ENABLED": "1",
            "C_LIVE_EARNINGS_ENABLED": "1",
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

        self.assertIsNotNone(bot.c_strategy)
        self.assertTrue(bot.config.market_c.enabled)
        self.assertEqual(tuple(bot.config.market_c.pm_symbols), ("R_HIKE", "R_HOLD", "R_CUT"))

    async def test_c_etf_projection_suppresses_conflicting_a_signal(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        bot.config = SimpleNamespace(
            etf=SimpleNamespace(
                enable_c_earnings=True,
                alpha_from_c_earnings=0.25,
                min_c_fair_shift_ticks=60,
                ac_conflict_policy="suppress",
                max_position=100,
            ),
            market_c=SimpleNamespace(symbol="C", pm_symbols=("R_HIKE", "R_HOLD", "R_CUT")),
        )
        bot._etf_trading_enabled = True
        bot.etf_strategy = SimpleNamespace(
            preview_projection=lambda projection: {"direction": 1, "target_inventory": 40},
            on_shock_projection=lambda projection, now_ms: None,
        )
        bot.c_strategy = SimpleNamespace(
            active_etf_projection=lambda: {
                "source_market": "C",
                "source_kind": "structured_earnings",
                "source_signal_id": "c_earnings_1",
                "fair_shift_ticks": 120.0,
                "source_target_inventory": 120,
                "source_direction": 1,
            }
        )
        bot.tracer = None
        bot._current_cash = lambda: None
        bot._trace_state_for_symbol = lambda symbol, now_ms: {"mode": "POST_EARNINGS_SHOCK", "shock_direction": -1, "symbol": symbol}
        bot._evaluate_and_sync_etf = _FakeAsync()

        await bot._maybe_emit_c_etf_projection("C earnings signal", now_ms=1_000)

        self.assertEqual(bot._evaluate_and_sync_etf.await_count, 0)

    async def test_a_trap_fill_routes_through_overlay_manager(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        overlay = AEarningsTrapOverlay(
            AConfig(earnings_trap_enabled=True),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=250),
        )
        overlay.order_manager.note_submitted(
            order_id="trap-1",
            side="SELL",
            px=1110,
            qty=20,
            now_ms=1_000,
            overlay="earnings",
            aggressive=False,
            intent="a_earnings_trap",
            mode_at_submit="POST_EARNINGS_SHOCK",
            evaluation_reason="earnings trap overlay",
            market_key="A",
            strategy_family="a_earnings_trap",
            action_class="trap_quote",
            pnl_owner="a_earnings_trap",
            signal_id="a_eps_1_trap",
            trade_group_id="a_eps_1_trap",
            leg_role="single",
        )

        sync_calls: list[int] = []
        journal_events: list[tuple[str, int]] = []

        strategy = SimpleNamespace(
            inventory=60,
            trace_state=lambda now_ms: {"symbol": "A", "mode": "POST_EARNINGS_SHOCK"},
            sync_inventory_from_exchange=lambda inventory: (sync_calls.append(inventory), setattr(strategy, "inventory", inventory)),
        )

        bot.strategy = strategy
        bot.a_trap_overlay = overlay
        bot.etf_strategy = None
        bot.b_option_strategy = None
        bot.b_strategy = None
        bot.c_strategy = None
        bot.tracer = None
        bot._ayush_port_mode = False
        bot.positions = {"cash": 0}
        bot._last_position_update_by_symbol = {}
        bot._now_ms = lambda: 1_005
        bot.journal = SimpleNamespace(
            record_fill=lambda order_id, qty, price: journal_events.append(("fill", qty)),
            record_inventory=lambda inventory, cash=None: journal_events.append(("inventory", inventory)),
        )
        eval_reasons: list[str] = []

        async def fake_eval(reason: str) -> None:
            eval_reasons.append(reason)

        bot._evaluate_and_sync = fake_eval

        await bot.bot_handle_order_fill("trap-1", 5, 1110)

        self.assertEqual(sync_calls, [55])
        self.assertEqual(strategy.inventory, 55)
        self.assertEqual(overlay.order_manager.orders["trap-1"].remaining_qty, 15)
        self.assertEqual(journal_events, [("fill", 5), ("inventory", 55)])
        self.assertEqual(eval_reasons, ["trap fill"])


class _FakeAsync:
    def __init__(self) -> None:
        self.await_count = 0

    async def __call__(self, *args, **kwargs) -> None:
        self.await_count += 1
