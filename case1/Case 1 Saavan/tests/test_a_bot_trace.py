from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import AConfig, RiskConfig, TraceConfig, load_bot_config
from a_bot_journal import TradingJournal, select_recovered_pricing_state
from a_bot_strategy import DesiredOrder, ManagedOrder, MarketAStrategy, QuotePlan
from a_bot_trace import TraceRecorder, load_trace_events, summarize_trace_events
from a_news_tracker import build_a_news_tracker_report
from ayush_a_port import AyushPortStrategy
from b_observer import MarketBObserver


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class TraceTests(unittest.TestCase):
    def make_strategy(self) -> MarketAStrategy:
        strategy = MarketAStrategy(
            a_config=AConfig(
                startup_assume_fresh_round=True,
                pre_news_pullback_ms=4_000,
                calibration_min_delay_ms=5_000,
                calibration_max_delay_ms=20_000,
                calibration_sample_period_ms=1_000,
                calibration_stability_band_ticks=8,
                recover_pricing_state=False,
                discovery_quote_size=2,
                steady_quote_size=2,
                steady_max_position=24,
                steady_take_inventory_guard=8,
                unwind_entry_position=24,
                unwind_exit_position=12,
                shock_quote_size=15,
                shock_base_max_position=80,
                shock_shift_max_position=160,
            ),
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
            book_depth_levels=10,
        )
        strategy.startup_ms = 0
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10, 1094: 8}, asks={1105: 9, 1106: 7}), now_ms=45_000)
        return strategy

    def test_trace_recorder_disabled_returns_none(self) -> None:
        recorder = TraceRecorder.create_if_enabled(TraceConfig(trace_enabled=False), session_prefix="test_trace")
        self.assertIsNone(recorder)

    def test_trace_state_exposes_live_orders_depth_and_allowed_sizes(self) -> None:
        strategy = self.make_strategy()
        strategy.order_manager.note_submitted(
            order_id="bid-1",
            side="BUY",
            px=1099,
            qty=2,
            now_ms=45_000,
            aggressive=False,
            intent="steady_mm_passive",
            mode_at_submit="STEADY_MM",
            evaluation_reason="steady-state bid around learned fair",
        )
        state = strategy.trace_state(45_000)

        self.assertEqual(state["mode"], "STEADY_MM")
        self.assertEqual(state["allowed_buy_size"], 178)
        self.assertEqual(state["buy_exposure"], 2)
        self.assertEqual(state["mm_position"], 0)
        self.assertEqual(state["earnings_budget"], 120)
        self.assertEqual(state["mm_budget"], 60)
        self.assertEqual(len(state["live_orders"]), 1)
        self.assertEqual(state["book"]["best_bid_px"], 1095)
        self.assertEqual(state["book"]["best_ask_px"], 1105)
        self.assertEqual(len(state["book"]["bid_levels"]), 2)
        self.assertIsNotNone(state["book"]["microprice"])

    def test_trace_snapshot_row_captures_ayush_news_fields(self) -> None:
        strategy = AyushPortStrategy(
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)
        strategy.on_news(
            {
                "tick": 321,
                "kind": "unstructured",
                "symbol": "A",
                "new_data": {
                    "content": "Analysts predict strong revenue growth for A.",
                    "type": "News",
                },
            },
            now_ms=1_100,
        )
        plan = strategy.compute_quotes(now_ms=1_100)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_snapshot_interval_ms=500,
                    trace_book_depth_levels=10,
                    trace_markout_windows_ms=(250, 1_000, 5_000),
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            now_ms = 1_100
            recorder.record_decision(
                now_ms=now_ms,
                state=strategy.trace_state(now_ms),
                cash=10_000,
                trigger="news",
                plan=plan,
            )
            recorder.finalize(now_ms=1_200, state=strategy.trace_state(1_200), cash=10_000, note="done")

            with recorder.snapshots_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["mode"], "NEWS_CONFIRMATION")
            self.assertEqual(row["news_sentiment_bucket"], "medium")
            self.assertEqual(row["news_confirmation_state"], "pending")
            self.assertEqual(row["pending_news_target_inventory"], "36")
            self.assertIn("strong revenue growth", row["pending_news_json"])

    def test_trace_recorder_writes_runtime_lifecycle_events(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_snapshot_interval_ms=500,
                    trace_book_depth_levels=10,
                    trace_markout_windows_ms=(250, 1_000, 5_000),
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(45_000)
            recorder.record_runtime_event(
                event_type="market_resolved_observed",
                now_ms=45_000,
                state=state,
                cash=10_000,
                reason="market_resolved",
                details={"market_id": "market-1", "winning_symbol": "A", "tick": 3975},
            )
            recorder.record_runtime_event(
                event_type="settlement_payout_observed",
                now_ms=45_100,
                state=state,
                cash=10_000,
                reason="settlement_payout",
                details={"market_id": "market-1", "amount": 100, "tick": 4000},
            )
            recorder.record_runtime_event(
                event_type="runtime_disconnect",
                now_ms=45_200,
                state=state,
                cash=10_000,
                reason="eof",
                details={"elapsed_runtime_ms": 1_000},
            )
            recorder.record_runtime_event(
                event_type="runtime_reconnect_attempt",
                now_ms=45_300,
                state=state,
                cash=10_000,
                reason="unexpected eof before round-end cutoff",
                details={"attempt": 1},
            )
            recorder.record_runtime_event(
                event_type="runtime_reconnect_succeeded",
                now_ms=45_400,
                state=state,
                cash=10_000,
                reason="first message received after reconnect",
                details={"message_type": "position_snapshot"},
            )
            recorder.finalize(now_ms=45_500, state=state, cash=10_000, note="done")

            event_types = [event["event_type"] for event in load_trace_events(recorder.run_dir)]
            self.assertIn("market_resolved_observed", event_types)
            self.assertIn("settlement_payout_observed", event_types)
            self.assertIn("runtime_disconnect", event_types)
            self.assertIn("runtime_reconnect_attempt", event_types)
            self.assertIn("runtime_reconnect_succeeded", event_types)

    def test_compact_trace_suppresses_book_updates_by_default(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(1_000)
            recorder.record_book_update(now_ms=1_000, state=state, cash=10_000, trigger="book_update")
            recorder.finalize(now_ms=1_100, state=state, cash=10_000, note="done")

            event_types = [event["event_type"] for event in load_trace_events(recorder.run_dir)]
            self.assertNotIn("book_update", event_types)

    def test_verbose_trace_can_record_book_updates(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_write_summary_on_shutdown=False,
                    trace_record_book_updates=True,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(1_000)
            recorder.record_book_update(now_ms=1_000, state=state, cash=10_000, trigger="book_update")
            recorder.finalize(now_ms=1_100, state=state, cash=10_000, note="done")

            event_types = [event["event_type"] for event in load_trace_events(recorder.run_dir)]
            self.assertIn("book_update", event_types)

    def test_compact_trace_suppresses_low_signal_observe_only_decisions(self) -> None:
        strategy = self.make_strategy()
        plan = QuotePlan(
            mode="POST_EARNINGS_SHOCK",
            bid=None,
            ask=None,
            aggressive_actions=(),
            observe_only=True,
            reason="already at the A shock target",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(1_000)
            recorder.record_decision(now_ms=1_000, state=state, cash=10_000, trigger="timer", plan=plan)
            recorder.finalize(now_ms=1_100, state=state, cash=10_000, note="done")

            event_types = [event["event_type"] for event in load_trace_events(recorder.run_dir)]
            self.assertNotIn("decision_evaluated", event_types)
            with recorder.snapshots_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_compact_trace_keeps_news_confirmation_decisions(self) -> None:
        strategy = self.make_strategy()
        plan = QuotePlan(
            mode="NEWS_CONFIRMATION",
            bid=None,
            ask=None,
            aggressive_actions=(),
            observe_only=True,
            reason="waiting for medium A-news confirmation",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(1_000)
            recorder.record_decision(now_ms=1_000, state=state, cash=10_000, trigger="news", plan=plan)
            recorder.finalize(now_ms=1_100, state=state, cash=10_000, note="done")

            event_types = [event["event_type"] for event in load_trace_events(recorder.run_dir)]
            self.assertIn("decision_evaluated", event_types)

    def test_trace_recorder_writes_run_files_and_summary(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_snapshot_interval_ms=500,
                    trace_book_depth_levels=10,
                    trace_markout_windows_ms=(250, 1_000, 5_000),
                    trace_write_summary_on_shutdown=True,
                ),
                session_prefix="test_trace",
            )
            now_ms = 45_000
            state = strategy.trace_state(now_ms)
            recorder.record_session_start(
                now_ms=now_ms,
                config_summary={"host": "example"},
                recovered_orders=[],
                state=state,
                cash=10_000,
            )
            plan = QuotePlan(
                mode="STEADY_MM",
                bid=DesiredOrder(
                    side="BUY",
                    px=1099,
                    qty=2,
                    aggressive=False,
                    reason="steady-state bid around learned fair",
                    intent="steady_mm_passive",
                    mode_at_submit="STEADY_MM",
                    evaluation_reason="book update",
                ),
                ask=DesiredOrder(
                    side="SELL",
                    px=1101,
                    qty=2,
                    aggressive=False,
                    reason="steady-state ask around learned fair",
                    intent="steady_mm_passive",
                    mode_at_submit="STEADY_MM",
                    evaluation_reason="book update",
                ),
                aggressive_actions=(),
                observe_only=False,
                reason="steady mm",
            )
            recorder.record_decision(now_ms=now_ms, state=state, cash=10_000, trigger="book update", plan=plan)
            recorder.record_order_submitted(
                now_ms=now_ms,
                state=state,
                cash=10_000,
                order={
                    "order_id": "bid-1",
                    "side": "BUY",
                    "px": 1099,
                    "qty": 2,
                    "remaining_qty": 2,
                    "overlay": "mm",
                    "aggressive": False,
                    "intent": "steady_mm_passive",
                    "mode_at_submit": "STEADY_MM",
                    "evaluation_reason": "book update",
                },
            )
            strategy.on_market_trade(1100, 1, now_ms=45_250)
            recorder.maybe_record_periodic_snapshot(now_ms=45_500, state=strategy.trace_state(45_500), cash=10_000, trigger="timer")
            recorder.record_fill(
                now_ms=45_750,
                state=strategy.trace_state(45_750),
                cash=10_200,
                order={
                    "order_id": "bid-1",
                    "side": "BUY",
                    "remaining_qty": 0,
                    "overlay": "mm",
                    "aggressive": False,
                    "intent": "steady_mm_passive",
                    "mode_at_submit": "STEADY_MM",
                    "evaluation_reason": "book update",
                },
                order_id="bid-1",
                qty=2,
                price=1099,
            )
            recorder.finalize(now_ms=46_000, state=strategy.trace_state(46_000), cash=10_200, note="done")

            self.assertTrue((recorder.run_dir / "trace_events.jsonl").exists())
            self.assertTrue((recorder.run_dir / "decision_snapshots.csv").exists())
            self.assertTrue((recorder.run_dir / "session_summary.json").exists())
            self.assertTrue((recorder.run_dir / "session_summary.md").exists())
            self.assertTrue((recorder.run_dir / "a_news_tracker.json").exists())
            self.assertTrue((recorder.run_dir / "a_news_tracker.md").exists())
            self.assertTrue((recorder.run_dir / "unknown_a_news_terms.json").exists())
            self.assertTrue((recorder.run_dir / "unknown_a_news_terms.md").exists())

            events = load_trace_events(recorder.run_dir)
            event_types = {event["event_type"] for event in events}
            self.assertIn("session_start", event_types)
            self.assertIn("decision_evaluated", event_types)
            self.assertIn("order_submitted", event_types)
            self.assertIn("order_filled", event_types)
            self.assertIn("session_end", event_types)
            fill_event = next(event for event in events if event["event_type"] == "order_filled")
            self.assertEqual(fill_event["fill_qty"], 2)
            self.assertEqual(fill_event["fill_price"], 1099)
            self.assertEqual(fill_event["inventory_after_fill"], 0)

            with (recorder.run_dir / "session_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["passive_fills"], 1)
            self.assertEqual(summary["aggressive_fills"], 0)
            self.assertEqual(summary["fills_by_overlay"]["mm"], 1)
            self.assertIn("a_episode_summaries", summary)
            self.assertIn("a_mm_loss_by_mode", summary)
            self.assertIn("pnl_by_action_class", summary)
            self.assertIn("a_strategy_breakdown", summary)
            self.assertIn("a_earnings_calibration_diagnostics", summary)
            self.assertIn("a_news_summary", summary)
            self.assertIn("a_news_episode_summaries", summary)
            self.assertIn("b_cost_adjusted_residual_stats", summary)
            self.assertIn("b_mean_reversion_summary", summary)
            self.assertIn("b_shadow_underlying_mm", summary)
            self.assertIn("b_strategy_block_reasons", summary)
            self.assertIn("trace_volume_summary", summary)

    def test_load_bot_config_trace_defaults_to_saavan_analysis_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_keys = {
                "UTC_HOST": "practice.uchicago.exchange:3333",
                "UTC_USERNAME": "user",
                "UTC_PASSWORD": "pass",
            }
            optional_keys = {
                "TRACE_ROOT",
                "TRACE_ENABLED",
                "TRACE_SNAPSHOT_INTERVAL_MS",
                "TRACE_BOOK_DEPTH_LEVELS",
                "TRACE_RECORD_BOOK_UPDATES",
                "TRACE_RECORD_OBSERVE_ONLY_DECISIONS",
                "B_MM_MIN_EVAL_INTERVAL_MS",
                "B_MM_REPRICE_THRESHOLD_TICKS",
                "B_MM_V2_MAX_POSITION",
                "B_MM_V2_QUOTE_SIZE",
                "B_MM_MIN_VALID_SPREAD_TICKS",
                "B_MM_MIN_HEALTHY_BOOK_AGE_MS",
                "B_MM_CANCEL_ON_BAD_BOOK",
                "B_MM_BAD_FILL_COOLDOWN_MS",
                "B_MEANREV_ENABLED",
                "B_MEANREV_MAX_POSITION",
                "B_MEANREV_QUOTE_SIZE",
                "B_MEANREV_EMA_FAST_MS",
                "B_MEANREV_EMA_SLOW_MS",
                "B_MEANREV_VOL_EWMA_MS",
                "B_MEANREV_SIGMA_FLOOR",
                "B_MEANREV_ENTRY_Z",
                "B_MEANREV_ENTRY_Z2",
                "B_MEANREV_EXIT_Z",
                "B_MEANREV_STOP_Z",
                "B_MEANREV_MIN_SPREAD_TICKS",
                "B_MEANREV_MAX_HOLD_MS",
                "B_MEANREV_COOLDOWN_MS",
                "B_MEANREV_AGGRESSIVE_ENTRY_Z",
                "B_MEANREV_AGGRESSIVE_EXIT",
                "B_OPTION_LOTTERY_ENABLED",
                "B_OPTION_LOTTERY_MAX_ASK",
                "B_OPTION_LOTTERY_TOTAL_PREMIUM_BUDGET",
                "B_OPTION_LOTTERY_WING_MAX_POSITION",
                "B_OPTION_LOTTERY_ATM_MAX_POSITION",
                "B_OPTION_LOTTERY_WING_PREMIUM_BUDGET",
                "B_OPTION_LOTTERY_ATM_TOTAL_PREMIUM_BUDGET",
                "B_OPTION_LOTTERY_PROFIT_TAKE_ENABLED",
                "B_OPTION_HEDGE_ENABLED",
                "B_OPTION_HEDGE_MAX_ASK",
                "B_OPTION_HEDGE_MIN_UNDERLYING_INVENTORY",
                "B_OPTION_HEDGE_TARGET_RATIO",
                "B_OPTION_HEDGE_PREMIUM_BUDGET",
                "ETF_ENABLED",
                "ETF_TRADING_ENABLED",
                "ETF_ALPHA_FROM_A",
                "ETF_ALPHA_FROM_A_EARNINGS",
                "ETF_ALPHA_FROM_A_NEWS",
                "ETF_MAX_POSITION",
                "ETF_QUOTE_SIZE",
                "ETF_TARGET_POSITION_PER_A_SHOCK_INVENTORY",
                "ETF_MIN_HOLD_MS",
                "ETF_MIN_EVAL_INTERVAL_MS",
                "ETF_UNWIND_REPRICE_THRESHOLD_TICKS",
                "ETF_ENTRY_RETRY_REPRICE_MS",
                "ETF_CHURN_WINDOW_MS",
                "ETF_CHURN_MAX_TOP_OF_BOOK_UPDATES",
                "ETF_CHURN_RESUME_STABLE_MS",
                "AUTO_STOP_ON_FOLLOWUP_POSITION_SNAPSHOT",
                "AUTO_STOP_ON_MARKET_RESOLVED",
            }
            old_values = {key: os.environ.get(key) for key in {*env_keys, *optional_keys}}
            try:
                os.environ.update(env_keys)
                for key in optional_keys:
                    os.environ.pop(key, None)
                config = load_bot_config(Path(temp_dir))
            finally:
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
            self.assertEqual(config.trace.trace_root, Path(temp_dir).resolve() / "analysis_runs")
            self.assertFalse(config.trace.trace_enabled)
            self.assertEqual(config.trace.trace_snapshot_interval_ms, 2_000)
            self.assertEqual(config.trace.trace_book_depth_levels, 1)
            self.assertFalse(config.trace.trace_record_book_updates)
            self.assertFalse(config.trace.trace_record_observe_only_decisions)
            self.assertFalse(config.market_a.recover_pricing_state)
            self.assertEqual(config.market_b.mm_min_eval_interval_ms, 150)
            self.assertEqual(config.market_b.max_position, 4)
            self.assertEqual(config.market_b.quote_size, 1)
            self.assertEqual(config.market_b.mm_reprice_threshold_ticks, 3)
            self.assertEqual(config.market_b.mm_min_valid_spread_ticks, 3)
            self.assertEqual(config.market_b.mm_min_healthy_book_age_ms, 500)
            self.assertTrue(config.market_b.mm_cancel_on_bad_book)
            self.assertEqual(config.market_b.mm_bad_fill_cooldown_ms, 750)
            self.assertTrue(config.market_b.meanrev_enabled)
            self.assertEqual(config.market_b.meanrev_max_position, 16)
            self.assertEqual(config.market_b.meanrev_quote_size, 2)
            self.assertEqual(config.market_b.meanrev_ema_fast_ms, 30_000)
            self.assertEqual(config.market_b.meanrev_ema_slow_ms, 180_000)
            self.assertEqual(config.market_b.meanrev_vol_ewma_ms, 60_000)
            self.assertEqual(config.market_b.meanrev_sigma_floor, 4.0)
            self.assertEqual(config.market_b.meanrev_entry_z, 1.25)
            self.assertEqual(config.market_b.meanrev_entry_z2, 2.25)
            self.assertEqual(config.market_b.meanrev_exit_z, 0.35)
            self.assertEqual(config.market_b.meanrev_stop_z, 5.0)
            self.assertEqual(config.market_b.meanrev_min_spread_ticks, 3)
            self.assertEqual(config.market_b.meanrev_max_hold_ms, 120_000)
            self.assertEqual(config.market_b.meanrev_cooldown_ms, 1_500)
            self.assertEqual(config.market_b.meanrev_aggressive_entry_z, 2.75)
            self.assertTrue(config.market_b.meanrev_aggressive_exit)
            self.assertEqual(config.market_b.meanrev_entry_ticks, 10)
            self.assertEqual(config.market_b.meanrev_full_entry_ticks, 15)
            self.assertEqual(config.market_b.meanrev_exit_ticks, 3)
            self.assertEqual(config.market_b.meanrev_base_target, 6)
            self.assertEqual(config.market_b.meanrev_full_target, 16)
            self.assertEqual(config.market_b.meanrev_extreme_entry_ticks, 20)
            self.assertEqual(config.market_b.meanrev_risk_off_deviation_ticks, 35)
            self.assertEqual(config.market_b.meanrev_turn_confirm_ms, 300)
            self.assertEqual(config.market_b.meanrev_min_healthy_book_age_ms, 500)
            self.assertEqual(config.market_b.meanrev_bad_fill_cooldown_ms, 1_000)
            self.assertTrue(config.market_b.option_lottery_enabled)
            self.assertEqual(config.market_b.option_lottery_max_ask, 3)
            self.assertEqual(config.market_b.option_lottery_total_premium_budget, 1_500)
            self.assertEqual(config.market_b.option_lottery_wing_max_position, 200)
            self.assertEqual(config.market_b.option_lottery_atm_max_position, 40)
            self.assertEqual(config.market_b.option_lottery_wing_premium_budget, 600)
            self.assertEqual(config.market_b.option_lottery_c1050_premium_budget, 0)
            self.assertEqual(config.market_b.option_lottery_p950_premium_budget, 450)
            self.assertEqual(config.market_b.option_lottery_atm_total_premium_budget, 300)
            self.assertTrue(config.market_b.option_lottery_profit_take_enabled)
            self.assertTrue(config.market_b.option_hedge_enabled)
            self.assertEqual(config.market_b.option_hedge_max_ask, 6)
            self.assertEqual(config.market_b.option_hedge_min_underlying_inventory, 4)
            self.assertEqual(config.market_b.option_hedge_target_ratio, 0.5)
            self.assertEqual(config.market_b.option_hedge_premium_budget, 300)
            self.assertTrue(config.etf.enabled)
            self.assertTrue(config.etf.trading_enabled)
            self.assertEqual(config.etf.alpha_from_a, 0.60)
            self.assertEqual(config.etf.max_position, 100)
            self.assertEqual(config.etf.quote_size, 16)
            self.assertEqual(config.etf.target_position_per_a_shock_inventory, 0.35)
            self.assertEqual(config.etf.min_hold_ms, 3_000)
            self.assertEqual(config.etf.min_eval_interval_ms, 100)
            self.assertEqual(config.etf.unwind_reprice_threshold_ticks, 8)
            self.assertEqual(config.etf.entry_retry_window_ms, 1_500)
            self.assertEqual(config.etf.entry_force_aggressive_ms, 250)
            self.assertEqual(config.etf.entry_retry_reprice_ms, 125)
            self.assertEqual(config.etf.churn_window_ms, 250)
            self.assertEqual(config.etf.churn_max_top_of_book_updates, 25)
            self.assertEqual(config.etf.churn_resume_stable_ms, 500)
            self.assertTrue(config.auto_stop_after_round_complete)
            self.assertEqual(config.assumed_round_duration_ms, 900_000)
            self.assertEqual(config.round_completion_grace_ms, 5_000)
            self.assertFalse(config.auto_stop_on_followup_position_snapshot)
            self.assertFalse(config.auto_stop_on_market_resolved)

    def test_recovered_pricing_state_is_ignored_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = TradingJournal(Path(temp_dir) / "journal.jsonl")
            submitted = ManagedOrder(
                order_id="order-1",
                side="SELL",
                px=901,
                qty=2,
                remaining_qty=2,
                submitted_ms=12,
            )
            journal.record_order_submitted(submitted)
            journal.record_multiplier(1050.0, confidence=2, source="updated", estimate=1055.0, method="stable_level")
            journal.record_fair_value(945, source="learned_multiplier", earnings_value=0.9)

            replay = journal.load_replay_state()
            ignored = select_recovered_pricing_state(replay, recover_pricing_state=False)
            restored = select_recovered_pricing_state(replay, recover_pricing_state=True)

            self.assertEqual(ignored, (None, 0, None, None))
            self.assertEqual(restored, (1050.0, 2, 945, 0.9))

    def test_summarize_trace_events_computes_fill_markouts(self) -> None:
        events = [
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "mode": "STEADY_MM",
                "mid": 1000.0,
                "inventory": 0,
                "quoted_spread": 2,
                "observe_only": False,
                "desired_bid": {"px": 999, "qty": 1},
                "desired_ask": {"px": 1001, "qty": 1},
                "aggressive_action_count": 0,
                "reason": "steady mm",
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "mode": "STEADY_MM",
                "side": "BUY",
                "price": 999,
                "qty": 1,
                "aggressive": False,
                "intent": "steady_mm_passive",
                "mid_at_event": 1000.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "mode": "STEADY_MM",
                "mid": 1002.0,
                "inventory": 1,
                "quoted_spread": 2,
            },
            {
                "event_type": "session_end",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "mode": "STEADY_MM",
            },
        ]
        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        self.assertEqual(summary["passive_fills"], 1)
        self.assertIn("steady_mm_passive", summary["fill_markouts_by_intent"])
        self.assertAlmostEqual(summary["fill_markouts_by_intent"]["steady_mm_passive"]["250ms"], 3.0, delta=0.001)

    def test_summarize_trace_events_tracks_earnings_prejump_intent(self) -> None:
        events = [
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "mode": "PRE_NEWS_PULLBACK",
                "overlay": "earnings",
                "intent": "earnings_prejump",
                "aggressive": True,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "mode": "PRE_NEWS_PULLBACK",
                "overlay": "earnings",
                "side": "BUY",
                "price": 999,
                "qty": 1,
                "aggressive": True,
                "intent": "earnings_prejump",
                "mid_at_event": 1000.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "mode": "POST_EARNINGS_SHOCK",
                "mid": 1006.0,
                "inventory": 1,
                "quoted_spread": 2,
                "budget_shift_active": True,
            },
        ]
        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        self.assertEqual(summary["fills_by_intent"]["earnings_prejump"], 1)
        self.assertEqual(summary["submits_by_intent"]["earnings_prejump"], 1)
        self.assertEqual(summary["activity_split"]["earnings_prejump"], 1)
        self.assertAlmostEqual(summary["fill_markouts_by_intent"]["earnings_prejump"]["250ms"], 7.0, delta=0.001)

    def test_summarize_trace_events_reports_inventory_divergence(self) -> None:
        events = [
            {
                "event_type": "inventory_updated",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "inventory": 12,
                "strategy_inventory": 12,
                "exchange_inventory": 12,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "symbol": "ETF",
                "market_key": "ETF",
                "inventory": 24,
                "strategy_inventory": 24,
                "exchange_inventory": 12,
            },
        ]

        summary = summarize_trace_events(events)

        self.assertEqual(summary["inventory_divergence_summary"]["ETF"]["divergent_sample_count"], 1)
        self.assertEqual(summary["inventory_divergence_summary"]["ETF"]["max_abs_difference"], 12)

    def test_summarize_trace_events_reports_b_option_lottery_summary(self) -> None:
        events = [
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "B_C_1050",
                "market_key": "B",
                "side": "BUY",
                "price": 3,
                "qty": 10,
                "fill_price": 3,
                "fill_qty": 10,
                "pnl_owner": "b_option_lottery:B_C_1050",
                "strategy_family": "b_option_lottery",
                "action_class": "cheap_option_buy",
                "inventory": 10,
                "mark_price": 3,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "B_C_1050",
                "market_key": "B",
                "inventory": 10,
                "mark_price": 9,
            },
        ]

        summary = summarize_trace_events(events)
        option_summary = summary["b_option_lottery_summary"]["by_symbol"]["B_C_1050"]

        self.assertEqual(option_summary["buy_qty"], 10)
        self.assertEqual(option_summary["premium_spent"], 30.0)
        self.assertEqual(option_summary["final_inventory"], 10)
        self.assertEqual(option_summary["mtm_pnl"], 60.0)
        self.assertEqual(option_summary["mtm_pnl_from_fills"], 60.0)

    def test_b_option_lottery_max_mark_ignores_pre_entry_marks(self) -> None:
        events = [
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 900,
                "symbol": "B_P_950",
                "market_key": "B",
                "inventory": 0,
                "mark_price": 20,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "B_P_950",
                "market_key": "B",
                "side": "BUY",
                "price": 3,
                "qty": 4,
                "fill_price": 3,
                "fill_qty": 4,
                "pnl_owner": "b_option_lottery:B_P_950",
                "strategy_family": "b_option_lottery",
                "action_class": "cheap_option_buy",
                "inventory": 4,
                "mark_price": 3,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "B_P_950",
                "market_key": "B",
                "inventory": 4,
                "mark_price": 9,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "B_P_950",
                "market_key": "B",
                "inventory": 0,
                "mark_price": 50,
            },
        ]

        summary = summarize_trace_events(events)
        option_summary = summary["b_option_lottery_summary"]["by_symbol"]["B_P_950"]

        self.assertEqual(option_summary["max_mark_after_first_buy"], 9.0)
        self.assertEqual(option_summary["open_qty_from_fills"], 4)
        self.assertEqual(option_summary["open_mark_value_from_fills"], 200.0)

    def test_summarize_trace_events_uses_latest_portfolio_marks_for_final_mtm(self) -> None:
        events = [
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "inventory": 5,
                "mid": 10.0,
                "cash": 100,
            },
            {
                "event_type": "session_end",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "A",
                "market_key": "A",
                "inventory": 0,
                "mid": 0.0,
                "cash": 100,
            },
        ]

        summary = summarize_trace_events(events)

        self.assertEqual(summary["estimated_final_mtm_pnl"], 150.0)
        self.assertEqual(summary["estimated_final_mtm_basis"], "portfolio_latest_marks")

    def test_summarize_trace_events_rolls_up_pnl_by_market_and_strategy(self) -> None:
        events = [
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "A",
                "market_key": "A",
                "pnl_owner": "a_earnings",
                "action_class": "shock_take",
                "intent": "post_earnings_shock_take",
                "side": "BUY",
                "fill_price": 1000,
                "fill_qty": 2,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "A",
                "market_key": "A",
                "mid": 1010.0,
                "inventory": 2,
            },
        ]
        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        self.assertAlmostEqual(summary["pnl_by_market"]["A"], 20.0, delta=0.001)
        self.assertAlmostEqual(summary["pnl_by_strategy_family"]["a_earnings"], 20.0, delta=0.001)

    def test_summarize_trace_events_suppresses_unreliable_fill_side_attribution(self) -> None:
        events = [
            {
                "event_type": "inventory_updated",
                "run_id": "run-1",
                "monotonic_ms": 900,
                "symbol": "A",
                "market_key": "A",
                "inventory": 0,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "A",
                "market_key": "A",
                "pnl_owner": "a_earnings",
                "action_class": "shock_take",
                "intent": "post_earnings_shock_take",
                "side": "BUY",
                "fill_price": 1000,
                "fill_qty": 10,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "A",
                "market_key": "A",
                "mid": 1100.0,
            },
            {
                "event_type": "session_end",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "A",
                "market_key": "A",
                "inventory": 0,
                "mid": 1100.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))

        self.assertFalse(summary["attribution_reliability"]["A"]["reliable"])
        self.assertNotIn("A", summary["pnl_by_market"])
        self.assertNotIn("a_earnings", summary["pnl_by_strategy_family"])

    def test_b_observer_computes_parity_residuals_and_synthetic_fair(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        observer.on_book_update("B", FakeOrderBook(bids={1098: 10}, asks={1102: 10}))
        observer.on_book_update("B_C_950", FakeOrderBook(bids={150: 10}, asks={154: 10}))
        observer.on_book_update("B_P_950", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={100: 10}, asks={104: 10}))
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={50: 10}, asks={54: 10}))
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={0: 10}, asks={4: 10}))

        bundle = observer.derived_signal_bundle()
        self.assertIsNotNone(bundle)
        payload = bundle.payload
        self.assertAlmostEqual(payload["synthetic_forward_by_strike"]["1000"], 1100.0, delta=0.001)
        self.assertAlmostEqual(payload["parity_residual_by_strike"]["1000"], 0.0, delta=0.001)
        self.assertAlmostEqual(payload["estimated_aggressive_crossing_cost_by_strike"]["1000"], 6.0, delta=0.001)
        self.assertAlmostEqual(payload["parity_edge_after_cost_by_strike"]["1000"], -6.0, delta=0.001)

    def test_b_observer_downsamples_until_threshold_or_interval(self) -> None:
        observer = MarketBObserver(depth_levels=5, signal_snapshot_interval_ms=250, signal_change_threshold_ticks=1)
        observer.on_book_update("B", FakeOrderBook(bids={1098: 10}, asks={1102: 10}))
        observer.on_book_update("B_C_950", FakeOrderBook(bids={150: 10}, asks={154: 10}))
        observer.on_book_update("B_P_950", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={100: 10}, asks={104: 10}))
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={50: 10}, asks={54: 10}))
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={0: 10}, asks={4: 10}))

        self.assertIsNotNone(observer.derived_signal_bundle(now_ms=1_000))
        self.assertIsNone(observer.derived_signal_bundle(now_ms=1_100))

        observer.on_book_update("B_C_1000", FakeOrderBook(bids={104: 10}, asks={108: 10}))
        self.assertIsNotNone(observer.derived_signal_bundle(now_ms=1_150))

    def test_summarize_trace_events_reports_b_adverse_selection_stats(self) -> None:
        events = [
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "B",
                "market_key": "B",
                "pnl_owner": "b_underlying_mm_v2",
                "action_class": "reduce_only",
                "intent": "b_underlying_mm_v2_passive",
                "side": "SELL",
                "fill_price": 1000,
                "fill_qty": 1,
                "spread": 0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "B",
                "market_key": "B",
                "mid": 999.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))

        self.assertEqual(summary["b_adverse_selection_stats"]["fill_count"], 1)
        self.assertEqual(summary["b_adverse_selection_stats"]["crossed_or_locked_fill_count"], 1)
        self.assertEqual(summary["b_adverse_selection_stats"]["reduce_only_fill_count"], 1)

    def test_summarize_trace_events_reports_b_mean_reversion_summary(self) -> None:
        events = [
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "B",
                "market_key": "B",
                "mode": "B_MEANREV_ENTRY",
                "inventory": 0,
                "b_meanrev_z": 1.5,
                "desired_bid": None,
                "desired_ask": {"strategy_family": "b_mean_reversion", "px": 1004, "qty": 1},
                "aggressive_action_count": 0,
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_010,
                "symbol": "B",
                "market_key": "B",
                "mode": "B_MEANREV_ENTRY",
                "strategy_family": "b_mean_reversion",
                "pnl_owner": "b_mean_reversion",
                "action_class": "mean_reversion_entry",
                "b_meanrev_z": 1.5,
                "b_meanrev_deviation_ticks": 10.0,
                "b_meanrev_mean_reference": 1000.0,
                "b_meanrev_mean_reference_ema_component": 650.0,
                "b_meanrev_mean_reference_synth_component": 350.0,
                "b_meanrev_entry_style": "normal",
                "side": "SELL",
                "price": 1004,
                "qty": 1,
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_015,
                "symbol": "B",
                "market_key": "B",
                "mode": "B_MEANREV_RISK_OFF",
                "strategy_family": "b_mean_reversion",
                "pnl_owner": "b_mean_reversion",
                "action_class": "mean_reversion_risk_off",
                "b_meanrev_risk_off_forced": False,
                "b_meanrev_deviation_ticks": 36.0,
                "side": "BUY",
                "price": 1001,
                "qty": 1,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_020,
                "symbol": "B",
                "market_key": "B",
                "strategy_family": "b_mean_reversion",
                "pnl_owner": "b_mean_reversion",
                "action_class": "mean_reversion_entry",
                "intent": "b_mean_reversion",
                "b_meanrev_entry_style": "normal",
                "side": "SELL",
                "fill_price": 1004,
                "fill_qty": 1,
                "spread": 6,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_270,
                "symbol": "B",
                "market_key": "B",
                "mid": 1000.0,
            },
            {
                "event_type": "session_end",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "B",
                "market_key": "B",
                "inventory": -1,
                "mid": 1000.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))

        meanrev = summary["b_mean_reversion_summary"]
        self.assertEqual(meanrev["decision_count"], 1)
        self.assertEqual(meanrev["quote_count"], 1)
        self.assertEqual(meanrev["entry_count"], 1)
        self.assertEqual(meanrev["risk_off_count"], 1)
        self.assertEqual(meanrev["risk_off_hold_or_passive_reduce_count"], 1)
        self.assertEqual(meanrev["risk_off_forced_exit_count"], 0)
        self.assertEqual(meanrev["fill_count"], 1)
        self.assertEqual(meanrev["fill_qty"], 1)
        self.assertEqual(meanrev["avg_entry_abs_z"], 1.5)
        self.assertEqual(meanrev["avg_entry_abs_deviation"], 10.0)
        self.assertEqual(meanrev["avg_entry_mean_reference"], 1000.0)
        self.assertEqual(meanrev["avg_entry_mean_reference_ema_component"], 650.0)
        self.assertEqual(meanrev["avg_entry_mean_reference_synth_component"], 350.0)
        self.assertEqual(meanrev["entry_style_counts"]["normal"], 1)
        self.assertGreater(meanrev["entry_markouts"]["250ms"], 0)
        self.assertGreater(meanrev["entry_markouts_normal"]["250ms"], 0)

    def test_summarize_trace_events_reports_a_episode_mm_loss_and_b_cost_stats(self) -> None:
        events = [
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 900,
                "symbol": "B",
                "market_key": "B",
                "mode": "OBSERVE_ONLY",
                "observe_only": True,
                "reason": "synthetic_dispersion_wide",
                "aggressive_action_count": 0,
                "desired_bid": None,
                "desired_ask": None,
            },
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_EARNINGS_SHOCK",
                "relevant": True,
                "current_earnings_signal_id": "a_eps_1",
                "mode_before_news": "STEADY_MM",
                "old_fair_value": 1000,
                "new_fair_value": 1080,
                "raw_payload": {
                    "kind": "structured",
                    "new_data": {"asset": "A", "structured_subtype": "earnings", "value": 1.08},
                },
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_EARNINGS_SHOCK",
                "pnl_owner": "a_earnings",
                "action_class": "shock_take",
                "intent": "post_earnings_shock_take",
                "signal_id": "a_eps_1",
                "side": "BUY",
                "fill_price": 1000,
                "fill_qty": 12,
                "inventory": 12,
                "earnings_position": 12,
                "active_earnings_cycle_id": "a_eps_1",
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_200,
                "symbol": "A",
                "market_key": "A",
                "mode": "UNWIND",
                "mode_at_submit": "STEADY_MM",
                "pnl_owner": "a_market_making",
                "action_class": "market_making",
                "intent": "steady_mm_passive",
                "side": "BUY",
                "fill_price": 1002,
                "fill_qty": 1,
                "inventory": 13,
                "earnings_position": 13,
                "active_earnings_cycle_id": "a_eps_1",
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "A",
                "market_key": "A",
                "mode": "UNWIND",
                "mid": 1010.0,
                "inventory": 40,
                "earnings_position": 40,
            },
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_EARNINGS_SHOCK",
                "relevant": True,
                "current_earnings_signal_id": "a_eps_2",
                "mode_before_news": "UNWIND",
                "old_fair_value": 1080,
                "new_fair_value": 1040,
                "raw_payload": {
                    "kind": "structured",
                    "new_data": {"asset": "A", "structured_subtype": "earnings", "value": 1.04},
                },
            },
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 2_100,
                "symbol": "B",
                "market_key": "B",
                "strategy_family": "b_observe_only",
                "payload": {
                    "parity_residual_by_strike": {"1000": 4.0},
                    "parity_edge_after_cost_by_strike": {"1000": 1.5},
                    "composite_basis": 0.75,
                    "call_monotonicity_violations": [],
                    "put_monotonicity_violations": [],
                    "vertical_spread_bound_violations": [],
                    "butterfly_convexity_violations": [],
                },
            },
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 2_400,
                "symbol": "B",
                "market_key": "B",
                "strategy_family": "b_observe_only",
                "payload": {
                    "parity_residual_by_strike": {"1000": 2.0},
                    "parity_edge_after_cost_by_strike": {"1000": -0.5},
                    "composite_basis": 0.25,
                    "call_monotonicity_violations": [],
                    "put_monotonicity_violations": [],
                    "vertical_spread_bound_violations": [],
                    "butterfly_convexity_violations": [],
                },
            },
        ]
        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        self.assertEqual(summary["a_episode_summaries"][0]["shock_take_qty"], 12)
        self.assertTrue(summary["a_episode_summaries"][0]["next_earnings_arrived_during_unwind"])
        self.assertEqual(summary["a_episode_summaries"][0]["peak_total_inventory"], 40)
        self.assertIn("STEADY_MM", summary["a_mm_loss_by_mode"])
        self.assertEqual(summary["a_mm_loss_by_mode"]["fills_inside_earnings_cycle"], 1)
        self.assertEqual(summary["pnl_by_action_class"]["A.shock_take"], 120.0)
        self.assertIn("shock_take_pnl", summary["a_strategy_breakdown"])
        self.assertEqual(summary["b_cost_adjusted_residual_stats"]["by_strike"]["1000"]["positive_edge_count"], 1)
        self.assertEqual(
            summary["b_cost_adjusted_residual_stats"]["by_strike"]["1000"]["positive_edge_persistence_ms"],
            300,
        )
        self.assertEqual(summary["b_strategy_block_reasons"]["synthetic_dispersion_wide"], 1)
        self.assertEqual(summary["trace_volume_summary"]["event_counts"]["derived_signal"], 2)

    def test_a_news_tracker_joins_ayush_port_orders_by_signal_id(self) -> None:
        events = [
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "exchange_tick": 100,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_NEWS_SHOCK",
                "news_kind": "unstructured",
                "relevant": True,
                "content": "A takes a leading position in a growing niche market.",
                "news_sentiment_score": 3.8,
                "news_sentiment_bucket": "strong",
                "news_matched_bigrams": ["leading position", "growing niche"],
                "active_news_signal_id": "ayush_news_1",
                "pending_news_signal_id": "ayush_news_1",
                "news_fair_value": 1048,
                "news_target_inventory": 90,
                "mid": 1000.0,
                "raw_payload": {
                    "kind": "unstructured",
                    "symbol": "A",
                    "new_data": {"content": "A takes a leading position in a growing niche market."},
                },
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_010,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_take",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "intent": "post_news_shock_take",
                "side": "BUY",
                "qty": 12,
                "price": 1002,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_020,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_take",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "intent": "post_news_shock_take",
                "side": "BUY",
                "fill_qty": 12,
                "fill_price": 1002,
                "mid": 1002.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 4_000,
                "symbol": "A",
                "market_key": "A",
                "mid": 1030.0,
            },
        ]

        report = build_a_news_tracker_report(events, config=AConfig())
        row = report["headline_analyses"][0]
        self.assertTrue(row["traded"])
        self.assertEqual(row["signal_id"], "ayush_news_1")
        self.assertEqual(row["first_desired_side"], "BUY")
        self.assertEqual(row["first_desired_qty"], 12)
        self.assertEqual(row["first_fill_side"], "BUY")
        self.assertEqual(row["first_fill_qty"], 12)
        self.assertEqual(row["target_inventory"], 90)
        self.assertNotEqual(row["verdict"], "missed_no_trade")

    def test_summarize_trace_events_reports_news_episodes_and_counterfactuals(self) -> None:
        events = [
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "exchange_tick": 100,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_NEWS_SHOCK",
                "news_kind": "unstructured",
                "relevant": True,
                "content": "A takes a leading position in a growing niche market.",
                "news_sentiment_score": 3.8,
                "news_sentiment_bucket": "strong",
                "active_news_signal_id": "ayush_news_1",
                "pending_news_signal_id": "ayush_news_1",
                "news_fair_value": 1048,
                "news_target_inventory": 90,
                "mid": 1000.0,
                "raw_payload": {
                    "kind": "unstructured",
                    "symbol": "A",
                    "new_data": {"content": "A takes a leading position in a growing niche market."},
                },
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_010,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_take",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "side": "BUY",
                "fill_qty": 12,
                "fill_price": 1001,
                "news_position": 12,
                "mid": 1001.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 3_000,
                "symbol": "A",
                "market_key": "A",
                "mode": "POST_NEWS_SHOCK",
                "news_position": 12,
                "mid": 1030.0,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 4_000,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_unwind",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "side": "SELL",
                "fill_qty": 12,
                "fill_price": 1020,
                "news_position": 0,
                "mid": 1020.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 6_000,
                "symbol": "A",
                "market_key": "A",
                "mode": "AYUSH_IDLE",
                "news_position": 0,
                "mid": 1040.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        row = summary["a_news_episode_summaries"][0]
        self.assertEqual(row["signal_id"], "ayush_news_1")
        self.assertEqual(row["target_inventory"], 90)
        self.assertEqual(row["fill_qty"], 24)
        self.assertEqual(row["peak_news_inventory"], 12)
        self.assertEqual(row["avg_entry_px"], 1001.0)
        self.assertEqual(row["avg_exit_px"], 1020.0)
        self.assertIn("hold_2s", row["counterfactual_holds"])
        self.assertGreater(row["counterfactual_holds"]["hold_2s"]["estimated_pnl"], row["realized_episode_pnl"])

    def test_a_news_episode_summary_ignores_later_signal_inventory(self) -> None:
        events = [
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "A",
                "market_key": "A",
                "news_kind": "unstructured",
                "relevant": True,
                "content": "A's revolutionary battery breakthrough is widely praised.",
                "news_sentiment_score": 2.5,
                "news_sentiment_bucket": "strong",
                "current_news_signal_id": "ayush_news_1",
                "news_target_inventory": 80,
                "raw_payload": {"kind": "unstructured", "new_data": {"asset": "A"}},
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_take",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "side": "BUY",
                "fill_qty": 8,
                "fill_price": 100,
                "news_position": 8,
                "mid": 100.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_400,
                "symbol": "A",
                "market_key": "A",
                "news_position": 8,
                "mid": 101.0,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_900,
                "symbol": "A",
                "market_key": "A",
                "strategy_family": "a_news",
                "action_class": "news_unwind",
                "pnl_owner": "a_news",
                "signal_id": "ayush_news_1",
                "side": "SELL",
                "fill_qty": 8,
                "fill_price": 103,
                "news_position": 0,
                "mid": 103.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 4_000,
                "symbol": "A",
                "market_key": "A",
                "news_position": 20,
                "mid": 95.0,
            },
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 5_000,
                "symbol": "A",
                "market_key": "A",
                "news_kind": "unstructured",
                "relevant": True,
                "content": "A expands revenue streams.",
                "news_sentiment_score": 1.8,
                "news_sentiment_bucket": "medium",
                "current_news_signal_id": "ayush_news_2",
                "news_target_inventory": 60,
                "raw_payload": {"kind": "unstructured", "new_data": {"asset": "A"}},
            },
        ]

        summary = summarize_trace_events(events)
        row = summary["a_news_episode_summaries"][0]

        self.assertEqual(row["signal_id"], "ayush_news_1")
        self.assertEqual(row["peak_news_inventory"], 8)

    def test_etf_episode_summary_tracks_handoff_unwind_and_pending_reason(self) -> None:
        events = [
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 900,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 3000.0,
                "inventory": 0,
            },
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "a_shock_projection",
                "signal_id": "etf_a_1",
                "etf_signal_id": "etf_a_1",
                "payload": {
                    "source_kind": "structured_earnings",
                    "alpha": 0.6,
                    "a_fair_shift": -120.0,
                    "base_mid": 3000.0,
                    "target_fair": 2928.0,
                    "target_inventory": -60,
                },
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_010,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "signal_id": "etf_a_1",
                "action_class": "etf_shock_take",
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_020,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_take",
                "signal_id": "etf_a_1",
                "side": "SELL",
                "fill_qty": 16,
                "fill_price": 2995,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_030,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 2990.0,
                "inventory": -16,
                "etf_signal_id": "etf_a_1",
            },
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "a_shock_projection",
                "signal_id": "etf_a_2",
                "etf_signal_id": "etf_a_1",
                "block_reason": "etf_signal_handoff_pending",
                "etf_missed_entry_terminal_reason": "handoff_flatten",
                "payload": {
                    "source_kind": "structured_earnings",
                    "alpha": 0.6,
                    "a_fair_shift": 140.0,
                    "base_mid": 2990.0,
                    "target_fair": 3074.0,
                    "target_inventory": 70,
                },
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_110,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "signal_id": "etf_a_1",
                "action_class": "etf_shock_unwind",
                "reason": "handoff_flatten",
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_120,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_unwind",
                "signal_id": "etf_a_1",
                "side": "BUY",
                "fill_qty": 16,
                "fill_price": 3005,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_130,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 3002.0,
                "inventory": 0,
            },
            {
                "event_type": "session_end",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 3002.0,
                "inventory": 0,
            },
        ]

        summary = summarize_trace_events(events)
        first_row, second_row = summary["etf_episode_summaries"]

        self.assertEqual(first_row["signal_id"], "etf_a_1")
        self.assertEqual(first_row["entry_qty"], 16)
        self.assertEqual(first_row["unwind_qty"], 16)
        self.assertEqual(first_row["peak_inventory"], -16)
        self.assertEqual(first_row["avg_exit"], 3005.0)

        self.assertEqual(second_row["signal_id"], "etf_a_2")
        self.assertEqual(second_row["entry_qty"], 0)
        self.assertEqual(second_row["peak_inventory"], 0)
        self.assertEqual(second_row["first_block_reason"], "etf_signal_handoff_pending")
        self.assertEqual(second_row["missed_entry_terminal_reason"], "handoff_flatten")
        self.assertEqual(summary["etf_missed_entry_summary"]["by_reason"], {"handoff_flatten": 1})

    def test_b_shadow_underlying_mm_summary_is_observe_only(self) -> None:
        events = [
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "B",
                "market_key": "B",
                "strategy_family": "b_observe_only",
                "best_bid_px": 999,
                "best_ask_px": 1005,
                "spread": 6,
                "microprice": 1002.0,
                "top_of_book_imbalance": 0.0,
                "payload": {
                    "composite_synthetic_fair": 1006.0,
                    "composite_basis": 4.0,
                    "synthetic_forward_by_strike": {"950": 1005.0, "1000": 1006.0, "1050": 1007.0},
                    "parity_residual_by_strike": {"1000": 1.0},
                    "parity_edge_after_cost_by_strike": {"1000": -5.0},
                },
            },
            {
                "event_type": "market_trade_observed",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "symbol": "B",
                "market_key": "B",
                "price": 999,
                "qty": 1,
                "mid": 1000.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 6_500,
                "symbol": "B",
                "market_key": "B",
                "mid": 1004.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        shadow = summary["b_shadow_underlying_mm"]
        self.assertTrue(shadow["observe_only"])
        self.assertEqual(shadow["candidate_quote_count"], 1)
        self.assertEqual(shadow["hypothetical_fill_count"], 1)
        self.assertIn("BUY", shadow["hypothetical_quote_counts"])

    def test_etf_a_shock_calibration_reports_realized_response_ratios(self) -> None:
        events = [
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "A",
                "market_key": "A",
                "mid": 1000.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 500.0,
            },
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "a_shock_projection",
                "signal_id": "etf_a_1",
                "payload": {
                    "source_kind": "structured_earnings",
                    "alpha": 0.25,
                    "a_fair_shift": 100.0,
                    "projected_etf_shift": 25.0,
                    "base_mid": 500.0,
                    "target_inventory": 25,
                },
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "A",
                "market_key": "A",
                "mid": 1080.0,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 520.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        calibration = summary["etf_a_shock_calibration"]

        self.assertEqual(calibration["signal_count"], 1)
        self.assertEqual(calibration["by_horizon_ms"]["1000"]["sample_count"], 1)
        self.assertEqual(calibration["by_horizon_ms"]["1000"]["mean_etf_over_a_fair_shift"], 0.2)
        self.assertEqual(calibration["by_horizon_ms"]["1000"]["mean_configured_alpha"], 0.25)
        self.assertEqual(calibration["by_source_kind"]["structured_earnings"]["1000"]["directional_hit_rate"], 1.0)

    def test_summarize_trace_events_reports_etf_entry_summary(self) -> None:
        events = [
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "a_shock_projection",
                "signal_id": "etf_a_1",
                "payload": {
                    "source_kind": "structured_earnings",
                    "alpha": 0.60,
                    "a_fair_shift": 100.0,
                    "projected_etf_shift": 60.0,
                    "base_mid": 500.0,
                    "target_inventory": 60,
                    "target_fair": 560.0,
                },
            },
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_010,
                "symbol": "ETF",
                "market_key": "ETF",
                "mode": "ETF_CHURN_GUARD",
                "etf_signal_id": "etf_a_1",
                "reason": "etf_quote_churn_guard",
                "observe_only": True,
                "aggressive_action_count": 0,
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_120,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_take",
                "signal_id": "etf_a_1",
                "side": "BUY",
                "qty": 16,
                "price": 502,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_160,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_take",
                "signal_id": "etf_a_1",
                "side": "BUY",
                "fill_qty": 16,
                "fill_price": 502,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 520.0,
                "inventory": 16,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))

        etf_entry = summary["etf_entry_summary"]
        self.assertEqual(etf_entry["signal_count"], 1)
        self.assertEqual(etf_entry["total_entry_attempt_count"], 1)
        self.assertEqual(etf_entry["avg_first_order_attempt_latency_ms"], 120)
        self.assertEqual(etf_entry["avg_first_fill_latency_ms"], 160)
        self.assertEqual(etf_entry["mean_target_fill_ratio"], round(16 / 60, 4))
        self.assertEqual(etf_entry["churn_guard_count"], 1)

    def test_summarize_trace_events_reports_c_strategy_summary(self) -> None:
        events = [
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "C",
                "market_key": "C",
                "strategy_family": "c_earnings",
                "action_class": "earnings_signal",
                "signal_id": "c_earnings_1",
                "payload": {
                    "signal_id": "c_earnings_1",
                    "tick": 200,
                    "fair_before": 1000.0,
                    "fair_after": 1080.0,
                    "fair_shift_ticks": 80.0,
                    "edge": 78.0,
                    "target_inventory": 80,
                    "live_trading_enabled": True,
                },
                "c_market_rate_bp": 0.0,
                "c_effective_rate_bp": 0.0,
                "c_cpi_bias_bp": 0.0,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_020,
                "symbol": "C",
                "market_key": "C",
                "strategy_family": "c_earnings",
                "pnl_owner": "c_earnings",
                "action_class": "shock_take",
                "signal_id": "c_earnings_1",
                "side": "BUY",
                "fill_qty": 10,
                "fill_price": 1002,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "C",
                "market_key": "C",
                "mid": 1040.0,
                "inventory": 10,
                "c_market_rate_bp": 0.0,
                "c_effective_rate_bp": 0.0,
                "c_cpi_bias_bp": 0.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))

        self.assertEqual(summary["pnl_by_market"]["C"], 380.0)
        self.assertEqual(summary["c_summary"]["earnings_signal_count"], 1)
        self.assertEqual(summary["c_summary"]["fill_count"], 1)
        self.assertEqual(summary["c_signal_episode_summaries"][0]["signal_id"], "c_earnings_1")
        self.assertEqual(summary["c_rate_context_summary"]["sample_count"], 2)

    def test_etf_episode_summary_tracks_c_source_attribution(self) -> None:
        events = [
            {
                "event_type": "derived_signal",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "c_shock_projection",
                "signal_id": "etf_c_1",
                "payload": {
                    "source_market": "C",
                    "source_combo": "C_only",
                    "source_kind": "structured_earnings",
                    "source_fair_shift": 120.0,
                    "alpha": 0.25,
                    "base_mid": 500.0,
                    "target_fair": 530.0,
                    "target_inventory": 30,
                },
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_050,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_take",
                "signal_id": "etf_c_1",
                "side": "BUY",
                "qty": 8,
                "price": 502,
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 1_060,
                "symbol": "ETF",
                "market_key": "ETF",
                "strategy_family": "etf_a_follower",
                "action_class": "etf_shock_take",
                "signal_id": "etf_c_1",
                "side": "BUY",
                "fill_qty": 8,
                "fill_price": 502,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "symbol": "ETF",
                "market_key": "ETF",
                "mid": 520.0,
                "inventory": 8,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        row = summary["etf_episode_summaries"][0]

        self.assertEqual(row["signal_id"], "etf_c_1")
        self.assertEqual(row["source_market"], "C")
        self.assertEqual(row["source_combo"], "C_only")
        self.assertEqual(row["source_fair_shift"], 120.0)


if __name__ == "__main__":
    unittest.main()
