from __future__ import annotations

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
                pre_news_arrival_grace_ms=1_200,
                calibration_min_delay_ms=5_000,
                calibration_max_delay_ms=20_000,
                calibration_sample_period_ms=1_000,
                calibration_stability_band_ticks=8,
                recover_pricing_state=False,
                max_order_qty=39,
                steady_quote_size=2,
                steady_max_position=24,
                steady_take_min_edge=2,
                steady_take_large_inventory_edge=4,
                discovery_max_position=4,
                discovery_half_spread_ticks=8,
                news_caution_max_position=4,
                news_caution_half_spread_ticks=8,
                steady_take_inventory_guard=8,
                unwind_entry_position=24,
                unwind_exit_position=12,
                earnings_unwind_quote_size=3,
                earnings_unwind_aggressive_quote_size=3,
                earnings_unwind_rapid_entry=9_999,
                earnings_unwind_rapid_exit=9_998,
                earnings_unwind_rapid_take_edge=4,
                earnings_unwind_aggressive_entry=48,
                earnings_unwind_aggressive_exit=24,
                earnings_unwind_passive_exit=8,
                earnings_unwind_passive_take_edge=8,
                shock_quote_size=12,
                shock_entry_window_ms=1_000,
                shock_entry_quote_size=24,
                shock_entry_min_edge=2,
                shock_entry_threshold_scale=0.50,
                shock_accumulate_target_position=180,
                shock_accumulate_min_quote_size=12,
                shock_accumulate_max_quote_size=12,
                shock_accumulate_min_edge=4,
                shock_accumulate_threshold_scale=1.0,
                shock_accumulate_window_ms=3_000,
                shock_base_max_position=80,
                shock_shift_max_position=160,
                shock_settle_min_hold_ms=1_200,
                shock_settle_max_hold_ms=4_000,
                shock_settle_band_ticks=8,
                shock_settle_drift_ticks=4,
                shock_settle_confirmations=2,
                shock_unwind_quote_size=3,
                shock_unwind_aggressive_quote_size=3,
                shock_unwind_take_edge=4,
                shock_unwind_exit_position=8,
                prejump_window_ms=1_200,
                prejump_max_position=24,
                prejump_quote_size=6,
                prejump_aggressive_edge=2,
            ),
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_min_rest_ms=0,
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
        self.assertEqual(state["earnings_phase"], "IDLE")
        self.assertEqual(state["mm_phase"], "STEADY_MM")
        self.assertFalse(state["discovery_contaminated"])
        self.assertFalse(state["pre_news_hold_active"])
        self.assertFalse(state["rapid_unwind_active"])
        self.assertEqual(state["shock_stage"], "NONE")
        self.assertEqual(state["shock_accumulate_target_position"], 180)
        self.assertIsNone(state["shock_target_total_inventory"])
        self.assertIsNone(state["shock_remaining_total_room"])
        self.assertIsNone(state["pre_news_expected_tick"])
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

    def test_trace_recorder_writes_run_files_and_summary(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_detail_level="lite",
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
            recorder.record_book_update(now_ms=now_ms, state=state, cash=10_000, trigger="book update")
            recorder.record_market_trade(now_ms=now_ms, state=state, cash=10_000, price=1100, qty=1)
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

            events = load_trace_events(recorder.run_dir)
            event_types = {event["event_type"] for event in events}
            self.assertIn("session_start", event_types)
            self.assertIn("decision_evaluated", event_types)
            self.assertIn("order_submitted", event_types)
            self.assertIn("order_filled", event_types)
            self.assertIn("session_end", event_types)
            self.assertNotIn("book_update", event_types)
            self.assertNotIn("market_trade_observed", event_types)
            fill_event = next(event for event in events if event["event_type"] == "order_filled")
            self.assertEqual(fill_event["fill_qty"], 2)
            self.assertEqual(fill_event["fill_price"], 1099)
            self.assertEqual(fill_event["inventory_after_fill"], 0)

            with (recorder.run_dir / "session_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["passive_fills"], 1)
            self.assertEqual(summary["aggressive_fills"], 0)
            self.assertEqual(summary["fills_by_overlay"]["mm"], 1)
            self.assertIn("local_processing_durations_ms", summary)
            self.assertIn("submit_to_fill_ms", summary)
            self.assertIn("shock_episode_unwind_metrics", summary)

    def test_load_bot_config_trace_defaults_to_saavan_analysis_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_keys = {
                "UTC_HOST": "34.197.188.76:3333",
                "UTC_USERNAME": "user",
                "UTC_PASSWORD": "pass",
            }
            old_values = {key: os.environ.get(key) for key in env_keys}
            try:
                os.environ.update(env_keys)
                os.environ.pop("TRACE_ROOT", None)
                os.environ.pop("TRACE_ENABLED", None)
                config = load_bot_config(Path(temp_dir))
            finally:
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
            self.assertEqual(config.trace.trace_root, Path(temp_dir).resolve() / "analysis_runs")
            self.assertFalse(config.trace.trace_enabled)
            self.assertEqual(config.trace.trace_detail_level, "lite")
            self.assertEqual(config.risk.passive_min_rest_ms, 0)
            self.assertEqual(config.risk.passive_reprice_threshold_ticks, 2)
            self.assertFalse(config.market_a.recover_pricing_state)

    def test_trace_recorder_full_mode_keeps_book_and_trade_events(self) -> None:
        strategy = self.make_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                TraceConfig(
                    trace_enabled=True,
                    trace_root=Path(temp_dir),
                    trace_detail_level="full",
                    trace_write_summary_on_shutdown=False,
                ),
                session_prefix="test_trace",
            )
            state = strategy.trace_state(45_000)
            recorder.record_book_update(now_ms=45_000, state=state, cash=10_000, trigger="book update")
            recorder.record_market_trade(now_ms=45_000, state=state, cash=10_000, price=1100, qty=1)
            recorder.finalize(now_ms=45_100, state=state, cash=10_000, note="done")

            event_types = {event["event_type"] for event in load_trace_events(recorder.run_dir)}
            self.assertIn("book_update", event_types)
            self.assertIn("market_trade_observed", event_types)

    def test_recovered_pricing_state_is_ignored_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = TradingJournal(Path(temp_dir) / "journal.jsonl")
            journal.record_session_started(startup_mode="clean_start", recovered_orders=0)
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

            self.assertEqual(replay.startup_mode, "crash_recovery")
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

    def test_summarize_trace_events_computes_order_lifecycle_latencies(self) -> None:
        events = [
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "order_id": "order-1",
                "mode": "STEADY_MM",
                "overlay": "mm",
                "intent": "steady_mm_passive",
            },
            {
                "event_type": "order_cancel_requested",
                "run_id": "run-1",
                "monotonic_ms": 1_250,
                "order_id": "order-1",
                "mode": "STEADY_MM",
                "cancel_reason": "test",
            },
            {
                "event_type": "order_cancel_response",
                "run_id": "run-1",
                "monotonic_ms": 1_450,
                "order_id": "order-1",
                "mode": "STEADY_MM",
                "success": True,
            },
            {
                "event_type": "order_submitted",
                "run_id": "run-1",
                "monotonic_ms": 2_000,
                "order_id": "order-2",
                "mode": "POST_EARNINGS_SHOCK",
                "overlay": "earnings",
                "intent": "post_earnings_shock_take",
            },
            {
                "event_type": "order_filled",
                "run_id": "run-1",
                "monotonic_ms": 2_275,
                "order_id": "order-2",
                "mode": "POST_EARNINGS_SHOCK",
                "overlay": "earnings",
                "side": "BUY",
                "price": 999,
                "qty": 1,
                "aggressive": True,
                "intent": "post_earnings_shock_take",
                "mid_at_event": 1000.0,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        self.assertAlmostEqual(summary["cancel_response_latency_ms"]["p50"], 200.0, delta=0.001)
        self.assertAlmostEqual(summary["submit_to_cancel_response_ms"]["p50"], 450.0, delta=0.001)
        self.assertAlmostEqual(summary["submit_to_fill_ms"]["p50"], 275.0, delta=0.001)

    def test_summarize_trace_events_tracks_settle_and_pre_settle_unwind(self) -> None:
        events = [
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "mode": "POST_EARNINGS_SHOCK",
                "exchange_tick": 150,
                "earnings_value": 1.1,
                "shock_direction": 1,
            },
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_100,
                "mode": "POST_EARNINGS_SHOCK",
                "shock_stage": "ACCUMULATE",
                "earnings_position": 120,
                "desired_ask": {"intent": "post_earnings_shock_unwind"},
                "desired_bid": None,
                "aggressive_actions": [],
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "mode": "POST_EARNINGS_SHOCK",
                "shock_stage": "HOLD",
                "earnings_position": 180,
                "inventory": 180,
                "quoted_spread": 2,
                "budget_shift_active": True,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_100,
                "mode": "UNWIND",
                "shock_stage": "SETTLED_UNWIND",
                "earnings_position": 20,
                "inventory": 20,
                "quoted_spread": 2,
                "budget_shift_active": False,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 2_400,
                "mode": "UNWIND",
                "shock_stage": "SETTLED_UNWIND",
                "earnings_position": 4,
                "inventory": 4,
                "quoted_spread": 2,
                "budget_shift_active": False,
            },
        ]

        summary = summarize_trace_events(events, markout_windows_ms=(250,))
        episode = summary["shock_episode_unwind_metrics"][0]
        self.assertEqual(episode["peak_signed_earnings_position"], 180)
        self.assertEqual(episode["time_to_settle_ms"], 1_100)
        self.assertEqual(episode["time_to_unwind_start_ms"], 1_100)
        self.assertEqual(episode["time_to_abs_20_ms"], 1_100)
        self.assertEqual(episode["time_to_abs_4_ms"], 1_400)
        self.assertEqual(episode["opposite_unwind_before_settle_count"], 1)
        self.assertTrue(episode["opposite_unwind_before_settle"])

    def test_summarize_trace_events_reports_episode_unwind_metrics(self) -> None:
        events = [
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "exchange_tick": 150,
                "mode": "POST_EARNINGS_SHOCK",
                "earnings_value": 1.2,
                "shock_direction": 1,
            },
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_050,
                "mode": "POST_EARNINGS_SHOCK",
                "inventory": 48,
                "earnings_position": 48,
                "mm_position": 0,
                "quoted_spread": 2,
                "observe_only": False,
                "aggressive_action_count": 1,
                "reason": "shock",
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "mode": "UNWIND",
                "inventory": 18,
                "earnings_position": 18,
                "mm_position": 0,
                "quoted_spread": 2,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_900,
                "mode": "STEADY_MM",
                "inventory": 3,
                "earnings_position": 3,
                "mm_position": 0,
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
        episode = summary["shock_episode_unwind_metrics"][0]
        self.assertEqual(episode["exchange_tick"], 150)
        self.assertEqual(episode["peak_signed_earnings_position"], 48)
        self.assertEqual(episode["time_to_abs_20_ms"], 500)
        self.assertEqual(episode["time_to_abs_4_ms"], 900)

    def test_summarize_trace_events_reports_inventory_ownership_transfer_metrics(self) -> None:
        events = [
            {
                "event_type": "inventory_ownership_transfer",
                "run_id": "run-1",
                "monotonic_ms": 900,
                "exchange_tick": 149,
                "ownership_transfer_qty": 14,
                "prior_earnings_position": -3,
                "prior_mm_position": 14,
                "resulting_earnings_position": 11,
                "resulting_mm_position": 0,
                "reason": "pre_news_pullback_start",
            },
            {
                "event_type": "news_received",
                "run_id": "run-1",
                "monotonic_ms": 1_000,
                "exchange_tick": 150,
                "mode": "POST_EARNINGS_SHOCK",
                "earnings_value": 1.2,
                "shock_direction": 1,
            },
            {
                "event_type": "decision_evaluated",
                "run_id": "run-1",
                "monotonic_ms": 1_050,
                "mode": "POST_EARNINGS_SHOCK",
                "shock_stage": "ACCUMULATE",
                "shock_target_total_inventory": 199,
                "inventory": 48,
                "earnings_position": 48,
                "mm_position": 0,
                "quoted_spread": 2,
                "observe_only": False,
                "aggressive_action_count": 1,
                "reason": "shock",
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_200,
                "mode": "POST_EARNINGS_SHOCK",
                "shock_stage": "HOLD",
                "shock_target_total_inventory": 199,
                "inventory": 180,
                "earnings_position": 180,
                "mm_position": 0,
                "quoted_spread": 2,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_300,
                "mode": "POST_EARNINGS_SHOCK",
                "shock_stage": "HOLD",
                "shock_target_total_inventory": 199,
                "inventory": 166,
                "earnings_position": 152,
                "mm_position": 14,
                "quoted_spread": 2,
            },
            {
                "event_type": "session_state_snapshot",
                "run_id": "run-1",
                "monotonic_ms": 1_500,
                "mode": "UNWIND",
                "shock_stage": "SETTLED_UNWIND",
                "shock_target_total_inventory": 199,
                "inventory": 18,
                "earnings_position": 18,
                "mm_position": 0,
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
        episode = summary["shock_episode_unwind_metrics"][0]
        self.assertEqual(episode["transferred_mm_inventory_at_cycle_start"], 14)
        self.assertEqual(episode["peak_signed_total_inventory"], 180)
        self.assertEqual(episode["peak_signed_earnings_position"], 180)
        self.assertEqual(episode["peak_residual_mm_position"], 14)
        self.assertEqual(episode["shock_target_total_inventory"], 199)
        self.assertFalse(episode["total_inventory_exceeded_target_cap"])
        self.assertEqual(episode["shock_stage_nonzero_mm_position_count"], 1)
        self.assertTrue(episode["shock_stage_nonzero_mm_position"])


if __name__ == "__main__":
    unittest.main()
