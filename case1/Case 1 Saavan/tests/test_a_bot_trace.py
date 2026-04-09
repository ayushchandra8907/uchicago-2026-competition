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
            self.assertIn("a_news_summary", summary)
            self.assertIn("b_cost_adjusted_residual_stats", summary)
            self.assertIn("b_strategy_block_reasons", summary)
            self.assertIn("trace_volume_summary", summary)

    def test_load_bot_config_trace_defaults_to_saavan_analysis_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_keys = {
                "UTC_HOST": "practice.uchicago.exchange:3333",
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
            self.assertFalse(config.market_a.recover_pricing_state)

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


if __name__ == "__main__":
    unittest.main()
