from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import AConfig, RiskConfig
from a_bot_journal import TradingJournal
from a_bot_strategy import (
    DesiredOrder,
    ManagedOrder,
    MarketAStrategy,
    OrderManager,
    QuotePlan,
)


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class MarketABotTests(unittest.TestCase):
    def make_strategy(
        self,
        *,
        initial_multiplier: float | None = None,
        initial_fair_value: int | None = None,
        recovered_multiplier: float | None = None,
        recovered_multiplier_confidence: int = 0,
        recovered_fair_value: int | None = None,
        recovered_earnings_value: float | None = None,
        restored_orders=(),
    ) -> MarketAStrategy:
        strategy = MarketAStrategy(
            a_config=AConfig(
                initial_multiplier=initial_multiplier,
                initial_fair_value=initial_fair_value,
                startup_assume_fresh_round=True,
                pre_news_pullback_ms=4_000,
                pre_news_arrival_grace_ms=1_200,
                calibration_min_delay_ms=5_000,
                calibration_max_delay_ms=20_000,
                calibration_sample_period_ms=1_000,
                calibration_stability_band_ticks=8,
                calibration_tolerance_fraction=0.10,
                calibration_min_tolerance_fraction=0.03,
                candidate_confirmations=2,
                total_position_limit=180,
                earnings_base_budget=120,
                mm_base_budget=60,
                earnings_shift_budget=180,
                mm_shift_budget=0,
                discovery_quote_size=1,
                discovery_max_position=4,
                discovery_half_spread_ticks=8,
                recover_pricing_state=False,
                max_order_qty=39,
                news_caution_quote_size=1,
                news_caution_max_position=4,
                news_caution_half_spread_ticks=8,
                steady_half_spread_ticks=1,
                steady_take_min_edge=2,
                steady_take_large_inventory_edge=4,
                opening_quote_size=1,
                opening_max_position=8,
                opening_half_spread_ticks=4,
                opening_min_book_spread=10,
                steady_quote_size=3,
                steady_max_position=32,
                steady_inventory_skew=0.75,
                steady_take_inventory_guard=8,
                steady_passive_reduce_start=8,
                steady_passive_reduce_full=20,
                unwind_inventory_skew=1.50,
                unwind_flatten_threshold=2,
                unwind_entry_position=24,
                unwind_exit_position=12,
                unwind_aggressive_entry=24,
                unwind_aggressive_exit=16,
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
                shock_base_max_position=100,
                shock_shift_max_position=180,
                shock_window_ms=3_000,
                shock_take_fraction=0.25,
                shock_take_min_edge=4,
                shock_settle_min_hold_ms=1_200,
                shock_settle_max_hold_ms=4_000,
                shock_settle_band_ticks=8,
                shock_settle_drift_ticks=4,
                shock_settle_confirmations=2,
                shock_unwind_quote_size=3,
                shock_unwind_aggressive_quote_size=3,
                shock_unwind_take_edge=4,
                shock_unwind_exit_position=8,
                prejump_enabled=True,
                prejump_window_ms=1_200,
                prejump_low_threshold=0.85,
                prejump_high_threshold=1.35,
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
            restored_orders=restored_orders,
            recovered_multiplier=recovered_multiplier,
            recovered_multiplier_confidence=recovered_multiplier_confidence,
            recovered_fair_value=recovered_fair_value,
            recovered_earnings_value=recovered_earnings_value,
        )
        strategy.startup_ms = 0
        return strategy

    @staticmethod
    def a_earnings_news(value: float, tick: int = 150) -> dict:
        return {
            "tick": tick,
            "kind": "structured",
            "symbol": "A",
            "new_data": {
                "structured_subtype": "earnings",
                "asset": "A",
                "value": value,
            },
        }

    @staticmethod
    def a_unstructured_news(tick: int = 170, content: str = "A loses a major supplier relationship.") -> dict:
        return {
            "tick": tick,
            "kind": "unstructured",
            "symbol": "A",
            "content": content,
        }

    def test_first_clean_earnings_bootstraps_multiplier_after_stable_samples(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=29_500)
        reaction = strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)

        self.assertTrue(reaction.relevant)
        self.assertIsNone(strategy.fair_value)

        for timestamp in (35_000, 36_000, 37_000, 38_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1088: 10}, asks={1092: 10}), now_ms=timestamp)

        plan = strategy.compute_quotes(now_ms=38_000)
        events = strategy.drain_learning_events()

        self.assertEqual(plan.mode, "STEADY_MM")
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1090.0, delta=0.5)
        self.assertEqual(strategy.fair_value, 1090)
        self.assertEqual(strategy.multiplier_confidence, 1)
        self.assertEqual(events[-1].status, "trusted")
        self.assertEqual(events[-1].method, "stable_level")

    def test_within_tolerance_earnings_updates_tighten_multiplier(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=1,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=59_500)
        strategy.on_news(self.a_earnings_news(1.02, tick=300), now_ms=60_000)

        for timestamp in (65_000, 66_000, 67_000, 68_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1117: 10}, asks={1121: 10}), now_ms=timestamp)

        strategy.compute_quotes(now_ms=68_000)
        events = strategy.drain_learning_events()

        self.assertEqual(events[-1].status, "updated")
        self.assertEqual(strategy.multiplier_confidence, 2)
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1098.0, delta=0.75)
        self.assertEqual(strategy.fair_value, round((strategy.trusted_multiplier or 0.0) * 1.02))

    def test_divergent_multiplier_requires_two_confirmations_before_replacing(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=29_500)
        strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)
        for timestamp in (35_000, 36_000, 37_000, 38_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1298: 10}, asks={1302: 10}), now_ms=timestamp)
        strategy.compute_quotes(now_ms=38_000)
        first_events = strategy.drain_learning_events()

        self.assertEqual(first_events[-1].status, "candidate")
        self.assertEqual(strategy.trusted_multiplier, 1100.0)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=59_500)
        strategy.on_news(self.a_earnings_news(1.0, tick=300), now_ms=60_000)
        for timestamp in (65_000, 66_000, 67_000, 68_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1296: 10}, asks={1300: 10}), now_ms=timestamp)
        strategy.compute_quotes(now_ms=68_000)
        second_events = strategy.drain_learning_events()

        self.assertEqual(second_events[-1].status, "replaced")
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1299.0, delta=1.0)
        self.assertEqual(strategy.multiplier_confidence, 1)

    def test_discovery_falls_back_at_twenty_seconds_when_not_stable(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=29_500)
        strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)

        samples = {
            35_000: (997, 1003),
            36_000: (1009, 1015),
            37_000: (1001, 1007),
            38_000: (1011, 1017),
            39_000: (1003, 1009),
            40_000: (1013, 1019),
            41_000: (1005, 1011),
            42_000: (1015, 1021),
            43_000: (1007, 1013),
            44_000: (1017, 1023),
            45_000: (1009, 1015),
            46_000: (1019, 1025),
            47_000: (1011, 1017),
            48_000: (1021, 1027),
            49_000: (1013, 1019),
            50_000: (1023, 1029),
        }
        for timestamp, (bid, ask) in samples.items():
            strategy.on_book_update_at("A", FakeOrderBook(bids={bid: 10}, asks={ask: 10}), now_ms=timestamp)

        plan = strategy.compute_quotes(now_ms=50_000)
        events = strategy.drain_learning_events()

        self.assertEqual(plan.mode, "STEADY_MM")
        self.assertEqual(events[-1].method, "fallback_level")
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1014.0, delta=0.5)

    def test_unstructured_news_persists_until_next_structured_earnings_reset(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=40_000)
        strategy.on_news(self.a_unstructured_news(tick=205), now_ms=40_100)

        cautious_plan = strategy.compute_quotes(now_ms=52_000)
        self.assertEqual(cautious_plan.mode, "NEWS_CAUTIOUS_MM")

        strategy.on_news(self.a_earnings_news(1.1, tick=300), now_ms=60_000)
        reset_plan = strategy.compute_quotes(now_ms=60_100)
        self.assertEqual(reset_plan.mode, "POST_EARNINGS_SHOCK")

    def test_structured_earnings_clears_prior_news_for_learning(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=29_000)
        strategy.on_news(self.a_unstructured_news(tick=145), now_ms=29_000)
        strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)

        for timestamp in (35_000, 36_000, 37_000, 38_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1088: 10}, asks={1092: 10}), now_ms=timestamp)

        strategy.compute_quotes(now_ms=38_000)
        events = strategy.drain_learning_events()

        self.assertEqual(events[-1].status, "trusted")
        self.assertFalse(strategy.news_caution_active)
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1090.0, delta=0.5)

    def test_pre_news_pullback_uses_30s_and_60s_schedule(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=26_500)
        plan = strategy.compute_quotes(now_ms=26_500)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")

        anchored = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        anchored.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=0)
        anchored.on_news(self.a_earnings_news(1.0, tick=150), now_ms=0)
        anchored.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=26_500)

        anchored_plan = anchored.compute_quotes(now_ms=26_500)
        self.assertEqual(anchored_plan.mode, "PRE_NEWS_PULLBACK")
        self.assertEqual(anchored.ms_until_next_scheduled_earnings(26_500), 3_500)

    def test_steady_quotes_tighten_around_learned_fair_and_skew_with_inventory(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.shock_started_ms = None
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=45_000)

        strategy.set_inventory(0)
        neutral = strategy.compute_quotes(now_ms=45_000)
        strategy.set_inventory(10)
        long_inventory = strategy.compute_quotes(now_ms=45_000)
        strategy.set_inventory(-10)
        short_inventory = strategy.compute_quotes(now_ms=45_000)

        self.assertEqual(neutral.mode, "STEADY_MM")
        self.assertEqual(neutral.bid.px, 1099)
        self.assertEqual(neutral.ask.px, 1101)
        self.assertLess(long_inventory.bid.px, neutral.bid.px)
        self.assertLess(long_inventory.ask.px, neutral.ask.px)
        self.assertGreater(short_inventory.bid.px, neutral.bid.px)
        self.assertGreater(short_inventory.ask.px, neutral.ask.px)

    def test_unwind_uses_hysteresis_before_returning_to_steady_mm(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=45_000)

        strategy.set_inventory(30)
        self.assertEqual(strategy.compute_quotes(now_ms=45_000).mode, "UNWIND")

        strategy.set_inventory(13)
        self.assertEqual(strategy.compute_quotes(now_ms=45_100).mode, "UNWIND")

        strategy.set_inventory(12)
        self.assertEqual(strategy.compute_quotes(now_ms=45_200).mode, "UNWIND")

        strategy.set_inventory(4)
        self.assertEqual(strategy.compute_quotes(now_ms=45_300).mode, "STEADY_MM")

    def test_steady_mode_suppresses_directional_takes_when_inventory_is_loaded(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )

        strategy.on_book_update_at("A", FakeOrderBook(bids={1090: 10}, asks={1098: 10}), now_ms=45_000)
        strategy.set_inventory(10)
        long_inventory_plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(long_inventory_plan.mode, "STEADY_MM")
        self.assertEqual(long_inventory_plan.aggressive_actions, ())
        self.assertIsNotNone(long_inventory_plan.bid)
        self.assertIsNotNone(long_inventory_plan.ask)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1102: 10}, asks={1110: 10}), now_ms=45_500)
        strategy.set_inventory(-10)
        short_inventory_plan = strategy.compute_quotes(now_ms=45_500)
        self.assertEqual(short_inventory_plan.mode, "STEADY_MM")
        self.assertEqual(short_inventory_plan.aggressive_actions, ())
        self.assertIsNotNone(short_inventory_plan.bid)
        self.assertIsNotNone(short_inventory_plan.ask)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1090: 10}, asks={1098: 10}), now_ms=46_000)
        strategy.set_inventory(15)
        passive_only_plan = strategy.compute_quotes(now_ms=46_000)
        self.assertEqual(passive_only_plan.mode, "STEADY_MM")
        self.assertEqual(passive_only_plan.aggressive_actions, ())
        self.assertIsNotNone(passive_only_plan.bid)
        self.assertIsNotNone(passive_only_plan.ask)

    def test_steady_mode_reclaims_two_tick_stale_edge_when_inventory_is_light(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={998: 10}), now_ms=45_000)

        plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(plan.mode, "STEADY_MM")
        self.assertTrue(any(action.side == "BUY" and action.intent == "steady_take" for action in plan.aggressive_actions))

    def test_steady_mode_shrinks_inventory_worsening_passive_side(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=45_000)

        strategy.set_inventory(10)
        long_inventory_plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(long_inventory_plan.mode, "STEADY_MM")
        self.assertEqual(long_inventory_plan.bid.qty, 2)
        self.assertEqual(long_inventory_plan.ask.qty, 3)

        strategy.set_inventory(12)
        more_loaded_long_plan = strategy.compute_quotes(now_ms=45_100)
        self.assertEqual(more_loaded_long_plan.bid.qty, 1)
        self.assertEqual(more_loaded_long_plan.ask.qty, 3)

        strategy.set_inventory(-10)
        short_inventory_plan = strategy.compute_quotes(now_ms=45_200)
        self.assertEqual(short_inventory_plan.bid.qty, 3)
        self.assertEqual(short_inventory_plan.ask.qty, 2)

    def test_order_manager_keeps_passive_quote_for_one_tick_move(self) -> None:
        manager = OrderManager(
            symbol="A",
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
        )
        manager.note_submitted(order_id="buy-1", side="BUY", px=1099, qty=2, now_ms=1)
        plan = QuotePlan(
            mode="STEADY_MM",
            bid=DesiredOrder(side="BUY", px=1100, qty=2, aggressive=False, reason="test"),
            ask=None,
            aggressive_actions=(),
            observe_only=False,
            reason="test",
        )

        actions = manager.build_actions(plan, now_ms=2_000)
        self.assertEqual(actions.cancels, ())
        self.assertEqual(actions.placements, ())

    def test_unwind_aggression_steps_down_before_unwind_exits(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1100: 10}, asks={1105: 10}), now_ms=45_000)

        strategy.set_inventory(60)
        self.assertTrue(strategy.unwind_active)
        self.assertTrue(strategy.unwind_aggressive_active)

        strategy.set_inventory(30)
        self.assertTrue(strategy.unwind_active)
        self.assertTrue(strategy.unwind_aggressive_active)

        strategy.set_inventory(12)
        self.assertTrue(strategy.unwind_active)
        self.assertFalse(strategy.unwind_aggressive_active)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1101: 10}, asks={1105: 10}), now_ms=45_100)
        moderate_unwind_plan = strategy.compute_quotes(now_ms=45_100)
        self.assertEqual(moderate_unwind_plan.mode, "UNWIND")
        self.assertEqual(moderate_unwind_plan.aggressive_actions, ())

        strategy.set_inventory(4)
        exited = strategy.compute_quotes(now_ms=45_200)
        self.assertEqual(exited.mode, "STEADY_MM")

    def test_recovery_blocks_until_restored_order_is_resolved(self) -> None:
        restored = ManagedOrder(
            order_id="restored-1",
            side="BUY",
            px=896,
            qty=4,
            remaining_qty=4,
            submitted_ms=10,
            restored=True,
        )
        strategy = self.make_strategy(
            restored_orders=(restored,),
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=990,
            recovered_earnings_value=0.9,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={985: 5}, asks={995: 5}), now_ms=45_000)

        before = strategy.compute_quotes(now_ms=45_000)
        self.assertTrue(before.observe_only)
        self.assertEqual([order.order_id for order in strategy.recovery_orders_to_cancel()], ["restored-1"])

        strategy.order_manager.mark_cancel_requested("restored-1", now_ms=45_010)
        strategy.on_cancel_response("restored-1", success=False)

        after = strategy.compute_quotes(now_ms=45_050)
        self.assertFalse(after.observe_only)
        self.assertFalse(strategy.recovery_active)
        self.assertIsNotNone(after.bid)

    def test_journal_replays_multiplier_inventory_and_live_orders(self) -> None:
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
            journal.record_fill("order-1", qty=1, price=901)
            journal.record_multiplier(1050.0, confidence=2, source="updated", estimate=1055.0, method="stable_level")
            journal.record_fair_value(945, source="learned_multiplier", earnings_value=0.9)
            journal.record_inventory(7, cash=1000)

            replay = journal.load_replay_state()
            self.assertEqual(replay.multiplier, 1050.0)
            self.assertEqual(replay.multiplier_confidence, 2)
            self.assertEqual(replay.fair_value, 945)
            self.assertAlmostEqual(replay.earnings_value, 0.9)
            self.assertEqual(replay.inventory, 7)
            self.assertEqual(len(replay.live_orders), 1)
            self.assertEqual(replay.live_orders[0].remaining_qty, 1)
            self.assertTrue(replay.live_orders[0].restored)

    def test_finished_session_is_ignored_and_prepare_for_startup_rotates_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_path = Path(temp_dir) / "journal.jsonl"
            journal = TradingJournal(journal_path)
            journal.record_session_started(startup_mode="clean_start", recovered_orders=0)
            journal.record_multiplier(1050.0, confidence=2, source="updated", estimate=1055.0, method="stable_level")
            journal.record_fair_value(945, source="learned_multiplier", earnings_value=0.9)
            journal.record_inventory(7, cash=1000)
            journal.record_session_finished(note="normal round end")

            replay = journal.load_replay_state()
            self.assertEqual(replay.startup_mode, "finished_session_ignored")
            self.assertIsNone(replay.multiplier)
            self.assertIsNone(replay.fair_value)
            self.assertEqual(replay.inventory, 0)
            self.assertEqual(replay.live_orders, ())

            prepared = journal.prepare_for_startup()
            self.assertEqual(prepared.startup_mode, "finished_session_ignored")
            self.assertIsNotNone(prepared.archived_path)
            assert prepared.archived_path is not None
            self.assertTrue(prepared.archived_path.exists())
            self.assertEqual(prepared.archived_path.parent.name, "journal_archive")

            with journal_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual([record["event_type"] for record in records], ["session_started"])
            self.assertEqual(records[0]["startup_mode"], "finished_session_ignored")

    def test_unfinished_session_recovery_keeps_live_orders_and_rotates_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_path = Path(temp_dir) / "journal.jsonl"
            journal = TradingJournal(journal_path)
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
            journal.record_inventory(7, cash=1000)

            replay = journal.load_replay_state()
            self.assertEqual(replay.startup_mode, "crash_recovery")
            self.assertEqual(replay.multiplier, 1050.0)
            self.assertEqual(replay.fair_value, 945)
            self.assertEqual(replay.inventory, 7)
            self.assertEqual(len(replay.live_orders), 1)

            prepared = journal.prepare_for_startup()
            self.assertEqual(prepared.startup_mode, "crash_recovery")
            self.assertEqual(len(prepared.live_orders), 1)
            self.assertIsNotNone(prepared.archived_path)

            with journal_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(
                [record["event_type"] for record in records],
                ["session_started", "order_submitted", "multiplier_updated", "fair_value_updated"],
            )
            self.assertEqual(records[0]["startup_mode"], "crash_recovery")
            self.assertEqual(records[0]["recovered_orders"], 1)
            self.assertEqual(records[1]["order_id"], "order-1")
            self.assertEqual(records[2]["source"], "recovered_journal_seed")
            self.assertEqual(records[3]["source"], "recovered_journal_seed")

    def test_overlay_fill_attribution_tracks_virtual_positions(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.order_manager.note_submitted(
            order_id="earn-1",
            side="BUY",
            px=1090,
            qty=4,
            now_ms=1,
            overlay="earnings",
            intent="post_earnings_shock_take",
            mode_at_submit="POST_EARNINGS_SHOCK",
            evaluation_reason="test",
        )
        strategy.on_fill("earn-1", qty=4, price=1090)
        self.assertEqual(strategy.inventory, 4)
        self.assertEqual(strategy.earnings_position, 4)
        self.assertEqual(strategy.mm_position, 0)

        strategy.order_manager.note_submitted(
            order_id="mm-1",
            side="SELL",
            px=1102,
            qty=3,
            now_ms=2,
            overlay="mm",
            intent="steady_mm_passive",
            mode_at_submit="STEADY_MM",
            evaluation_reason="test",
        )
        strategy.on_fill("mm-1", qty=3, price=1102)
        self.assertEqual(strategy.inventory, 1)
        self.assertEqual(strategy.earnings_position, 4)
        self.assertEqual(strategy.mm_position, -3)

    def test_budget_shift_applies_during_pre_news_and_shock_then_reverts(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=26_500)
        pre_news_state = strategy.trace_state(26_500)
        self.assertEqual(pre_news_state["mode"], "PRE_NEWS_PULLBACK")
        self.assertEqual(pre_news_state["earnings_budget"], 180)
        self.assertEqual(pre_news_state["mm_budget"], 0)
        self.assertTrue(pre_news_state["budget_shift_active"])

        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        shock_state = strategy.trace_state(30_100)
        self.assertEqual(shock_state["mode"], "POST_EARNINGS_SHOCK")
        self.assertEqual(shock_state["earnings_budget"], 180)
        self.assertEqual(shock_state["mm_budget"], 0)
        self.assertEqual(shock_state["overlay_exposures"]["earnings"]["allowed_buy"], 180)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1209: 10}, asks={1211: 10}), now_ms=32_100)
        strategy.compute_quotes(now_ms=32_100)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1209: 10}, asks={1211: 10}), now_ms=33_400)
        strategy.compute_quotes(now_ms=33_400)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1209: 10}, asks={1211: 10}), now_ms=33_600)
        reverted_state = strategy.trace_state(33_600)
        self.assertEqual(reverted_state["mode"], "MULTIPLIER_DISCOVERY")
        self.assertEqual(reverted_state["earnings_budget"], 120)
        self.assertEqual(reverted_state["mm_budget"], 60)
        self.assertFalse(reverted_state["budget_shift_active"])

    def test_shock_entry_uses_base_threshold_and_base_quote_size(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1070: 20}, asks={1074: 40}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)

        plan = strategy.compute_quotes(now_ms=30_100)
        self.assertEqual(plan.mode, "POST_EARNINGS_SHOCK")
        self.assertTrue(any(action.intent == "post_earnings_shock_take" and action.side == "BUY" for action in plan.aggressive_actions))
        buy_action = next(action for action in plan.aggressive_actions if action.side == "BUY")
        self.assertEqual(buy_action.qty, 12)

    def test_shock_mode_posts_passive_unwind_quote_when_inventory_is_already_long(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.set_inventory(18)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1088: 20}, asks={1102: 20}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)

        plan = strategy.compute_quotes(now_ms=30_100)
        self.assertEqual(plan.mode, "POST_EARNINGS_SHOCK")
        self.assertEqual(plan.aggressive_actions, ())
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.overlay, "earnings")
        self.assertEqual(plan.ask.intent, "post_earnings_shock_unwind")

    def test_shock_mode_can_post_unwind_quote_while_shock_window_is_active(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1088: 20}, asks={1092: 20}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        strategy.set_inventory(18)

        plan = strategy.compute_quotes(now_ms=30_100)
        self.assertEqual(plan.mode, "POST_EARNINGS_SHOCK")
        self.assertEqual(strategy.shock_stage, "NONE")
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.intent, "post_earnings_shock_unwind")

    def test_shock_buy_size_is_limited_by_remaining_room_under_180_cap(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1070: 20}, asks={1074: 100}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        strategy.set_inventory(170)

        plan = strategy.compute_quotes(now_ms=30_100)
        self.assertEqual(plan.mode, "POST_EARNINGS_SHOCK")
        buy_action = next(action for action in plan.aggressive_actions if action.side == "BUY")
        self.assertEqual(buy_action.qty, 10)

    def test_shock_cycle_exits_into_ordinary_unwind_after_accumulate_window(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1078: 20}, asks={1082: 20}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        strategy.set_inventory(100)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 20}, asks={1102: 20}), now_ms=33_100)
        plan = strategy.compute_quotes(now_ms=33_100)

        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.shock_stage, "NONE")
        self.assertEqual(strategy.mm_phase, "MULTIPLIER_DISCOVERY")
        self.assertEqual(plan.ask.overlay, "earnings")
        self.assertEqual(plan.ask.intent, "unwind")
        self.assertEqual(plan.ask.qty, 3)

    def test_shock_unwind_returns_to_mm_participation_after_accumulate_phase(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1078: 20}, asks={1082: 20}), now_ms=29_900)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        strategy.set_inventory(30)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1099: 100}, asks={1101: 20}), now_ms=33_100)
        plan = strategy.compute_quotes(now_ms=33_100)

        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.shock_stage, "NONE")
        self.assertEqual(strategy.mm_phase, "MULTIPLIER_DISCOVERY")
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.overlay, "mm")
        self.assertEqual(plan.ask.qty, 3)

    def test_pre_news_hold_stays_active_until_grace_expires(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=26_500)
        self.assertTrue(strategy.pre_news_hold_active)
        self.assertEqual(strategy.earnings_phase, "PRE_NEWS_PULLBACK")

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=30_500)
        self.assertTrue(strategy.pre_news_hold_active)
        held_state = strategy.trace_state(30_500)
        self.assertEqual(held_state["mode"], "PRE_NEWS_PULLBACK")
        self.assertEqual(held_state["mm_phase"], "SHIFTED_OFF")

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=31_250)
        self.assertFalse(strategy.pre_news_hold_active)
        released_state = strategy.trace_state(31_250)
        self.assertNotEqual(released_state["mode"], "PRE_NEWS_PULLBACK")

    def test_structured_earnings_clears_pre_news_hold_into_shock(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=26_500)
        self.assertTrue(strategy.pre_news_hold_active)

        strategy.on_news(self.a_earnings_news(0.90, tick=150), now_ms=30_000)
        self.assertFalse(strategy.pre_news_hold_active)
        self.assertEqual(strategy.earnings_phase, "POST_EARNINGS_SHOCK")
        self.assertEqual(strategy.mode, "POST_EARNINGS_SHOCK")

    def test_pre_news_hold_keeps_mm_shifted_off_and_earnings_only(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=924,
            recovered_earnings_value=0.84,
        )
        strategy.order_manager.note_submitted(order_id="mm-bid", side="BUY", px=920, qty=3, now_ms=28_000, overlay="mm")
        strategy.order_manager.note_submitted(order_id="mm-ask", side="SELL", px=928, qty=3, now_ms=28_000, overlay="mm")
        strategy.on_book_update_at("A", FakeOrderBook(bids={922: 10}, asks={925: 10}), now_ms=29_000)

        plan = strategy.compute_quotes(now_ms=29_000)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertEqual(strategy.mm_phase, "SHIFTED_OFF")
        self.assertTrue(all(action.overlay == "earnings" for action in plan.aggressive_actions))
        self.assertTrue(plan.bid is None or plan.bid.overlay == "earnings")
        self.assertIsNone(plan.ask)

    def test_pre_news_hold_transfers_mm_inventory_into_earnings_ownership(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=924,
            recovered_earnings_value=0.84,
        )
        strategy.inventory = 11
        strategy.earnings_position = -3
        strategy.mm_position = 14

        strategy.on_book_update_at("A", FakeOrderBook(bids={922: 10}, asks={925: 10}), now_ms=29_000)

        self.assertTrue(strategy.pre_news_hold_active)
        self.assertEqual(strategy.earnings_position, 11)
        self.assertEqual(strategy.mm_position, 0)
        transfer_events = strategy.drain_inventory_transfer_events()
        self.assertEqual(len(transfer_events), 1)
        self.assertEqual(transfer_events[0]["ownership_transfer_qty"], 14)
        self.assertEqual(transfer_events[0]["resulting_earnings_position"], 11)

    def test_shock_accumulation_targets_total_inventory_not_split_overlay_inventory(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.inventory = 170
        strategy.earnings_position = 160
        strategy.mm_position = 10
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=26_500)
        strategy.on_news(self.a_earnings_news(1.1, tick=150), now_ms=30_000)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1070: 100}, asks={1074: 100}), now_ms=30_050)

        plan = strategy.compute_quotes(now_ms=30_050)

        self.assertEqual(strategy.earnings_position, 170)
        self.assertEqual(strategy.mm_position, 0)
        buy_action = next(action for action in plan.aggressive_actions if action.side == "BUY")
        self.assertEqual(buy_action.qty, 10)

    def test_stale_settled_shock_state_clears_back_to_ordinary_unwind(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.inventory = 22
        strategy.earnings_position = 22
        strategy.mm_position = 0
        strategy.shock_started_ms = 30_000
        strategy.shock_direction = 1
        strategy.shock_threshold = 1
        strategy.shock_target_fair = 1100
        strategy.shock_stage = "SETTLED_UNWIND"
        strategy.on_book_update_at("A", FakeOrderBook(bids={1099: 100}, asks={1101: 20}), now_ms=33_600)

        plan = strategy.compute_quotes(now_ms=33_600)

        self.assertEqual(strategy.shock_stage, "NONE")
        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.earnings_position, 22)
        self.assertEqual(strategy.mm_position, 0)
        self.assertEqual(strategy.mm_phase, "STEADY_MM")
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.qty, 3)

    def test_unwind_keeps_mm_bid_live_while_earnings_overlay_reduces_inventory(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=45_000)
        strategy.set_inventory(30)

        plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(plan.mode, "UNWIND")
        self.assertIsNotNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.bid.overlay, "mm")
        self.assertEqual(plan.bid.intent, "steady_mm_passive")
        self.assertEqual(plan.ask.overlay, "earnings")
        self.assertEqual(plan.ask.intent, "unwind")
        self.assertEqual(plan.ask.qty, 3)

    def test_large_unwind_keeps_mm_active_under_baseline_settings(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1099: 100}, asks={1105: 10}), now_ms=45_000)
        strategy.set_inventory(100)

        plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.mm_phase, "STEADY_MM")
        self.assertFalse(strategy.rapid_unwind_active)
        self.assertIsNotNone(plan.bid)
        self.assertTrue(plan.ask is None or plan.ask.qty == 3)
        self.assertEqual(plan.aggressive_actions, ())

    def test_news_caution_does_not_suppress_large_earnings_unwind(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_news(self.a_unstructured_news(), now_ms=40_100)
        strategy.set_inventory(40)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=45_000)

        plan = strategy.compute_quotes(now_ms=45_000)
        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.earnings_phase, "UNWIND")
        self.assertEqual(strategy.mm_phase, "NEWS_CAUTIOUS_MM")
        self.assertEqual(plan.ask.overlay, "earnings")
        self.assertEqual(plan.ask.intent, "unwind")
        self.assertEqual(plan.bid.overlay, "mm")
        self.assertEqual(plan.bid.intent, "news_cautious_mm")

    def test_order_manager_waits_for_min_rest_before_passive_reprice(self) -> None:
        manager = OrderManager(
            symbol="A",
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_min_rest_ms=500,
                passive_reprice_threshold_ticks=3,
                passive_quote_ttl_ms=3_000,
            ),
        )
        manager.note_submitted(order_id="buy-1", side="BUY", px=1099, qty=2, now_ms=1_000)
        plan = QuotePlan(
            mode="STEADY_MM",
            bid=DesiredOrder(side="BUY", px=1096, qty=2, aggressive=False, reason="test"),
            ask=None,
            aggressive_actions=(),
            observe_only=False,
            reason="test",
        )

        early_actions = manager.build_actions(plan, now_ms=1_300)
        late_actions = manager.build_actions(plan, now_ms=1_600)

        self.assertEqual(early_actions.cancels, ())
        self.assertEqual(late_actions.cancels[0].order_id, "buy-1")

    def test_order_manager_keeps_passive_quote_for_two_tick_move_under_new_threshold(self) -> None:
        manager = OrderManager(
            symbol="A",
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_min_rest_ms=0,
                passive_reprice_threshold_ticks=3,
                passive_quote_ttl_ms=3_000,
            ),
        )
        manager.note_submitted(order_id="buy-1", side="BUY", px=1099, qty=2, now_ms=1_000)
        plan = QuotePlan(
            mode="STEADY_MM",
            bid=DesiredOrder(side="BUY", px=1097, qty=2, aggressive=False, reason="test"),
            ask=None,
            aggressive_actions=(),
            observe_only=False,
            reason="test",
        )

        actions = manager.build_actions(plan, now_ms=1_600)
        self.assertEqual(actions.cancels, ())

    def test_discovery_does_not_suppress_large_earnings_unwind(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)
        strategy.set_inventory(40)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=33_100)

        plan = strategy.compute_quotes(now_ms=33_100)
        self.assertEqual(plan.mode, "UNWIND")
        self.assertEqual(strategy.earnings_phase, "UNWIND")
        self.assertEqual(strategy.mm_phase, "MULTIPLIER_DISCOVERY")
        self.assertEqual(plan.ask.overlay, "earnings")
        self.assertEqual(plan.ask.intent, "unwind")
        self.assertEqual(plan.bid.overlay, "mm")
        self.assertEqual(plan.bid.intent, "multiplier_discovery_mm")

    def test_boundary_low_earnings_prejump_emits_bullish_earnings_orders(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=924,
            recovered_earnings_value=0.84,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={922: 10}, asks={925: 10}), now_ms=29_000)

        plan = strategy.compute_quotes(now_ms=29_000)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertTrue(any(action.intent == "earnings_prejump" and action.side == "BUY" for action in plan.aggressive_actions))
        self.assertTrue(plan.bid is None or (plan.bid.overlay == "earnings" and plan.bid.intent == "earnings_prejump"))

    def test_boundary_high_earnings_prejump_emits_bearish_earnings_orders(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1496,
            recovered_earnings_value=1.36,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={1495: 10}, asks={1498: 10}), now_ms=29_000)

        plan = strategy.compute_quotes(now_ms=29_000)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertTrue(any(action.intent == "earnings_prejump" and action.side == "SELL" for action in plan.aggressive_actions))
        self.assertTrue(plan.ask is None or (plan.ask.overlay == "earnings" and plan.ask.intent == "earnings_prejump"))

    def test_bearish_prejump_can_flip_through_flat_toward_target(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1496,
            recovered_earnings_value=1.36,
        )
        strategy.set_inventory(60)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1495: 50}, asks={1498: 10}), now_ms=29_000)

        state = strategy.trace_state(29_000)
        plan = strategy.compute_quotes(now_ms=29_000)

        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertEqual(state["overlay_exposures"]["earnings"]["allowed_sell"], 84)
        self.assertTrue(any(action.intent == "earnings_prejump" and action.side == "SELL" for action in plan.aggressive_actions))
        self.assertTrue(plan.ask is None or plan.ask.qty == 6)

    def test_prejump_stays_off_without_extreme_earnings_or_with_news_caution(self) -> None:
        neutral = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1100,
            recovered_earnings_value=1.0,
        )
        neutral.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=29_000)
        neutral_plan = neutral.compute_quotes(now_ms=29_000)
        self.assertEqual(neutral_plan.mode, "PRE_NEWS_PULLBACK")
        self.assertIsNone(neutral_plan.bid)
        self.assertIsNone(neutral_plan.ask)
        self.assertEqual(neutral_plan.aggressive_actions, ())

        cautious = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=924,
            recovered_earnings_value=0.84,
        )
        cautious.news_caution_active = True
        cautious.on_book_update_at("A", FakeOrderBook(bids={922: 10}, asks={925: 10}), now_ms=29_000)
        cautious_plan = cautious.compute_quotes(now_ms=29_000)
        self.assertEqual(cautious_plan.mode, "PRE_NEWS_PULLBACK")
        self.assertIsNone(cautious_plan.bid)
        self.assertIsNone(cautious_plan.ask)
        self.assertEqual(cautious_plan.aggressive_actions, ())

    def test_contaminated_discovery_blocks_prejump_until_next_structured_earnings(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1100.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=924,
            recovered_earnings_value=0.84,
        )
        strategy.discovery_contaminated = True
        strategy.on_book_update_at("A", FakeOrderBook(bids={922: 10}, asks={925: 10}), now_ms=29_000)
        plan = strategy.compute_quotes(now_ms=29_000)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertIsNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.aggressive_actions, ())

        strategy.on_news(self.a_earnings_news(0.90, tick=300), now_ms=60_000)
        self.assertFalse(strategy.discovery_contaminated)


if __name__ == "__main__":
    unittest.main()
