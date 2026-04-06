from __future__ import annotations

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
                calibration_min_delay_ms=5_000,
                calibration_max_delay_ms=20_000,
                calibration_sample_period_ms=1_000,
                calibration_stability_band_ticks=8,
                calibration_tolerance_fraction=0.10,
                calibration_min_tolerance_fraction=0.03,
                candidate_confirmations=2,
                discovery_quote_size=1,
                discovery_max_position=4,
                discovery_half_spread_ticks=8,
                recover_pricing_state=False,
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
                earnings_unwind_aggressive_entry=48,
                earnings_unwind_aggressive_exit=24,
                earnings_unwind_passive_exit=8,
                earnings_unwind_passive_take_edge=8,
                shock_quote_size=12,
                shock_base_max_position=100,
                shock_shift_max_position=180,
                shock_window_ms=3_000,
                shock_take_fraction=0.25,
                shock_take_min_edge=4,
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

        strategy.set_inventory(8)
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

        strategy.set_inventory(24)
        self.assertTrue(strategy.unwind_active)
        self.assertFalse(strategy.unwind_aggressive_active)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1101: 10}, asks={1105: 10}), now_ms=45_100)
        moderate_unwind_plan = strategy.compute_quotes(now_ms=45_100)
        self.assertEqual(moderate_unwind_plan.mode, "UNWIND")
        self.assertEqual(moderate_unwind_plan.aggressive_actions, ())

        strategy.set_inventory(8)
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

        strategy.on_news(self.a_earnings_news(1.0, tick=150), now_ms=30_000)
        shock_state = strategy.trace_state(30_100)
        self.assertEqual(shock_state["mode"], "POST_EARNINGS_SHOCK")
        self.assertEqual(shock_state["earnings_budget"], 180)
        self.assertEqual(shock_state["mm_budget"], 0)
        self.assertEqual(shock_state["overlay_exposures"]["earnings"]["allowed_buy"], 180)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=33_500)
        reverted_state = strategy.trace_state(33_500)
        self.assertEqual(reverted_state["mode"], "MULTIPLIER_DISCOVERY")
        self.assertEqual(reverted_state["earnings_budget"], 120)
        self.assertEqual(reverted_state["mm_budget"], 60)
        self.assertFalse(reverted_state["budget_shift_active"])

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


if __name__ == "__main__":
    unittest.main()
