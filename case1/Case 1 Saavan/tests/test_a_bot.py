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
                calibration_delay_ms=2_000,
                calibration_window_ms=2_000,
                calibration_min_samples=4,
                calibration_min_eps_jump=0.03,
                calibration_tolerance_fraction=0.10,
                calibration_min_tolerance_fraction=0.03,
                candidate_confirmations=2,
                news_caution_ms=8_000,
                news_caution_quote_size=1,
                news_caution_max_position=8,
                news_caution_half_spread_ticks=5,
                steady_half_spread_ticks=1,
                steady_take_min_edge=1,
                opening_quote_size=1,
                opening_max_position=8,
                opening_half_spread_ticks=4,
                opening_min_book_spread=10,
                steady_quote_size=2,
                steady_max_position=24,
                steady_inventory_skew=0.75,
                unwind_inventory_skew=1.50,
                unwind_flatten_threshold=2,
                shock_quote_size=12,
                shock_max_position=80,
                shock_window_ms=3_000,
                shock_take_fraction=0.25,
                shock_take_min_edge=4,
            ),
            risk=RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=2),
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
    def a_unstructured_news(tick: int = 170, content: str = "A loses key supplier relationship.") -> dict:
        return {
            "tick": tick,
            "kind": "unstructured",
            "symbol": "A",
            "content": content,
        }

    def test_first_clean_earnings_learns_round_multiplier_after_settle(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=29_500)
        reaction = strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)

        self.assertTrue(reaction.relevant)
        self.assertIsNone(strategy.fair_value)

        for timestamp in (32_100, 32_400, 32_700, 33_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1085: 10}, asks={1095: 10}), now_ms=timestamp)
        plan = strategy.compute_quotes(now_ms=34_100)
        events = strategy.drain_learning_events()

        self.assertEqual(plan.mode, "STEADY_MM")
        self.assertIsNotNone(strategy.trusted_multiplier)
        self.assertAlmostEqual(strategy.trusted_multiplier or 0.0, 1090.0, delta=0.5)
        self.assertEqual(strategy.fair_value, 1090)
        self.assertEqual(events[-1].status, "trusted")
        self.assertEqual(events[-1].method, "level")

    def test_second_earnings_uses_trusted_multiplier_for_immediate_shock(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=59_900)
        reaction = strategy.on_news(self.a_earnings_news(1.1, tick=300), now_ms=60_000)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1052: 10}, asks={1060: 7}), now_ms=60_100)

        plan = strategy.compute_quotes(now_ms=60_100)

        self.assertTrue(reaction.fair_value_updated)
        self.assertEqual(reaction.new_fair_value, 1100)
        self.assertEqual(plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(plan.aggressive_actions[0].px, 1060)

    def test_unstructured_a_news_contaminates_ratio_calibration(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=29_500)
        strategy.on_news(self.a_earnings_news(1.0), now_ms=30_000)
        strategy.on_news(self.a_unstructured_news(), now_ms=31_000)

        for timestamp in (32_100, 32_400, 32_700, 33_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={1085: 10}, asks={1095: 10}), now_ms=timestamp)
        strategy.compute_quotes(now_ms=34_100)
        events = strategy.drain_learning_events()

        self.assertIsNone(strategy.trusted_multiplier)
        self.assertEqual(events[-1].status, "skipped")
        self.assertIn("contaminated", events[-1].reason)

    def test_pre_news_pullback_uses_30s_and_60s_schedule(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=26_500)
        plan = strategy.compute_quotes(now_ms=26_500)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")

        anchored = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        anchored.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=0)
        anchored.on_news(self.a_earnings_news(1.0, tick=150), now_ms=0)
        anchored.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=26_500)

        anchored_plan = anchored.compute_quotes(now_ms=26_500)
        self.assertEqual(anchored_plan.mode, "PRE_NEWS_PULLBACK")
        self.assertEqual(anchored.ms_until_next_scheduled_earnings(26_500), 3_500)

    def test_news_caution_mode_quotes_small_and_wide(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=40_000)
        strategy.on_news(self.a_unstructured_news(tick=205), now_ms=40_100)

        plan = strategy.compute_quotes(now_ms=40_200)

        self.assertEqual(plan.mode, "NEWS_CAUTIOUS_MM")
        self.assertIsNotNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.bid.qty, 1)
        self.assertEqual(plan.ask.qty, 1)
        self.assertLess(plan.bid.px, 995)
        self.assertGreater(plan.ask.px, 1005)

    def test_steady_quotes_tighten_around_learned_fair_and_skew_with_inventory(self) -> None:
        strategy = self.make_strategy(
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=3,
            recovered_fair_value=1000,
            recovered_earnings_value=1.0,
        )
        strategy.shock_started_ms = None
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=45_000)

        strategy.set_inventory(0)
        neutral = strategy.compute_quotes(now_ms=45_000)
        strategy.set_inventory(10)
        long_inventory = strategy.compute_quotes(now_ms=45_000)
        strategy.set_inventory(-10)
        short_inventory = strategy.compute_quotes(now_ms=45_000)

        self.assertEqual(neutral.mode, "STEADY_MM")
        self.assertEqual(neutral.bid.px, 999)
        self.assertEqual(neutral.ask.px, 1001)
        self.assertLess(long_inventory.bid.px, neutral.bid.px)
        self.assertLess(long_inventory.ask.px, neutral.ask.px)
        self.assertGreater(short_inventory.bid.px, neutral.bid.px)
        self.assertGreater(short_inventory.ask.px, neutral.ask.px)

    def test_order_manager_keeps_passive_quote_for_one_tick_move(self) -> None:
        manager = OrderManager(symbol="A", risk=RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=2))
        manager.note_submitted(order_id="buy-1", side="BUY", px=999, qty=2, now_ms=1)
        plan = QuotePlan(
            mode="STEADY_MM",
            bid=DesiredOrder(side="BUY", px=1000, qty=2, aggressive=False, reason="test"),
            ask=None,
            aggressive_actions=(),
            observe_only=False,
            reason="test",
        )

        actions = manager.build_actions(plan, now_ms=100)
        self.assertEqual(actions.cancels, ())
        self.assertEqual(actions.placements, ())

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
            recovered_multiplier=1000.0,
            recovered_multiplier_confidence=2,
            recovered_fair_value=900,
            recovered_earnings_value=0.9,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 5}, asks={905: 5}), now_ms=45_000)

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
            journal.record_multiplier(1050.0, confidence=2, source="updated", estimate=1055.0, method="jump")
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


if __name__ == "__main__":
    unittest.main()
