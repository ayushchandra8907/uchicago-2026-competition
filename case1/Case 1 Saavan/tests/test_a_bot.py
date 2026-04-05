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
from a_bot_strategy import ManagedOrder, MarketAStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class MarketABotTests(unittest.TestCase):
    def make_strategy(
        self,
        *,
        initial_fair_value: int | None = None,
        recovered_fair_value: int | None = None,
        recovered_earnings_value: float | None = None,
        restored_orders=(),
    ) -> MarketAStrategy:
        strategy = MarketAStrategy(
            a_config=AConfig(
                pe_ratio=10.0,
                price_scale=100,
                initial_fair_value=initial_fair_value,
                startup_assume_fresh_round=True,
                pre_news_pullback_ms=4_000,
                steady_half_spread_ticks=1,
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
            risk=RiskConfig(reprice_cooldown_ms=0),
            restored_orders=restored_orders,
            recovered_fair_value=recovered_fair_value,
            recovered_earnings_value=recovered_earnings_value,
        )
        strategy.startup_ms = 0
        return strategy

    @staticmethod
    def a_earnings_news(value: float, tick: int = 110) -> dict:
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

    def test_earnings_news_updates_fair_value_with_fixed_ratio(self) -> None:
        strategy = self.make_strategy()
        reaction = strategy.on_news(self.a_earnings_news(0.9), now_ms=22_000)
        self.assertTrue(reaction.relevant)
        self.assertEqual(strategy.fair_value, 900)

        reaction = strategy.on_news(self.a_earnings_news(1.1, tick=440), now_ms=88_000)
        self.assertEqual(strategy.fair_value, 1100)
        self.assertEqual(reaction.old_fair_value, 900)
        self.assertEqual(reaction.new_fair_value, 1100)

    def test_opening_micro_mm_quotes_tiny_and_wide_before_first_earnings(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=5_000)
        plan = strategy.compute_quotes(now_ms=5_000)

        self.assertEqual(plan.mode, "OPENING_MICRO_MM")
        self.assertFalse(plan.observe_only)
        self.assertEqual(plan.bid.qty, 1)
        self.assertEqual(plan.ask.qty, 1)
        self.assertEqual(plan.bid.px, 996)
        self.assertEqual(plan.ask.px, 1004)

    def test_pre_news_pullback_kicks_in_before_first_scheduled_earnings(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=18_500)
        plan = strategy.compute_quotes(now_ms=18_500)

        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertIsNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.aggressive_actions, ())

    def test_tick_aligned_pullback_uses_exchange_schedule_after_news_sync(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=0)
        strategy.on_news(self.a_earnings_news(0.9, tick=110), now_ms=0)
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=62_500)

        plan = strategy.compute_quotes(now_ms=62_500)
        self.assertEqual(plan.mode, "PRE_NEWS_PULLBACK")
        self.assertEqual(strategy.ms_until_next_scheduled_earnings(62_500), 3_500)

    def test_upside_shock_buys_stale_asks_then_posts_unwind_sell(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=0)
        strategy.on_news(self.a_earnings_news(1.1, tick=110), now_ms=0)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1030: 10}, asks={1040: 7}), now_ms=100)

        plan = strategy.compute_quotes(now_ms=100)
        self.assertEqual(plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(plan.aggressive_actions[0].px, 1040)
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertGreaterEqual(plan.ask.px, 1101)

    def test_downside_shock_sells_stale_bids_then_posts_unwind_buy(self) -> None:
        strategy = self.make_strategy(initial_fair_value=1100)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1095: 10}, asks={1105: 10}), now_ms=0)
        strategy.on_news(self.a_earnings_news(0.9, tick=440), now_ms=0)
        strategy.on_book_update_at("A", FakeOrderBook(bids={960: 8}, asks={970: 10}), now_ms=100)

        plan = strategy.compute_quotes(now_ms=100)
        self.assertEqual(plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "SELL")
        self.assertEqual(plan.aggressive_actions[0].px, 960)
        self.assertIsNotNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertLessEqual(plan.bid.px, 899)

    def test_steady_quotes_tighten_around_fair_and_skew_with_inventory(self) -> None:
        strategy = self.make_strategy(initial_fair_value=900)
        strategy.shock_started_ms = None
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=30_000)

        strategy.set_inventory(0)
        neutral = strategy.compute_quotes(now_ms=30_000)
        strategy.set_inventory(10)
        long_inventory = strategy.compute_quotes(now_ms=30_000)
        strategy.set_inventory(-10)
        short_inventory = strategy.compute_quotes(now_ms=30_000)

        self.assertEqual(neutral.mode, "STEADY_MM")
        self.assertEqual(neutral.bid.px, 899)
        self.assertEqual(neutral.ask.px, 901)
        self.assertLess(long_inventory.bid.px, neutral.bid.px)
        self.assertLess(long_inventory.ask.px, neutral.ask.px)
        self.assertGreater(short_inventory.bid.px, neutral.bid.px)
        self.assertGreater(short_inventory.ask.px, neutral.ask.px)

    def test_steady_mode_uses_steady_cap_not_shock_cap(self) -> None:
        strategy = self.make_strategy(initial_fair_value=900)
        strategy.shock_started_ms = None
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=30_000)
        strategy.set_inventory(24)

        plan = strategy.compute_quotes(now_ms=30_000)
        self.assertEqual(plan.mode, "STEADY_MM")
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)

    def test_unwind_biases_quotes_back_toward_flat(self) -> None:
        strategy = self.make_strategy(initial_fair_value=900)
        strategy.shock_started_ms = 0
        strategy.on_book_update_at("A", FakeOrderBook(bids={900: 10}, asks={902: 10}), now_ms=5_000)
        strategy.set_inventory(6)

        plan = strategy.compute_quotes(now_ms=5_000)
        self.assertEqual(plan.mode, "UNWIND")
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)

    def test_fill_updates_remaining_and_inventory(self) -> None:
        strategy = self.make_strategy(initial_fair_value=900)
        strategy.order_manager.note_submitted(
            order_id="buy-1",
            side="BUY",
            px=899,
            qty=2,
            now_ms=1,
        )
        strategy.set_inventory(0)

        order = strategy.on_fill("buy-1", qty=1, price=899)
        self.assertIsNotNone(order)
        self.assertEqual(order.remaining_qty, 1)
        self.assertEqual(strategy.inventory, 1)

        strategy.on_fill("buy-1", qty=1, price=899)
        self.assertIsNone(strategy.order_manager.live_order("BUY"))
        self.assertEqual(strategy.inventory, 2)

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
            recovered_fair_value=900,
            recovered_earnings_value=0.9,
        )
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 5}, asks={905: 5}), now_ms=30_000)

        before = strategy.compute_quotes(now_ms=30_000)
        self.assertTrue(before.observe_only)
        self.assertEqual([order.order_id for order in strategy.recovery_orders_to_cancel()], ["restored-1"])

        strategy.order_manager.mark_cancel_requested("restored-1", now_ms=30_010)
        strategy.on_cancel_response("restored-1", success=False)

        after = strategy.compute_quotes(now_ms=30_050)
        self.assertFalse(after.observe_only)
        self.assertFalse(strategy.recovery_active)
        self.assertIsNotNone(after.bid)

    def test_journal_replays_live_orders_inventory_and_fair_value(self) -> None:
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
            journal.record_fair_value(900, source="earnings", earnings_value=0.9)
            journal.record_inventory(7, cash=1000)

            replay = journal.load_replay_state()
            self.assertEqual(replay.fair_value, 900)
            self.assertAlmostEqual(replay.earnings_value, 0.9)
            self.assertEqual(replay.inventory, 7)
            self.assertEqual(len(replay.live_orders), 1)
            self.assertEqual(replay.live_orders[0].remaining_qty, 1)
            self.assertTrue(replay.live_orders[0].restored)


if __name__ == "__main__":
    unittest.main()
