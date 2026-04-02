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
from a_bot_strategy import BookLevel, BookSnapshot, ManagedOrder, MarketAStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class MarketABotTests(unittest.TestCase):
    def make_strategy(
        self,
        *,
        initial_fair_value: int | None = 100,
        take_edge: int = 4,
        inventory_skew: float = 0.35,
        restored_orders=(),
        recovered_fair_value: int | None = None,
    ) -> MarketAStrategy:
        return MarketAStrategy(
            a_config=AConfig(pe_ratio=7.0, initial_fair_value=initial_fair_value),
            risk=RiskConfig(
                max_position=20,
                quote_size=5,
                min_edge=2,
                take_edge=take_edge,
                inventory_skew=inventory_skew,
                reprice_cooldown_ms=0,
            ),
            restored_orders=restored_orders,
            recovered_fair_value=recovered_fair_value,
        )

    def test_earnings_news_updates_fair_value(self) -> None:
        strategy = self.make_strategy(initial_fair_value=None)
        changed = strategy.on_news(
            {
                "kind": "structured",
                "symbol": "A",
                "new_data": {
                    "structured_subtype": "earnings",
                    "asset": "A",
                    "value": 12.5,
                },
            }
        )
        self.assertTrue(changed)
        self.assertEqual(strategy.fair_value, round(7.0 * 12.5))

    def test_missing_fair_value_keeps_observe_only(self) -> None:
        strategy = self.make_strategy(initial_fair_value=None)
        strategy.on_book_update("A", FakeOrderBook(bids={99: 10}, asks={101: 10}))
        plan = strategy.compute_quotes()
        self.assertTrue(plan.observe_only)
        self.assertIsNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.aggressive_actions, ())

    def test_quote_skews_with_inventory(self) -> None:
        strategy = self.make_strategy(initial_fair_value=100, inventory_skew=0.5)
        strategy.on_book_update("A", FakeOrderBook(bids={97: 10}, asks={103: 10}))

        strategy.set_inventory(0)
        neutral_plan = strategy.compute_quotes()

        strategy.set_inventory(6)
        long_plan = strategy.compute_quotes()

        strategy.set_inventory(-6)
        short_plan = strategy.compute_quotes()

        self.assertLess(long_plan.bid.px, neutral_plan.bid.px)
        self.assertLess(long_plan.ask.px, neutral_plan.ask.px)
        self.assertGreater(short_plan.bid.px, neutral_plan.bid.px)
        self.assertGreater(short_plan.ask.px, neutral_plan.ask.px)

    def test_aggressive_take_requires_edge(self) -> None:
        strategy = self.make_strategy(initial_fair_value=100, take_edge=3)
        strategy.on_book_update("A", FakeOrderBook(bids={95: 5}, asks={97: 2}))
        aggressive_plan = strategy.compute_quotes()
        self.assertEqual(len(aggressive_plan.aggressive_actions), 1)
        self.assertEqual(aggressive_plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(aggressive_plan.aggressive_actions[0].px, 97)

        strategy.on_book_update("A", FakeOrderBook(bids={95: 5}, asks={98: 2}))
        passive_plan = strategy.compute_quotes()
        self.assertEqual(passive_plan.aggressive_actions, ())
        self.assertIsNotNone(passive_plan.bid)

    def test_fill_updates_remaining_and_inventory(self) -> None:
        strategy = self.make_strategy(initial_fair_value=100)
        strategy.order_manager.note_submitted(
            order_id="buy-1",
            side="BUY",
            px=98,
            qty=5,
            now_ms=1,
        )
        strategy.set_inventory(0)

        order = strategy.on_fill("buy-1", qty=2, price=98)
        self.assertIsNotNone(order)
        self.assertEqual(order.remaining_qty, 3)
        self.assertEqual(strategy.inventory, 2)

        strategy.on_fill("buy-1", qty=3, price=98)
        self.assertIsNone(strategy.order_manager.live_order("BUY"))
        self.assertEqual(strategy.inventory, 5)

    def test_recovery_blocks_until_restored_order_is_resolved(self) -> None:
        restored = ManagedOrder(
            order_id="restored-1",
            side="BUY",
            px=96,
            qty=4,
            remaining_qty=4,
            submitted_ms=10,
            restored=True,
        )
        strategy = self.make_strategy(
            initial_fair_value=None,
            restored_orders=(restored,),
            recovered_fair_value=102,
        )
        strategy.on_book_update("A", FakeOrderBook(bids={99: 5}, asks={103: 5}))

        before = strategy.compute_quotes()
        self.assertTrue(before.observe_only)
        self.assertEqual([order.order_id for order in strategy.recovery_orders_to_cancel()], ["restored-1"])

        strategy.order_manager.mark_cancel_requested("restored-1", now_ms=20)
        strategy.on_cancel_response("restored-1", success=False)

        after = strategy.compute_quotes()
        self.assertFalse(after.observe_only)
        self.assertFalse(strategy.recovery_active)
        self.assertIsNotNone(after.bid)

    def test_journal_replays_live_orders_inventory_and_fair_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = TradingJournal(Path(temp_dir) / "journal.jsonl")
            submitted = ManagedOrder(
                order_id="order-1",
                side="SELL",
                px=104,
                qty=5,
                remaining_qty=5,
                submitted_ms=12,
            )
            journal.record_order_submitted(submitted)
            journal.record_fill("order-1", qty=2, price=104)
            journal.record_fair_value(111, source="earnings", earnings_value=15.8)
            journal.record_inventory(7, cash=1000)

            replay = journal.load_replay_state()
            self.assertEqual(replay.fair_value, 111)
            self.assertEqual(replay.inventory, 7)
            self.assertEqual(len(replay.live_orders), 1)
            self.assertEqual(replay.live_orders[0].remaining_qty, 3)
            self.assertTrue(replay.live_orders[0].restored)


if __name__ == "__main__":
    unittest.main()
