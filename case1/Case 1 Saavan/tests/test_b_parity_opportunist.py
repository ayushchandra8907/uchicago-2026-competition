from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import BConfig
from b_observer import MarketBObserver
from b_parity_opportunist import BParityOpportunist


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class BParityOpportunistTests(unittest.TestCase):
    def make_observer_with_conversion_edge(self) -> MarketBObserver:
        observer = MarketBObserver(depth_levels=5)
        observer.on_book_update("B", FakeOrderBook(bids={999: 10}, asks={1000: 10}), now_ms=1_000)
        observer.on_book_update("B_C_950", FakeOrderBook(bids={80: 10}, asks={82: 10}), now_ms=1_000)
        observer.on_book_update("B_P_950", FakeOrderBook(bids={18: 10}, asks={20: 10}), now_ms=1_000)
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={50: 10}, asks={54: 10}), now_ms=1_000)
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={45: 10}, asks={49: 10}), now_ms=1_000)
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={25: 10}, asks={29: 10}), now_ms=1_000)
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={70: 10}, asks={74: 10}), now_ms=1_000)
        return observer

    def test_observer_computes_tradeable_conversion_and_reversal_edges(self) -> None:
        observer = self.make_observer_with_conversion_edge()

        payload = observer.compute_residuals(now_ms=1_010)

        self.assertIsNotNone(payload)
        strike_950 = payload["tradeable_parity_by_strike"]["950"]
        self.assertEqual(strike_950["conversion_cost"], 940)
        self.assertAlmostEqual(strike_950["conversion_edge"], 10.0, delta=0.001)
        self.assertAlmostEqual(strike_950["reversal_edge"], -15.0, delta=0.001)
        self.assertEqual(strike_950["max_quote_age_ms"], 10)

    def test_opportunist_triggers_only_above_threshold(self) -> None:
        observer = self.make_observer_with_conversion_edge()
        payload = observer.compute_residuals(now_ms=1_010)

        quiet = BParityOpportunist(BConfig(parity_edge_threshold_ticks=12, parity_trade_size=1))
        active = BParityOpportunist(BConfig(parity_edge_threshold_ticks=8, parity_trade_size=1))

        self.assertIsNone(quiet.evaluate(payload, now_ms=1_010))
        opportunity = active.evaluate(payload, now_ms=1_010)
        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.kind, "conversion")
        self.assertEqual(opportunity.strike, 950)
        self.assertEqual(len(opportunity.legs), 3)

    def test_desired_orders_are_tagged_for_attribution(self) -> None:
        observer = self.make_observer_with_conversion_edge()
        payload = observer.compute_residuals(now_ms=1_010)
        opportunist = BParityOpportunist(BConfig(parity_edge_threshold_ticks=8, parity_trade_size=1))
        opportunity = opportunist.evaluate(payload, now_ms=1_010)

        orders = opportunist.desired_orders_for(opportunity)

        self.assertEqual(len(orders), 3)
        self.assertTrue(all(order.market_key == "B" for order in orders))
        self.assertTrue(all(order.strategy_family == "b_parity_opportunist" for order in orders))
        self.assertTrue(all(order.pnl_owner == "b_parity_opportunist" for order in orders))
        self.assertTrue(all(order.action_class == "conversion" for order in orders))
        self.assertTrue(all(order.aggressive for order in orders))


if __name__ == "__main__":
    unittest.main()
