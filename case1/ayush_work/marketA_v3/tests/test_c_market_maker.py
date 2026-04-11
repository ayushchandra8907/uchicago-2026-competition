from __future__ import annotations

import unittest

from case1.ayush_work.marketA_v3.config import StrategyConfig
from case1.ayush_work.marketA_v3.core.types import BookLevel, BookSnapshot, NewsEvent, StrategySnapshot
from case1.ayush_work.marketA_v3.market_C_maker_strategy import CMarketMakerStrategy


def snapshot(
    *,
    now_ms: int = 1_000,
    inventory: int = 0,
    bid: int = 1000,
    ask: int = 1008,
    last_trade_px: int | None = None,
) -> StrategySnapshot:
    return StrategySnapshot(
        now_ms=now_ms,
        exchange_tick=now_ms // 100,
        book=BookSnapshot(best_bid=BookLevel(bid, 20), best_ask=BookLevel(ask, 20)),
        inventory=inventory,
        cash=0,
        fair_value=None,
        trusted_multiplier=None,
        latest_earnings=None,
        mode="IDLE",
        open_orders=(),
        last_trade_px=last_trade_px,
        message_index=None,
    )


class CMarketMakerStrategyTests(unittest.TestCase):
    def test_no_quote_when_spread_too_tight_and_flat(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C", maker_min_spread_ticks=6))
        decision = strategy.on_book(snapshot(bid=1000, ask=1004))
        self.assertTrue(decision.observe_only)
        self.assertIsNone(decision.desired_order)

    def test_sell_quote_after_buy_burst(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C", maker_min_spread_ticks=6, maker_min_recent_trade_count=3, maker_flow_trigger=2))
        for idx in range(3):
            decision = strategy.on_trade(snapshot(now_ms=1_000 + idx * 100, bid=1000, ask=1008, last_trade_px=1008))
        self.assertFalse(decision.observe_only)
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.px, 1008)
        self.assertEqual(decision.desired_order.intent, "stock_c_market_make")

    def test_buy_quote_after_sell_burst(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C", maker_min_spread_ticks=6, maker_min_recent_trade_count=3, maker_flow_trigger=2))
        for idx in range(3):
            decision = strategy.on_trade(snapshot(now_ms=1_000 + idx * 100, bid=1000, ask=1008, last_trade_px=1000))
        self.assertFalse(decision.observe_only)
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(decision.desired_order.px, 1000)

    def test_inventory_skew_recycles_even_without_burst(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C", maker_min_spread_ticks=6, maker_inventory_soft_limit=20))
        decision = strategy.on_book(snapshot(inventory=25, bid=1000, ask=1008))
        self.assertFalse(decision.observe_only)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.px, 1008)

    def test_aggressive_flatten_when_inventory_too_large(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C", maker_aggressive_flatten_inventory=60))
        decision = strategy.on_book(snapshot(inventory=70, bid=1000, ask=1008))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.px, 1000)
        self.assertEqual(decision.desired_order.intent, "stock_c_inventory_flatten")

    def test_news_is_ignored(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        decision = strategy.on_news(
            snapshot(),
            NewsEvent(now_ms=1_000, tick=1, kind="unstructured", symbol="C", content="noise", raw_payload={}),
        )
        self.assertTrue(decision.observe_only)


if __name__ == "__main__":
    unittest.main()
