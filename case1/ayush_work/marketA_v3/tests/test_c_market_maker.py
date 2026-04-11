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
    books_by_symbol: dict[str, BookSnapshot] | None = None,
    open_orders=(),
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
        open_orders=open_orders,
        last_trade_px=last_trade_px,
        message_index=None,
        books_by_symbol=books_by_symbol or {},
    )


def with_rate_books(
    *,
    c_bid: int = 1000,
    c_ask: int = 1008,
    hike_bid: int = 300,
    hike_ask: int = 304,
    hold_bid: int = 392,
    hold_ask: int = 396,
    cut_bid: int = 296,
    cut_ask: int = 300,
    **kwargs,
) -> StrategySnapshot:
    books_by_symbol = {
        "C": BookSnapshot(best_bid=BookLevel(c_bid, 20), best_ask=BookLevel(c_ask, 20)),
        "R_HIKE": BookSnapshot(best_bid=BookLevel(hike_bid, 20), best_ask=BookLevel(hike_ask, 20)),
        "R_HOLD": BookSnapshot(best_bid=BookLevel(hold_bid, 20), best_ask=BookLevel(hold_ask, 20)),
        "R_CUT": BookSnapshot(best_bid=BookLevel(cut_bid, 20), best_ask=BookLevel(cut_ask, 20)),
    }
    return snapshot(bid=c_bid, ask=c_ask, books_by_symbol=books_by_symbol, **kwargs)


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

    def test_non_earnings_news_is_ignored(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        decision = strategy.on_news(
            snapshot(),
            NewsEvent(now_ms=1_000, tick=1, kind="unstructured", symbol="C", content="noise", raw_payload={}),
        )
        self.assertTrue(decision.observe_only)

    def test_first_c_earnings_report_sets_baseline_without_trading(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        decision = strategy.on_news(
            with_rate_books(),
            NewsEvent(
                now_ms=1_000,
                tick=10,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.0,
                raw_payload={},
            ),
        )
        self.assertTrue(decision.observe_only)
        self.assertIsNone(decision.desired_order)
        self.assertTrue(strategy.have_real_eps_c)
        self.assertEqual(strategy.latest_earnings, 2.0)
        self.assertIsNotNone(strategy.anchor_price)

    def test_second_c_earnings_report_can_start_long_shock(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        baseline = with_rate_books(c_bid=1000, c_ask=1008)
        strategy.on_news(
            baseline,
            NewsEvent(
                now_ms=1_000,
                tick=10,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.0,
                raw_payload={},
            ),
        )
        decision = strategy.on_news(
            with_rate_books(now_ms=2_000, c_bid=1000, c_ask=1008),
            NewsEvent(
                now_ms=2_000,
                tick=20,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.12,
                raw_payload={},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(decision.desired_order.symbol, "C")
        self.assertEqual(strategy.active_signal_kind, "structured_c_earnings")

    def test_second_c_earnings_report_can_start_short_shock(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        baseline = with_rate_books(c_bid=1000, c_ask=1008)
        strategy.on_news(
            baseline,
            NewsEvent(
                now_ms=1_000,
                tick=10,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.0,
                raw_payload={},
            ),
        )
        decision = strategy.on_news(
            with_rate_books(now_ms=2_000, c_bid=1000, c_ask=1008),
            NewsEvent(
                now_ms=2_000,
                tick=20,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=1.90,
                raw_payload={},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.symbol, "C")

    def test_c_earnings_takeover_flattens_existing_inventory_first(self):
        strategy = CMarketMakerStrategy(StrategyConfig(symbol="C"))
        strategy.on_news(
            with_rate_books(),
            NewsEvent(
                now_ms=1_000,
                tick=10,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.0,
                raw_payload={},
            ),
        )
        decision = strategy.on_news(
            with_rate_books(now_ms=2_000, inventory=15),
            NewsEvent(
                now_ms=2_000,
                tick=20,
                kind="structured",
                symbol="C",
                structured_subtype="earnings",
                asset="C",
                value=2.12,
                raw_payload={},
            ),
        )
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(strategy.pending_earnings_value, 2.12)


if __name__ == "__main__":
    unittest.main()
