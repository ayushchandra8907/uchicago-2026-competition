from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import AConfig, RiskConfig
from ayush_a_port import AyushPortStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class AyushPortStrategyTests(unittest.TestCase):
    def make_strategy(self, *, initial_fair_value: int | None = None) -> AyushPortStrategy:
        return AyushPortStrategy(
            a_config=AConfig(),
            risk=RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
            initial_fair_value=initial_fair_value,
            book_depth_levels=5,
        )

    def seed_baseline(self, strategy: AyushPortStrategy, *, mid: int, start_ms: int = 0, samples: int = 10) -> None:
        for index in range(samples):
            strategy.on_book_update_at(
                "A",
                FakeOrderBook(bids={mid - 2: 10}, asks={mid + 2: 10}),
                now_ms=start_ms + (index * 1_000),
            )

    def send_unknown_a_news(self, strategy: AyushPortStrategy, *, now_ms: int = 10_000) -> None:
        strategy.on_news(
            {
                "tick": 50,
                "kind": "unstructured",
                "symbol": "A",
                "new_data": {
                    "content": "A mentions a completely novel undecidable update.",
                    "type": "News",
                },
            },
            now_ms=now_ms,
        )

    def test_strong_a_news_immediately_enters_news_shock(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)

        reaction = strategy.on_news(
            {
                "tick": 123,
                "kind": "unstructured",
                "symbol": "A",
                "new_data": {
                    "content": "A secures a leading position in a growing niche market.",
                    "type": "News",
                },
            },
            now_ms=1_100,
        )
        plan = strategy.compute_quotes(now_ms=1_100)

        self.assertTrue(reaction.relevant)
        self.assertEqual(plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        order = plan.aggressive_actions[0]
        self.assertEqual(order.strategy_family, "a_news")
        self.assertEqual(order.action_class, "news_take")
        self.assertEqual(order.side, "BUY")

    def test_structured_earnings_emits_earnings_shock_order(self) -> None:
        strategy = self.make_strategy()
        for timestamp in range(0, 10_000, 1_000):
            strategy.on_book_update_at("A", FakeOrderBook(bids={995: 10}, asks={1005: 10}), now_ms=timestamp)
        strategy.on_book_update_at("A", FakeOrderBook(bids={895: 10}, asks={905: 10}), now_ms=10_000)

        reaction = strategy.on_news(
            {
                "tick": 150,
                "kind": "structured",
                "symbol": "A",
                "new_data": {
                    "structured_subtype": "earnings",
                    "asset": "A",
                    "value": 1.0,
                },
            },
            now_ms=10_100,
        )
        plan = strategy.compute_quotes(now_ms=10_100)

        self.assertTrue(reaction.relevant)
        self.assertEqual(plan.mode, "POST_EARNINGS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        order = plan.aggressive_actions[0]
        self.assertEqual(order.strategy_family, "a_earnings")
        self.assertEqual(order.action_class, "shock_take")
        self.assertEqual(order.side, "BUY")
        self.assertGreater(order.qty, 0)

    def test_a_news_uses_faithful_ayush_permanent_pe_freeze(self) -> None:
        strategy = self.make_strategy(initial_fair_value=1000)
        strategy.on_book_update_at("A", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_000)

        self.send_unknown_a_news(strategy, now_ms=1_100)
        self.assertTrue(strategy._strategy.pe_frozen)

        strategy.evaluate_runtime(now_ms=1_250)
        self.assertTrue(strategy._strategy.pe_frozen)

    def test_medium_news_confirms_then_enters_news_shock(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)

        reaction = strategy.on_news(
            {
                "tick": 321,
                "kind": "unstructured",
                "symbol": "A",
                "new_data": {
                    "content": "Analysts predict strong revenue growth for A.",
                    "type": "News",
                },
            },
            now_ms=1_100,
        )
        plan = strategy.compute_quotes(now_ms=1_100)
        self.assertTrue(reaction.relevant)
        self.assertEqual(plan.mode, "NEWS_CONFIRMATION")
        self.assertTrue(plan.observe_only)
        self.assertEqual(len(plan.aggressive_actions), 0)

        strategy.on_book_update_at("A", FakeOrderBook(bids={1102: 10}, asks={1106: 10}), now_ms=1_250)
        confirmed_plan = strategy.compute_quotes(now_ms=1_250)
        self.assertEqual(confirmed_plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(confirmed_plan.aggressive_actions), 1)
        self.assertEqual(confirmed_plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(confirmed_plan.aggressive_actions[0].strategy_family, "a_news")

    def test_takeover_flatten_stays_one_sided_until_inventory_clears(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(10, now_ms=1_000)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)

        strategy.on_news(
            {
                "tick": 456,
                "kind": "unstructured",
                "symbol": "A",
                "new_data": {
                    "content": "A secures a leading position in a growing niche market.",
                    "type": "News",
                },
            },
            now_ms=1_100,
        )
        first_plan = strategy.compute_quotes(now_ms=1_100)
        self.assertEqual(first_plan.mode, "UNWIND")
        self.assertEqual(len(first_plan.aggressive_actions), 1)
        self.assertEqual(first_plan.aggressive_actions[0].side, "SELL")
        self.assertEqual(first_plan.aggressive_actions[0].action_class, "news_takeover_flatten")

        strategy.on_book_update_at("A", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_150)
        second_plan = strategy.compute_quotes(now_ms=1_150)
        self.assertEqual(second_plan.mode, "UNWIND")
        self.assertEqual(len(second_plan.aggressive_actions), 1)
        self.assertEqual(second_plan.aggressive_actions[0].side, "SELL")
        self.assertEqual(second_plan.aggressive_actions[0].action_class, "news_takeover_flatten")

        strategy.sync_inventory_from_exchange(0, now_ms=1_200)
        strategy.on_book_update_at("A", FakeOrderBook(bids={1102: 10}, asks={1106: 10}), now_ms=1_250)
        shock_plan = strategy.compute_quotes(now_ms=1_250)
        self.assertEqual(shock_plan.mode, "POST_NEWS_SHOCK")
        self.assertEqual(len(shock_plan.aggressive_actions), 1)
        self.assertEqual(shock_plan.aggressive_actions[0].side, "BUY")

    def test_earnings_shock_buy_above_fair_guard_is_blocked(self) -> None:
        strategy = AyushPortStrategy(
            a_config=AConfig(earnings_shock_entry_guard_ticks=24),
            risk=RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=2, passive_quote_ttl_ms=3_000),
            initial_fair_value=1_000,
            book_depth_levels=5,
        )
        strategy._strategy.fair_value = 1_000

        plan = strategy._translate_decision(
            SimpleNamespace(
                mode="SHOCK",
                observe_only=False,
                reason="shock",
                desired_order=SimpleNamespace(
                    side="BUY",
                    px=1_040,
                    qty=12,
                    aggressive=True,
                    reason="shock buy",
                    intent="post_earnings_shock_take",
                ),
            )
        )

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "a_shock_price_guard_blocked")
        self.assertEqual(strategy.trace_state(0)["a_shock_entry_price_vs_fair"], 40)
        self.assertTrue(strategy.trace_state(0)["a_shock_price_guard_blocked"])

    def test_earnings_shock_sell_below_fair_guard_is_blocked(self) -> None:
        strategy = AyushPortStrategy(
            a_config=AConfig(earnings_shock_entry_guard_ticks=24),
            risk=RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=2, passive_quote_ttl_ms=3_000),
            initial_fair_value=1_000,
            book_depth_levels=5,
        )
        strategy._strategy.fair_value = 1_000

        plan = strategy._translate_decision(
            SimpleNamespace(
                mode="SHOCK",
                observe_only=False,
                reason="shock",
                desired_order=SimpleNamespace(
                    side="SELL",
                    px=960,
                    qty=12,
                    aggressive=True,
                    reason="shock sell",
                    intent="post_earnings_shock_take",
                ),
            )
        )

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "a_shock_price_guard_blocked")
        self.assertEqual(strategy.trace_state(0)["a_shock_entry_price_vs_fair"], 40)
        self.assertEqual(strategy.trace_state(0)["a_shock_entry_block_count"], 1)


if __name__ == "__main__":
    unittest.main()
