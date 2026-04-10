from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import ETFConfig, RiskConfig
from a_bot_strategy import NewsReaction
from etf_a_follower import ETFAFollowerStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class ETFAFollowerTests(unittest.TestCase):
    def make_strategy(self, *, alpha: float = 0.25) -> ETFAFollowerStrategy:
        strategy = ETFAFollowerStrategy(
            ETFConfig(
                alpha_from_a=alpha,
                max_position=50,
                quote_size=8,
                target_position_per_etf_tick=1.0,
                min_a_fair_shift_ticks=20,
                min_projected_edge_ticks=3,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_000)
        return strategy

    def test_structured_a_shock_creates_damped_etf_buy_signal(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        reaction = NewsReaction(
            relevant=True,
            fair_value_updated=True,
            earnings_value=1.1,
            old_fair_value=1000,
            new_fair_value=1100,
        )

        signal = strategy.on_a_news_reaction(reaction, now_ms=1_100)
        plan = strategy.compute_quotes(now_ms=1_100)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.projected_etf_shift, 25.0)
        self.assertEqual(signal.target_inventory, 50)
        self.assertEqual(plan.mode, "ETF_A_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(plan.aggressive_actions[0].strategy_family, "etf_a_follower")
        self.assertEqual(plan.aggressive_actions[0].action_class, "etf_shock_take")

    def test_structured_a_shock_target_inventory_can_scale_etf_target(self) -> None:
        strategy = ETFAFollowerStrategy(
            ETFConfig(
                alpha_from_a=0.25,
                max_position=100,
                quote_size=16,
                target_position_per_etf_tick=1.0,
                target_position_per_a_shock_inventory=0.35,
                min_a_fair_shift_ticks=20,
                min_projected_edge_ticks=3,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_000)

        signal = strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.04,
                old_fair_value=1000,
                new_fair_value=1040,
                shock_target_inventory=200,
            ),
            now_ms=1_100,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.target_from_a_position, 70)
        self.assertEqual(signal.target_inventory, 70)

    def test_unconfirmed_a_news_does_not_start_etf_signal(self) -> None:
        strategy = self.make_strategy()
        reaction = NewsReaction(
            relevant=True,
            fair_value_updated=False,
            news_sentiment_score=2.0,
            news_sentiment_bucket="medium",
            base_fair_value=1000,
            news_fair_value=1048,
            pending_news_target_inventory=36,
            news_confirmation_state="pending",
        )

        signal = strategy.on_a_news_reaction(reaction, now_ms=1_100)

        self.assertIsNone(signal)
        self.assertIsNone(strategy.active_signal)

    def test_reaches_target_then_holds_instead_of_unwinding(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        signal = strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.1,
                old_fair_value=1000,
                new_fair_value=1100,
            ),
            now_ms=1_100,
        )
        self.assertIsNotNone(signal)
        strategy.sync_inventory_from_exchange(50)
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={1023: 10}, asks={1027: 10}), now_ms=1_500)

        plan = strategy.compute_quotes(now_ms=1_500, a_state={"mode": "POST_EARNINGS_SHOCK", "shock_direction": 1})

        self.assertEqual(plan.mode, "ETF_A_HOLD")
        self.assertEqual(len(plan.aggressive_actions), 0)

    def test_fill_after_position_update_does_not_double_count_inventory(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        strategy.order_manager.note_submitted(
            order_id="etf-buy-1",
            side="BUY",
            px=1002,
            qty=8,
            now_ms=1_100,
            aggressive=False,
            intent="etf_a_follower",
            mode_at_submit="ETF_A_SHOCK",
            action_class="etf_shock_take",
        )
        strategy.sync_inventory_from_exchange(8, now_ms=1_120)

        strategy.on_fill("etf-buy-1", 8, 1002, authoritative_inventory=8, now_ms=1_121)

        self.assertEqual(strategy.inventory, 8)

    def test_unwind_does_not_cross_zero_after_authoritative_fill(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.1,
                old_fair_value=1000,
                new_fair_value=1100,
            ),
            now_ms=1_100,
        )
        strategy.sync_inventory_from_exchange(-8, now_ms=4_400)
        plan = strategy.compute_quotes(now_ms=4_500, a_state={"mode": "AYUSH_IDLE", "shock_direction": 0})
        order = plan.aggressive_actions[0]
        strategy.order_manager.note_submitted(
            order_id="etf-unwind-1",
            side=order.side,
            px=order.px,
            qty=order.qty,
            now_ms=4_500,
            aggressive=order.aggressive,
            intent=order.intent,
            mode_at_submit=order.mode_at_submit,
            action_class=order.action_class,
        )
        strategy.sync_inventory_from_exchange(0, now_ms=4_520)
        strategy.on_fill("etf-unwind-1", 8, order.px, authoritative_inventory=0, now_ms=4_521)

        next_plan = strategy.compute_quotes(now_ms=4_522, a_state={"mode": "AYUSH_IDLE", "shock_direction": 0})

        self.assertEqual(strategy.inventory, 0)
        self.assertEqual(next_plan.mode, "ETF_OBSERVE_ONLY")
        self.assertEqual(len(next_plan.aggressive_actions), 0)

    def test_unwinds_after_min_hold_when_a_shock_inactive(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.1,
                old_fair_value=1000,
                new_fair_value=1100,
            ),
            now_ms=1_100,
        )
        strategy.sync_inventory_from_exchange(50)
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={1023: 10}, asks={1027: 10}), now_ms=4_500)

        plan = strategy.compute_quotes(now_ms=4_500, a_state={"mode": "AYUSH_IDLE", "shock_direction": 0})

        self.assertEqual(plan.mode, "ETF_UNWIND")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "SELL")

    def test_crossed_etf_book_does_not_submit_orders(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.1,
                old_fair_value=1000,
                new_fair_value=1100,
            ),
            now_ms=1_100,
        )
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={1003: 10}, asks={1001: 10}), now_ms=1_200)

        plan = strategy.compute_quotes(now_ms=1_200)

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "etf_book_crossed_or_locked")
        self.assertEqual(len(plan.aggressive_actions), 0)


if __name__ == "__main__":
    unittest.main()
