from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import BConfig, RiskConfig
from b_option_lottery import BOptionLotteryStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class BOptionLotteryTests(unittest.TestCase):
    def make_strategy(self) -> BOptionLotteryStrategy:
        strategy = BOptionLotteryStrategy(
            BConfig(
                option_lottery_enabled=True,
                option_lottery_max_ask=3,
                option_lottery_floor_ask=0,
                option_lottery_quote_size=2,
                option_lottery_max_position_per_symbol=20,
                option_lottery_wing_max_position=200,
                option_lottery_atm_max_position=40,
                option_lottery_total_premium_budget=12,
                option_lottery_wing_premium_budget=9,
                option_lottery_atm_total_premium_budget=6,
                option_lottery_near_strike_ticks=80,
                option_lottery_min_momentum_ticks=1,
                option_lottery_rebuy_cooldown_ms=0,
                option_lottery_profit_take_enabled=True,
                option_lottery_profit_take_min_edge=6,
                option_lottery_profit_take_multiple=2.0,
                option_lottery_profit_take_quote_size=20,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("B", FakeOrderBook(bids={999: 10}, asks={1001: 10}), now_ms=1_000)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1002: 10}, asks={1004: 10}), now_ms=1_100)
        return strategy

    def test_zero_ask_call_produces_small_buy_order(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={0: 10}, asks={0: 10}), now_ms=1_200)

        plan = strategy.compute_quote_plan("B_C_1050", now_ms=1_200, residual_payload=None)

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.side, "BUY")
        self.assertEqual(plan.bid.px, 0)
        self.assertEqual(plan.bid.strategy_family, "b_option_lottery")
        self.assertEqual(plan.bid.action_class, "cheap_option_buy")

    def test_cheap_call_requires_directional_setup(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={1: 10}, asks={3: 10}), now_ms=1_200)

        plan = strategy.compute_quote_plan(
            "B_C_1050",
            now_ms=1_200,
            residual_payload={"composite_basis": 2.0, "underlying_mid": 1003.0},
        )

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.qty, 2)

    def test_expensive_option_does_not_trade(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={4: 10}, asks={5: 10}), now_ms=1_200)

        plan = strategy.compute_quote_plan("B_C_1050", now_ms=1_200, residual_payload={"composite_basis": 5.0})

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "option_ask_above_lottery_threshold")

    def test_premium_budget_caps_quantity(self) -> None:
        strategy = self.make_strategy()
        strategy.premium_spent = 10
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={1: 10}, asks={3: 10}), now_ms=1_200)

        plan = strategy.compute_quote_plan("B_C_1050", now_ms=1_200, residual_payload={"composite_basis": 5.0})

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "option_lottery_premium_budget_spent")

    def test_wing_symbol_can_scale_to_large_cap_when_budget_allows(self) -> None:
        strategy = BOptionLotteryStrategy(
            BConfig(
                option_lottery_quote_size=50,
                option_lottery_wing_max_position=200,
                option_lottery_total_premium_budget=1_500,
                option_lottery_wing_premium_budget=600,
                option_lottery_rebuy_cooldown_ms=0,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("B", FakeOrderBook(bids={999: 10}, asks={1001: 10}), now_ms=1_000)
        strategy.on_book_update_at("B", FakeOrderBook(bids={996: 10}, asks={998: 10}), now_ms=1_100)
        strategy.on_book_update_at("B_P_950", FakeOrderBook(bids={1: 10}, asks={3: 200}), now_ms=1_200)

        plan = strategy.compute_quote_plan("B_P_950", now_ms=1_200, residual_payload={"composite_basis": -2.0, "underlying_mid": 997.0})

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.qty, 50)
        self.assertEqual(strategy.trace_state("B_P_950", 1_200)["position_cap"], 200)

    def test_atm_symbols_share_smaller_premium_budget(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B_C_1000", FakeOrderBook(bids={1: 10}, asks={3: 10}), now_ms=1_200)
        strategy.open_cost_basis["B_P_1000"] = 6.0

        plan = strategy.compute_quote_plan(
            "B_C_1000",
            now_ms=1_200,
            residual_payload={"composite_basis": 2.0, "underlying_mid": 1003.0},
        )

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "option_lottery_premium_budget_spent")

    def test_fill_after_position_update_does_not_double_count_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={1: 10}, asks={3: 10}), now_ms=1_200)
        plan = strategy.compute_quote_plan(
            "B_C_1050",
            now_ms=1_200,
            residual_payload={"composite_basis": 2.0, "underlying_mid": 1003.0},
        )
        self.assertIsNotNone(plan.bid)
        managed = strategy.note_submitted("B_C_1050", order_id="opt-buy-1", desired=plan.bid, now_ms=1_200)
        strategy.sync_inventory("B_C_1050", 2)

        strategy.on_fill(
            managed.order_id,
            2,
            3,
            now_ms=1_210,
            authoritative_inventory=2,
            pre_fill_inventory=0,
        )

        self.assertEqual(strategy.inventory("B_C_1050"), 2)
        self.assertEqual(strategy.premium_spent, 6)

    def test_profit_take_sells_existing_long_only(self) -> None:
        strategy = self.make_strategy()
        strategy.positions["B_C_1050"] = 10
        strategy.open_cost_basis["B_C_1050"] = 30.0
        strategy.on_book_update_at("B_C_1050", FakeOrderBook(bids={9: 10}, asks={10: 10}), now_ms=1_200)

        plan = strategy.compute_quote_plan("B_C_1050", now_ms=1_200, residual_payload={"composite_basis": 0.0})

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertLessEqual(plan.ask.qty, 5)
        self.assertEqual(plan.ask.action_class, "cheap_option_profit_take")


if __name__ == "__main__":
    unittest.main()
