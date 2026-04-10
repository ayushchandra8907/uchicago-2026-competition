from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import BConfig, RiskConfig
from b_mean_reversion import BMeanReversionStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class BMeanReversionStrategyTests(unittest.TestCase):
    def make_strategy(self) -> BMeanReversionStrategy:
        return BMeanReversionStrategy(
            BConfig(
                meanrev_max_position=8,
                meanrev_quote_size=1,
                meanrev_entry_z=1.25,
                meanrev_entry_z2=2.25,
                meanrev_exit_z=0.35,
                meanrev_stop_z=5.0,
                meanrev_min_spread_ticks=3,
                meanrev_sigma_floor=4.0,
                meanrev_aggressive_entry_z=2.75,
                meanrev_aggressive_exit=True,
                meanrev_max_hold_ms=120_000,
                meanrev_cooldown_ms=0,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )

    @staticmethod
    def payload(*, dispersion: float = 0.0, basis: float = 0.0) -> dict:
        return {"synthetic_dispersion": dispersion, "composite_basis": basis}

    def seed_signal(self, strategy: BMeanReversionStrategy, *, mid: float, ema_slow: float = 1000.0) -> None:
        bid = int(mid - 3)
        ask = int(mid + 3)
        strategy.on_book_update_at("B", FakeOrderBook(bids={bid: 10}, asks={ask: 10}), now_ms=1_000)
        strategy.ema_fast = ema_slow
        strategy.ema_slow = ema_slow
        strategy.last_sigma = 4.0
        strategy.ewma_var = 16.0

    def test_ema_and_sigma_update_from_book_stream(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_000)
        first_slow = strategy.ema_slow

        strategy.on_book_update_at("B", FakeOrderBook(bids={1010: 10}, asks={1014: 10}), now_ms=31_000)

        self.assertIsNotNone(first_slow)
        self.assertGreater(strategy.ema_slow, first_slow)
        self.assertGreaterEqual(strategy.last_sigma, 4.0)

    def test_positive_z_creates_sell_target(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1006.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertEqual(plan.ask.strategy_family, "b_mean_reversion")
        self.assertEqual(plan.ask.action_class, "mean_reversion_entry")
        self.assertEqual(strategy.last_target_inventory, -3)

    def test_negative_z_creates_buy_target(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=994.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.side, "BUY")
        self.assertEqual(plan.bid.action_class, "mean_reversion_entry")
        self.assertEqual(strategy.last_target_inventory, 3)

    def test_small_z_exits_existing_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(4)
        self.seed_signal(strategy, mid=1001.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.mode, "B_MEANREV_EXIT")
        self.assertEqual(plan.ask.action_class, "mean_reversion_exit")

    def test_locked_or_tight_book_blocks_new_entries(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1000: 10}, asks={1001: 10}), now_ms=1_000)
        strategy.ema_slow = 994.0
        strategy.last_sigma = 4.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "b_meanrev_book_too_tight")

    def test_stop_z_blocks_new_entries_and_passively_reduces_small_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(5)
        self.seed_signal(strategy, mid=1024.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(plan.mode, "B_MEANREV_RISK_OFF")
        self.assertEqual(len(plan.aggressive_actions), 0)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertFalse(plan.ask.aggressive)
        self.assertEqual(plan.ask.action_class, "mean_reversion_risk_off")
        self.assertFalse(strategy.trace_state(1_000)["b_meanrev_risk_off_forced"])

    def test_stop_z_forces_exit_at_full_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(8)
        self.seed_signal(strategy, mid=1024.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(plan.mode, "B_MEANREV_RISK_OFF")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "SELL")
        self.assertTrue(plan.aggressive_actions[0].aggressive)
        self.assertEqual(plan.aggressive_actions[0].action_class, "mean_reversion_risk_off")
        self.assertTrue(strategy.trace_state(1_000)["b_meanrev_risk_off_forced"])

    def test_max_hold_triggers_exit(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(-4)
        strategy.position_entry_ms = 1_000
        self.seed_signal(strategy, mid=990.0)

        plan = strategy.compute_quotes(now_ms=130_000, residual_payload=self.payload())

        self.assertEqual(plan.mode, "B_MEANREV_EXIT")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(plan.aggressive_actions[0].action_class, "mean_reversion_exit")

    def test_synthetic_filter_blocks_new_fade_entries(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1010.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload(basis=4.0))

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "b_meanrev_synthetic_confirms_deviation")

    def test_fill_after_position_update_does_not_double_count_inventory(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=994.0)
        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())
        self.assertIsNotNone(plan.bid)
        managed = strategy.order_manager.note_submitted(
            order_id="b-meanrev-buy-1",
            side=plan.bid.side,
            px=plan.bid.px,
            qty=plan.bid.qty,
            now_ms=1_000,
            aggressive=plan.bid.aggressive,
            intent=plan.bid.intent,
            mode_at_submit=plan.bid.mode_at_submit,
            action_class=plan.bid.action_class,
        )
        strategy.sync_inventory_from_exchange(1)

        strategy.on_fill(managed.order_id, 1, plan.bid.px, authoritative_inventory=1)

        self.assertEqual(strategy.inventory, 1)


if __name__ == "__main__":
    unittest.main()
