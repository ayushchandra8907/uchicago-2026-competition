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
                meanrev_max_position=16,
                meanrev_quote_size=2,
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
                meanrev_entry_ticks=10,
                meanrev_full_entry_ticks=15,
                meanrev_exit_ticks=3,
                meanrev_base_target=6,
                meanrev_full_target=16,
                meanrev_extreme_entry_ticks=20,
                meanrev_risk_off_deviation_ticks=35,
                meanrev_turn_confirm_ms=300,
                meanrev_min_healthy_book_age_ms=0,
                meanrev_bad_fill_cooldown_ms=1_000,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )

    @staticmethod
    def payload(*, dispersion: float = 0.0, basis: float = 0.0, synthetic_fair: float | None = None) -> dict:
        payload = {"synthetic_dispersion": dispersion, "composite_basis": basis}
        if synthetic_fair is not None:
            payload["composite_synthetic_fair"] = synthetic_fair
        return payload

    def seed_signal(self, strategy: BMeanReversionStrategy, *, mid: float, ema_slow: float = 1000.0) -> None:
        bid = int(mid - 3)
        ask = int(mid + 3)
        strategy.on_book_update_at("B", FakeOrderBook(bids={bid: 10}, asks={ask: 10}), now_ms=1_000)
        strategy.ema_fast = ema_slow
        strategy.ema_slow = ema_slow
        strategy.last_sigma = 4.0
        strategy.ewma_var = 16.0
        deviation = mid - ema_slow
        strategy.last_deviation_ticks = deviation
        strategy.last_abs_deviation = abs(deviation)
        strategy.last_deviation_sign = 1 if deviation > 0 else (-1 if deviation < 0 else 0)
        strategy.last_extension_ms = 0
        strategy.last_fast_slope = -1.0 if deviation > 0 else 1.0

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
        self.seed_signal(strategy, mid=1010.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertEqual(plan.ask.strategy_family, "b_mean_reversion")
        self.assertEqual(plan.ask.action_class, "mean_reversion_entry")
        self.assertEqual(strategy.last_target_inventory, -6)

    def test_negative_z_creates_buy_target(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=990.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertEqual(plan.bid.side, "BUY")
        self.assertEqual(plan.bid.action_class, "mean_reversion_entry")
        self.assertEqual(strategy.last_target_inventory, 6)

    def test_fifteen_tick_deviation_creates_full_aggressive_target(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1015.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(strategy.last_target_inventory, -16)
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertTrue(plan.aggressive_actions[0].aggressive)
        self.assertEqual(plan.aggressive_actions[0].side, "SELL")

    def test_full_entry_uses_passive_follow_up_when_already_loaded(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(-2)
        self.seed_signal(strategy, mid=1015.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(strategy.last_target_inventory, -16)
        self.assertEqual(len(plan.aggressive_actions), 0)
        self.assertIsNotNone(plan.ask)
        self.assertFalse(plan.ask.aggressive)
        self.assertEqual(plan.ask.side, "SELL")

    def test_small_z_exits_existing_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(4)
        self.seed_signal(strategy, mid=1001.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertEqual(plan.mode, "B_MEANREV_EXIT")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "SELL")
        self.assertTrue(plan.aggressive_actions[0].aggressive)
        self.assertEqual(plan.aggressive_actions[0].action_class, "mean_reversion_exit")

    def test_exit_z_triggers_aggressive_exit_even_outside_tick_band(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(-4)
        self.seed_signal(strategy, mid=1002.0)
        strategy.last_sigma = 8.0
        strategy.ewma_var = 64.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(plan.mode, "B_MEANREV_EXIT")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertTrue(plan.aggressive_actions[0].aggressive)
        self.assertEqual(plan.aggressive_actions[0].action_class, "mean_reversion_exit")

    def test_low_z_blocks_non_extreme_entry_even_when_ticks_are_large_enough(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1010.0)
        strategy.last_sigma = 16.0
        strategy.ewma_var = 256.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "B mean-reversion waiting for z-score entry")
        self.assertEqual(strategy.last_block_reason, "b_meanrev_waiting_for_z_entry")

    def test_high_tick_low_z_signal_downshifts_to_base_target(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1015.0)
        strategy.last_sigma = 8.0
        strategy.ewma_var = 64.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(strategy.last_target_inventory, -6)
        self.assertEqual(len(plan.aggressive_actions), 0)
        self.assertIsNotNone(plan.ask)
        self.assertFalse(plan.ask.aggressive)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertEqual(plan.ask.action_class, "mean_reversion_entry")

    def test_waits_for_turn_confirmation_for_normal_fade(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1010.0)
        strategy.last_extension_ms = 950
        strategy.last_fast_slope = 1.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "B mean-reversion waiting for turn confirmation")

    def test_extreme_deviation_bypasses_turn_confirmation(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1020.0)
        strategy.last_sigma = 10.0
        strategy.last_extension_ms = 999
        strategy.last_fast_slope = 1.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(strategy.last_target_inventory, -16)

    def test_locked_or_tight_book_blocks_new_entries(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1000: 10}, asks={1001: 10}), now_ms=1_000)
        strategy.ema_slow = 994.0
        strategy.last_sigma = 4.0

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "b_meanrev_book_too_tight")

    def test_bad_book_requests_immediate_live_order_cancel(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1015.0)
        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())
        order = plan.aggressive_actions[0]
        strategy.order_manager.note_submitted(
            order_id="b-sell-1",
            side=order.side,
            px=order.px,
            qty=order.qty,
            now_ms=1_000,
            aggressive=order.aggressive,
            intent=order.intent,
            mode_at_submit=order.mode_at_submit,
            action_class=order.action_class,
        )

        strategy.on_book_update_at("B", FakeOrderBook(bids={1002: 10}, asks={1001: 10}), now_ms=1_100)

        self.assertTrue(strategy.should_force_bad_book_cancel())

    def test_single_intent_actions_wait_for_opposite_cancel(self) -> None:
        strategy = self.make_strategy()
        strategy.order_manager.note_submitted(
            order_id="b-buy-1",
            side="BUY",
            px=998,
            qty=2,
            now_ms=900,
            aggressive=False,
            intent="b_mean_reversion",
            mode_at_submit="B_MEANREV_EXIT",
            action_class="mean_reversion_exit",
        )
        self.seed_signal(strategy, mid=1015.0)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())
        actions = strategy.build_actions(plan, 1_000)

        self.assertTrue(any(cancel.order_id == "b-buy-1" for cancel in actions.cancels))
        self.assertFalse(any(placement.side == "SELL" for placement in actions.placements))
        self.assertEqual(strategy.last_block_reason, "b_meanrev_waiting_for_opposite_cancel")

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
        strategy.sync_inventory_from_exchange(16)
        self.seed_signal(strategy, mid=1036.0)

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

    def test_mean_reference_blends_synthetic_anchor_when_stable(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1010.0, ema_slow=1000.0)

        strategy.compute_quotes(
            now_ms=1_000,
            residual_payload=self.payload(dispersion=3.0, basis=2.0, synthetic_fair=1008.0),
        )

        state = strategy.trace_state(1_000)
        self.assertAlmostEqual(state["b_meanrev_mean_reference"], 1002.8, places=4)
        self.assertAlmostEqual(state["b_meanrev_mean_reference_ema_component"], 650.0, places=4)
        self.assertAlmostEqual(state["b_meanrev_mean_reference_synth_component"], 352.8, places=4)
        self.assertTrue(state["b_meanrev_used_synthetic_reference"])

    def test_mean_reference_ignores_synthetic_anchor_when_basis_too_wide(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=1010.0, ema_slow=1000.0)

        strategy.compute_quotes(
            now_ms=1_000,
            residual_payload=self.payload(dispersion=3.0, basis=7.0, synthetic_fair=1008.0),
        )

        state = strategy.trace_state(1_000)
        self.assertEqual(state["b_meanrev_mean_reference"], 1000.0)
        self.assertEqual(state["b_meanrev_mean_reference_ema_component"], 1000.0)
        self.assertIsNone(state["b_meanrev_mean_reference_synth_component"])
        self.assertFalse(state["b_meanrev_used_synthetic_reference"])

    def test_fill_after_position_update_does_not_double_count_inventory(self) -> None:
        strategy = self.make_strategy()
        self.seed_signal(strategy, mid=990.0)
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
