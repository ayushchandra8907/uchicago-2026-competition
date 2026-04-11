from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import BConfig, RiskConfig
from b_underlying_mm_v2 import BUnderlyingMMv2


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class BUnderlyingMMv2Tests(unittest.TestCase):
    def make_strategy(self) -> BUnderlyingMMv2:
        return BUnderlyingMMv2(
            BConfig(
                quote_size=1,
                max_position=8,
                base_half_spread_ticks=2,
                inventory_skew_ticks_per_unit=0.5,
                passive_reduce_start=4,
                passive_reduce_full=8,
                max_synthetic_dispersion=4,
                mm_v2_book_weight=0.7,
                mm_v2_synth_weight=0.3,
                mm_v2_min_half_spread_ticks=1,
                mm_v2_inside_improve_ticks=1,
                mm_v2_dispersion_widen_factor=0.5,
                mm_v2_reduce_size_bonus=1,
                mm_min_healthy_book_age_ms=0,
                mm_min_valid_spread_ticks=3,
                mm_bad_fill_cooldown_ms=750,
            ),
            RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
            book_depth_levels=5,
        )

    @staticmethod
    def payload(*, synthetic_fair: float = 1100.0, dispersion: float = 0.0) -> dict:
        return {
            "composite_synthetic_fair": synthetic_fair,
            "synthetic_dispersion": dispersion,
            "composite_basis": synthetic_fair - 1100.0,
            "synthetic_forward_by_strike": {"950": synthetic_fair, "1000": synthetic_fair, "1050": synthetic_fair},
        }

    def test_quotes_even_when_book_spread_is_tight(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertFalse(plan.observe_only)
        self.assertEqual(plan.mode, "B_UNDERLYING_MM_V2")
        self.assertIsNotNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertLess(plan.bid.px, plan.ask.px)

    def test_locked_or_one_tick_book_does_not_open_new_risk_when_flat(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1100: 10}, asks={1101: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertIsNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.reason, "b_book_too_tight")

    def test_one_tick_book_cancels_without_reduce_only(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(5)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1100: 10}, asks={1101: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.mode, "REDUCE_ONLY")
        self.assertIsNone(plan.bid)
        self.assertIsNone(plan.ask)

    def test_crossed_book_waits_even_when_inventory_is_nonzero(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(-5)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1102: 10}, asks={1100: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "b_book_crossed_or_locked")

    def test_healthy_book_age_delay_blocks_fresh_quotes(self) -> None:
        strategy = BUnderlyingMMv2(
            BConfig(mm_min_healthy_book_age_ms=250),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=2, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)

        early = strategy.compute_quotes(now_ms=1_100, residual_payload=self.payload())
        late = strategy.compute_quotes(now_ms=1_300, residual_payload=self.payload())

        self.assertTrue(early.observe_only)
        self.assertEqual(early.reason, "b_book_waiting_for_healthy_age")
        self.assertFalse(late.observe_only)

    def test_clamps_quotes_to_not_cross_or_over_improve(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertIsNotNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertLessEqual(plan.bid.px, 1091)
        self.assertGreaterEqual(plan.ask.px, 1109)
        self.assertLess(plan.bid.px, plan.ask.px)

    def test_inventory_skew_moves_quotes_away_from_long_inventory(self) -> None:
        flat = self.make_strategy()
        flat.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)
        flat_plan = flat.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        long = self.make_strategy()
        long.sync_inventory_from_exchange(3)
        long.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)
        long_plan = long.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertIsNotNone(flat_plan.bid)
        self.assertIsNotNone(long_plan.bid)
        self.assertLessEqual(long_plan.bid.px, flat_plan.bid.px)

    def test_reduce_only_stops_inventory_increasing_side(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(4)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=self.payload())

        self.assertEqual(plan.mode, "REDUCE_ONLY")
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.action_class, "reduce_only")

    def test_fill_after_position_update_does_not_double_count_inventory(self) -> None:
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_000)
        strategy.order_manager.note_submitted(
            order_id="b-buy-1",
            side="BUY",
            px=1099,
            qty=1,
            now_ms=1_000,
            aggressive=False,
            intent="b_underlying_mm_v2_passive",
            mode_at_submit="B_UNDERLYING_MM_V2",
            action_class="market_making",
        )
        strategy.sync_inventory_from_exchange(1)

        strategy.on_fill("b-buy-1", 1, 1099, authoritative_inventory=1)

        self.assertEqual(strategy.inventory, 1)

    def test_recent_bad_book_fill_cooldown_blocks_quotes(self) -> None:
        strategy = self.make_strategy()
        strategy.last_bad_book_fill_ms = 1_000
        strategy.on_book_update_at("B", FakeOrderBook(bids={1098: 10}, asks={1102: 10}), now_ms=1_100)

        plan = strategy.compute_quotes(now_ms=1_100, residual_payload=self.payload())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "recent_bad_book_fill_cooldown")

    def test_wide_synthetic_dispersion_widens_quotes_but_does_not_block(self) -> None:
        normal = self.make_strategy()
        normal.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)
        normal.compute_quotes(now_ms=1_000, residual_payload=self.payload(dispersion=0.0))

        wide = self.make_strategy()
        wide.on_book_update_at("B", FakeOrderBook(bids={1090: 10}, asks={1110: 10}), now_ms=1_000)
        plan = wide.compute_quotes(now_ms=1_000, residual_payload=self.payload(dispersion=10.0))

        self.assertFalse(plan.observe_only)
        self.assertGreater(wide.last_fair.dynamic_half_spread, normal.last_fair.dynamic_half_spread)


if __name__ == "__main__":
    unittest.main()
