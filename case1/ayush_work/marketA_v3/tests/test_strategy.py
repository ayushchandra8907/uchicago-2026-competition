from __future__ import annotations

from collections import deque
from statistics import median
import tempfile
import unittest
from pathlib import Path

from case1.ayush_work.marketA_v3.A_strategy import AStrategy
from case1.ayush_work.marketA_v3.config import LoggerConfig, StrategyConfig
from case1.ayush_work.marketA_v3.core.graphs import generate_run_graphs
from case1.ayush_work.marketA_v3.core.logger import RunLogger
from case1.ayush_work.marketA_v3.core.types import BookLevel, BookSnapshot, NewsEvent, StrategySnapshot


def snapshot(
    *,
    now_ms: int,
    inventory: int = 0,
    bid: int = 1000,
    ask: int = 1002,
    fair_value: int | None = None,
    trusted_multiplier: float | None = None,
    latest_earnings: float | None = None,
    mode: str = "IDLE",
):
    return StrategySnapshot(
        now_ms=now_ms,
        exchange_tick=now_ms // 200,
        book=BookSnapshot(best_bid=BookLevel(bid, 10), best_ask=BookLevel(ask, 10)),
        inventory=inventory,
        cash=0,
        fair_value=fair_value,
        trusted_multiplier=trusted_multiplier,
        latest_earnings=latest_earnings,
        mode=mode,
        open_orders=(),
        last_trade_px=None,
        message_index=None,
    )


class StrategyTests(unittest.TestCase):
    def test_first_structured_earnings_bootstraps_and_triggers_shock(self):
        strategy = AStrategy(StrategyConfig(position_cap=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertGreater(strategy.fair_value, 1000)

    def test_shock_target_respects_cap(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, shock_position_scale=5.0, shock_full_confidence_edge_ticks=20))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.50,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(abs(decision.target_inventory), 40)

    def test_default_shock_target_respects_new_default_cap(self):
        strategy = AStrategy(StrategyConfig(shock_position_scale=5.0, shock_full_confidence_edge_ticks=20))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.60,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(abs(decision.target_inventory), 200)

    def test_small_earnings_move_stays_small(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                shock_min_edge_ticks=10,
                shock_full_confidence_edge_ticks=80,
                shock_position_scale=0.3,
                shock_min_position=4,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.02,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertLessEqual(abs(decision.target_inventory), 8)

    def test_small_fair_revision_can_still_take_large_inventory_when_edge_is_large(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                shock_min_edge_ticks=10,
                shock_position_scale=1.2,
                shock_change_position_scale=0.5,
                shock_full_confidence_change_ticks=40,
                shock_min_position=4,
            )
        )
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.12
        strategy.fair_value = 1120
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.11,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertGreater(abs(strategy.fair_value - 1000), 100)
        self.assertEqual(abs(decision.target_inventory), 40)

    def test_more_than_two_earnings_events_are_supported(self):
        strategy = AStrategy(StrategyConfig())
        for idx in range(15):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        for offset, value in enumerate((1.1, 0.9, 1.3), start=1):
            strategy.on_news(
                snapshot(now_ms=4000 * offset, bid=999, ask=1001),
                NewsEvent(
                    now_ms=4000 * offset,
                    tick=offset,
                    kind="structured",
                    symbol="A",
                    structured_subtype="earnings",
                    asset="A",
                    value=value,
                    raw_payload={"kind": "structured"},
                ),
            )
            for sample_idx in range(8):
                strategy.on_book(snapshot(now_ms=(4000 * offset) + 200 * sample_idx, bid=1099, ask=1101))
        self.assertEqual(strategy.structured_event_count, 3)

    def test_unwind_waits_for_equilibrium(self):
        strategy = AStrategy(StrategyConfig())
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=20, bid=1098, ask=1102))
        self.assertEqual(decision.mode, "SHOCK")

    def test_emergency_dump_exits_long_shock_before_equilibrium_on_wrong_way_drop(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=5_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=5_000,
                shock_emergency_dump_min_elapsed_ms=250,
                shock_emergency_dump_ticks=40,
                shock_emergency_dump_fraction=0.20,
                shock_emergency_dump_min_inventory=12,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=40, bid=949, ask=951))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.qty, 40)
        self.assertEqual(decision.target_inventory, 0)

    def test_emergency_dump_exits_short_shock_before_equilibrium_on_wrong_way_rally(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=5_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=5_000,
                shock_emergency_dump_min_elapsed_ms=250,
                shock_emergency_dump_ticks=40,
                shock_emergency_dump_fraction=0.20,
                shock_emergency_dump_min_inventory=12,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=0.80,
                raw_payload={"kind": "structured"},
            ),
        )
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=-40, bid=1049, ask=1051))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(decision.desired_order.qty, 40)
        self.assertEqual(decision.target_inventory, 0)

    def test_large_shock_inventory_decays_before_equilibrium(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=5_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=5_000,
                shock_decay_start_ms=800,
                shock_decay_interval_ms=600,
                shock_decay_fraction=0.10,
                shock_decay_min_qty=4,
                shock_decay_max_qty=8,
                shock_decay_min_inventory=16,
                shock_decay_min_residual_fraction=0.25,
                shock_decay_stall_window_ms=600,
                shock_decay_stall_threshold_ticks=12,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2500, inventory=40, bid=1149, ask=1151))
        strategy.on_book(snapshot(now_ms=2850, inventory=40, bid=1149, ask=1151))
        decision = strategy.on_book(snapshot(now_ms=3200, inventory=40, bid=1149, ask=1151))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.qty, 4)
        self.assertEqual(decision.target_inventory, 36)

    def test_large_shock_inventory_does_not_decay_while_price_is_still_moving(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=5_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=5_000,
                shock_decay_start_ms=800,
                shock_decay_interval_ms=600,
                shock_decay_fraction=0.10,
                shock_decay_min_qty=4,
                shock_decay_max_qty=8,
                shock_decay_min_inventory=16,
                shock_decay_min_residual_fraction=0.25,
                shock_decay_stall_window_ms=800,
                shock_decay_stall_threshold_ticks=12,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2500, inventory=40, bid=1109, ask=1111))
        strategy.on_book(snapshot(now_ms=2850, inventory=40, bid=1129, ask=1131))
        decision = strategy.on_book(snapshot(now_ms=3200, inventory=40, bid=1149, ask=1151))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertTrue(decision.observe_only)
        self.assertEqual(decision.target_inventory, 40)
        self.assertEqual(strategy.shock_decay_trimmed_qty_total, 0)

    def test_default_like_decay_trusts_large_position_for_longer(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=50,
                max_absolute_position=50,
                shock_initial_clip=50,
                equilibrium_hold_ms=6_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=6_000,
                shock_decay_start_ms=3_000,
                shock_decay_interval_ms=1_200,
                shock_decay_fraction=0.08,
                shock_decay_min_qty=4,
                shock_decay_max_qty=6,
                shock_decay_min_inventory=24,
                shock_decay_min_residual_fraction=0.55,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        early = strategy.on_book(snapshot(now_ms=4200, inventory=50, bid=1149, ask=1151))
        self.assertEqual(early.mode, "SHOCK")
        self.assertTrue(early.observe_only)
        self.assertEqual(early.target_inventory, 50)

        later = strategy.on_book(snapshot(now_ms=5400, inventory=50, bid=1149, ask=1151))
        self.assertEqual(later.mode, "SHOCK")
        self.assertIsNotNone(later.desired_order)
        self.assertEqual(later.desired_order.side, "SELL")
        self.assertEqual(later.desired_order.qty, 4)
        self.assertEqual(later.target_inventory, 46)

    def test_shock_target_ratchets_inward_after_inventory_moves_toward_zero(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=6_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=6_000,
                shock_decay_start_ms=10_000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=0.80,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(strategy.shock_target_inventory, -40)
        strategy.on_book(snapshot(now_ms=2400, inventory=-40, bid=979, ask=981))
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=-20, bid=979, ask=981))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertTrue(decision.observe_only)
        self.assertEqual(decision.target_inventory, -20)
        self.assertEqual(strategy.shock_target_inventory, -20)

    def test_partial_fill_buildup_does_not_ratchet_target_inward(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=150,
                max_absolute_position=150,
                shock_initial_clip=150,
                equilibrium_hold_ms=6_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=6_000,
                shock_decay_start_ms=10_000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.60,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(strategy.shock_target_inventory, 150)
        decision = strategy.on_book(snapshot(now_ms=2400, inventory=12, bid=1078, ask=1080))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertFalse(decision.observe_only)
        self.assertEqual(decision.target_inventory, 150)
        self.assertEqual(strategy.shock_target_inventory, 150)

    def test_large_default_decay_no_longer_pins_halfway_to_zero(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=85,
                max_absolute_position=85,
                shock_initial_clip=85,
                equilibrium_hold_ms=20_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=20_000,
                shock_decay_start_ms=3_000,
                shock_decay_interval_ms=1_200,
                shock_decay_fraction=0.08,
                shock_decay_min_qty=4,
                shock_decay_max_qty=6,
                shock_decay_min_inventory=24,
                shock_decay_min_residual_fraction=0.10,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.60,
                raw_payload={"kind": "structured"},
            ),
        )
        for t, inv in ((5400, 85), (6600, 79), (7800, 73), (9000, 67), (10200, 61), (11400, 55), (12600, 49)):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=inv, bid=1199, ask=1201))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertLessEqual(decision.target_inventory, 43)

    def test_small_shock_inventory_does_not_decay(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                equilibrium_hold_ms=5_000,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=5_000,
                shock_position_scale=0.30,
                shock_min_position=4,
                shock_decay_start_ms=800,
                shock_decay_interval_ms=600,
                shock_decay_min_inventory=16,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.02,
                raw_payload={"kind": "structured"},
            ),
        )
        target = strategy.shock_target_inventory
        decision = strategy.on_book(snapshot(now_ms=3200, inventory=target, bid=1017, ask=1019))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertTrue(decision.observe_only)
        self.assertEqual(decision.target_inventory, target)
        self.assertEqual(strategy.shock_decay_trimmed_qty_total, 0)

    def test_equilibrium_unwind_beats_decay(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
                shock_decay_start_ms=200,
                shock_decay_interval_ms=200,
                shock_decay_fraction=0.10,
                shock_decay_min_qty=4,
                flatten_deadline_ms=2_000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        for t in (2600, 2800, 3000, 3200):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=40, bid=1189, ask=1191))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.qty, 40)

    def test_unwind_targets_zero_after_equilibrium(self):
        strategy = AStrategy(
            StrategyConfig(
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
                flatten_deadline_ms=2000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        for t in (2600, 2800, 3000, 3200):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=20, bid=1189, ask=1191))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertEqual(decision.target_inventory, 0)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.qty, 20)

    def test_negative_unwind_buys_back_immediately_after_equilibrium(self):
        strategy = AStrategy(
            StrategyConfig(
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
                flatten_deadline_ms=2000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=0.80,
                raw_payload={"kind": "structured"},
            ),
        )
        for t in (2600, 2800, 3000, 3200):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=-20, bid=809, ask=811))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertEqual(decision.target_inventory, 0)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(decision.desired_order.qty, 20)

    def test_equilibrium_still_holds_if_stable_but_far_from_fair(self):
        strategy = AStrategy(
            StrategyConfig(
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        for t in (2600, 2800, 3000, 3200):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=20, bid=1089, ask=1091))
        self.assertEqual(decision.mode, "SHOCK")

    def test_equilibrium_requires_real_elapsed_time(self):
        strategy = AStrategy(
            StrategyConfig(
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        for t in (2250, 2300, 2350, 2400):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=20, bid=1099, ask=1101))
        self.assertEqual(decision.mode, "SHOCK")

    def test_overshoot_does_not_trim_before_crossing_fair(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, max_absolute_position=40, shock_initial_clip=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=40, bid=1189, ask=1191))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNone(decision.desired_order)

    def test_crossing_fair_alone_does_not_trigger_overshoot_trim(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, max_absolute_position=40, shock_initial_clip=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        for t, bid, ask in ((2400, 1241, 1243), (2500, 1242, 1244), (2600, 1243, 1245)):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=40, bid=bid, ask=ask))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNone(decision.desired_order)

    def test_crossed_fair_waits_briefly_for_overshoot_before_unwind(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                overshoot_hold_ms=350,
                overshoot_max_wait_ms=900,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        decision = strategy.on_book(snapshot(now_ms=2800, inventory=40, bid=1199, ask=1201))
        self.assertEqual(decision.mode, "SHOCK")

    def test_crossed_fair_without_overshoot_unwinds_after_short_timeout(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                overshoot_hold_ms=350,
                overshoot_max_wait_ms=900,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        # Cross fair, remain stable, but never reach the overshoot threshold.
        for t, bid, ask in (
            (2600, 1199, 1201),
            (3200, 1200, 1202),
            (3400, 1198, 1200),
            (3600, 1199, 1201),
            (3800, 1200, 1202),
        ):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=40, bid=bid, ask=ask))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertEqual(decision.target_inventory, 0)

    def test_long_overshoot_trim_sells_partial_basket(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, max_absolute_position=40, shock_initial_clip=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2400, inventory=40, bid=1245, ask=1247))
        strategy.on_book(snapshot(now_ms=2500, inventory=40, bid=1244, ask=1246))
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=40, bid=1241, ask=1243))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.intent, "overshoot_trim")
        self.assertLessEqual(decision.desired_order.qty, 12)
        self.assertEqual(decision.target_inventory, 28)

    def test_short_overshoot_trim_buys_partial_basket(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, max_absolute_position=40, shock_initial_clip=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=0.80,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2400, inventory=-40, bid=755, ask=757))
        strategy.on_book(snapshot(now_ms=2500, inventory=-40, bid=756, ask=758))
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=-40, bid=759, ask=761))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(decision.desired_order.intent, "overshoot_trim")
        self.assertLessEqual(decision.desired_order.qty, 12)
        self.assertEqual(decision.target_inventory, -28)

    def test_overshoot_trims_preserve_residual_runner_floor(self):
        strategy = AStrategy(StrategyConfig(position_cap=40, max_absolute_position=40, shock_initial_clip=40))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        windows = [
            ((2400, 1245, 1247), (2500, 1244, 1246), (2600, 1241, 1243), 40),
            ((3000, 1275, 1277), (3100, 1274, 1276), (3200, 1271, 1273), 32),
            ((3600, 1305, 1307), (3700, 1304, 1306), (3800, 1301, 1303), 24),
        ]
        for samples in windows:
            inventory = samples[-1]
            for t, bid, ask in samples[:-1]:
                decision = strategy.on_book(snapshot(now_ms=t, inventory=inventory, bid=bid, ask=ask))
        self.assertEqual(strategy.shock_target_inventory, 12)
        self.assertEqual(strategy.overshoot_stage_index, 3)
        self.assertLessEqual(decision.desired_order.qty, 12)

    def test_overshoot_trim_precedes_unwind_then_regular_unwind_still_works(self):
        strategy = AStrategy(
            StrategyConfig(
                position_cap=40,
                max_absolute_position=40,
                shock_initial_clip=40,
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
                flatten_deadline_ms=2000,
            )
        )
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2400, inventory=40, bid=1245, ask=1247))
        strategy.on_book(snapshot(now_ms=2500, inventory=40, bid=1244, ask=1246))
        overshoot = strategy.on_book(snapshot(now_ms=2600, inventory=40, bid=1241, ask=1243))
        self.assertEqual(overshoot.desired_order.side, "SELL")
        for t in (3200, 3400, 3600, 3800):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=32, bid=1189, ask=1191))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertEqual(decision.target_inventory, 0)

    def test_large_basket_overshoot_can_trim_half_and_keep_large_runner(self):
        strategy = AStrategy(StrategyConfig(position_cap=150, max_absolute_position=150, shock_initial_clip=150))
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.60,
                raw_payload={"kind": "structured"},
            ),
        )
        strategy.on_book(snapshot(now_ms=2400, inventory=150, bid=1725, ask=1727))
        strategy.on_book(snapshot(now_ms=2500, inventory=150, bid=1724, ask=1726))
        decision = strategy.on_book(snapshot(now_ms=2600, inventory=150, bid=1721, ask=1723))
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.intent, "overshoot_trim")
        self.assertEqual(decision.target_inventory, 75)

    def test_clean_multiplier_samples_use_median_and_stop_at_limit(self):
        strategy = AStrategy(StrategyConfig(multiplier_clean_sample_limit=5))
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        samples = [980.0, 1020.0, 1100.0, 970.0, 1010.0, 900.0]
        for sample in samples:
            strategy.post_event_mids = deque([(0, sample), (1, sample), (2, sample)])
            strategy.current_event_contaminated = False
            strategy._update_multiplier_from_equilibrium()
        self.assertEqual(len(strategy.clean_multiplier_samples), 5)
        self.assertEqual(strategy.trusted_multiplier, median(strategy.clean_multiplier_samples))
        self.assertNotIn(900.0, strategy.clean_multiplier_samples)

    def test_a_unstructured_news_freezes_multiplier_updates(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.post_event_mids = deque([(0, 1030.0), (1, 1030.0), (2, 1030.0)])
        strategy.on_news(
            snapshot(now_ms=1000),
            NewsEvent(
                now_ms=1000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A product issue",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        strategy._update_multiplier_from_equilibrium()
        self.assertTrue(strategy.pe_frozen)
        self.assertEqual(strategy.clean_multiplier_samples, [])
        self.assertEqual(strategy.trusted_multiplier, 1000.0)

    def test_frozen_multiplier_still_trades_later_structured_earnings(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.pe_frozen = True
        for idx in range(10):
            strategy.on_book(snapshot(now_ms=idx * 200, bid=999, ask=1001))
        decision = strategy.on_news(
            snapshot(now_ms=2200, bid=999, ask=1001),
            NewsEvent(
                now_ms=2200,
                tick=11,
                kind="structured",
                symbol="A",
                structured_subtype="earnings",
                asset="A",
                value=1.20,
                raw_payload={"kind": "structured"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertEqual(strategy.fair_value, 1200)

    def test_non_structured_news_is_ignored(self):
        strategy = AStrategy(StrategyConfig())
        decision = strategy.on_news(
            snapshot(now_ms=1000),
            NewsEvent(
                now_ms=1000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A rumor",
                raw_payload={"kind": "unstructured"},
            ),
        )
        self.assertTrue(decision.observe_only)
        self.assertEqual(decision.mode, "IDLE")

    def test_market_making_stub_emits_no_orders(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.fair_value = 1000
        strategy.mode = "MARKET_MAKING_STUB"
        decision = strategy.on_book(snapshot(now_ms=1000, bid=999, ask=1001, fair_value=1000, trusted_multiplier=1000.0, latest_earnings=1.0, mode="MARKET_MAKING_STUB"))
        self.assertIsNone(decision.desired_order)
        self.assertTrue(decision.observe_only)


class LoggerAndGraphsTests(unittest.TestCase):
    def test_logger_and_graph_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(LoggerConfig(enabled=True, run_root=Path(temp_dir), queue_max_events=32))
            logger.record_event(
                "book_update",
                now_ms=1_000,
                exchange_tick=5,
                best_bid_px=999,
                best_bid_qty=10,
                best_ask_px=1001,
                best_ask_qty=8,
                mid=1000.0,
                spread=2,
                inventory=0,
                cash=0,
                mtm_pnl_estimate=0.0,
                shock_pnl=0.0,
                mm_pnl=0.0,
            )
            logger.record_event(
                "news_received",
                now_ms=1_200,
                exchange_tick=6,
                news_kind="structured",
                raw_payload={
                    "kind": "structured",
                    "symbol": "A",
                    "new_data": {"structured_subtype": "earnings", "asset": "A", "value": 1.2},
                },
                mtm_pnl_estimate=0.0,
                shock_pnl=0.0,
                mm_pnl=0.0,
            )
            logger.record_decision_snapshot(
                now_ms=1_300,
                exchange_tick=6,
                message_index=1,
                row={
                    "mode": "SHOCK",
                    "trigger": "news",
                    "reason": "test",
                    "observe_only": False,
                    "inventory": 0,
                    "target_inventory": 20,
                    "best_bid_px": 999,
                    "best_bid_qty": 10,
                    "best_ask_px": 1001,
                    "best_ask_qty": 8,
                    "mid": 1000.0,
                    "spread": 2,
                    "fair_value": 1200,
                    "trusted_multiplier": 1000.0,
                    "latest_earnings": 1.2,
                    "equilibrium_reached": False,
                    "desired_side": "BUY",
                    "desired_px": 1001,
                    "desired_qty": 20,
                    "cash": 0,
                    "mtm_pnl_estimate": 0.0,
                    "shock_pnl": 0.0,
                    "mm_pnl": 0.0,
                },
            )
            logger.record_event(
                "fill_state",
                now_ms=1_500,
                exchange_tick=7,
                mid=1080.0,
                inventory=20,
                cash=-20_020,
                mtm_pnl_estimate=1_580.0,
                shock_pnl=1_580.0,
                mm_pnl=0.0,
            )
            logger.close()
            generated = generate_run_graphs(logger.run_dir)
            self.assertTrue(generated)
            self.assertTrue((logger.run_dir / "graphs" / "pnl_A.png").exists())


if __name__ == "__main__":
    unittest.main()
