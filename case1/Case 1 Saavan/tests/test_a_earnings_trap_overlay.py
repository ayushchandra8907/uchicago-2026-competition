from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import AConfig, RiskConfig
from a_earnings_trap_overlay import AEarningsTrapOverlay


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class AEarningsTrapOverlayTests(unittest.TestCase):
    def make_overlay(self) -> AEarningsTrapOverlay:
        return AEarningsTrapOverlay(
            AConfig(),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=250),
            book_depth_levels=5,
        )

    def make_state(self, **overrides: object) -> dict[str, object]:
        state = {
            "mode": "POST_EARNINGS_SHOCK",
            "fair_value": 1000,
            "shock_direction": 1,
            "inventory": 80,
            "shock_reference_mid": 900.0,
            "active_earnings_cycle_id": "a_eps_1",
            "current_earnings_signal_id": "a_eps_1",
        }
        state.update(overrides)
        return state

    def test_overlay_ignores_non_earnings_mode(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state(mode="STEADY_MM"))

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_inactive_non_earnings_mode")

    def test_overlay_waits_for_large_fair_shift(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state(fair_value=1000, shock_reference_mid=960.0))

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_fair_shift_below_threshold")

    def test_overlay_waits_for_aligned_inventory(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state(inventory=20))

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_waiting_for_aligned_inventory")

    def test_overlay_requires_already_stretched_upside_book(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1050: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state())

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_waiting_for_stretched_ask")

    def test_upside_overlay_posts_passive_sell_at_best_ask(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state())

        self.assertFalse(plan.observe_only)
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")
        self.assertEqual(plan.ask.px, 1110)
        self.assertEqual(plan.ask.qty, 20)
        self.assertEqual(plan.ask.strategy_family, "a_earnings_trap")

    def test_downside_overlay_posts_passive_buy_at_best_bid(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={890: 10}, asks={1001: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(
            1_000,
            self.make_state(
                shock_direction=-1,
                inventory=-75,
                fair_value=1000,
                shock_reference_mid=1100.0,
            ),
        )

        self.assertFalse(plan.observe_only)
        self.assertIsNotNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.bid.side, "BUY")
        self.assertEqual(plan.bid.px, 890)
        self.assertEqual(plan.bid.qty, 20)

    def test_overlay_qty_respects_inventory_reserve(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state(inventory=47))

        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.qty, 20 if 47 - 20 > 20 else 27)
        self.assertEqual(plan.ask.qty, 20)

        smaller = overlay.compute_quotes(1_000, self.make_state(inventory=44))
        self.assertIsNotNone(smaller.ask)
        self.assertEqual(smaller.ask.qty, 20)

        tight = overlay.compute_quotes(1_000, self.make_state(inventory=41))
        self.assertIsNotNone(tight.ask)
        self.assertEqual(tight.ask.qty, 20 if 41 - 20 > 20 else 21)
        self.assertEqual(tight.ask.qty, 20)

    def test_overlay_qty_caps_at_inventory_minus_reserve(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)

        plan = overlay.compute_quotes(1_000, self.make_state(inventory=38))

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_waiting_for_aligned_inventory")

        plan = overlay.compute_quotes(1_000, self.make_state(inventory=42))
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.qty, 20 if 42 - 20 > 20 else 22)
        self.assertEqual(plan.ask.qty, 20)

        plan = overlay.compute_quotes(1_000, self.make_state(inventory=55))
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.qty, 20)

    def test_overlay_ttl_forces_cancel(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)
        plan = overlay.compute_quotes(1_000, self.make_state())
        actions = overlay.order_manager.build_actions(plan, 1_000)
        self.assertEqual(len(actions.placements), 1)
        placed = overlay.order_manager.note_submitted(
            order_id="trap-1",
            side="SELL",
            px=1110,
            qty=20,
            now_ms=1_000,
            overlay="earnings",
            aggressive=False,
            intent="a_earnings_trap",
            mode_at_submit="POST_EARNINGS_SHOCK",
            evaluation_reason="earnings trap overlay",
            market_key="A",
            strategy_family="a_earnings_trap",
            action_class="trap_quote",
            pnl_owner="a_earnings_trap",
            signal_id="a_eps_1_trap",
            trade_group_id="a_eps_1_trap",
            leg_role="single",
        )
        overlay.on_order_submitted(placed.signal_id)

        expired = overlay.compute_quotes(2_600, self.make_state())
        cancel_actions = overlay.order_manager.build_actions(expired, 2_600)

        self.assertTrue(expired.observe_only)
        self.assertEqual(expired.reason, "trap_order_expired")
        self.assertEqual(len(cancel_actions.cancels), 1)

    def test_overlay_cancels_when_cycle_changes(self) -> None:
        overlay = self.make_overlay()
        overlay.on_book_update_at("A", FakeOrderBook(bids={999: 10}, asks={1110: 10}), now_ms=1_000)
        overlay.order_manager.note_submitted(
            order_id="trap-1",
            side="SELL",
            px=1110,
            qty=20,
            now_ms=1_000,
            overlay="earnings",
            aggressive=False,
            intent="a_earnings_trap",
            mode_at_submit="POST_EARNINGS_SHOCK",
            evaluation_reason="earnings trap overlay",
            market_key="A",
            strategy_family="a_earnings_trap",
            action_class="trap_quote",
            pnl_owner="a_earnings_trap",
            signal_id="a_eps_1_trap",
            trade_group_id="a_eps_1_trap",
            leg_role="single",
        )

        plan = overlay.compute_quotes(1_100, self.make_state(active_earnings_cycle_id="a_eps_2", current_earnings_signal_id="a_eps_2"))
        cancel_actions = overlay.order_manager.build_actions(plan, 1_100)

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "trap_waiting_for_cycle_roll_cancel")
        self.assertEqual(len(cancel_actions.cancels), 1)
