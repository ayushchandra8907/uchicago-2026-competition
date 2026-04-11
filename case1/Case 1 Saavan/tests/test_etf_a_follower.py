from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import ETFConfig, RiskConfig
from a_bot_strategy import NewsReaction
from etf_a_follower import ETFAFollowerStrategy, ETFShockProjection


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
                entry_retry_window_ms=1_500,
                entry_force_aggressive_ms=250,
                entry_retry_reprice_ms=125,
                churn_window_ms=250,
                churn_max_top_of_book_updates=25,
                churn_resume_stable_ms=500,
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
        self.assertTrue(plan.aggressive_actions[0].aggressive)
        self.assertEqual(plan.aggressive_actions[0].strategy_family, "etf_a_follower")
        self.assertEqual(plan.aggressive_actions[0].action_class, "etf_shock_take")
        self.assertEqual(strategy.trace_state(1_100)["etf_entry_order_attempt_count"], 1)

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
                entry_retry_window_ms=1_500,
                entry_force_aggressive_ms=250,
                entry_retry_reprice_ms=125,
                churn_window_ms=250,
                churn_max_top_of_book_updates=25,
                churn_resume_stable_ms=500,
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
        self.assertEqual(strategy.trace_state(1_121)["etf_first_fill_latency_ms"], 21)

    def test_signal_retries_until_entry_window_expires(self) -> None:
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

        first = strategy.compute_quotes(now_ms=1_100)
        second = strategy.compute_quotes(now_ms=1_500)
        expired = strategy.compute_quotes(now_ms=2_700)

        self.assertEqual(len(first.aggressive_actions), 1)
        self.assertEqual(len(second.aggressive_actions), 1)
        self.assertTrue(second.aggressive_actions[0].aggressive)
        self.assertEqual(expired.reason, "entry_retry_window_expired")
        self.assertIsNone(strategy.active_signal)

    def test_c_projection_creates_c_origin_signal(self) -> None:
        strategy = self.make_strategy(alpha=0.25)

        signal = strategy.on_shock_projection(
            ETFShockProjection(
                source_market="C",
                source_kind="structured_earnings",
                source_signal_id="c_earnings_1",
                fair_shift_ticks=100.0,
                alpha=0.25,
                source_target_inventory=120,
                source_combo="C_only",
                source_direction=1,
            ),
            now_ms=1_100,
        )
        assert signal is not None
        strategy.activate_signal(signal)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.source_market, "C")
        self.assertEqual(signal.source_combo, "C_only")
        self.assertEqual(signal.target_inventory, 50)
        state = strategy.trace_state(1_100)
        self.assertEqual(state["etf_source_market"], "C")
        self.assertEqual(state["etf_source_combo"], "C_only")

    def test_preview_projection_reports_target_inventory(self) -> None:
        strategy = self.make_strategy(alpha=0.25)

        preview = strategy.preview_projection(
            ETFShockProjection(
                source_market="C",
                source_kind="structured_earnings",
                source_signal_id="c_earnings_2",
                fair_shift_ticks=120.0,
                alpha=0.25,
                source_target_inventory=200,
                source_combo="A_C_aligned",
                source_direction=1,
            )
        )

        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview["direction"], 1)
        self.assertEqual(preview["target_inventory"], 50)
        self.assertEqual(preview["target_from_source_position"], 70)

    def test_cancel_response_does_not_clear_unfilled_signal(self) -> None:
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
        plan = strategy.compute_quotes(now_ms=1_100)
        order = plan.aggressive_actions[0]
        managed = strategy.order_manager.note_submitted(
            order_id="etf-entry-1",
            side=order.side,
            px=order.px,
            qty=order.qty,
            now_ms=1_100,
            aggressive=order.aggressive,
            intent=order.intent,
            mode_at_submit=order.mode_at_submit,
            action_class=order.action_class,
        )

        strategy.on_cancel_response(managed.order_id, True)
        retry = strategy.compute_quotes(now_ms=1_250)

        self.assertIsNotNone(strategy.active_signal)
        self.assertEqual(len(retry.aggressive_actions), 1)

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
        self.assertEqual(plan.mode, "ETF_CHURN_GUARD")
        self.assertEqual(plan.reason, "crossed_or_locked_etf_book")
        self.assertEqual(len(plan.aggressive_actions), 0)
        self.assertIsNotNone(strategy.active_signal)

    def test_quote_churn_guard_blocks_entry_without_clearing_signal(self) -> None:
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
        for offset in range(26):
            strategy.on_book_update_at(
                "ETF",
                FakeOrderBook(
                    bids={998 + offset: 10},
                    asks={1002 + offset: 10},
                ),
                now_ms=1_200 + offset,
            )

        plan = strategy.compute_quotes(now_ms=1_230)

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.mode, "ETF_CHURN_GUARD")
        self.assertEqual(plan.reason, "etf_quote_churn_guard")
        self.assertIsNotNone(strategy.active_signal)

    def test_churn_guard_clears_after_stable_book_period(self) -> None:
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
        blocked = strategy.compute_quotes(now_ms=1_200)
        strategy.on_book_update_at("ETF", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_300)
        still_blocked = strategy.compute_quotes(now_ms=1_600)
        resumed = strategy.compute_quotes(now_ms=1_800)

        self.assertEqual(blocked.reason, "crossed_or_locked_etf_book")
        self.assertEqual(still_blocked.reason, "crossed_or_locked_etf_book")
        self.assertEqual(len(resumed.aggressive_actions), 1)
        self.assertTrue(resumed.aggressive_actions[0].aggressive)

    def test_opposite_signal_flattens_before_handoff(self) -> None:
        strategy = self.make_strategy(alpha=0.25)
        first = strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=1.1,
                old_fair_value=1000,
                new_fair_value=1100,
            ),
            now_ms=1_100,
        )
        self.assertIsNotNone(first)
        strategy.sync_inventory_from_exchange(20, now_ms=1_200)

        second = strategy.on_a_news_reaction(
            NewsReaction(
                relevant=True,
                fair_value_updated=True,
                earnings_value=0.9,
                old_fair_value=1100,
                new_fair_value=1000,
            ),
            now_ms=1_300,
        )

        self.assertIsNotNone(second)
        self.assertEqual(strategy.active_signal.signal_id, first.signal_id)
        self.assertEqual(strategy.pending_signal.signal_id, second.signal_id)

        flatten_plan = strategy.compute_quotes(now_ms=1_301, a_state={"mode": "POST_EARNINGS_SHOCK", "shock_direction": 1})
        self.assertEqual(flatten_plan.mode, "ETF_UNWIND")
        self.assertEqual(len(flatten_plan.aggressive_actions), 1)
        self.assertEqual(flatten_plan.aggressive_actions[0].side, "SELL")

        strategy.sync_inventory_from_exchange(0, now_ms=1_400)
        next_plan = strategy.compute_quotes(now_ms=1_401, a_state={"mode": "POST_EARNINGS_SHOCK", "shock_direction": -1})

        self.assertIsNotNone(strategy.active_signal)
        self.assertEqual(strategy.active_signal.signal_id, second.signal_id)
        self.assertIsNone(strategy.pending_signal)
        self.assertEqual(next_plan.mode, "ETF_A_SHOCK")
        self.assertEqual(len(next_plan.aggressive_actions), 1)
        self.assertEqual(next_plan.aggressive_actions[0].side, "SELL")

    def test_retry_window_reports_guard_reason_if_guard_persists(self) -> None:
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

        expired = strategy.compute_quotes(now_ms=2_700)

        self.assertEqual(expired.reason, "crossed_or_locked_etf_book")
        self.assertIsNone(strategy.active_signal)


if __name__ == "__main__":
    unittest.main()
