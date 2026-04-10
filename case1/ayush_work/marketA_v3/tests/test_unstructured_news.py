from __future__ import annotations

import unittest

from case1.ayush_work.marketA_v3.A_strategy import AStrategy
from case1.ayush_work.marketA_v3.analyze_run import build_unknown_news_term_report
from case1.ayush_work.marketA_v3.config import StrategyConfig
from case1.ayush_work.marketA_v3.core.a_news_sentiment import score_a_unstructured_headline
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
) -> StrategySnapshot:
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


class UnstructuredNewsTests(unittest.TestCase):
    def test_positive_a_headline_trades_immediately(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A's mobile services division exceeds subscriber growth forecasts.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(strategy.active_signal_kind, "unstructured")
        self.assertGreater(strategy.fair_value, 1000)

    def test_negative_a_headline_trades_immediately(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="Significant insider selling raises concerns among A investors.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertLess(strategy.fair_value, 1000)

    def test_negative_a_news_can_trade_before_first_structured_earnings_using_market_baseline(self):
        strategy = AStrategy(StrategyConfig())

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=1126, ask=1140),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="Investors are warned by A about growing insolvency risks.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(strategy.active_signal_kind, "unstructured")
        self.assertIsNotNone(strategy.base_fair_value)
        self.assertLess(strategy.fair_value or 10**9, strategy.base_fair_value or 0)

    def test_neutral_or_unknown_a_headline_does_not_trade(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A rumor about weekend scheduling changes.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertTrue(decision.observe_only)
        self.assertEqual(decision.mode, "IDLE")
        self.assertEqual(strategy.active_signal_kind, None)

    def test_non_a_tagged_unstructured_is_ignored_for_trading(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol=None,
                content="A's mobile services division exceeds subscriber growth forecasts.",
                raw_payload={"kind": "unstructured"},
            ),
        )
        self.assertTrue(decision.observe_only)
        self.assertEqual(strategy.pending_unstructured_news, None)

    def test_medium_a_headline_requires_confirmation(self):
        strategy = AStrategy(StrategyConfig(news_confirmation_timeout_ms=1_200, news_confirmation_move_ticks=4))
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        initial = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A expands margins.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertTrue(initial.observe_only)
        confirmed = strategy.on_book(snapshot(now_ms=1_200, bid=1005, ask=1007))
        self.assertEqual(confirmed.mode, "SHOCK")
        self.assertEqual(confirmed.desired_order.side, "BUY")

    def test_light_a_headline_trades_small_after_confirmation(self):
        strategy = AStrategy(StrategyConfig(news_confirmation_timeout_ms=1_200, news_confirmation_move_ticks=4))
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        initial = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A launches innovative service.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertTrue(initial.observe_only)
        self.assertEqual(strategy.news_sentiment_bucket, "light")

        confirmed = strategy.on_book(snapshot(now_ms=1_200, bid=1005, ask=1007))
        self.assertEqual(confirmed.mode, "SHOCK")
        self.assertIsNotNone(confirmed.desired_order)
        self.assertEqual(confirmed.desired_order.side, "BUY")
        self.assertGreater(abs(confirmed.target_inventory), 3)
        self.assertLessEqual(abs(confirmed.target_inventory), 8)

    def test_repeated_earnings_beat_headline_takes_near_max_long(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1008.0
        strategy.latest_earnings = 0.8999727459564282
        strategy.base_fair_value = 907
        strategy.fair_value = 907

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=871, ask=887),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A surpasses earnings expectations for fifth consecutive quarter.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(abs(decision.target_inventory), 200)

    def test_news_zero_position_threshold_is_the_new_effective_neutral(self):
        strategy = AStrategy(
            StrategyConfig(
                news_light_position=3,
                news_zero_position_threshold=3,
                news_confirmation_timeout_ms=1_200,
                news_confirmation_move_ticks=4,
            )
        )
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A launches innovative service.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertTrue(decision.observe_only)
        self.assertEqual(strategy.pending_unstructured_news, None)

    def test_positive_news_keeps_positive_direction_even_if_mid_is_above_news_fair(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1142.0
        strategy.latest_earnings = 1.0455992170779476
        strategy.base_fair_value = 1194
        strategy.fair_value = 1194

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=1231, ask=1245),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="Amid strong competition, A posts impressive customer retention figures.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        if decision.observe_only:
            self.assertGreater(strategy.pending_news_target_inventory or 0, 0)
        else:
            self.assertEqual(decision.mode, "SHOCK")
            self.assertIsNotNone(decision.desired_order)
            self.assertEqual(decision.desired_order.side, "BUY")

    def test_failed_medium_confirmation_aborts_trade(self):
        strategy = AStrategy(StrategyConfig(news_confirmation_timeout_ms=800, news_confirmation_move_ticks=4))
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A expands margins.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        decision = strategy.on_timer(snapshot(now_ms=1_900, bid=1000, ask=1002))
        self.assertTrue(decision.observe_only)
        self.assertEqual(strategy.pending_unstructured_news, None)

    def test_a_news_arriving_while_loaded_flattens_first_then_takes_over(self):
        strategy = AStrategy(StrategyConfig(news_takeover_near_flat_threshold=4, news_takeover_flatten_ms=1_200))
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1200
        strategy.mode = "SHOCK"
        strategy.active_signal_kind = "structured"
        strategy.shock_target_inventory = 20
        strategy.original_shock_target_inventory = 20
        strategy.shock_direction = 1

        flatten = strategy.on_news(
            snapshot(now_ms=1_000, inventory=20, bid=1199, ask=1201),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="Significant insider selling raises concerns among A investors.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(flatten.mode, "UNWIND")
        self.assertEqual(flatten.desired_order.side, "SELL")
        self.assertEqual(flatten.desired_order.qty, 20)

        still_flattening = strategy.on_book(snapshot(now_ms=1_200, inventory=4, bid=999, ask=1001))
        self.assertEqual(still_flattening.mode, "UNWIND")
        self.assertIsNotNone(still_flattening.desired_order)
        self.assertEqual(still_flattening.desired_order.side, "SELL")
        self.assertEqual(still_flattening.desired_order.qty, 4)

        takeover = strategy.on_book(snapshot(now_ms=1_300, inventory=0, bid=999, ask=1001))
        self.assertEqual(takeover.mode, "SHOCK")
        self.assertEqual(takeover.desired_order.side, "SELL")
        self.assertEqual(strategy.active_signal_kind, "unstructured")

    def test_latest_pending_news_replaces_older_one(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1200
        strategy.mode = "SHOCK"
        strategy.active_signal_kind = "structured"
        strategy.shock_target_inventory = 20
        strategy.original_shock_target_inventory = 20
        strategy.shock_direction = 1

        strategy.on_news(
            snapshot(now_ms=1_000, inventory=20, bid=1199, ask=1201),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A improves retention among subscribers.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        strategy.on_news(
            snapshot(now_ms=1_100, inventory=20, bid=1198, ask=1200),
            NewsEvent(
                now_ms=1_100,
                tick=6,
                kind="unstructured",
                symbol="A",
                content="Significant insider selling raises concerns among A investors.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertTrue(
            any(term in strategy.pending_news_matched_phrases for term in ("insider selling", "insider", "selling"))
        )
        self.assertLess(strategy.pending_news_target_inventory or 0, 0)

    def test_no_base_fair_uses_live_market_baseline(self):
        strategy = AStrategy(StrategyConfig())
        decision = strategy.on_news(
            snapshot(now_ms=1_000),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A's mobile services division exceeds subscriber growth forecasts.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")
        self.assertEqual(strategy.active_signal_kind, "unstructured")
        self.assertIsNotNone(strategy.base_fair_value)

    def test_news_equilibrium_uses_structured_timing_after_initial_window(self):
        strategy = AStrategy(
            StrategyConfig(
                equilibrium_hold_ms=600,
                equilibrium_min_samples=3,
                equilibrium_min_elapsed_ms=600,
                news_equilibrium_residual_edge_ticks=40,
                news_equilibrium_min_capture_fraction=0.55,
                equilibrium_residual_edge_ticks=20,
                equilibrium_min_capture_fraction=0.70,
                flatten_deadline_ms=2_000,
            )
        )
        initial = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A new partnership bolsters A’s international presence.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(initial.mode, "SHOCK")
        inventory = abs(initial.target_inventory)
        decision = initial
        for t in (1_300, 1_500, 1_700, 1_900):
            decision = strategy.on_book(snapshot(now_ms=t, inventory=inventory, bid=1089, ask=1091))
        self.assertEqual(decision.mode, "UNWIND")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_unstructured_news_overshoot_trim_is_less_strict_than_earnings(self):
        strategy = AStrategy(
            StrategyConfig(
                news_overshoot_hold_ms=250,
                news_overshoot_band_ticks=8,
                news_overshoot_reversal_ticks=3,
            )
        )
        initial = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A new partnership bolsters A’s international presence.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(initial.mode, "SHOCK")
        inventory = abs(initial.target_inventory)

        strategy.on_book(snapshot(now_ms=1_250, inventory=inventory, bid=1139, ask=1141))
        strategy.on_book(snapshot(now_ms=1_350, inventory=inventory, bid=1138, ask=1140))
        decision = strategy.on_book(snapshot(now_ms=1_450, inventory=inventory, bid=1136, ask=1138))

        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.intent, "overshoot_trim")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_unknown_terms_are_captured(self):
        result = score_a_unstructured_headline("A wins kudos for resilient logistics execution.")
        self.assertEqual(result.bucket, "none")
        self.assertIn("kudos", result.unknown_candidate_phrases)
        self.assertIn("resilient", result.unknown_candidate_phrases)

    def test_observed_negative_live_headline_is_no_longer_neutral(self):
        result = score_a_unstructured_headline("A damning investigative report levels serious allegations at A.")
        self.assertLess(result.score, -2.75)
        self.assertNotEqual(result.bucket, "none")
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_observed_positive_live_headline_is_no_longer_neutral(self):
        result = score_a_unstructured_headline("A’s major product clears rigorous industry testing with success.")
        self.assertGreater(result.score, 2.75)
        self.assertNotEqual(result.bucket, "none")
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_additional_positive_headlines_are_no_longer_neutral(self):
        for headline in (
            "Promising commercial applications result from A’s successful R&D efforts.",
            "A experiences unprecedented holiday season sales surge.",
            "A named strategic supplier to major international corporation.",
        ):
            result = score_a_unstructured_headline(headline)
            self.assertGreater(result.score, 1.0, headline)
            self.assertEqual(result.direction, 1, headline)

    def test_additional_negative_headlines_are_no_longer_neutral(self):
        for headline in (
            "A struggles with significant drop in new subscriptions.",
            "Misleading marketing claims cause reputational damage to A.",
        ):
            result = score_a_unstructured_headline(headline)
            self.assertLess(result.score, -1.0, headline)
            self.assertEqual(result.direction, -1, headline)

    def test_recent_missed_positive_headline_is_no_longer_neutral(self):
        result = score_a_unstructured_headline("A selected for high-profile federal technology initiative.")
        self.assertGreater(result.score, 1.0)
        self.assertEqual(result.direction, 1)

    def test_recent_missed_negative_headlines_are_no_longer_neutral(self):
        for headline in (
            "Operational inefficiencies remain unaddressed at A.",
            "Regulatory bans threaten the future of A’s flagship product.",
        ):
            result = score_a_unstructured_headline(headline)
            self.assertLess(result.score, -1.0, headline)
            self.assertEqual(result.direction, -1, headline)

    def test_growth_strategy_doubts_headline_is_negative(self):
        result = score_a_unstructured_headline("Analysts express doubts over A's long-term growth strategy.")
        self.assertLess(result.score, -2.5)
        self.assertEqual(result.direction, -1)

    def test_suffers_wording_is_negative(self):
        result = score_a_unstructured_headline("A suffers renewed demand pressure in key market.")
        self.assertLess(result.score, 0.0)
        self.assertEqual(result.direction, -1)

    def test_setbacks_in_overseas_markets_headline_is_now_immediate_negative(self):
        result = score_a_unstructured_headline("A suffers setbacks in critical overseas markets.")
        self.assertLess(result.score, -3.0)
        self.assertEqual(result.direction, -1)
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_investor_sentiment_decline_headline_is_immediate_negative(self):
        result = score_a_unstructured_headline("Investor sentiment sours, sending A’s stock into decline.")
        self.assertLess(result.score, -3.0)
        self.assertEqual(result.direction, -1)
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_leading_position_niche_market_headline_is_immediate_positive(self):
        result = score_a_unstructured_headline("A takes a leading position in a growing niche market.")
        self.assertGreater(result.score, 3.0)
        self.assertEqual(result.direction, 1)
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_contract_slips_to_rival_headline_is_immediate_negative(self):
        result = score_a_unstructured_headline("A vital contract slips through A's fingers, won by a rival firm instead.")
        self.assertLess(result.score, -3.0)
        self.assertEqual(result.direction, -1)
        self.assertIn(result.bucket, {"strong", "extreme"})

    def test_high_value_strategic_alliance_headline_is_immediate_positive(self):
        result = score_a_unstructured_headline("A enters into high-value strategic alliance with industry leaders.")
        self.assertGreater(result.score, 4.0)
        self.assertEqual(result.direction, 1)
        self.assertEqual(result.bucket, "extreme")

    def test_setbacks_headline_trades_immediately(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A suffers setbacks in critical overseas markets.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_leading_position_niche_market_headline_trades_immediately(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A takes a leading position in a growing niche market.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "BUY")

    def test_contract_slips_to_rival_headline_trades_immediately(self):
        strategy = AStrategy(StrategyConfig())
        strategy.trusted_multiplier = 1000.0
        strategy.latest_earnings = 1.0
        strategy.base_fair_value = 1000
        strategy.fair_value = 1000

        decision = strategy.on_news(
            snapshot(now_ms=1_000, bid=999, ask=1001),
            NewsEvent(
                now_ms=1_000,
                tick=5,
                kind="unstructured",
                symbol="A",
                content="A vital contract slips through A's fingers, won by a rival firm instead.",
                raw_payload={"kind": "unstructured", "symbol": "A"},
            ),
        )
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_loses_out_on_exclusive_partnership_is_negative(self):
        result = score_a_unstructured_headline("A loses out on exclusive partnership.")
        self.assertLess(result.score, -2.0)
        self.assertEqual(result.direction, -1)

    def test_partnership_headline_is_positive(self):
        result = score_a_unstructured_headline("A new partnership bolsters A’s international presence.")
        self.assertGreater(result.score, 2.5)
        self.assertEqual(result.direction, 1)

    def test_falling_demand_headline_is_negative(self):
        result = score_a_unstructured_headline("A warns of falling demand amid evolving consumer behavior.")
        self.assertLess(result.score, -2.5)
        self.assertEqual(result.direction, -1)

    def test_supply_chain_disruption_headline_is_negative(self):
        result = score_a_unstructured_headline("Quarterly earnings take a hit due to A's ongoing supply-chain disruptions.")
        self.assertLess(result.score, -2.5)
        self.assertEqual(result.direction, -1)

    def test_sustainability_award_headline_is_positive(self):
        result = score_a_unstructured_headline("For its outstanding contributions to sustainability, A earns a prestigious award.")
        self.assertGreater(result.score, 2.5)
        self.assertEqual(result.direction, 1)

    def test_consumer_satisfaction_all_time_high_is_positive(self):
        result = score_a_unstructured_headline(
            "Consumer satisfaction ratings hit an all-time high, according to the latest figures from A."
        )
        self.assertGreater(result.score, 2.0)
        self.assertEqual(result.direction, 1)

    def test_delayed_stalling_progress_headline_is_negative(self):
        result = score_a_unstructured_headline(
            "Deployment of a key strategic technology is delayed, stalling progress at A."
        )
        self.assertLess(result.score, -2.0)
        self.assertEqual(result.direction, -1)

    def test_profit_margins_shrink_due_to_rising_costs_is_negative(self):
        result = score_a_unstructured_headline("A's profit margins shrink significantly due to rising costs.")
        self.assertLess(result.score, -2.0)
        self.assertEqual(result.direction, -1)

    def test_environmental_violations_and_fines_is_negative(self):
        result = score_a_unstructured_headline("A accused of environmental violations, potential fines looming.")
        self.assertLess(result.score, -3.0)
        self.assertEqual(result.direction, -1)

    def test_termination_of_major_strategic_alliance_is_negative(self):
        result = score_a_unstructured_headline("The termination of a major strategic alliance is confirmed by A.")
        self.assertLess(result.score, -2.0)
        self.assertEqual(result.direction, -1)

    def test_expansion_proves_unsuccessful_is_negative(self):
        result = score_a_unstructured_headline("A's expansion into new market proves unsuccessful.")
        self.assertLess(result.score, -2.5)
        self.assertEqual(result.direction, -1)

    def test_diluted_shareholder_value_is_negative(self):
        result = score_a_unstructured_headline("A's stock tumbles after news breaks of diluted shareholder value.")
        self.assertLess(result.score, -2.5)
        self.assertEqual(result.direction, -1)

    def test_payment_system_adopted_by_financial_institutions_is_positive(self):
        result = score_a_unstructured_headline("A's new payment system adopted by leading financial institutions.")
        self.assertGreater(result.score, 2.5)
        self.assertEqual(result.direction, 1)

    def test_unknown_term_report_summarizes_price_response(self):
        events = [
            {
                "event_type": "book_update",
                "monotonic_ms": 1_000,
                "mid": 1000.0,
            },
            {
                "event_type": "news_received",
                "news_kind": "unstructured",
                "monotonic_ms": 1_100,
                "content": "A wins kudos for resilient logistics execution.",
                "unknown_candidate_phrases": ["kudos"],
                "raw_payload": {"kind": "unstructured", "symbol": "A", "new_data": {"content": "A wins kudos for resilient logistics execution."}},
            },
            {
                "event_type": "book_update",
                "monotonic_ms": 1_200,
                "mid": 1002.0,
            },
            {
                "event_type": "book_update",
                "monotonic_ms": 4_200,
                "mid": 1014.0,
            },
            {
                "event_type": "book_update",
                "monotonic_ms": 9_300,
                "mid": 1018.0,
            },
        ]
        report = build_unknown_news_term_report(events)
        self.assertEqual(report["terms"][0]["candidate"], "kudos")
        self.assertEqual(report["terms"][0]["suggestion"], "positive")


if __name__ == "__main__":
    unittest.main()
