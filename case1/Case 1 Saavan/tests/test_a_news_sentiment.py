from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BOT_DIR.parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a_news_sentiment import score_a_unstructured_headline
from case1.ayush_work.marketA_v3.market_A_strategy.a_news_sentiment import (
    score_a_unstructured_headline as ayush_score_a_unstructured_headline,
)


class ANewsSentimentParityTests(unittest.TestCase):
    def test_scoring_matches_ayush_reference_on_curated_headlines(self) -> None:
        headlines = [
            "A's mobile services division exceeds subscriber growth forecasts.",
            "Significant insider selling raises concerns among A investors.",
            "A expands margins.",
            "A launches innovative service.",
            "Consumer satisfaction ratings hit an all-time high for A.",
            "Deployment of A's strategic technology initiative is delayed, stalling progress.",
            "A surpasses earnings expectations for fifth consecutive quarter.",
            "A rumor about weekend scheduling changes.",
        ]

        for headline in headlines:
            with self.subTest(headline=headline):
                ours = score_a_unstructured_headline(headline)
                ayush = ayush_score_a_unstructured_headline(headline)
                self.assertEqual(ours.score, ayush.score)
                self.assertEqual(ours.bucket, ayush.bucket)
                self.assertEqual(ours.direction, ayush.direction)
                self.assertEqual(ours.matched_phrases, ayush.matched_phrases)
                self.assertEqual(ours.matched_unigrams, ayush.matched_unigrams)
                self.assertEqual(ours.matched_bigrams, ayush.matched_bigrams)
                self.assertEqual(ours.unknown_candidate_phrases, ayush.unknown_candidate_phrases)
                self.assertEqual(ours.unknown_candidate_unigrams, ayush.unknown_candidate_unigrams)
                self.assertEqual(ours.unknown_candidate_bigrams, ayush.unknown_candidate_bigrams)

    def test_restored_scorer_matches_ayush_for_recent_review_headlines(self) -> None:
        headlines = [
            "International markets continue to drag down A's financials.",
            "An innovative virtual reality platform is successfully launched by A.",
            "Significant expansion is reported by A in its subscription-based revenue streams.",
        ]

        for headline in headlines:
            with self.subTest(headline=headline):
                ours = score_a_unstructured_headline(headline)
                ayush = ayush_score_a_unstructured_headline(headline)
                self.assertEqual(ours.score, ayush.score)
                self.assertEqual(ours.bucket, ayush.bucket)
                self.assertEqual(ours.direction, ayush.direction)

    def test_new_positive_phrase_family_scores_positive(self) -> None:
        result = score_a_unstructured_headline(
            "Revenue growth surges in a previously underperforming division at A."
        )

        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.direction, 1)
        self.assertTrue(
            any(
                term in result.matched_bigrams
                for term in ("growth surges", "previously underperforming", "underperforming division")
            )
        )

    def test_automation_cost_reduction_phrase_scores_positive(self) -> None:
        result = score_a_unstructured_headline(
            "Through advanced automation, A successfully reduces costs across its operations."
        )

        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.direction, 1)
        self.assertIn("advanced automation", result.matched_bigrams)
        self.assertTrue(
            "successfully reduces" in result.matched_bigrams or "reduces costs" in result.matched_bigrams
        )

    def test_analyst_commendation_phrase_scores_positive(self) -> None:
        result = score_a_unstructured_headline(
            "Analysts commend A for successful product differentiation in a competitive market."
        )

        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.direction, 1)
        self.assertIn("analysts commend", result.matched_bigrams)
        self.assertIn("product differentiation", result.matched_bigrams)
