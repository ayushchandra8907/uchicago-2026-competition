from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from case1.ayush_work.marketA_v3.analyze_run import write_run_outputs
from case1.ayush_work.marketA_v3.config import StrategyConfig
from case1.ayush_work.marketA_v3.market_A_strategy.a_news_tracker import build_a_news_tracker_report


def _event(event_type: str, monotonic_ms: int, **fields):
    payload = {
        "event_type": event_type,
        "monotonic_ms": monotonic_ms,
    }
    payload.update(fields)
    return payload


def _synthetic_events() -> list[dict]:
    return [
        _event("book_update", 900, mid=1000.0),
        _event(
            "news_received",
            1_000,
            news_kind="unstructured",
            content="A wins kudos for resilient logistics execution.",
            raw_payload={"kind": "unstructured", "symbol": "A", "new_data": {"content": "A wins kudos for resilient logistics execution."}},
            news_sentiment_score=0.0,
            news_sentiment_bucket="none",
            news_matched_unigrams=[],
            news_matched_bigrams=[],
            unknown_candidate_unigrams=["kudos", "resilient", "logistics", "execution"],
            unknown_candidate_bigrams=["resilient logistics", "logistics execution"],
            base_fair_value=1000,
            news_fair_value=1000,
            pending_news_target_inventory=0,
        ),
        _event("book_update", 2_000, mid=1005.0),
        _event("book_update", 4_000, mid=1012.0),
        _event("book_update", 8_500, mid=1014.0),
        _event("book_update", 9_900, mid=1000.0),
        _event(
            "news_received",
            10_000,
            news_kind="unstructured",
            content="A new partnership bolsters A's international presence.",
            raw_payload={"kind": "unstructured", "symbol": "A", "new_data": {"content": "A new partnership bolsters A's international presence."}},
            news_sentiment_score=3.4,
            news_sentiment_bucket="strong",
            news_matched_unigrams=["bolsters"],
            news_matched_bigrams=["new partnership", "international presence"],
            unknown_candidate_unigrams=[],
            unknown_candidate_bigrams=[],
            base_fair_value=1000,
            news_fair_value=1048,
            pending_news_target_inventory=8,
        ),
        _event(
            "decision_evaluated",
            10_100,
            desired_intent="post_news_shock_take",
            desired_side="BUY",
            desired_qty=8,
            target_inventory=8,
        ),
        _event(
            "order_filled",
            10_120,
            intent="post_news_shock_take",
            side="BUY",
            fill_qty=8,
        ),
        _event("book_update", 11_000, mid=1018.0),
        _event("book_update", 13_100, mid=1030.0),
        _event("book_update", 18_500, mid=1034.0),
        _event("book_update", 19_900, mid=1040.0),
        _event(
            "news_received",
            20_000,
            news_kind="unstructured",
            content="Significant insider selling raises concerns among A investors.",
            raw_payload={"kind": "unstructured", "symbol": "A", "new_data": {"content": "Significant insider selling raises concerns among A investors."}},
            news_sentiment_score=-4.5,
            news_sentiment_bucket="extreme",
            news_matched_unigrams=["concerns"],
            news_matched_bigrams=["insider selling"],
            unknown_candidate_unigrams=[],
            unknown_candidate_bigrams=[],
            base_fair_value=1040,
            news_fair_value=960,
            pending_news_target_inventory=-40,
        ),
        _event(
            "decision_evaluated",
            20_100,
            desired_intent="post_news_shock_take",
            desired_side="BUY",
            desired_qty=20,
            target_inventory=20,
        ),
        _event(
            "order_filled",
            20_120,
            intent="post_news_shock_take",
            side="BUY",
            fill_qty=20,
        ),
        _event("book_update", 21_000, mid=1032.0),
        _event("book_update", 23_100, mid=1008.0),
        _event("book_update", 28_500, mid=1005.0),
        _event("book_update", 29_900, mid=1050.0),
        _event(
            "news_received",
            30_000,
            news_kind="unstructured",
            content="A wins kudos from observers after resilient execution.",
            raw_payload={"kind": "unstructured", "symbol": "A", "new_data": {"content": "A wins kudos from observers after resilient execution."}},
            news_sentiment_score=0.0,
            news_sentiment_bucket="none",
            news_matched_unigrams=[],
            news_matched_bigrams=[],
            unknown_candidate_unigrams=["kudos", "resilient", "execution"],
            unknown_candidate_bigrams=["resilient execution"],
            base_fair_value=1050,
            news_fair_value=1050,
            pending_news_target_inventory=0,
        ),
        _event("book_update", 31_000, mid=1058.0),
        _event("book_update", 33_200, mid=1064.0),
        _event("book_update", 38_500, mid=1067.0),
    ]


class ANewsTrackerTests(unittest.TestCase):
    def test_tracker_ignores_non_a_unstructured_news_even_if_logger_symbol_is_a(self):
        events = _synthetic_events() + [
            _event(
                "news_received",
                39_000,
                symbol="A",
                news_kind="unstructured",
                content="Fed keeps options open as inflation and growth signals conflict.",
                raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak"}},
                news_sentiment_score=0.0,
                news_sentiment_bucket="none",
                news_matched_unigrams=[],
                news_matched_bigrams=[],
                unknown_candidate_unigrams=[],
                unknown_candidate_bigrams=[],
                base_fair_value=1000,
                news_fair_value=1000,
                pending_news_target_inventory=0,
            )
        ]
        report = build_a_news_tracker_report(events, StrategyConfig())
        headlines = {row["headline"] for row in report["headline_analyses"]}
        self.assertNotIn("Fed keeps options open as inflation and growth signals conflict.", headlines)

    def test_tracker_records_headlines_and_flags_verdicts(self):
        report = build_a_news_tracker_report(_synthetic_events(), StrategyConfig())
        rows = report["headline_analyses"]
        self.assertEqual(len(rows), 4)
        verdicts = {row["headline"]: row["verdict"] for row in rows}
        self.assertEqual(verdicts["A wins kudos for resilient logistics execution."], "missed_no_trade")
        self.assertEqual(verdicts["A new partnership bolsters A's international presence."], "undersized")
        self.assertEqual(verdicts["Significant insider selling raises concerns among A investors."], "wrong_direction")

    def test_tracker_emits_add_increase_and_decrease_recommendations(self):
        report = build_a_news_tracker_report(_synthetic_events(), StrategyConfig())
        recommendations = {row["term"]: row for row in report["term_recommendations"]}
        self.assertEqual(recommendations["kudos"]["suggested_action"], "add")
        self.assertGreater(recommendations["kudos"]["suggested_weight_delta"], 0.0)
        self.assertEqual(recommendations["new partnership"]["suggested_action"], "increase")
        self.assertGreater(recommendations["new partnership"]["suggested_weight_delta"], 0.0)
        self.assertEqual(recommendations["insider selling"]["suggested_action"], "decrease")
        self.assertGreater(recommendations["insider selling"]["suggested_weight_delta"], 0.0)

    def test_write_run_outputs_emits_a_news_tracker_artifacts(self):
        events = _synthetic_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "marketA_v3_test_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            trace_path = run_dir / "trace_events.jsonl"
            trace_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            write_run_outputs(run_dir, write_summary=True, write_graphs=False)
            self.assertTrue((run_dir / "a_news_tracker.json").exists())
            self.assertTrue((run_dir / "a_news_tracker.md").exists())

    def test_write_run_outputs_can_write_checkpoint_artifacts_to_separate_directory(self):
        events = _synthetic_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "marketA_v3_test_run"
            checkpoint_dir = run_dir / "checkpoints" / "halfway"
            run_dir.mkdir(parents=True, exist_ok=True)
            trace_path = run_dir / "trace_events.jsonl"
            trace_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            write_run_outputs(run_dir, output_dir=checkpoint_dir, write_summary=True, write_graphs=False)
            self.assertTrue((checkpoint_dir / "session_summary.json").exists())
            self.assertTrue((checkpoint_dir / "a_news_tracker.json").exists())
            self.assertTrue((checkpoint_dir / "unknown_a_news_terms.json").exists())
            self.assertFalse((run_dir / "session_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
