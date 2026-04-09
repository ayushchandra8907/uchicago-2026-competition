from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from case1.ayush_work.marketA_v3.analyze_run import write_run_outputs
from case1.ayush_work.marketA_v3.config import MarketCStrategyConfig
from case1.ayush_work.marketA_v3.market_C_strategy.c_prediction_tracker import build_c_prediction_tracker_report


def _event(event_type: str, monotonic_ms: int, **fields):
    payload = {"event_type": event_type, "monotonic_ms": monotonic_ms}
    payload.update(fields)
    return payload


def _contract_snapshot(hike: float, hold: float, cut: float) -> dict[str, dict[str, int | float | None]]:
    return {
        "R_HIKE": {"mid": hike, "best_bid_px": int(hike - 2), "best_bid_qty": 20, "best_ask_px": int(hike + 2), "best_ask_qty": 20, "inventory": 0, "open_order_count": 0},
        "R_HOLD": {"mid": hold, "best_bid_px": int(hold - 2), "best_bid_qty": 20, "best_ask_px": int(hold + 2), "best_ask_qty": 20, "inventory": 0, "open_order_count": 0},
        "R_CUT": {"mid": cut, "best_bid_px": int(cut - 2), "best_bid_qty": 20, "best_ask_px": int(cut + 2), "best_ask_qty": 20, "inventory": 0, "open_order_count": 0},
    }


def _book_triplet(monotonic_ms: int, *, hike: float, hold: float, cut: float) -> list[dict]:
    return [
        _event("book_update", monotonic_ms, symbol="R_HIKE", mid=hike, inventory=0),
        _event("book_update", monotonic_ms + 10, symbol="R_HOLD", mid=hold, inventory=0),
        _event("book_update", monotonic_ms + 20, symbol="R_CUT", mid=cut, inventory=0),
    ]


def _synthetic_events() -> list[dict]:
    events: list[dict] = []
    events.extend(_book_triplet(900, hike=300.0, hold=330.0, cut=250.0))
    events.append(
        _event(
            "news_received",
            1_000,
            news_kind="unstructured",
            content="Officials adopt a patient stance while assessing incoming data.",
            raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak", "content": "Officials adopt a patient stance while assessing incoming data."}},
            rate_macro_event_id="FedSpeak:1",
            rate_no_trade_reason="headline_below_relevance_threshold",
            rate_bucket="none",
            rate_relevance_score=1.1,
            prior_hike=0.33,
            prior_hold=0.34,
            prior_cut=0.33,
            posterior_hike=0.28,
            posterior_hold=0.47,
            posterior_cut=0.25,
            fair_value_hike=280,
            fair_value_hold=470,
            fair_value_cut=250,
            rate_target_symbol="R_HOLD",
            rate_target_inventory=0,
            rate_macro_pair_symbols=["R_HOLD", "R_HIKE"],
            rate_baseline_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_macro_pair_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_trading_phase_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_final_phase_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_combined_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_macro_leg_reference_mids={"R_HIKE": 300.0, "R_HOLD": 330.0, "R_CUT": None},
            rate_macro_leg_fairs={"R_HIKE": 250.0, "R_HOLD": 380.0, "R_CUT": None},
            rate_macro_leg_bucket="medium",
            rate_unknown_candidate_unigrams=["patient", "stance"],
            rate_unknown_candidate_bigrams=["patient stance"],
            rate_matched_unigrams=[],
            rate_matched_bigrams=[],
            rate_contract_snapshot_t0=_contract_snapshot(300.0, 330.0, 250.0),
        )
    )
    events.extend(_book_triplet(2_000, hike=292.0, hold=342.0, cut=250.0))
    events.extend(_book_triplet(4_000, hike=280.0, hold=355.0, cut=248.0))
    events.extend(_book_triplet(8_500, hike=276.0, hold=356.0, cut=248.0))

    events.extend(_book_triplet(11_900, hike=298.0, hold=332.0, cut=252.0))
    events.append(
        _event(
            "news_received",
            12_000,
            news_kind="unstructured",
            content="Officials reiterate a patient stance as they monitor the economy.",
            raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak", "content": "Officials reiterate a patient stance as they monitor the economy."}},
            rate_macro_event_id="FedSpeak:2",
            rate_no_trade_reason="headline_below_relevance_threshold",
            rate_bucket="none",
            rate_relevance_score=1.0,
            prior_hike=0.31,
            prior_hold=0.40,
            prior_cut=0.29,
            posterior_hike=0.27,
            posterior_hold=0.49,
            posterior_cut=0.24,
            fair_value_hike=270,
            fair_value_hold=490,
            fair_value_cut=240,
            rate_target_symbol="R_HOLD",
            rate_target_inventory=0,
            rate_macro_pair_symbols=["R_HOLD", "R_HIKE"],
            rate_baseline_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_macro_pair_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_trading_phase_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_final_phase_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_combined_targets_by_symbol={"R_HIKE": -80, "R_HOLD": 80, "R_CUT": 0},
            rate_macro_leg_reference_mids={"R_HIKE": 298.0, "R_HOLD": 332.0, "R_CUT": None},
            rate_macro_leg_fairs={"R_HIKE": 248.0, "R_HOLD": 382.0, "R_CUT": None},
            rate_macro_leg_bucket="medium",
            rate_unknown_candidate_unigrams=["patient", "stance"],
            rate_unknown_candidate_bigrams=["patient stance"],
            rate_matched_unigrams=[],
            rate_matched_bigrams=[],
            rate_contract_snapshot_t0=_contract_snapshot(298.0, 332.0, 252.0),
        )
    )
    events.extend(_book_triplet(13_000, hike=290.0, hold=345.0, cut=252.0))
    events.extend(_book_triplet(15_000, hike=281.0, hold=357.0, cut=250.0))
    events.extend(_book_triplet(20_500, hike=279.0, hold=360.0, cut=248.0))

    events.extend(_book_triplet(22_900, hike=220.0, hold=330.0, cut=220.0))
    events.append(
        _event(
            "news_received",
            23_000,
            news_kind="unstructured",
            content="Officials emphasize balanced risks amid mixed indicators.",
            raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak", "content": "Officials emphasize balanced risks amid mixed indicators."}},
            rate_macro_event_id="FedSpeak:3",
            rate_bucket="strong",
            rate_relevance_score=2.4,
            prior_hike=0.24,
            prior_hold=0.50,
            prior_cut=0.26,
            posterior_hike=0.22,
            posterior_hold=0.56,
            posterior_cut=0.22,
            fair_value_hike=220,
            fair_value_hold=560,
            fair_value_cut=220,
            rate_target_symbol="R_HOLD",
            rate_target_inventory=20,
            rate_macro_pair_symbols=["R_HOLD", "R_HIKE"],
            rate_baseline_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_macro_pair_targets_by_symbol={"R_HIKE": -20, "R_HOLD": 20, "R_CUT": 0},
            rate_trading_phase_targets_by_symbol={"R_HIKE": -20, "R_HOLD": 20, "R_CUT": 0},
            rate_final_phase_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_combined_targets_by_symbol={"R_HIKE": -20, "R_HOLD": 20, "R_CUT": 0},
            rate_macro_leg_reference_mids={"R_HIKE": 220.0, "R_HOLD": 330.0, "R_CUT": None},
            rate_macro_leg_fairs={"R_HIKE": 170.0, "R_HOLD": 380.0, "R_CUT": None},
            rate_macro_leg_bucket="strong",
            rate_matched_unigrams=["balanced"],
            rate_matched_bigrams=["balanced risks"],
            rate_unknown_candidate_unigrams=[],
            rate_unknown_candidate_bigrams=[],
            rate_contract_snapshot_t0=_contract_snapshot(220.0, 330.0, 220.0),
        )
    )
    events.append(
        _event(
            "decision_evaluated",
            23_100,
            symbol="R_HOLD",
            inventory=0,
            desired_symbol="R_HOLD",
            desired_intent="prediction_market_take_macro",
            desired_side="BUY",
            desired_qty=20,
            target_inventory=20,
            rate_macro_event_id="FedSpeak:3",
        )
    )
    events.append(
        _event(
            "order_submitted",
            23_110,
            symbol="R_HOLD",
            side="BUY",
            px=332,
            qty=20,
            intent="prediction_market_take_macro",
            inventory=0,
            rate_macro_event_id="FedSpeak:3",
        )
    )
    events.append(
        _event(
            "order_filled",
            23_120,
            symbol="R_HOLD",
            intent="prediction_market_take_macro",
            side="BUY",
            fill_qty=20,
            inventory=20,
            rate_macro_event_id="FedSpeak:3",
        )
    )
    events.extend(_book_triplet(24_000, hike=208.0, hold=352.0, cut=220.0))
    events.extend(_book_triplet(26_500, hike=195.0, hold=378.0, cut=220.0))
    events.extend(_book_triplet(31_000, hike=192.0, hold=382.0, cut=218.0))

    events.extend(_book_triplet(34_900, hike=320.0, hold=280.0, cut=140.0))
    events.append(
        _event(
            "news_received",
            35_000,
            news_kind="unstructured",
            content="Officials argue rates may stay higher for longer.",
            raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak", "content": "Officials argue rates may stay higher for longer."}},
            rate_macro_event_id="FedSpeak:4",
            rate_bucket="strong",
            rate_relevance_score=2.3,
            prior_hike=0.46,
            prior_hold=0.33,
            prior_cut=0.21,
            posterior_hike=0.58,
            posterior_hold=0.28,
            posterior_cut=0.14,
            fair_value_hike=580,
            fair_value_hold=280,
            fair_value_cut=140,
            rate_target_symbol="R_HIKE",
            rate_target_inventory=80,
            rate_macro_pair_symbols=["R_HIKE", "R_CUT"],
            rate_baseline_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_macro_pair_targets_by_symbol={"R_HIKE": 80, "R_HOLD": 0, "R_CUT": -80},
            rate_trading_phase_targets_by_symbol={"R_HIKE": 80, "R_HOLD": 0, "R_CUT": -80},
            rate_final_phase_targets_by_symbol={"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0},
            rate_combined_targets_by_symbol={"R_HIKE": 80, "R_HOLD": 0, "R_CUT": -80},
            rate_macro_leg_reference_mids={"R_HIKE": 320.0, "R_HOLD": None, "R_CUT": 140.0},
            rate_macro_leg_fairs={"R_HIKE": 400.0, "R_HOLD": None, "R_CUT": 60.0},
            rate_macro_leg_bucket="strong",
            rate_matched_unigrams=[],
            rate_matched_bigrams=["higher longer"],
            rate_unknown_candidate_unigrams=[],
            rate_unknown_candidate_bigrams=[],
            rate_contract_snapshot_t0=_contract_snapshot(320.0, 280.0, 140.0),
        )
    )
    events.append(
        _event(
            "decision_evaluated",
            35_100,
            symbol="R_CUT",
            inventory=0,
            desired_symbol="R_CUT",
            desired_intent="prediction_market_take_macro",
            desired_side="BUY",
            desired_qty=40,
            target_inventory=80,
            rate_macro_event_id="FedSpeak:4",
        )
    )
    events.append(
        _event(
            "order_submitted",
            35_110,
            symbol="R_CUT",
            side="BUY",
            px=142,
            qty=40,
            intent="prediction_market_take_macro",
            inventory=0,
            rate_macro_event_id="FedSpeak:4",
        )
    )
    events.append(
        _event(
            "order_filled",
            35_120,
            symbol="R_CUT",
            intent="prediction_market_take_macro",
            side="BUY",
            fill_qty=40,
            inventory=40,
            rate_macro_event_id="FedSpeak:4",
        )
    )
    events.extend(_book_triplet(36_000, hike=340.0, hold=278.0, cut=120.0))
    events.extend(_book_triplet(38_100, hike=374.0, hold=276.0, cut=84.0))
    events.extend(_book_triplet(43_500, hike=380.0, hold=276.0, cut=80.0))
    return events


class CPredictionTrackerTests(unittest.TestCase):
    def test_tracker_flags_missed_undersized_and_wrong_direction(self):
        report = build_c_prediction_tracker_report(_synthetic_events(), MarketCStrategyConfig())
        rows = {row["headline"]: row for row in report["event_analyses"]}
        self.assertEqual(rows["Officials adopt a patient stance while assessing incoming data."]["verdict"], "missed_no_trade")
        self.assertEqual(rows["Officials reiterate a patient stance as they monitor the economy."]["verdict"], "missed_no_trade")
        self.assertEqual(rows["Officials emphasize balanced risks amid mixed indicators."]["verdict"], "undersized")
        self.assertEqual(rows["Officials argue rates may stay higher for longer."]["verdict"], "wrong_direction")

    def test_tracker_records_macro_pair_and_per_leg_moves(self):
        report = build_c_prediction_tracker_report(_synthetic_events(), MarketCStrategyConfig())
        row = {row["headline"]: row for row in report["event_analyses"]}["Officials emphasize balanced risks amid mixed indicators."]
        self.assertEqual(row["positive_leg_symbol"], "R_HOLD")
        self.assertEqual(row["negative_leg_symbol"], "R_HIKE")
        self.assertEqual(row["macro_pair_targets_by_symbol"]["R_HOLD"], 20)
        self.assertEqual(row["macro_pair_targets_by_symbol"]["R_HIKE"], -20)
        self.assertGreater(row["positive_leg_delta_3s"], 0.0)
        self.assertLess(row["negative_leg_delta_3s"], 0.0)
        self.assertEqual(len(row["desired_order_path"]), 1)
        self.assertEqual(row["desired_order_path"][0]["intent"], "prediction_market_take_macro")

    def test_tracker_emits_add_increase_and_decrease_recommendations(self):
        report = build_c_prediction_tracker_report(_synthetic_events(), MarketCStrategyConfig())
        recommendations = {row["term"]: row for row in report["term_recommendations"]}
        self.assertEqual(recommendations["patient stance"]["suggested_action"], "add")
        self.assertEqual(recommendations["balanced risks"]["suggested_action"], "increase")
        self.assertEqual(recommendations["higher longer"]["suggested_action"], "decrease")

    def test_write_run_outputs_emits_c_prediction_tracker_artifacts(self):
        events = _synthetic_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "marketC_v1_test_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            trace_path = run_dir / "trace_events.jsonl"
            with trace_path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
            write_run_outputs(run_dir, events=events, write_summary=True, write_graphs=False)
            self.assertTrue((run_dir / "c_prediction_tracker.json").exists())
            self.assertTrue((run_dir / "c_prediction_tracker.md").exists())


if __name__ == "__main__":
    unittest.main()
