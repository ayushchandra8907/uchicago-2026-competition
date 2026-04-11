from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import MarketCStrategyConfig
from .c_news_sentiment import get_c_term_polarity, get_c_term_weight


@dataclass(frozen=True)
class CPredictionEventAnalysis:
    event_kind: str
    tick: int | None
    monotonic_ms: int
    macro_event_id: str | None
    headline: str
    no_trade_reason: str | None
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unmatched_candidate_unigrams: tuple[str, ...]
    unmatched_candidate_bigrams: tuple[str, ...]
    bucket: str
    relevance_score: float
    prior_hike: float | None
    prior_hold: float | None
    prior_cut: float | None
    posterior_hike: float | None
    posterior_hold: float | None
    posterior_cut: float | None
    fair_value_hike: int | None
    fair_value_hold: int | None
    fair_value_cut: int | None
    positive_leg_symbol: str | None
    negative_leg_symbol: str | None
    baseline_targets_by_symbol: dict[str, int]
    macro_pair_targets_by_symbol: dict[str, int]
    trading_phase_targets_by_symbol: dict[str, int]
    final_phase_targets_by_symbol: dict[str, int]
    combined_targets_by_symbol: dict[str, int]
    macro_leg_reference_mids: dict[str, float | None]
    macro_leg_fairs: dict[str, float | None]
    macro_leg_bucket: str | None
    long_edges: dict[str, float | None]
    short_edges: dict[str, float | None]
    target_symbol: str | None
    target_inventory: int | None
    traded: bool
    first_desired_symbol: str | None
    first_desired_side: str | None
    first_desired_qty: int | None
    first_fill_symbol: str | None
    first_fill_side: str | None
    first_fill_qty: int | None
    contract_snapshot_t0: dict[str, dict[str, int | float | None]]
    contract_snapshot_1s: dict[str, dict[str, int | float | None]]
    contract_snapshot_3s: dict[str, dict[str, int | float | None]]
    contract_snapshot_8s: dict[str, dict[str, int | float | None]]
    desired_order_path: tuple[dict[str, Any], ...]
    order_path: tuple[dict[str, Any], ...]
    fill_path: tuple[dict[str, Any], ...]
    inventory_path: tuple[dict[str, Any], ...]
    ending_inventory_by_symbol: dict[str, int | None]
    positive_leg_delta_1s: float | None
    positive_leg_delta_3s: float | None
    positive_leg_delta_8s: float | None
    negative_leg_delta_1s: float | None
    negative_leg_delta_3s: float | None
    negative_leg_delta_8s: float | None
    max_favorable_excursion: float
    max_adverse_excursion: float
    verdict: str


@dataclass(frozen=True)
class CPredictionTermRecommendation:
    term: str
    polarity: str
    count: int
    avg_pair_move_3s: float
    avg_abs_pair_move_3s: float
    same_sign_rate: float
    missed_no_trade_count: int
    undersized_count: int
    wrong_leg_count: int
    wrong_direction_count: int
    current_live_weight: float | None
    suggested_action: str
    suggested_weight_delta: float
    example_headlines: tuple[str, ...]


def build_c_prediction_tracker_report(events: list[dict[str, Any]], config: MarketCStrategyConfig | None = None) -> dict[str, Any]:
    live_config = config or MarketCStrategyConfig()
    ordered = sorted(events, key=lambda event: int(event.get("monotonic_ms", 0)))
    state_series = _contract_state_series(ordered, live_config.tracked_symbols)
    state_times = {symbol: [item[0] for item in rows] for symbol, rows in state_series.items()}
    decisions = [event for event in ordered if event.get("event_type") == "decision_evaluated"]
    submitted_orders = [event for event in ordered if event.get("event_type") == "order_submitted"]
    fills = [event for event in ordered if event.get("event_type") == "order_filled"]
    macro_events = [event for event in ordered if _is_c_macro_event(event)]
    last_event_ms = 0 if not ordered else int(ordered[-1].get("monotonic_ms", 0))

    analyses: list[CPredictionEventAnalysis] = []
    for index, macro_event in enumerate(macro_events):
        next_event_ms = None
        if index + 1 < len(macro_events):
            next_event_ms = int(macro_events[index + 1].get("monotonic_ms", 0))
        analyses.append(
            _analyze_c_event(
                macro_event,
                decisions=decisions,
                submitted_orders=submitted_orders,
                fills=fills,
                state_series_by_symbol=state_series,
                state_times_by_symbol=state_times,
                config=live_config,
                next_event_ms=next_event_ms,
                last_event_ms=last_event_ms,
            )
        )

    recommendations = _build_term_recommendations(analyses, live_config)
    return {
        "event_analyses": [asdict(row) for row in analyses],
        "reversion_events": [],
        "term_recommendations": [asdict(row) for row in recommendations],
    }


def render_c_prediction_tracker_markdown(report: dict[str, Any], run_dir: Path) -> str:
    event_rows = report.get("event_analyses") or []
    recommendation_rows = report.get("term_recommendations") or []
    lines = [
        "# C Prediction Tracker",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Macro events analyzed: `{len(event_rows)}`",
        "",
        "## Per-Event Review",
    ]
    if not event_rows:
        lines.append("- No CPI / FedSpeak events were captured.")
    else:
        for row in event_rows:
            lines.extend(
                [
                    f"### `{row['headline']}`",
                    f"- Event kind / id: `{row['event_kind']}` / `{row['macro_event_id']}`",
                    f"- Time: `{row['monotonic_ms']}`",
                    f"- Bucket / relevance / no-trade reason: `{row['bucket']}` / `{row['relevance_score']}` / `{row['no_trade_reason']}`",
                    f"- Positive / negative leg: `{row['positive_leg_symbol']}` / `{row['negative_leg_symbol']}`",
                    f"- Prior H/Hd/C: `{row['prior_hike']}` / `{row['prior_hold']}` / `{row['prior_cut']}`",
                    f"- Posterior H/Hd/C: `{row['posterior_hike']}` / `{row['posterior_hold']}` / `{row['posterior_cut']}`",
                    f"- Fair H/Hd/C: `{row['fair_value_hike']}` / `{row['fair_value_hold']}` / `{row['fair_value_cut']}`",
                    f"- Baseline targets: `{row['baseline_targets_by_symbol']}`",
                    f"- Macro pair targets: `{row['macro_pair_targets_by_symbol']}`",
                    f"- Trading-phase targets: `{row['trading_phase_targets_by_symbol']}`",
                    f"- Final-phase targets: `{row['final_phase_targets_by_symbol']}`",
                    f"- Combined targets: `{row['combined_targets_by_symbol']}`",
                    f"- Macro leg reference mids: `{row['macro_leg_reference_mids']}`",
                    f"- Macro leg fairs: `{row['macro_leg_fairs']}`",
                    f"- Target symbol / inventory: `{row['target_symbol']}` / `{row['target_inventory']}`",
                    f"- First desired order: `{row['first_desired_symbol']}` `{row['first_desired_side']}` x `{row['first_desired_qty']}`",
                    f"- First fill: `{row['first_fill_symbol']}` `{row['first_fill_side']}` x `{row['first_fill_qty']}`",
                    f"- Positive leg 1s / 3s / 8s: `{row['positive_leg_delta_1s']}` / `{row['positive_leg_delta_3s']}` / `{row['positive_leg_delta_8s']}`",
                    f"- Negative leg 1s / 3s / 8s: `{row['negative_leg_delta_1s']}` / `{row['negative_leg_delta_3s']}` / `{row['negative_leg_delta_8s']}`",
                    f"- Max favorable / adverse excursion: `{row['max_favorable_excursion']}` / `{row['max_adverse_excursion']}`",
                    f"- Verdict: `{row['verdict']}`",
                    "",
                ]
            )

    lines.extend(["## Term Recommendations"])
    if not recommendation_rows:
        lines.append("- No term recommendations for this run.")
    else:
        for row in recommendation_rows:
            lines.extend(
                [
                    f"### `{row['term']}`",
                    f"- Polarity: `{row['polarity']}`",
                    f"- Count: `{row['count']}`",
                    f"- Avg pair move 3s / abs 3s: `{row['avg_pair_move_3s']}` / `{row['avg_abs_pair_move_3s']}`",
                    f"- Same-sign rate: `{row['same_sign_rate']}`",
                    f"- Missed / undersized / wrong-leg / wrong-direction: `{row['missed_no_trade_count']}` / `{row['undersized_count']}` / `{row['wrong_leg_count']}` / `{row['wrong_direction_count']}`",
                    f"- Current live weight: `{row['current_live_weight']}`",
                    f"- Suggested action / delta: `{row['suggested_action']}` / `{row['suggested_weight_delta']}`",
                    f"- Example headlines: `{row['example_headlines']}`",
                    "",
                ]
            )
    return "\n".join(lines)


def _analyze_c_event(
    event: dict[str, Any],
    *,
    decisions: list[dict[str, Any]],
    submitted_orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    state_series_by_symbol: dict[str, list[tuple[int, dict[str, int | float | None]]]],
    state_times_by_symbol: dict[str, list[int]],
    config: MarketCStrategyConfig,
    next_event_ms: int | None,
    last_event_ms: int,
) -> CPredictionEventAnalysis:
    event_ms = int(event.get("monotonic_ms", 0))
    order_window_end_ms = last_event_ms if next_event_ms is None else max(event_ms, next_event_ms)
    macro_event_id = None if event.get("rate_macro_event_id") is None else str(event.get("rate_macro_event_id"))
    headline = str(event.get("content") or event.get("raw_payload") or "")
    matched_unigrams = tuple(_as_term_list(event.get("rate_matched_unigrams")))
    matched_bigrams = tuple(_as_term_list(event.get("rate_matched_bigrams")))
    unknown_unigrams = tuple(_as_term_list(event.get("rate_unknown_candidate_unigrams")))
    unknown_bigrams = tuple(_as_term_list(event.get("rate_unknown_candidate_bigrams")))
    pair_symbols = _as_term_list(event.get("rate_macro_pair_symbols"))
    positive_leg_symbol = pair_symbols[0] if len(pair_symbols) >= 1 else None
    negative_leg_symbol = pair_symbols[1] if len(pair_symbols) >= 2 else None

    desired_order_rows = [
        row
        for row in decisions
        if _matches_macro_context(row, macro_event_id, event_ms, order_window_end_ms)
        and str(row.get("desired_intent") or "").startswith("prediction_market_")
        and row.get("desired_side") in {"BUY", "SELL"}
    ]
    submitted_order_rows = [
        row
        for row in submitted_orders
        if _matches_macro_context(row, macro_event_id, event_ms, order_window_end_ms)
        and str(row.get("intent") or "").startswith("prediction_market_")
        and row.get("side") in {"BUY", "SELL"}
    ]
    fill_rows = [
        row
        for row in fills
        if _matches_macro_context(row, macro_event_id, event_ms, order_window_end_ms)
        and str(row.get("intent") or "").startswith("prediction_market_")
        and row.get("side") in {"BUY", "SELL"}
    ]

    first_trade_decision = desired_order_rows[0] if desired_order_rows else None
    first_fill = fill_rows[0] if fill_rows else None

    target_symbol = None if event.get("rate_target_symbol") is None else str(event.get("rate_target_symbol"))
    if target_symbol is None and first_trade_decision is not None:
        target_symbol = None if first_trade_decision.get("desired_symbol") is None else str(first_trade_decision.get("desired_symbol"))
    target_inventory = _optional_int(event.get("rate_target_inventory"))
    if target_inventory is None and first_trade_decision is not None:
        target_inventory = _optional_int(first_trade_decision.get("target_inventory"))

    snapshot_t0 = _normalize_contract_snapshot(event.get("rate_contract_snapshot_t0"), config.tracked_symbols)
    if not any(snapshot_t0[symbol]["mid"] is not None for symbol in config.tracked_symbols):
        snapshot_t0 = _contract_snapshot_bundle_at(state_series_by_symbol, state_times_by_symbol, config.tracked_symbols, event_ms)

    snapshot_1s = _contract_snapshot_bundle_at(state_series_by_symbol, state_times_by_symbol, config.tracked_symbols, event_ms + 1_000)
    snapshot_3s = _contract_snapshot_bundle_at(state_series_by_symbol, state_times_by_symbol, config.tracked_symbols, event_ms + 3_000)
    snapshot_8s = _contract_snapshot_bundle_at(state_series_by_symbol, state_times_by_symbol, config.tracked_symbols, event_ms + 8_000)
    ending_snapshot = _contract_snapshot_bundle_at(state_series_by_symbol, state_times_by_symbol, config.tracked_symbols, order_window_end_ms)

    positive_1s = _leg_delta(snapshot_t0, snapshot_1s, positive_leg_symbol)
    positive_3s = _leg_delta(snapshot_t0, snapshot_3s, positive_leg_symbol)
    positive_8s = _leg_delta(snapshot_t0, snapshot_8s, positive_leg_symbol)
    negative_1s = _leg_delta(snapshot_t0, snapshot_1s, negative_leg_symbol)
    negative_3s = _leg_delta(snapshot_t0, snapshot_3s, negative_leg_symbol)
    negative_8s = _leg_delta(snapshot_t0, snapshot_8s, negative_leg_symbol)

    favorable = max(
        _favorable_move(positive_1s, positive_3s, positive_8s, long_leg=True),
        _favorable_move(negative_1s, negative_3s, negative_8s, long_leg=False),
    )
    adverse = max(
        _adverse_move(positive_1s, positive_3s, positive_8s, long_leg=True),
        _adverse_move(negative_1s, negative_3s, negative_8s, long_leg=False),
    )

    ideal_inventory = _ideal_inventory_from_pair_move(
        max(abs(_or_zero(positive_3s)), abs(_or_zero(negative_3s))),
        config,
    )
    traded = bool(desired_order_rows or submitted_order_rows or fill_rows)
    first_desired_symbol = None if first_trade_decision is None else _optional_str(first_trade_decision.get("desired_symbol") or first_trade_decision.get("symbol"))
    first_desired_side = None if first_trade_decision is None else _optional_str(first_trade_decision.get("desired_side"))
    first_fill_symbol = None if first_fill is None else _optional_str(first_fill.get("symbol"))
    first_fill_side = None if first_fill is None else _optional_str(first_fill.get("side"))

    verdict = _classify_event(
        traded=traded,
        first_desired_symbol=first_desired_symbol,
        first_desired_side=first_desired_side,
        positive_leg_symbol=positive_leg_symbol,
        negative_leg_symbol=negative_leg_symbol,
        positive_leg_delta_3s=positive_3s,
        negative_leg_delta_3s=negative_3s,
        target_inventory=target_inventory,
        ideal_inventory=ideal_inventory,
    )

    raw_payload = event.get("raw_payload") or {}
    event_kind = "cpi_print" if _is_cpi_event(raw_payload) else "FedSpeak"
    return CPredictionEventAnalysis(
        event_kind=event_kind,
        tick=_optional_int(event.get("exchange_tick")),
        monotonic_ms=event_ms,
        macro_event_id=macro_event_id,
        headline=headline,
        no_trade_reason=_optional_str(event.get("rate_no_trade_reason")),
        matched_unigrams=matched_unigrams,
        matched_bigrams=matched_bigrams,
        unmatched_candidate_unigrams=unknown_unigrams,
        unmatched_candidate_bigrams=unknown_bigrams,
        bucket=str(event.get("rate_bucket") or "none"),
        relevance_score=float(event.get("rate_relevance_score") or 0.0),
        prior_hike=_optional_float(event.get("prior_hike")),
        prior_hold=_optional_float(event.get("prior_hold")),
        prior_cut=_optional_float(event.get("prior_cut")),
        posterior_hike=_optional_float(event.get("posterior_hike")),
        posterior_hold=_optional_float(event.get("posterior_hold")),
        posterior_cut=_optional_float(event.get("posterior_cut")),
        fair_value_hike=_optional_int(event.get("fair_value_hike")),
        fair_value_hold=_optional_int(event.get("fair_value_hold")),
        fair_value_cut=_optional_int(event.get("fair_value_cut")),
        positive_leg_symbol=positive_leg_symbol,
        negative_leg_symbol=negative_leg_symbol,
        baseline_targets_by_symbol=_int_dict(event.get("rate_baseline_targets_by_symbol")),
        macro_pair_targets_by_symbol=_int_dict(event.get("rate_macro_pair_targets_by_symbol") or event.get("rate_macro_targets_by_symbol")),
        trading_phase_targets_by_symbol=_int_dict(event.get("rate_trading_phase_targets_by_symbol")),
        final_phase_targets_by_symbol=_int_dict(event.get("rate_final_phase_targets_by_symbol")),
        combined_targets_by_symbol=_int_dict(event.get("rate_combined_targets_by_symbol")),
        macro_leg_reference_mids=_float_dict(event.get("rate_macro_leg_reference_mids")),
        macro_leg_fairs=_float_dict(event.get("rate_macro_leg_fairs")),
        macro_leg_bucket=_optional_str(event.get("rate_macro_leg_bucket")),
        long_edges={
            config.fed_hike: _optional_float(event.get("rate_long_edge_hike")),
            config.fed_hold: _optional_float(event.get("rate_long_edge_hold")),
            config.fed_cut: _optional_float(event.get("rate_long_edge_cut")),
        },
        short_edges={
            config.fed_hike: _optional_float(event.get("rate_short_edge_hike")),
            config.fed_hold: _optional_float(event.get("rate_short_edge_hold")),
            config.fed_cut: _optional_float(event.get("rate_short_edge_cut")),
        },
        target_symbol=target_symbol,
        target_inventory=target_inventory,
        traded=traded,
        first_desired_symbol=first_desired_symbol,
        first_desired_side=first_desired_side,
        first_desired_qty=None if first_trade_decision is None else _optional_int(first_trade_decision.get("desired_qty")),
        first_fill_symbol=first_fill_symbol,
        first_fill_side=first_fill_side,
        first_fill_qty=None if first_fill is None else _optional_int(first_fill.get("fill_qty")),
        contract_snapshot_t0=snapshot_t0,
        contract_snapshot_1s=snapshot_1s,
        contract_snapshot_3s=snapshot_3s,
        contract_snapshot_8s=snapshot_8s,
        desired_order_path=tuple(_condense_decision_row(row) for row in desired_order_rows),
        order_path=tuple(_condense_order_row(row) for row in submitted_order_rows),
        fill_path=tuple(_condense_fill_row(row) for row in fill_rows),
        inventory_path=tuple(_build_inventory_path(desired_order_rows, submitted_order_rows, fill_rows)),
        ending_inventory_by_symbol={symbol: _optional_int(ending_snapshot[symbol].get("inventory")) for symbol in config.tracked_symbols},
        positive_leg_delta_1s=_rounded_optional(positive_1s),
        positive_leg_delta_3s=_rounded_optional(positive_3s),
        positive_leg_delta_8s=_rounded_optional(positive_8s),
        negative_leg_delta_1s=_rounded_optional(negative_1s),
        negative_leg_delta_3s=_rounded_optional(negative_3s),
        negative_leg_delta_8s=_rounded_optional(negative_8s),
        max_favorable_excursion=round(favorable, 3),
        max_adverse_excursion=round(adverse, 3),
        verdict=verdict,
    )


def _build_term_recommendations(rows: list[CPredictionEventAnalysis], config: MarketCStrategyConfig) -> list[CPredictionTermRecommendation]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        realized_pair_move = _pair_realized_score(row)
        for term in [*row.matched_bigrams, *row.matched_unigrams, *row.unmatched_candidate_bigrams, *row.unmatched_candidate_unigrams]:
            bucket = aggregates.setdefault(
                term,
                {
                    "count": 0,
                    "sum_pair_move_3s": 0.0,
                    "sum_abs_pair_move_3s": 0.0,
                    "same_sign_hits": 0,
                    "same_sign_total": 0,
                    "missed_no_trade_count": 0,
                    "undersized_count": 0,
                    "wrong_leg_count": 0,
                    "wrong_direction_count": 0,
                    "examples": [],
                },
            )
            bucket["count"] += 1
            bucket["sum_pair_move_3s"] += realized_pair_move
            bucket["sum_abs_pair_move_3s"] += abs(realized_pair_move)
            if realized_pair_move != 0.0:
                bucket["same_sign_total"] += 1
                if realized_pair_move > 0.0:
                    bucket["same_sign_hits"] += 1
            if row.verdict == "missed_no_trade":
                bucket["missed_no_trade_count"] += 1
            elif row.verdict == "undersized":
                bucket["undersized_count"] += 1
            elif row.verdict == "wrong_leg":
                bucket["wrong_leg_count"] += 1
            elif row.verdict == "wrong_direction":
                bucket["wrong_direction_count"] += 1
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(row.headline)

    recommendations: list[CPredictionTermRecommendation] = []
    for term, bucket in sorted(aggregates.items()):
        count = int(bucket["count"])
        avg_pair_move = bucket["sum_pair_move_3s"] / count
        avg_abs_pair_move = bucket["sum_abs_pair_move_3s"] / count
        same_sign_total = int(bucket["same_sign_total"])
        same_sign_rate = 0.0 if same_sign_total == 0 else float(bucket["same_sign_hits"]) / float(same_sign_total)
        current_weight = get_c_term_weight(term)
        polarity = get_c_term_polarity(term)
        missed_count = int(bucket["missed_no_trade_count"])
        undersized_count = int(bucket["undersized_count"])
        wrong_leg_count = int(bucket["wrong_leg_count"])
        wrong_direction_count = int(bucket["wrong_direction_count"])
        suggestion, delta = _suggest_term_action(
            current_weight=current_weight,
            avg_abs_pair_move_3s=avg_abs_pair_move,
            count=count,
            same_sign_rate=same_sign_rate,
            missed_count=missed_count,
            undersized_count=undersized_count,
            wrong_leg_count=wrong_leg_count,
            wrong_direction_count=wrong_direction_count,
        )
        recommendations.append(
            CPredictionTermRecommendation(
                term=term,
                polarity=polarity,
                count=count,
                avg_pair_move_3s=round(avg_pair_move, 3),
                avg_abs_pair_move_3s=round(avg_abs_pair_move, 3),
                same_sign_rate=round(same_sign_rate, 3),
                missed_no_trade_count=missed_count,
                undersized_count=undersized_count,
                wrong_leg_count=wrong_leg_count,
                wrong_direction_count=wrong_direction_count,
                current_live_weight=None if current_weight is None else round(current_weight, 3),
                suggested_action=suggestion,
                suggested_weight_delta=round(delta, 3),
                example_headlines=tuple(bucket["examples"]),
            )
        )
    return recommendations


def _suggest_term_action(
    *,
    current_weight: float | None,
    avg_abs_pair_move_3s: float,
    count: int,
    same_sign_rate: float,
    missed_count: int,
    undersized_count: int,
    wrong_leg_count: int,
    wrong_direction_count: int,
) -> tuple[str, float]:
    if current_weight is None:
        if count >= 2 and same_sign_rate >= 0.70:
            if avg_abs_pair_move_3s >= 28.0:
                return "add", 2.0
            if avg_abs_pair_move_3s >= 16.0:
                return "add", 1.5
            if avg_abs_pair_move_3s >= 8.0:
                return "add", 1.0
        return "review", 0.0

    if wrong_leg_count > 0 or wrong_direction_count > 0:
        return "decrease", min(0.75, 0.25 * max(1, wrong_leg_count + wrong_direction_count))
    if (missed_count > 0 or undersized_count > 0) and same_sign_rate >= 0.70 and avg_abs_pair_move_3s >= 8.0:
        return "increase", min(0.75, 0.25 * max(1, missed_count + undersized_count))
    return "review", 0.0


def _classify_event(
    *,
    traded: bool,
    first_desired_symbol: str | None,
    first_desired_side: str | None,
    positive_leg_symbol: str | None,
    negative_leg_symbol: str | None,
    positive_leg_delta_3s: float | None,
    negative_leg_delta_3s: float | None,
    target_inventory: int | None,
    ideal_inventory: int,
) -> str:
    favorable_pair_move = max(
        _or_zero(positive_leg_delta_3s),
        -_or_zero(negative_leg_delta_3s),
    )
    if not traded and favorable_pair_move >= 8.0:
        return "missed_no_trade"
    if traded and first_desired_symbol not in {positive_leg_symbol, negative_leg_symbol}:
        return "wrong_leg"
    if traded and first_desired_symbol == positive_leg_symbol and first_desired_side == "SELL" and favorable_pair_move >= 8.0:
        return "wrong_direction"
    if traded and first_desired_symbol == negative_leg_symbol and first_desired_side == "BUY" and favorable_pair_move >= 8.0:
        return "wrong_direction"
    if traded and target_inventory is not None and ideal_inventory > 0 and abs(target_inventory) < (0.60 * ideal_inventory) and favorable_pair_move >= 8.0:
        return "undersized"
    if traded:
        return "traded_ok"
    return "unclear"


def _ideal_inventory_from_pair_move(abs_delta_ticks: float, config: MarketCStrategyConfig) -> int:
    if abs_delta_ticks >= float(config.macro_move_strong_ticks):
        return int(config.trading_macro_target_cap)
    if abs_delta_ticks >= float(config.macro_move_medium_ticks):
        return min(int(config.trading_macro_target_cap), int(config.signal_medium_position))
    if abs_delta_ticks >= float(config.macro_move_light_ticks):
        return min(int(config.trading_macro_target_cap), int(config.signal_light_position))
    return 0


def _contract_state_series(
    events: list[dict[str, Any]],
    symbols: tuple[str, ...],
) -> dict[str, list[tuple[int, dict[str, int | float | None]]]]:
    symbol_set = set(symbols)
    states = {symbol: _blank_contract_state() for symbol in symbols}
    series = {symbol: [] for symbol in symbols}
    for event in events:
        symbol = event.get("symbol")
        if symbol not in symbol_set:
            continue
        state = dict(states[str(symbol)])
        updated = False
        for key in ("best_bid_px", "best_bid_qty", "best_ask_px", "best_ask_qty", "mid", "inventory", "open_order_count"):
            value = event.get(key)
            if value is None:
                continue
            if key in {"best_bid_px", "best_bid_qty", "best_ask_px", "best_ask_qty", "inventory", "open_order_count"}:
                state[key] = int(value)
            else:
                state[key] = float(value)
            updated = True
        if state["mid"] is None and state["best_bid_px"] is not None and state["best_ask_px"] is not None:
            state["mid"] = (float(state["best_bid_px"]) + float(state["best_ask_px"])) / 2.0
            updated = True
        if updated:
            states[str(symbol)] = state
            series[str(symbol)].append((int(event.get("monotonic_ms", 0)), dict(state)))
    return series


def _blank_contract_state() -> dict[str, int | float | None]:
    return {
        "best_bid_px": None,
        "best_bid_qty": None,
        "best_ask_px": None,
        "best_ask_qty": None,
        "mid": None,
        "inventory": None,
        "open_order_count": None,
    }


def _normalize_contract_snapshot(value: Any, symbols: tuple[str, ...]) -> dict[str, dict[str, int | float | None]]:
    snapshot = {symbol: _blank_contract_state() for symbol in symbols}
    if not isinstance(value, dict):
        return snapshot
    for symbol in symbols:
        raw = value.get(symbol)
        if not isinstance(raw, dict):
            continue
        normalized = _blank_contract_state()
        for key in normalized:
            if raw.get(key) is None:
                continue
            if key in {"best_bid_px", "best_bid_qty", "best_ask_px", "best_ask_qty", "inventory", "open_order_count"}:
                normalized[key] = int(raw[key])
            else:
                normalized[key] = float(raw[key])
        if normalized["mid"] is None and normalized["best_bid_px"] is not None and normalized["best_ask_px"] is not None:
            normalized["mid"] = (float(normalized["best_bid_px"]) + float(normalized["best_ask_px"])) / 2.0
        snapshot[symbol] = normalized
    return snapshot


def _contract_snapshot_bundle_at(
    state_series_by_symbol: dict[str, list[tuple[int, dict[str, int | float | None]]]],
    state_times_by_symbol: dict[str, list[int]],
    symbols: tuple[str, ...],
    target_ms: int,
) -> dict[str, dict[str, int | float | None]]:
    return {
        symbol: _contract_state_at_or_nearest(
            state_series_by_symbol.get(symbol, []),
            state_times_by_symbol.get(symbol, []),
            target_ms,
        )
        for symbol in symbols
    }


def _contract_state_at_or_nearest(
    rows: list[tuple[int, dict[str, int | float | None]]],
    times: list[int],
    target_ms: int,
) -> dict[str, int | float | None]:
    if not rows:
        return _blank_contract_state()
    index = bisect_left(times, target_ms)
    if index < len(rows):
        return dict(rows[index][1])
    return dict(rows[-1][1])


def _leg_delta(
    base_snapshot: dict[str, dict[str, int | float | None]],
    target_snapshot: dict[str, dict[str, int | float | None]],
    symbol: str | None,
) -> float | None:
    if symbol is None:
        return None
    base_mid = base_snapshot.get(symbol, {}).get("mid")
    target_mid = target_snapshot.get(symbol, {}).get("mid")
    if base_mid is None or target_mid is None:
        return None
    return float(target_mid) - float(base_mid)


def _favorable_move(delta_1s: float | None, delta_3s: float | None, delta_8s: float | None, *, long_leg: bool) -> float:
    values = [_or_zero(delta_1s), _or_zero(delta_3s), _or_zero(delta_8s)]
    return max(values) if long_leg else max(-value for value in values)


def _adverse_move(delta_1s: float | None, delta_3s: float | None, delta_8s: float | None, *, long_leg: bool) -> float:
    values = [_or_zero(delta_1s), _or_zero(delta_3s), _or_zero(delta_8s)]
    return max(-value for value in values) if long_leg else max(values)


def _pair_realized_score(row: CPredictionEventAnalysis) -> float:
    positive = _or_zero(row.positive_leg_delta_3s)
    negative = _or_zero(row.negative_leg_delta_3s)
    return positive - negative


def _build_inventory_path(
    desired_order_rows: list[dict[str, Any]],
    submitted_order_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = [*_condensed_rows(desired_order_rows, _condense_inventory_decision_row)]
    path.extend(_condensed_rows(submitted_order_rows, _condense_inventory_order_row))
    path.extend(_condensed_rows(fill_rows, _condense_inventory_fill_row))
    return sorted(path, key=lambda row: (int(row.get("monotonic_ms", 0)), str(row.get("event_type") or "")))


def _condensed_rows(rows: list[dict[str, Any]], formatter) -> list[dict[str, Any]]:
    return [formatter(row) for row in rows]


def _condense_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("desired_symbol") or row.get("symbol"),
        "side": row.get("desired_side"),
        "qty": _optional_int(row.get("desired_qty")),
        "intent": row.get("desired_intent"),
        "target_inventory": _optional_int(row.get("target_inventory")),
        "inventory": _optional_int(row.get("inventory")),
        "baseline_targets_by_symbol": _int_dict(row.get("rate_baseline_targets_by_symbol")),
        "macro_pair_targets_by_symbol": _int_dict(row.get("rate_macro_pair_targets_by_symbol") or row.get("rate_macro_targets_by_symbol")),
    }


def _condense_order_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "px": _optional_int(row.get("px")),
        "qty": _optional_int(row.get("qty")),
        "intent": row.get("intent"),
        "inventory": _optional_int(row.get("inventory")),
        "order_id": row.get("order_id"),
    }


def _condense_fill_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "fill_qty": _optional_int(row.get("fill_qty")),
        "fill_price": _optional_int(row.get("fill_price")),
        "intent": row.get("intent"),
        "inventory": _optional_int(row.get("inventory")),
        "order_id": row.get("order_id"),
    }


def _condense_inventory_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "desired_order",
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("desired_symbol") or row.get("symbol"),
        "target_inventory": _optional_int(row.get("target_inventory")),
        "desired_qty": _optional_int(row.get("desired_qty")),
        "desired_side": row.get("desired_side"),
        "inventory": _optional_int(row.get("inventory")),
    }


def _condense_inventory_order_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "order_submitted",
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("symbol"),
        "qty": _optional_int(row.get("qty")),
        "side": row.get("side"),
        "inventory": _optional_int(row.get("inventory")),
    }


def _condense_inventory_fill_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "order_filled",
        "monotonic_ms": int(row.get("monotonic_ms", 0)),
        "symbol": row.get("symbol"),
        "fill_qty": _optional_int(row.get("fill_qty")),
        "side": row.get("side"),
        "inventory": _optional_int(row.get("inventory")),
    }


def _matches_macro_context(row: dict[str, Any], macro_event_id: str | None, start_ms: int, end_ms: int) -> bool:
    row_ms = int(row.get("monotonic_ms", 0))
    if row_ms < start_ms or row_ms > end_ms:
        return False
    row_context = None if row.get("rate_macro_event_id") is None else str(row.get("rate_macro_event_id"))
    if macro_event_id is not None and row_context == macro_event_id:
        return True
    return macro_event_id is None and row_context is None


def _is_c_macro_event(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "news_received":
        return False
    raw_payload = event.get("raw_payload") or {}
    if not isinstance(raw_payload, dict):
        return False
    return _is_cpi_event(raw_payload) or str((raw_payload.get("new_data") or {}).get("type") or raw_payload.get("type") or "").lower() == "fedspeak"


def _is_cpi_event(raw_payload: dict[str, Any]) -> bool:
    if not isinstance(raw_payload, dict):
        return False
    new_data = raw_payload.get("new_data") or {}
    return raw_payload.get("kind") == "structured" and new_data.get("structured_subtype") == "cpi_print"


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if item is None:
            continue
        result[str(key)] = int(item)
    return result


def _float_dict(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float | None] = {}
    for key, item in value.items():
        result[str(key)] = None if item is None else float(item)
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _rounded_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _as_term_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)
