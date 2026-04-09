from __future__ import annotations

# Adapted from Ayush's marketA_v3 tracker.
# Keep the headline verdict and term recommendation logic aligned with the source unless intentionally retuned.

from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from a_bot_config import AConfig
from a_news_sentiment import get_a_news_term_polarity, get_a_news_term_weight


@dataclass(frozen=True)
class ANewsHeadlineAnalysis:
    tick: int | None
    monotonic_ms: int
    signal_id: str | None
    headline: str
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unmatched_candidate_unigrams: tuple[str, ...]
    unmatched_candidate_bigrams: tuple[str, ...]
    sentiment_score: float
    sentiment_bucket: str
    base_fair_value: int | None
    news_fair_value: int | None
    target_inventory: int | None
    traded: bool
    first_desired_side: str | None
    first_desired_qty: int | None
    first_fill_side: str | None
    first_fill_qty: int | None
    delta_1s: float
    delta_3s: float
    delta_8s: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    verdict: str


@dataclass(frozen=True)
class ANewsTermRecommendation:
    term: str
    polarity: str
    count: int
    avg_delta_1s: float
    avg_delta_3s: float
    avg_delta_8s: float
    same_sign_rate: float
    missed_no_trade_count: int
    undersized_count: int
    wrong_direction_count: int
    current_live_weight: float | None
    suggested_action: str
    suggested_weight_delta: float
    example_headlines: tuple[str, ...]


def build_a_news_tracker_report(events: list[dict[str, Any]], config: AConfig | None = None) -> dict[str, Any]:
    live_config = config or AConfig()
    ordered = sorted(events, key=lambda event: int(event.get("monotonic_ms", 0)))
    mid_points = _mid_series(ordered)
    midpoint_times = [item[0] for item in mid_points]
    decisions = [event for event in ordered if event.get("event_type") == "decision_evaluated"]
    submissions = [event for event in ordered if event.get("event_type") == "order_submitted"]
    fills = [event for event in ordered if event.get("event_type") == "order_filled"]
    a_news_events = [event for event in ordered if _is_a_unstructured_news_event(event)]

    headlines: list[ANewsHeadlineAnalysis] = []
    for index, news_event in enumerate(a_news_events):
        next_event_ms = None
        if index + 1 < len(a_news_events):
            next_event_ms = int(a_news_events[index + 1].get("monotonic_ms", 0))
        headlines.append(
            _analyze_a_news_event(
                news_event,
                decisions=decisions,
                submissions=submissions,
                fills=fills,
                mid_points=mid_points,
                midpoint_times=midpoint_times,
                config=live_config,
                next_event_ms=next_event_ms,
            )
        )

    recommendations = _build_term_recommendations(headlines)
    return {
        "headline_analyses": [asdict(row) for row in headlines],
        "term_recommendations": [asdict(row) for row in recommendations],
    }


def render_a_news_tracker_markdown(report: dict[str, Any], run_dir: Path) -> str:
    headline_rows = report.get("headline_analyses") or []
    recommendation_rows = report.get("term_recommendations") or []
    lines = [
        "# A-News Tracker",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Headlines analyzed: `{len(headline_rows)}`",
        "",
        "## Per-Headline Review",
    ]
    if not headline_rows:
        lines.append("- No A-tagged unstructured headlines found.")
    else:
        for row in headline_rows:
            lines.extend(
                [
                    f"### `{row['headline']}`",
                    f"- Time: `{row['monotonic_ms']}`",
                    f"- Signal ID: `{row['signal_id']}`",
                    f"- Score / bucket: `{row['sentiment_score']}` / `{row['sentiment_bucket']}`",
                    f"- Matched bigrams: `{row['matched_bigrams']}`",
                    f"- Matched unigrams: `{row['matched_unigrams']}`",
                    f"- Unknown bigrams: `{row['unmatched_candidate_bigrams']}`",
                    f"- Unknown unigrams: `{row['unmatched_candidate_unigrams']}`",
                    f"- Base fair / news fair: `{row['base_fair_value']}` / `{row['news_fair_value']}`",
                    f"- Target inventory: `{row['target_inventory']}`",
                    f"- Traded: `{row['traded']}`",
                    f"- First desired order: `{row['first_desired_side']}` x `{row['first_desired_qty']}`",
                    f"- First fill: `{row['first_fill_side']}` x `{row['first_fill_qty']}`",
                    f"- Delta 1s / 3s / 8s: `{row['delta_1s']}` / `{row['delta_3s']}` / `{row['delta_8s']}`",
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
                    f"- Avg delta 1s / 3s / 8s: `{row['avg_delta_1s']}` / `{row['avg_delta_3s']}` / `{row['avg_delta_8s']}`",
                    f"- Same-sign rate: `{row['same_sign_rate']}`",
                    f"- Missed / undersized / wrong-direction: `{row['missed_no_trade_count']}` / `{row['undersized_count']}` / `{row['wrong_direction_count']}`",
                    f"- Current live weight: `{row['current_live_weight']}`",
                    f"- Suggested action / delta: `{row['suggested_action']}` / `{row['suggested_weight_delta']}`",
                    f"- Example headlines: `{row['example_headlines']}`",
                    "",
                ]
            )
    return "\n".join(lines)


def build_unknown_news_term_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    mid_points = _mid_series(sorted(events, key=lambda entry: int(entry.get("monotonic_ms", 0))))
    midpoint_times = [point[0] for point in mid_points]
    aggregated: dict[str, dict[str, Any]] = {}

    for event in events:
        if not _is_a_unstructured_news_event(event):
            continue
        unknown_candidates = list(event.get("unknown_candidate_phrases") or [])
        if not unknown_candidates:
            continue
        event_ms = int(event.get("monotonic_ms", 0))
        base_mid = _mid_at_or_after(mid_points, midpoint_times, event_ms)
        if base_mid is None:
            continue
        delta_1s = _mid_delta(mid_points, midpoint_times, event_ms, 1_000, base_mid)
        delta_3s = _mid_delta(mid_points, midpoint_times, event_ms, 3_000, base_mid)
        delta_8s = _mid_delta(mid_points, midpoint_times, event_ms, 8_000, base_mid)
        for candidate in unknown_candidates:
            bucket = aggregated.setdefault(
                candidate,
                {
                    "count": 0,
                    "sum_delta_1s": 0.0,
                    "sum_delta_3s": 0.0,
                    "sum_delta_8s": 0.0,
                    "sum_abs_delta_3s": 0.0,
                    "same_sign_hits": 0,
                    "same_sign_total": 0,
                    "examples": [],
                },
            )
            bucket["count"] += 1
            bucket["sum_delta_1s"] += delta_1s
            bucket["sum_delta_3s"] += delta_3s
            bucket["sum_delta_8s"] += delta_8s
            bucket["sum_abs_delta_3s"] += abs(delta_3s)
            if delta_3s != 0:
                bucket["same_sign_total"] += 1
                if delta_3s > 0:
                    bucket["same_sign_hits"] += 1
            if len(bucket["examples"]) < 3 and event.get("content"):
                bucket["examples"].append(str(event.get("content")))

    result_rows: list[dict[str, Any]] = []
    for candidate, bucket in sorted(aggregated.items()):
        count = int(bucket["count"])
        avg_delta_1s = bucket["sum_delta_1s"] / count
        avg_delta_3s = bucket["sum_delta_3s"] / count
        avg_delta_8s = bucket["sum_delta_8s"] / count
        avg_abs_delta_3s = bucket["sum_abs_delta_3s"] / count
        same_sign_total = int(bucket["same_sign_total"])
        same_sign_rate = 0.0 if same_sign_total == 0 else float(bucket["same_sign_hits"]) / float(same_sign_total)
        suggestion = "unclear"
        if avg_delta_3s >= 8.0 and same_sign_rate >= 0.70:
            suggestion = "positive"
        elif avg_delta_3s <= -8.0 and same_sign_rate >= 0.70:
            suggestion = "negative"
        result_rows.append(
            {
                "candidate": candidate,
                "count": count,
                "avg_delta_1s": round(avg_delta_1s, 3),
                "avg_delta_3s": round(avg_delta_3s, 3),
                "avg_delta_8s": round(avg_delta_8s, 3),
                "avg_abs_delta_3s": round(avg_abs_delta_3s, 3),
                "same_sign_rate": round(same_sign_rate, 3),
                "suggestion": suggestion,
                "examples": bucket["examples"],
            }
        )
    return {"terms": result_rows}


def render_unknown_terms_markdown(report: dict[str, Any], run_dir: Path) -> str:
    rows = report.get("terms") or []
    lines = [
        "# Unknown A-News Terms",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Candidates: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.append("- No unknown A-news terms found.")
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                f"## `{row['candidate']}`",
                f"- Count: `{row['count']}`",
                f"- Avg delta 1s / 3s / 8s: `{row['avg_delta_1s']}` / `{row['avg_delta_3s']}` / `{row['avg_delta_8s']}`",
                f"- Avg abs delta 3s: `{row['avg_abs_delta_3s']}`",
                f"- Same-sign rate: `{row['same_sign_rate']}`",
                f"- Suggestion: `{row['suggestion']}`",
                f"- Examples: `{tuple(row['examples'])}`",
                "",
            ]
        )
    return "\n".join(lines)


def _analyze_a_news_event(
    news_event: dict[str, Any],
    *,
    decisions: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    mid_points: list[tuple[int, float]],
    midpoint_times: list[int],
    config: AConfig,
    next_event_ms: int | None,
) -> ANewsHeadlineAnalysis:
    event_ms = int(news_event.get("monotonic_ms", 0))
    window_end_ms = min(event_ms + 8_000, next_event_ms) if next_event_ms is not None else event_ms + 8_000
    signal_id = _news_signal_id(news_event)
    headline = _headline_from_news_event(news_event)
    matched_unigrams = tuple(_as_term_list(news_event.get("news_matched_unigrams"), fallback=_split_terms(news_event.get("news_matched_phrases"), size=1)))
    matched_bigrams = tuple(_as_term_list(news_event.get("news_matched_bigrams"), fallback=_split_terms(news_event.get("news_matched_phrases"), size=2)))
    unknown_unigrams = tuple(_as_term_list(news_event.get("unknown_candidate_unigrams"), fallback=_split_terms(news_event.get("unknown_candidate_phrases"), size=1)))
    unknown_bigrams = tuple(_as_term_list(news_event.get("unknown_candidate_bigrams"), fallback=_split_terms(news_event.get("unknown_candidate_phrases"), size=2)))

    base_mid = _mid_before(mid_points, midpoint_times, event_ms)
    if base_mid is None:
        base_mid = _mid_at_or_after(mid_points, midpoint_times, event_ms)
    base_mid = 0.0 if base_mid is None else float(base_mid)

    delta_1s = _mid_delta(mid_points, midpoint_times, event_ms, 1_000, base_mid)
    delta_3s = _mid_delta(mid_points, midpoint_times, event_ms, 3_000, base_mid)
    delta_8s = _mid_delta(mid_points, midpoint_times, event_ms, min(8_000, max(0, window_end_ms - event_ms)), base_mid)
    move_direction = _sign(delta_3s)
    if move_direction == 0:
        move_direction = _sign(float(news_event.get("news_sentiment_score") or 0.0))
    favorable_excursion, adverse_excursion = _excursions(mid_points, midpoint_times, event_ms, window_end_ms, base_mid, move_direction)

    first_trade_decision = _first_matching_event(
        decisions,
        lambda event: (
            event_ms <= int(event.get("monotonic_ms", 0)) <= window_end_ms
            and _matches_signal(event, signal_id)
            and any(
                (
                    _is_news_take_event(action)
                    and action.get("side") in {"BUY", "SELL"}
                )
                for action in (event.get("aggressive_actions") or [])
            )
        ),
    )
    first_submit = _first_matching_event(
        submissions,
        lambda event: (
            event_ms <= int(event.get("monotonic_ms", 0)) <= window_end_ms
            and _matches_signal(event, signal_id)
            and _is_news_take_event(event)
            and event.get("side") in {"BUY", "SELL"}
        ),
    )
    first_fill = _first_matching_event(
        fills,
        lambda event: (
            event_ms <= int(event.get("monotonic_ms", 0)) <= window_end_ms
            and _matches_signal(event, signal_id)
            and _is_news_take_event(event)
            and event.get("side") in {"BUY", "SELL"}
        ),
    )

    first_action = None
    if first_trade_decision is not None:
        for action in first_trade_decision.get("aggressive_actions") or []:
            if _is_news_take_event(action) and action.get("side") in {"BUY", "SELL"}:
                first_action = action
                break

    first_desired_event = first_action or first_submit
    first_desired_side = None if first_desired_event is None else str(first_desired_event.get("side"))
    first_desired_qty = None if first_desired_event is None or first_desired_event.get("qty") is None else int(first_desired_event.get("qty"))
    first_fill_side = None if first_fill is None else str(first_fill.get("side"))
    first_fill_qty = None if first_fill is None or first_fill.get("fill_qty") is None else int(first_fill.get("fill_qty"))

    target_inventory = news_event.get("pending_news_target_inventory")
    if target_inventory is None:
        target_inventory = news_event.get("news_target_inventory")
    if target_inventory is None:
        target_inventory = news_event.get("original_shock_target_inventory")
    if target_inventory is None:
        target_inventory = news_event.get("shock_target_inventory")
    if first_trade_decision is not None and first_trade_decision.get("news_target_inventory") is not None:
        target_inventory = int(first_trade_decision.get("news_target_inventory"))
    elif first_submit is not None and first_submit.get("news_target_inventory") is not None:
        target_inventory = int(first_submit.get("news_target_inventory"))
    elif target_inventory is not None:
        target_inventory = int(target_inventory)
    elif news_event.get("news_fair_value") is not None:
        target_inventory = _ideal_inventory_from_move(
            delta_ticks=float(news_event.get("news_fair_value")) - base_mid,
            score=float(news_event.get("news_sentiment_score") or 0.0),
            bucket=str(news_event.get("news_sentiment_bucket") or "none"),
            config=config,
        )

    traded = first_trade_decision is not None or first_submit is not None or first_fill is not None
    first_direction_side = first_desired_side or first_fill_side
    first_trade_direction = 0 if first_direction_side is None else (1 if first_direction_side == "BUY" else -1)
    ideal_inventory = _ideal_inventory_from_move(
        delta_ticks=delta_3s,
        score=float(news_event.get("news_sentiment_score") or 0.0),
        bucket=str(news_event.get("news_sentiment_bucket") or "none"),
        config=config,
    )
    verdict = _classify_headline(
        traded=traded,
        first_trade_direction=first_trade_direction,
        target_inventory=target_inventory,
        ideal_inventory=ideal_inventory,
        delta_3s=delta_3s,
        favorable_excursion=favorable_excursion,
    )

    return ANewsHeadlineAnalysis(
        tick=None if news_event.get("exchange_tick") is None else int(news_event.get("exchange_tick")),
        monotonic_ms=event_ms,
        signal_id=signal_id,
        headline=headline,
        matched_unigrams=matched_unigrams,
        matched_bigrams=matched_bigrams,
        unmatched_candidate_unigrams=unknown_unigrams,
        unmatched_candidate_bigrams=unknown_bigrams,
        sentiment_score=float(news_event.get("news_sentiment_score") or 0.0),
        sentiment_bucket=str(news_event.get("news_sentiment_bucket") or "none"),
        base_fair_value=None if news_event.get("base_fair_value") is None else int(news_event.get("base_fair_value")),
        news_fair_value=None if news_event.get("news_fair_value") is None else int(news_event.get("news_fair_value")),
        target_inventory=target_inventory,
        traded=traded,
        first_desired_side=first_desired_side,
        first_desired_qty=first_desired_qty,
        first_fill_side=first_fill_side,
        first_fill_qty=first_fill_qty,
        delta_1s=round(delta_1s, 3),
        delta_3s=round(delta_3s, 3),
        delta_8s=round(delta_8s, 3),
        max_favorable_excursion=round(favorable_excursion, 3),
        max_adverse_excursion=round(adverse_excursion, 3),
        verdict=verdict,
    )


def _build_term_recommendations(rows: list[ANewsHeadlineAnalysis]) -> list[ANewsTermRecommendation]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        known_terms = [*row.matched_bigrams, *row.matched_unigrams]
        unknown_terms = [*row.unmatched_candidate_bigrams, *row.unmatched_candidate_unigrams]
        for term in [*known_terms, *unknown_terms]:
            bucket = aggregates.setdefault(
                term,
                {
                    "count": 0,
                    "sum_delta_1s": 0.0,
                    "sum_delta_3s": 0.0,
                    "sum_delta_8s": 0.0,
                    "sign_hits": 0,
                    "sign_total": 0,
                    "positive_moves": 0,
                    "negative_moves": 0,
                    "missed_no_trade_count": 0,
                    "undersized_count": 0,
                    "wrong_direction_count": 0,
                    "examples": [],
                    "current_live_weight": get_a_news_term_weight(term),
                    "current_live_polarity": get_a_news_term_polarity(term),
                },
            )
            bucket["count"] += 1
            bucket["sum_delta_1s"] += row.delta_1s
            bucket["sum_delta_3s"] += row.delta_3s
            bucket["sum_delta_8s"] += row.delta_8s
            observed_sign = _sign(row.delta_3s)
            if observed_sign > 0:
                bucket["positive_moves"] += 1
            elif observed_sign < 0:
                bucket["negative_moves"] += 1
            if bucket["current_live_polarity"] is not None and observed_sign != 0:
                bucket["sign_total"] += 1
                expected_sign = 1 if bucket["current_live_polarity"] == "positive" else -1
                if observed_sign == expected_sign:
                    bucket["sign_hits"] += 1
            if row.verdict == "missed_no_trade":
                bucket["missed_no_trade_count"] += 1
            elif row.verdict == "undersized":
                bucket["undersized_count"] += 1
            elif row.verdict == "wrong_direction":
                bucket["wrong_direction_count"] += 1
            if len(bucket["examples"]) < 3 and row.headline not in bucket["examples"]:
                bucket["examples"].append(row.headline)

    recommendations: list[ANewsTermRecommendation] = []
    for term, bucket in sorted(aggregates.items()):
        count = int(bucket["count"])
        avg_delta_1s = bucket["sum_delta_1s"] / count
        avg_delta_3s = bucket["sum_delta_3s"] / count
        avg_delta_8s = bucket["sum_delta_8s"] / count
        current_weight = bucket["current_live_weight"]
        current_polarity = bucket["current_live_polarity"]
        if current_polarity is None:
            polarity = "positive" if avg_delta_3s > 0 else "negative" if avg_delta_3s < 0 else "unclear"
            same_sign_total = bucket["positive_moves"] + bucket["negative_moves"]
            same_sign_hits = max(bucket["positive_moves"], bucket["negative_moves"])
        else:
            polarity = current_polarity
            same_sign_total = bucket["sign_total"]
            same_sign_hits = bucket["sign_hits"]
        same_sign_rate = 0.0 if same_sign_total == 0 else float(same_sign_hits) / float(same_sign_total)
        suggested_action, suggested_delta = _recommend_term_adjustment(
            current_weight=current_weight,
            avg_delta_3s=avg_delta_3s,
            same_sign_rate=same_sign_rate,
            count=count,
            missed_no_trade_count=int(bucket["missed_no_trade_count"]),
            undersized_count=int(bucket["undersized_count"]),
            wrong_direction_count=int(bucket["wrong_direction_count"]),
        )
        recommendations.append(
            ANewsTermRecommendation(
                term=term,
                polarity=polarity,
                count=count,
                avg_delta_1s=round(avg_delta_1s, 3),
                avg_delta_3s=round(avg_delta_3s, 3),
                avg_delta_8s=round(avg_delta_8s, 3),
                same_sign_rate=round(same_sign_rate, 3),
                missed_no_trade_count=int(bucket["missed_no_trade_count"]),
                undersized_count=int(bucket["undersized_count"]),
                wrong_direction_count=int(bucket["wrong_direction_count"]),
                current_live_weight=None if current_weight is None else round(float(current_weight), 3),
                suggested_action=suggested_action,
                suggested_weight_delta=round(suggested_delta, 3),
                example_headlines=tuple(bucket["examples"]),
            )
        )
    return recommendations


def _recommend_term_adjustment(
    *,
    current_weight: float | None,
    avg_delta_3s: float,
    same_sign_rate: float,
    count: int,
    missed_no_trade_count: int,
    undersized_count: int,
    wrong_direction_count: int,
) -> tuple[str, float]:
    if current_weight is None:
        avg_abs = abs(avg_delta_3s)
        if count >= 2 and same_sign_rate >= 0.70 and avg_abs >= 8.0:
            base_weight = 1.0 if avg_abs < 16.0 else 1.5 if avg_abs < 28.0 else 2.0
            sign = 1.0 if avg_delta_3s > 0 else -1.0
            return "add", sign * base_weight
        return "review", 0.0

    sign = 1.0 if current_weight > 0 else -1.0
    increase_hits = missed_no_trade_count + undersized_count
    if increase_hits > 0 and same_sign_rate >= 0.70:
        return "increase", sign * min(0.75, 0.25 * increase_hits)
    if wrong_direction_count > 0 or (same_sign_rate > 0.0 and same_sign_rate < 0.50):
        return "decrease", (-sign) * min(0.75, 0.25 * max(1, wrong_direction_count))
    return "review", 0.0


def _classify_headline(
    *,
    traded: bool,
    first_trade_direction: int,
    target_inventory: int | None,
    ideal_inventory: int,
    delta_3s: float,
    favorable_excursion: float,
) -> str:
    if not traded and (abs(delta_3s) >= 8.0 or abs(favorable_excursion) >= 12.0):
        return "missed_no_trade"
    if traded and abs(delta_3s) >= 8.0 and first_trade_direction != 0 and first_trade_direction != _sign(delta_3s):
        return "wrong_direction"
    if (
        traded
        and abs(delta_3s) >= 8.0
        and target_inventory is not None
        and ideal_inventory > 0
        and abs(target_inventory) < (0.60 * float(ideal_inventory))
        and first_trade_direction == _sign(delta_3s)
    ):
        return "undersized"
    if traded:
        return "traded_ok"
    return "unclear"


def _ideal_inventory_from_move(*, delta_ticks: float, score: float, bucket: str, config: AConfig) -> int:
    direction = _sign(delta_ticks)
    if direction == 0:
        return 0
    edge_abs = abs(delta_ticks)
    min_edge = max(1, config.shock_take_min_edge)
    if edge_abs < min_edge:
        return 0
    scaled_cap = max(0, min(config.total_position_limit, config.news_very_extreme_position))
    confidence_span = max(1, 80 - min_edge)
    confidence = min(1.0, max(0.0, edge_abs - min_edge) / confidence_span)
    base_target = max(4, round(scaled_cap * confidence))
    scaled_target = max(base_target, round(edge_abs * 1.20))
    target_abs = min(scaled_cap, scaled_target)

    change_span = max(1, 40 - min_edge)
    change_confidence = min(1.0, max(0.0, edge_abs - min_edge) / change_span)
    change_base_target = max(4, round(scaled_cap * change_confidence))
    change_scaled_target = max(change_base_target, round(edge_abs * 0.75))
    target_abs = min(target_abs, min(scaled_cap, change_scaled_target))
    target_abs = min(target_abs, _news_position_cap(config, score=score, bucket=bucket))
    if target_abs <= config.news_zero_position_threshold:
        return 0
    return direction * target_abs


def _news_position_cap(config: AConfig, *, score: float, bucket: str) -> int:
    absolute = abs(score)
    if absolute >= 5.0:
        return config.news_very_extreme_position
    if bucket == "extreme":
        return config.news_extreme_position
    if bucket == "strong":
        return config.news_strong_position
    if bucket == "medium":
        return config.news_medium_position
    if bucket == "light":
        return config.news_light_position
    return 0


def _mid_series(events: list[dict[str, Any]]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        mid = event.get("mid")
        if mid is None:
            bid = event.get("best_bid_px")
            ask = event.get("best_ask_px")
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2.0
        if mid is None and event_type == "trade_print" and event.get("price") is not None:
            mid = float(event.get("price"))
        if mid is None:
            mark_price = event.get("mark_price")
            if mark_price is not None:
                mid = float(mark_price)
        if mid is None:
            continue
        series.append((int(event.get("monotonic_ms", 0)), float(mid)))
    return series


def _is_a_unstructured_news_event(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "news_received" or event.get("news_kind") != "unstructured":
        return False
    raw_payload = event.get("raw_payload") or {}
    symbol = str(raw_payload.get("symbol") or raw_payload.get("asset") or "").upper()
    new_data = raw_payload.get("new_data") or {}
    asset = str(new_data.get("asset") or "").upper()
    return symbol == "A" or asset == "A"


def _news_signal_id(event: dict[str, Any]) -> str | None:
    for key in ("active_news_signal_id", "pending_news_signal_id", "current_news_signal_id", "signal_id", "trade_group_id"):
        value = event.get(key)
        if value:
            return str(value)
    return None


def _event_signal_ids(event: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for key in ("signal_id", "trade_group_id", "active_news_signal_id", "pending_news_signal_id", "current_news_signal_id")
        for value in (event.get(key),)
        if value
    }


def _matches_signal(event: dict[str, Any], signal_id: str | None) -> bool:
    if signal_id is None:
        return True
    event_ids = _event_signal_ids(event)
    return not event_ids or signal_id in event_ids


def _is_news_take_event(event: dict[str, Any]) -> bool:
    action_class = str(event.get("action_class") or "")
    intent = str(event.get("intent") or "")
    strategy_family = str(event.get("strategy_family") or "")
    return (
        action_class == "news_take"
        or intent in {"news_take", "post_news_shock_take"}
        or (strategy_family == "a_news" and action_class in {"", "unknown"})
    )


def _headline_from_news_event(event: dict[str, Any]) -> str:
    raw_payload = event.get("raw_payload") or {}
    new_data = raw_payload.get("new_data") or {}
    candidates = (
        event.get("content"),
        new_data.get("content"),
        raw_payload.get("content"),
        raw_payload.get("headline"),
        raw_payload.get("text"),
    )
    for raw_value in candidates:
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if text:
            return text
    return ""


def _as_term_list(raw_value: Any, *, fallback: list[str]) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item)]
    return fallback


def _split_terms(raw_value: Any, *, size: int) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    return [str(item) for item in raw_value if str(item) and len(str(item).split()) == size]


def _first_matching_event(events: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for event in events:
        if predicate(event):
            return event
    return None


def _mid_at_or_after(mid_points: list[tuple[int, float]], midpoint_times: list[int], target_ms: int) -> float | None:
    index = bisect_left(midpoint_times, target_ms)
    if index >= len(mid_points):
        return None
    return float(mid_points[index][1])


def _mid_before(mid_points: list[tuple[int, float]], midpoint_times: list[int], target_ms: int) -> float | None:
    index = bisect_left(midpoint_times, target_ms) - 1
    if index < 0 or index >= len(mid_points):
        return None
    return float(mid_points[index][1])


def _mid_delta(mid_points: list[tuple[int, float]], midpoint_times: list[int], event_ms: int, offset_ms: int, base_mid: float) -> float:
    target_mid = _mid_at_or_after(mid_points, midpoint_times, event_ms + offset_ms)
    if target_mid is None:
        target_mid = _mid_before(mid_points, midpoint_times, event_ms + offset_ms)
    if target_mid is None:
        target_mid = base_mid
    return float(target_mid) - float(base_mid)


def _excursions(
    mid_points: list[tuple[int, float]],
    midpoint_times: list[int],
    start_ms: int,
    end_ms: int,
    base_mid: float,
    direction: int,
) -> tuple[float, float]:
    start_index = bisect_left(midpoint_times, start_ms)
    end_index = bisect_left(midpoint_times, end_ms + 1)
    deltas = [float(mid) - float(base_mid) for _, mid in mid_points[start_index:end_index]]
    if not deltas:
        return 0.0, 0.0
    if direction < 0:
        favorable = max((float(base_mid) - (float(base_mid) + delta)) for delta in deltas)
        adverse = max((float(base_mid) + delta - float(base_mid)) for delta in deltas)
        return favorable, adverse
    favorable = max(deltas)
    adverse = max((-delta) for delta in deltas)
    return favorable, adverse


def _sign(value: float | int | None) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
