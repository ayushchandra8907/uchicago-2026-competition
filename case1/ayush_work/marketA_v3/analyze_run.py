from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from bisect import bisect_left

from .config import MarketCStrategyConfig, StrategyConfig
from .core.graphs import generate_run_graphs
from .core.logger import load_trace_events
from .market_A_strategy.a_news_tracker import build_a_news_tracker_report, render_a_news_tracker_markdown
from .market_C_strategy.c_prediction_tracker import build_c_prediction_tracker_report, render_c_prediction_tracker_markdown


def resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    return path.parent if path.is_file() else path


def summarize_events(events: list[dict]) -> dict:
    fills_by_intent: Counter[str] = Counter()
    fills_by_symbol: Counter[str] = Counter()
    news_by_kind: Counter[str] = Counter()
    mode_durations_ms: dict[str, int] = defaultdict(int)
    inventory_values: list[int] = []
    latest_mtm = None
    latest_basis = "mid"
    last_mode = None
    last_mode_started_ms = None
    active_strategy = None
    tracked_symbols = None
    final_inventory_by_symbol: dict[str, int] = {}
    final_contract_mids: dict[str, float] = {}

    ordered = sorted(events, key=lambda event: int(event.get("monotonic_ms", 0)))
    for event in ordered:
        if event.get("event_type") == "runner_started":
            if event.get("active_strategy") is not None:
                active_strategy = str(event.get("active_strategy"))
            if event.get("tracked_symbols") is not None:
                tracked_symbols = list(event.get("tracked_symbols") or [])
        mode = event.get("mode")
        now_ms = int(event.get("monotonic_ms", 0))
        if mode is not None:
            mode = str(mode)
            if last_mode is None:
                last_mode = mode
                last_mode_started_ms = now_ms
            elif mode != last_mode:
                if last_mode_started_ms is not None:
                    mode_durations_ms[last_mode] += max(0, now_ms - last_mode_started_ms)
                last_mode = mode
                last_mode_started_ms = now_ms

        inventory = event.get("inventory")
        if inventory is not None:
            inventory_values.append(int(inventory))
            symbol = event.get("symbol")
            if symbol is not None:
                final_inventory_by_symbol[str(symbol)] = int(inventory)
        mtm = event.get("mtm_pnl_estimate")
        if mtm is not None:
            latest_mtm = float(mtm)

        symbol = event.get("symbol")
        if symbol is not None:
            mid = event.get("mid")
            if mid is None:
                bid = event.get("best_bid_px")
                ask = event.get("best_ask_px")
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2.0
            if mid is not None:
                final_contract_mids[str(symbol)] = float(mid)

        if event.get("event_type") == "order_filled":
            fills_by_intent[str(event.get("intent") or "unknown")] += 1
            fills_by_symbol[str(event.get("symbol") or "unknown")] += 1
        elif event.get("event_type") == "news_received":
            raw = event.get("raw_payload") or {}
            new_data = raw.get("new_data") or {} if isinstance(raw, dict) else {}
            if raw.get("kind") == "structured" and new_data.get("structured_subtype") == "cpi_print":
                news_by_kind["cpi_print"] += 1
            elif str(new_data.get("type") or raw.get("type") or "").lower() == "fedspeak":
                news_by_kind["FedSpeak"] += 1
            else:
                news_by_kind[str(event.get("news_kind") or "unknown")] += 1

    if last_mode is not None and last_mode_started_ms is not None and ordered:
        mode_durations_ms[last_mode] += max(0, int(ordered[-1].get("monotonic_ms", 0)) - last_mode_started_ms)

    final_market_leader_symbol = None
    final_market_leader_mid = None
    if final_contract_mids:
        final_market_leader_symbol, final_market_leader_mid = max(final_contract_mids.items(), key=lambda item: item[1])

    return {
        "active_strategy": active_strategy,
        "tracked_symbols": tracked_symbols or [],
        "total_events": len(ordered),
        "fills_total": sum(fills_by_intent.values()),
        "fills_by_intent": dict(sorted(fills_by_intent.items())),
        "fills_by_symbol": dict(sorted(fills_by_symbol.items())),
        "news_events_by_kind": dict(sorted(news_by_kind.items())),
        "mode_durations_ms": dict(sorted(mode_durations_ms.items())),
        "largest_inventory_long": max(inventory_values) if inventory_values else 0,
        "largest_inventory_short": min(inventory_values) if inventory_values else 0,
        "average_inventory": (sum(inventory_values) / len(inventory_values)) if inventory_values else 0.0,
        "estimated_final_mtm_pnl": latest_mtm,
        "estimated_final_mtm_basis": latest_basis,
        "final_inventory_by_symbol": dict(sorted(final_inventory_by_symbol.items())),
        "final_contract_mids": {symbol: round(value, 3) for symbol, value in sorted(final_contract_mids.items())},
        "final_market_leader_symbol": final_market_leader_symbol,
        "final_market_leader_mid": None if final_market_leader_mid is None else round(final_market_leader_mid, 3),
    }


def render_summary_markdown(summary: dict, run_dir: Path) -> str:
    active_strategy = str(summary.get("active_strategy") or "unknown").upper()
    if active_strategy == "C":
        title = "# Market C Prediction Run Summary"
    elif active_strategy == "CM":
        title = "# Market C Equity MM Run Summary"
    else:
        title = "# Market A v3 Run Summary"
    c_sections = []
    if active_strategy == "C":
        c_sections = [
            "",
            "## Final Contract State",
            *(f"- Final inventory `{symbol}`: `{qty}`" for symbol, qty in (summary.get("final_inventory_by_symbol") or {}).items()),
            *(f"- Final mid `{symbol}`: `{mid}`" for symbol, mid in (summary.get("final_contract_mids") or {}).items()),
            f"- Final market leader: `{summary.get('final_market_leader_symbol')}` @ `{summary.get('final_market_leader_mid')}`",
        ]
    return "\n".join(
        [
            title,
            "",
            f"- Run folder: `{run_dir}`",
            f"- Active strategy: `{summary.get('active_strategy')}`",
            f"- Tracked symbols: `{summary.get('tracked_symbols')}`",
            f"- Total events: `{summary.get('total_events', 0)}`",
            f"- Estimated final MTM PnL: `{summary.get('estimated_final_mtm_pnl')}`",
            f"- Largest long inventory: `{summary.get('largest_inventory_long', 0)}`",
            f"- Largest short inventory: `{summary.get('largest_inventory_short', 0)}`",
            f"- Average inventory: `{summary.get('average_inventory', 0.0):.2f}`",
            "",
            "## Fills By Intent",
            *(f"- `{intent}`: `{count}`" for intent, count in (summary.get("fills_by_intent") or {}).items()),
            "",
            "## Fills By Symbol",
            *(f"- `{symbol}`: `{count}`" for symbol, count in (summary.get("fills_by_symbol") or {}).items()),
            "",
            "## News Events By Kind",
            *(f"- `{kind}`: `{count}`" for kind, count in (summary.get("news_events_by_kind") or {}).items()),
            "",
            "## Mode Durations",
            *(f"- `{mode}`: `{duration}`" for mode, duration in (summary.get("mode_durations_ms") or {}).items()),
            *c_sections,
        ]
    )


def build_unknown_news_term_report(events: list[dict]) -> dict:
    mid_points: list[tuple[int, float]] = []
    for event in sorted(events, key=lambda entry: int(entry.get("monotonic_ms", 0))):
        mid = event.get("mid")
        if mid is None:
            bid = event.get("best_bid_px")
            ask = event.get("best_ask_px")
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2.0
        if mid is None:
            continue
        mid_points.append((int(event.get("monotonic_ms", 0)), float(mid)))

    midpoint_times = [point[0] for point in mid_points]
    aggregated: dict[str, dict] = {}

    for event in events:
        if event.get("event_type") != "news_received" or event.get("news_kind") != "unstructured":
            continue
        raw_payload = event.get("raw_payload") or {}
        symbol = str(raw_payload.get("symbol") or "").upper()
        asset = str((raw_payload.get("new_data") or {}).get("asset") or "").upper()
        if symbol != "A" and asset != "A":
            continue
        unknown_candidates = list(event.get("unknown_candidate_phrases") or [])
        if not unknown_candidates:
            continue
        event_ms = int(event.get("monotonic_ms", 0))
        base_mid = _first_mid_at_or_after(mid_points, midpoint_times, event_ms)
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

    result_rows: list[dict] = []
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


def _first_mid_at_or_after(mid_points: list[tuple[int, float]], midpoint_times: list[int], target_ms: int) -> float | None:
    index = bisect_left(midpoint_times, target_ms)
    if index >= len(mid_points):
        return None
    return float(mid_points[index][1])


def _mid_delta(mid_points: list[tuple[int, float]], midpoint_times: list[int], event_ms: int, offset_ms: int, base_mid: float) -> float:
    target_mid = _first_mid_at_or_after(mid_points, midpoint_times, event_ms + offset_ms)
    if target_mid is None:
        target_mid = base_mid
    return float(target_mid) - float(base_mid)


def render_unknown_terms_markdown(report: dict, run_dir: Path) -> str:
    rows = report.get("terms") or []
    lines = [
        "# Unknown A-News Terms",
        "",
        f"- Run folder: `{run_dir}`",
        "",
    ]
    if not rows:
        lines.append("- No unknown A-news terms were captured.")
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                f"## `{row['candidate']}`",
                f"- Count: `{row['count']}`",
                f"- Avg delta 1s: `{row['avg_delta_1s']}`",
                f"- Avg delta 3s: `{row['avg_delta_3s']}`",
                f"- Avg delta 8s: `{row['avg_delta_8s']}`",
                f"- Avg abs delta 3s: `{row['avg_abs_delta_3s']}`",
                f"- Same-sign rate: `{row['same_sign_rate']}`",
                f"- Suggestion: `{row['suggestion']}`",
                "- Examples:",
                *(f"  - {example}" for example in row.get("examples") or []),
                "",
            ]
        )
    return "\n".join(lines)


def write_run_outputs(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    events: list[dict] | None = None,
    write_summary: bool = True,
    write_graphs: bool = True,
) -> tuple[dict, list[Path]]:
    resolved = resolve_run_dir(str(run_dir))
    output_path = resolved if output_dir is None else Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    loaded_events = load_trace_events(resolved) if events is None else list(events)
    summary = summarize_events(loaded_events)
    unknown_term_report = build_unknown_news_term_report(loaded_events)
    tracker_report = build_a_news_tracker_report(loaded_events, StrategyConfig())
    c_tracker_report = build_c_prediction_tracker_report(loaded_events, MarketCStrategyConfig())
    generated: list[Path] = []
    (output_path / "unknown_a_news_terms.json").write_text(json.dumps(unknown_term_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "unknown_a_news_terms.md").write_text(render_unknown_terms_markdown(unknown_term_report, resolved), encoding="utf-8")
    (output_path / "a_news_tracker.json").write_text(json.dumps(tracker_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "a_news_tracker.md").write_text(render_a_news_tracker_markdown(tracker_report, resolved), encoding="utf-8")
    (output_path / "c_prediction_tracker.json").write_text(json.dumps(c_tracker_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "c_prediction_tracker.md").write_text(render_c_prediction_tracker_markdown(c_tracker_report, resolved), encoding="utf-8")
    if write_summary:
        (output_path / "session_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (output_path / "session_summary.md").write_text(render_summary_markdown(summary, resolved), encoding="utf-8")
    if write_graphs and _has_a_symbol_activity(loaded_events):
        generated = generate_run_graphs(resolved, output_dir=output_path, events=loaded_events)
    return summary, generated


def _has_a_symbol_activity(events: list[dict]) -> bool:
    for event in events:
        symbol = str(event.get("symbol") or "").upper()
        if symbol == "A":
            return True
        raw = event.get("raw_payload") or {}
        if not isinstance(raw, dict):
            continue
        new_data = raw.get("new_data") or {}
        event_symbol = str(raw.get("symbol") or new_data.get("asset") or "").upper()
        if event_symbol == "A":
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline analyzer for marketA_v3 runs.")
    parser.add_argument("path", help="Run directory or trace_events.jsonl path.")
    parser.add_argument("--rewrite-summary", action="store_true", help="Rewrite session_summary.json and session_summary.md.")
    parser.add_argument("--graphs", action="store_true", help="Generate pnl_A graph(s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.path)
    events = load_trace_events(run_dir)
    summary = summarize_events(events)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.rewrite_summary or args.graphs:
        _, generated = write_run_outputs(run_dir, write_summary=args.rewrite_summary, write_graphs=args.graphs)
        if generated:
            for path in generated:
                print(path)


if __name__ == "__main__":
    main()
