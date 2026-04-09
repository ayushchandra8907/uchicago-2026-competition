from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable
import uuid

from a_bot_config import AConfig, TraceConfig
from a_news_tracker import (
    build_a_news_tracker_report,
    build_unknown_news_term_report,
    render_a_news_tracker_markdown,
    render_unknown_terms_markdown,
)


SNAPSHOT_FIELDNAMES = [
    "event_type",
    "run_id",
    "wall_time_iso",
    "monotonic_ms",
    "exchange_tick",
    "symbol",
    "market_key",
    "mode",
    "trigger",
    "observe_only",
    "reason",
    "best_bid_px",
    "best_bid_qty",
    "best_ask_px",
    "best_ask_qty",
    "spread",
    "mid",
    "microprice",
    "fair_value",
    "trusted_multiplier",
    "multiplier_confidence",
    "inventory",
    "earnings_position",
    "news_position",
    "mm_position",
    "buy_exposure",
    "sell_exposure",
    "earnings_budget",
    "mm_budget",
    "budget_shift_active",
    "allowed_buy_size",
    "allowed_sell_size",
    "position_cap",
    "live_bid_px",
    "live_bid_qty",
    "live_ask_px",
    "live_ask_qty",
    "desired_bid_px",
    "desired_bid_qty",
    "desired_ask_px",
    "desired_ask_qty",
    "aggressive_action_count",
    "aggressive_actions_json",
    "quoted_spread",
    "cash",
    "mtm_pnl_estimate",
    "mtm_basis",
    "mark_price",
    "news_caution_remaining_ms",
    "last_relevant_a_earnings_ms",
    "active_earnings_cycle_id",
]


class AppendSafeCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        needs_header = (not file_exists) or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if needs_header:
            self._writer.writeheader()
            self._handle.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def _wall_clock_fields() -> tuple[str, int]:
    wall_time_ns = time.time_ns()
    wall_iso = datetime.fromtimestamp(wall_time_ns / 1_000_000_000, tz=timezone.utc).isoformat()
    return wall_iso, wall_time_ns


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _coalesce_mark_price(*values: Any) -> tuple[float | None, str]:
    bases = ("mid", "fair", "last_trade")
    for basis, value in zip(bases, values):
        if value is not None:
            return float(value), basis
    return None, "unknown"


def estimate_mtm(state: dict[str, Any], cash: int | None) -> tuple[float | None, str, float | None]:
    book = state.get("book") or {}
    mark_price, basis = _coalesce_mark_price(book.get("mid"), state.get("fair_value"), state.get("last_trade_px"))
    inventory = int(state.get("inventory", 0) or 0)
    if cash is None and mark_price is None:
        return None, "unknown", None
    if mark_price is None:
        return float(cash or 0), "cash_only", None
    return float(cash or 0) + (inventory * mark_price), basis, mark_price


def _quoted_spread(desired_bid: dict[str, Any] | None, desired_ask: dict[str, Any] | None, live_orders: list[dict[str, Any]]) -> int | None:
    if desired_bid is not None and desired_ask is not None:
        return int(desired_ask["px"]) - int(desired_bid["px"])
    live_bid = next((order for order in live_orders if order.get("side") == "BUY" and not order.get("cancel_pending")), None)
    live_ask = next((order for order in live_orders if order.get("side") == "SELL" and not order.get("cancel_pending")), None)
    if live_bid is None or live_ask is None:
        return None
    return int(live_ask["px"]) - int(live_bid["px"])


def _resolve_run_dir(path_or_run_dir: str | Path) -> Path:
    path = Path(path_or_run_dir).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def load_trace_events(path_or_run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = _resolve_run_dir(path_or_run_dir)
    trace_path = run_dir if run_dir.name.endswith(".jsonl") else run_dir / "trace_events.jsonl"
    if trace_path.is_dir():
        trace_path = trace_path / "trace_events.jsonl"
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize_trace_events(events: Iterable[dict[str, Any]], *, markout_windows_ms: tuple[int, ...] = (250, 1_000, 5_000)) -> dict[str, Any]:
    event_list = sorted(list(events), key=lambda event: (int(event.get("monotonic_ms", 0)), str(event.get("event_type", ""))))
    mode_durations_by_market: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_mode_by_market: dict[str, str] = {}
    last_mode_started_ms_by_market: dict[str, int] = {}

    fills_by_intent: Counter[str] = Counter()
    fills_by_overlay: Counter[str] = Counter()
    fills_by_action_class: Counter[str] = Counter()
    fill_qty_by_action_class: Counter[str] = Counter()
    submits_by_intent: Counter[str] = Counter()
    submits_by_overlay: Counter[str] = Counter()
    submits_by_action_class: Counter[str] = Counter()
    cancel_reason_counts: Counter[str] = Counter()
    reject_reason_counts: Counter[str] = Counter()
    no_action_reasons: Counter[str] = Counter()
    passive_fills = 0
    aggressive_fills = 0
    spread_cross_count = 0
    observe_only_count = 0
    inventory_values: list[int] = []
    earnings_inventory_values: list[int] = []
    news_inventory_values: list[int] = []
    mm_inventory_values: list[int] = []
    quoted_spreads: list[int] = []
    latest_mtm: float | None = None
    latest_mtm_basis: str | None = None
    budget_shift_active_ms = 0
    last_budget_shift_state: bool | None = None
    last_budget_shift_started_ms: int | None = None
    a_relevant_structured_earnings_count = 0
    a_irrelevant_structured_count = 0
    a_relevant_unstructured_count = 0

    mark_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    latest_mark_by_symbol: dict[str, float] = {}
    fill_markouts_by_intent: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fill_markouts_by_action_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    derived_signal_counts: Counter[str] = Counter()
    latest_derived_signals: dict[str, dict[str, Any]] = {}

    owner_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    action_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    signal_buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    a_cycle_events: list[dict[str, Any]] = []
    mm_mode_markouts: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    mm_fills_inside_earnings_cycle = 0
    b_strategy_block_reasons: Counter[str] = Counter()
    trace_event_counts: Counter[str] = Counter()
    trace_symbol_counts: Counter[str] = Counter()

    def _mark_from_event(event: dict[str, Any]) -> float | None:
        for key in ("mark_price", "mid", "mid_price", "fair_value", "last_trade_px", "price"):
            value = event.get(key)
            if value is not None:
                return float(value)
        return None

    def _bucket() -> dict[str, Any]:
        return {
            "position": 0,
            "avg_cost": 0.0,
            "realized_pnl": 0.0,
            "fill_count": 0,
            "fill_qty": 0,
            "peak_abs_inventory": 0,
            "time_in_position_ms": 0,
            "open_started_ms": None,
        }

    def _apply_fill(bucket: dict[str, Any], signed_qty: int, price: float, now_ms: int) -> None:
        if signed_qty == 0:
            return
        prior_position = int(bucket["position"])
        if prior_position == 0:
            bucket["open_started_ms"] = now_ms
        bucket["fill_count"] += 1
        bucket["fill_qty"] += abs(signed_qty)

        if prior_position == 0 or (prior_position > 0 and signed_qty > 0) or (prior_position < 0 and signed_qty < 0):
            new_position = prior_position + signed_qty
            base_qty = abs(prior_position)
            add_qty = abs(signed_qty)
            if new_position != 0:
                if base_qty == 0:
                    bucket["avg_cost"] = float(price)
                else:
                    bucket["avg_cost"] = ((bucket["avg_cost"] * base_qty) + (float(price) * add_qty)) / abs(new_position)
            bucket["position"] = new_position
        else:
            closing_qty = min(abs(prior_position), abs(signed_qty))
            if prior_position > 0:
                bucket["realized_pnl"] += closing_qty * (float(price) - bucket["avg_cost"])
            else:
                bucket["realized_pnl"] += closing_qty * (bucket["avg_cost"] - float(price))
            new_position = prior_position + signed_qty
            bucket["position"] = new_position
            if new_position == 0:
                if bucket["open_started_ms"] is not None:
                    bucket["time_in_position_ms"] += max(0, now_ms - int(bucket["open_started_ms"]))
                bucket["open_started_ms"] = None
                bucket["avg_cost"] = 0.0
            elif (prior_position > 0 > new_position) or (prior_position < 0 < new_position):
                bucket["avg_cost"] = float(price)
        bucket["peak_abs_inventory"] = max(bucket["peak_abs_inventory"], abs(int(bucket["position"])))

    def _nearest_mark(symbol: str, target_ms: int) -> float | None:
        lookup_symbol = symbol or "_ALL"
        series = mark_series.get(lookup_symbol)
        if not series:
            return None
        times = [item[0] for item in series]
        index = bisect_left(times, target_ms)
        if index >= len(series):
            return None
        return float(series[index][1])

    for event in event_list:
        symbol = str(event.get("symbol") or "")
        mark = _mark_from_event(event)
        if mark is not None:
            event_ms = int(event.get("monotonic_ms", 0))
            mark_series["_ALL"].append((event_ms, mark))
            if symbol:
                mark_series[symbol].append((event_ms, mark))
                latest_mark_by_symbol[symbol] = mark

    for event in event_list:
        event_type = str(event.get("event_type") or "")
        symbol = str(event.get("symbol") or "")
        market_key = str(event.get("market_key") or symbol or "unknown")
        mode = event.get("mode")
        now_ms = int(event.get("monotonic_ms", 0))

        trace_event_counts[event_type] += 1
        trace_symbol_counts[symbol or "_ALL"] += 1

        if mode:
            last_mode = last_mode_by_market.get(market_key)
            last_mode_started_ms = last_mode_started_ms_by_market.get(market_key)
            if last_mode is None:
                last_mode_by_market[market_key] = str(mode)
                last_mode_started_ms_by_market[market_key] = now_ms
            elif mode != last_mode:
                if last_mode_started_ms is not None:
                    mode_durations_by_market[market_key][last_mode] += max(0, now_ms - last_mode_started_ms)
                last_mode_by_market[market_key] = str(mode)
                last_mode_started_ms_by_market[market_key] = now_ms

        if market_key == "A":
            budget_shift_active = bool(event.get("budget_shift_active"))
            if last_budget_shift_state is None:
                last_budget_shift_state = budget_shift_active
                last_budget_shift_started_ms = now_ms
            elif budget_shift_active != last_budget_shift_state:
                if last_budget_shift_state and last_budget_shift_started_ms is not None:
                    budget_shift_active_ms += max(0, now_ms - last_budget_shift_started_ms)
                last_budget_shift_state = budget_shift_active
                last_budget_shift_started_ms = now_ms

        if market_key == "A" and event_type in {"decision_evaluated", "session_state_snapshot", "book_update", "inventory_updated"}:
            inventory = event.get("inventory")
            if inventory is not None:
                inventory_values.append(int(inventory))
            earnings_inventory = event.get("earnings_position")
            if earnings_inventory is not None:
                earnings_inventory_values.append(int(earnings_inventory))
            news_inventory = event.get("news_position")
            if news_inventory is not None:
                news_inventory_values.append(int(news_inventory))
            mm_inventory = event.get("mm_position")
            if mm_inventory is not None:
                mm_inventory_values.append(int(mm_inventory))
            quoted_spread = event.get("quoted_spread")
            if quoted_spread is not None:
                quoted_spreads.append(int(quoted_spread))
            mtm = event.get("mtm_pnl_estimate")
            if mtm is not None:
                latest_mtm = float(mtm)
                latest_mtm_basis = str(event.get("mtm_basis") or "unknown")

        if event_type == "decision_evaluated":
            if bool(event.get("observe_only")):
                observe_only_count += 1
            if int(event.get("aggressive_action_count") or 0) == 0 and event.get("desired_bid") is None and event.get("desired_ask") is None:
                no_action_reasons[str(event.get("reason") or "unknown")] += 1
                if market_key == "B":
                    b_strategy_block_reasons[str(event.get("reason") or "unknown")] += 1

        elif event_type == "news_received":
            payload = event.get("raw_payload") or event.get("raw_news_payload") or {}
            kind = str(payload.get("kind") or event.get("news_kind") or "")
            if kind == "structured":
                new_data = payload.get("new_data") or {}
                asset = str(new_data.get("asset") or payload.get("symbol") or "").upper()
                subtype = str(new_data.get("structured_subtype") or "")
                if asset == "A" and subtype == "earnings" and bool(event.get("relevant")):
                    a_relevant_structured_earnings_count += 1
                    signal_id = str(event.get("current_earnings_signal_id") or f"a_eps_event_{a_relevant_structured_earnings_count}")
                    old_fair_value = event.get("old_fair_value")
                    new_fair_value = event.get("new_fair_value")
                    fair_jump = None
                    if old_fair_value is not None and new_fair_value is not None:
                        fair_jump = float(new_fair_value) - float(old_fair_value)
                    a_cycle_events.append(
                        {
                            "signal_id": signal_id,
                            "event_ms": now_ms,
                            "tick": event.get("exchange_tick"),
                            "mode_before_news": event.get("mode_before_news"),
                            "fair_jump": fair_jump,
                        }
                    )
                else:
                    a_irrelevant_structured_count += 1
            elif bool(event.get("relevant")):
                a_relevant_unstructured_count += 1

        elif event_type == "derived_signal":
            strategy_family = str(event.get("strategy_family") or "unknown")
            derived_signal_counts[strategy_family] += 1
            latest_derived_signals[symbol or strategy_family] = {
                "strategy_family": strategy_family,
                "action_class": event.get("action_class"),
                "signal_id": event.get("signal_id"),
                "payload": event.get("payload"),
            }

        elif event_type == "order_submitted":
            intent = str(event.get("intent") or "unknown")
            action_class = str(event.get("action_class") or "unknown")
            submits_by_intent[intent] += 1
            submits_by_overlay[str(event.get("overlay") or "unknown")] += 1
            submits_by_action_class[action_class] += 1
            if bool(event.get("aggressive")):
                spread_cross_count += 1

        elif event_type == "order_cancel_requested":
            cancel_reason_counts[str(event.get("cancel_reason") or "unknown")] += 1

        elif event_type == "order_rejected":
            reject_reason_counts[str(event.get("rejection_reason") or "unknown")] += 1

        elif event_type == "order_filled":
            intent = str(event.get("intent") or "unknown")
            overlay = str(event.get("overlay") or "unknown")
            action_class = str(event.get("action_class") or "unknown")
            pnl_owner = str(event.get("pnl_owner") or "unknown")
            signal_id = str(event.get("signal_id") or event.get("trade_group_id") or "ungrouped")
            fills_by_intent[intent] += 1
            fills_by_overlay[overlay] += 1
            fills_by_action_class[action_class] += 1
            if bool(event.get("aggressive")):
                aggressive_fills += 1
            else:
                passive_fills += 1

            fill_price = float(event.get("fill_price") or event.get("price") or 0.0)
            fill_qty = int(event.get("fill_qty") or event.get("qty") or 0)
            fill_qty_by_action_class[action_class] += fill_qty
            signed_qty = fill_qty if str(event.get("side")) == "BUY" else -fill_qty
            owner_key = (market_key, pnl_owner, symbol)
            action_key = (market_key, action_class, symbol)
            signal_key = (market_key, pnl_owner, symbol, signal_id)
            owner_bucket = owner_buckets.setdefault(owner_key, _bucket())
            action_bucket = action_buckets.setdefault(action_key, _bucket())
            signal_bucket = signal_buckets.setdefault(signal_key, _bucket())
            _apply_fill(owner_bucket, signed_qty, fill_price, now_ms)
            _apply_fill(action_bucket, signed_qty, fill_price, now_ms)
            _apply_fill(signal_bucket, signed_qty, fill_price, now_ms)
            mm_fill_inside_active_cycle = pnl_owner == "a_market_making" and market_key == "A" and bool(event.get("active_earnings_cycle_id"))
            if mm_fill_inside_active_cycle:
                mm_fills_inside_earnings_cycle += 1

            signed_side = 1 if str(event.get("side")) == "BUY" else -1
            for window_ms in markout_windows_ms:
                future_mark = _nearest_mark(symbol, now_ms + window_ms)
                if future_mark is None:
                    continue
                markout = signed_side * (future_mark - fill_price)
                fill_markouts_by_intent[intent][f"{window_ms}ms"].append(markout)
                fill_markouts_by_action_class[action_class][f"{window_ms}ms"].append(markout)
                if pnl_owner == "a_market_making" and market_key == "A":
                    mode_key = str(event.get("mode_at_submit") or event.get("mode") or "unknown")
                    mm_mode_markouts[mode_key][f"{window_ms}ms"].append(markout)

    final_ms = int(event_list[-1].get("monotonic_ms", 0)) if event_list else 0
    for market_key, last_mode in last_mode_by_market.items():
        last_mode_started_ms = last_mode_started_ms_by_market.get(market_key)
        if last_mode_started_ms is not None:
            mode_durations_by_market[market_key][last_mode] += max(0, final_ms - last_mode_started_ms)
    if last_budget_shift_state and last_budget_shift_started_ms is not None:
        budget_shift_active_ms += max(0, final_ms - last_budget_shift_started_ms)

    mode_durations_ms = dict(sorted(mode_durations_by_market.get("A", {}).items()))

    for bucket in list(owner_buckets.values()) + list(signal_buckets.values()):
        if bucket["open_started_ms"] is not None:
            bucket["time_in_position_ms"] += max(0, final_ms - int(bucket["open_started_ms"]))
            bucket["open_started_ms"] = final_ms

    pnl_by_market: dict[str, float] = defaultdict(float)
    pnl_by_strategy_family: dict[str, float] = defaultdict(float)
    pnl_by_action_class: dict[str, float] = defaultdict(float)
    signal_episode_rows: list[dict[str, Any]] = []
    strategy_rows: dict[str, dict[str, Any]] = {}

    for (market_key, pnl_owner, symbol), bucket in owner_buckets.items():
        position = int(bucket["position"])
        avg_cost = float(bucket["avg_cost"])
        latest_mark = latest_mark_by_symbol.get(symbol)
        unrealized = 0.0
        if latest_mark is not None:
            if position > 0:
                unrealized = position * (latest_mark - avg_cost)
            elif position < 0:
                unrealized = abs(position) * (avg_cost - latest_mark)
        total_pnl = float(bucket["realized_pnl"]) + unrealized
        pnl_by_market[market_key] += total_pnl
        pnl_by_strategy_family[pnl_owner] += total_pnl
        row = strategy_rows.setdefault(
            pnl_owner,
            {
                "strategy_family": pnl_owner,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "fill_count": 0,
                "fill_qty": 0,
                "peak_abs_inventory": 0,
                "time_in_position_ms": 0,
            },
        )
        row["realized_pnl"] += float(bucket["realized_pnl"])
        row["unrealized_pnl"] += unrealized
        row["total_pnl"] += total_pnl
        row["fill_count"] += int(bucket["fill_count"])
        row["fill_qty"] += int(bucket["fill_qty"])
        row["peak_abs_inventory"] = max(int(row["peak_abs_inventory"]), int(bucket["peak_abs_inventory"]))
        row["time_in_position_ms"] += int(bucket["time_in_position_ms"])

    for (market_key, action_class, symbol), bucket in action_buckets.items():
        position = int(bucket["position"])
        avg_cost = float(bucket["avg_cost"])
        latest_mark = latest_mark_by_symbol.get(symbol)
        unrealized = 0.0
        if latest_mark is not None:
            if position > 0:
                unrealized = position * (latest_mark - avg_cost)
            elif position < 0:
                unrealized = abs(position) * (avg_cost - latest_mark)
        total_pnl = float(bucket["realized_pnl"]) + unrealized
        pnl_by_action_class[f"{market_key}.{action_class}"] += total_pnl

    for (market_key, pnl_owner, symbol, signal_id), bucket in signal_buckets.items():
        position = int(bucket["position"])
        avg_cost = float(bucket["avg_cost"])
        latest_mark = latest_mark_by_symbol.get(symbol)
        unrealized = 0.0
        if latest_mark is not None:
            if position > 0:
                unrealized = position * (latest_mark - avg_cost)
            elif position < 0:
                unrealized = abs(position) * (avg_cost - latest_mark)
        total_pnl = float(bucket["realized_pnl"]) + unrealized
        signal_episode_rows.append(
            {
                "market_key": market_key,
                "strategy_family": pnl_owner,
                "symbol": symbol,
                "signal_id": signal_id,
                "realized_pnl": round(float(bucket["realized_pnl"]), 4),
                "unrealized_pnl": round(unrealized, 4),
                "total_pnl": round(total_pnl, 4),
                "fill_count": int(bucket["fill_count"]),
                "fill_qty": int(bucket["fill_qty"]),
                "peak_abs_inventory": int(bucket["peak_abs_inventory"]),
                "time_in_position_ms": int(bucket["time_in_position_ms"]),
            }
        )

    def _average_markouts(source: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for key, windows in source.items():
            summary[key] = {}
            for label, values in windows.items():
                if values:
                    summary[key][label] = sum(values) / len(values)
        return summary

    top_losing_strategy_families = [
        {
            "strategy_family": name,
            "total_pnl": round(row["total_pnl"], 4),
            "realized_pnl": round(row["realized_pnl"], 4),
            "unrealized_pnl": round(row["unrealized_pnl"], 4),
            "fill_count": row["fill_count"],
            "fill_qty": row["fill_qty"],
        }
        for name, row in sorted(strategy_rows.items(), key=lambda item: item[1]["total_pnl"])[:5]
    ]
    top_losing_signal_episodes = sorted(signal_episode_rows, key=lambda row: row["total_pnl"])[:10]

    a_episode_summaries: list[dict[str, Any]] = []
    if a_cycle_events:
        cycle_end_times = [cycle["event_ms"] for cycle in a_cycle_events[1:]] + [final_ms]
        a_state_events = [
            event
            for event in event_list
            if str(event.get("market_key") or event.get("symbol") or "") == "A"
        ]
        for index, cycle in enumerate(a_cycle_events):
            cycle_start = int(cycle["event_ms"])
            cycle_end = int(cycle_end_times[index])
            cycle_signal_id = str(cycle["signal_id"])
            peak_total_inventory = 0
            peak_earnings_inventory = 0
            shock_take_qty = 0
            unwind_time_ms = 0
            cycle_last_mode: str | None = None
            cycle_last_mode_started_ms: int | None = None
            for event in a_state_events:
                event_ms = int(event.get("monotonic_ms", 0))
                if event_ms < cycle_start or event_ms >= cycle_end:
                    continue
                inventory = event.get("inventory")
                if inventory is not None:
                    peak_total_inventory = max(peak_total_inventory, abs(int(inventory)))
                earnings_position = event.get("earnings_position")
                if earnings_position is not None:
                    peak_earnings_inventory = max(peak_earnings_inventory, abs(int(earnings_position)))
                if str(event.get("event_type") or "") == "order_filled":
                    action_class = str(event.get("action_class") or "")
                    if action_class == "shock_take" or str(event.get("intent") or "") == "post_earnings_shock_take":
                        shock_take_qty += int(event.get("fill_qty") or event.get("qty") or 0)
                mode = str(event.get("mode") or "")
                if not mode:
                    continue
                if cycle_last_mode is None:
                    cycle_last_mode = mode
                    cycle_last_mode_started_ms = max(event_ms, cycle_start)
                    continue
                if mode != cycle_last_mode:
                    if cycle_last_mode == "UNWIND" and cycle_last_mode_started_ms is not None:
                        unwind_time_ms += max(0, event_ms - cycle_last_mode_started_ms)
                    cycle_last_mode = mode
                    cycle_last_mode_started_ms = event_ms
            if cycle_last_mode == "UNWIND" and cycle_last_mode_started_ms is not None:
                unwind_time_ms += max(0, cycle_end - cycle_last_mode_started_ms)

            cycle_pnl_by_strategy_owner: dict[str, float] = defaultdict(float)
            for row in signal_episode_rows:
                if str(row.get("market_key")) != "A" or str(row.get("signal_id")) != cycle_signal_id:
                    continue
                cycle_pnl_by_strategy_owner[str(row.get("strategy_family") or "unknown")] += float(row.get("total_pnl") or 0.0)

            next_arrived_during_unwind = False
            if index + 1 < len(a_cycle_events):
                next_arrived_during_unwind = str(a_cycle_events[index + 1].get("mode_before_news") or "") == "UNWIND"
            a_episode_summaries.append(
                {
                    "signal_id": cycle_signal_id,
                    "tick": cycle.get("tick"),
                    "fair_jump": cycle.get("fair_jump"),
                    "mode_before_news": cycle.get("mode_before_news"),
                    "shock_take_qty": shock_take_qty,
                    "peak_total_inventory": peak_total_inventory,
                    "peak_earnings_inventory": peak_earnings_inventory,
                    "unwind_time_ms": unwind_time_ms,
                    "next_earnings_arrived_during_unwind": next_arrived_during_unwind,
                    "cycle_pnl_by_strategy_owner": {
                        key: round(value, 4) for key, value in sorted(cycle_pnl_by_strategy_owner.items())
                    },
                }
            )

    a_mm_loss_by_mode: dict[str, Any] = {}
    for mode_name, windows in sorted(mm_mode_markouts.items()):
        row: dict[str, Any] = {}
        for label, values in sorted(windows.items()):
            if values:
                row[label] = sum(values) / len(values)
                row[f"{label}_samples"] = len(values)
        a_mm_loss_by_mode[mode_name] = row
    if a_mm_loss_by_mode:
        a_mm_loss_by_mode["fills_inside_earnings_cycle"] = mm_fills_inside_earnings_cycle

    a_strategy_breakdown = {
        "shock_take_qty": int(fill_qty_by_action_class.get("shock_take", 0)),
        "shock_take_pnl": round(pnl_by_action_class.get("A.shock_take", 0.0), 4),
        "shock_unwind_pnl": round(pnl_by_action_class.get("A.shock_unwind", 0.0), 4),
        "shock_total_pnl": round(
            pnl_by_action_class.get("A.shock_take", 0.0) + pnl_by_action_class.get("A.shock_unwind", 0.0),
            4,
        ),
        "news_take_pnl": round(pnl_by_action_class.get("A.news_take", 0.0), 4),
        "news_unwind_pnl": round(pnl_by_action_class.get("A.news_unwind", 0.0), 4),
        "news_takeover_flatten_pnl": round(pnl_by_action_class.get("A.news_takeover_flatten", 0.0), 4),
        "news_caution_pnl": round(pnl_by_action_class.get("A.news_caution_mm", 0.0), 4),
        "steady_mm_pnl": round(pnl_by_action_class.get("A.market_making", 0.0), 4),
        "steady_take_pnl": round(pnl_by_action_class.get("A.steady_take", 0.0), 4),
        "mode_durations_ms": mode_durations_ms,
    }

    b_cost_adjusted_residual_stats: dict[str, Any] = {}
    b_residual_events = [
        event
        for event in event_list
        if str(event.get("event_type") or "") == "derived_signal"
        and str(event.get("strategy_family") or "") == "b_observe_only"
    ]
    if b_residual_events:
        per_strike: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "abs_residual_sum": 0.0,
                "abs_residual_count": 0,
                "max_abs_residual": 0.0,
                "edge_after_cost_sum": 0.0,
                "edge_after_cost_count": 0,
                "max_edge_after_cost": 0.0,
                "positive_edge_count": 0,
                "positive_edge_persistence_ms": 0,
            }
        )
        composite_basis_abs_sum = 0.0
        composite_basis_count = 0
        composite_basis_max_abs = 0.0
        for idx, event in enumerate(b_residual_events):
            payload = event.get("payload") or {}
            residuals = payload.get("parity_residual_by_strike") or {}
            edges = payload.get("parity_edge_after_cost_by_strike") or {}
            current_ms = int(event.get("monotonic_ms", 0))
            next_ms = current_ms
            if idx + 1 < len(b_residual_events):
                next_ms = int(b_residual_events[idx + 1].get("monotonic_ms", current_ms))
            delta_ms = max(0, next_ms - current_ms)
            for strike, residual in residuals.items():
                bucket = per_strike[str(strike)]
                abs_residual = abs(float(residual))
                bucket["abs_residual_sum"] += abs_residual
                bucket["abs_residual_count"] += 1
                bucket["max_abs_residual"] = max(bucket["max_abs_residual"], abs_residual)
                edge_after_cost = float(edges.get(strike, 0.0) or 0.0)
                bucket["edge_after_cost_sum"] += edge_after_cost
                bucket["edge_after_cost_count"] += 1
                bucket["max_edge_after_cost"] = max(bucket["max_edge_after_cost"], edge_after_cost)
                if edge_after_cost > 0:
                    bucket["positive_edge_count"] += 1
                    bucket["positive_edge_persistence_ms"] += delta_ms
            composite_basis = payload.get("composite_basis")
            if composite_basis is not None:
                composite_abs = abs(float(composite_basis))
                composite_basis_abs_sum += composite_abs
                composite_basis_count += 1
                composite_basis_max_abs = max(composite_basis_max_abs, composite_abs)

        b_cost_adjusted_residual_stats = {
            "by_strike": {
                strike: {
                    "mean_abs_residual": round(
                        bucket["abs_residual_sum"] / bucket["abs_residual_count"],
                        4,
                    ) if bucket["abs_residual_count"] else 0.0,
                    "max_abs_residual": round(bucket["max_abs_residual"], 4),
                    "mean_edge_after_cost": round(
                        bucket["edge_after_cost_sum"] / bucket["edge_after_cost_count"],
                        4,
                    ) if bucket["edge_after_cost_count"] else 0.0,
                    "max_edge_after_cost": round(bucket["max_edge_after_cost"], 4),
                    "positive_edge_count": int(bucket["positive_edge_count"]),
                    "positive_edge_persistence_ms": int(bucket["positive_edge_persistence_ms"]),
                }
                for strike, bucket in sorted(per_strike.items())
            },
            "composite_basis": {
                "mean_abs_basis": round(composite_basis_abs_sum / composite_basis_count, 4) if composite_basis_count else 0.0,
                "max_abs_basis": round(composite_basis_max_abs, 4),
            },
        }

    trace_volume_summary = {
        "event_counts": dict(sorted(trace_event_counts.items())),
        "symbol_counts": dict(sorted(trace_symbol_counts.items())),
    }

    return {
        "total_events": len(event_list),
        "fills_total": sum(fills_by_intent.values()),
        "cancels_total": sum(cancel_reason_counts.values()),
        "rejects_total": sum(reject_reason_counts.values()),
        "passive_fills": passive_fills,
        "aggressive_fills": aggressive_fills,
        "fills_by_intent": dict(sorted(fills_by_intent.items())),
        "fills_by_overlay": dict(sorted(fills_by_overlay.items())),
        "fills_by_action_class": dict(sorted(fills_by_action_class.items())),
        "submits_by_intent": dict(sorted(submits_by_intent.items())),
        "submits_by_overlay": dict(sorted(submits_by_overlay.items())),
        "submits_by_action_class": dict(sorted(submits_by_action_class.items())),
        "cancel_reasons": dict(cancel_reason_counts.most_common()),
        "reject_reasons": dict(reject_reason_counts.most_common()),
        "mode_durations_ms": mode_durations_ms,
        "mode_durations_by_market": {
            market_key: dict(sorted(mode_map.items()))
            for market_key, mode_map in sorted(mode_durations_by_market.items())
        },
        "largest_inventory_long": max(inventory_values) if inventory_values else 0,
        "largest_inventory_short": min(inventory_values) if inventory_values else 0,
        "average_inventory": (sum(inventory_values) / len(inventory_values)) if inventory_values else 0.0,
        "average_earnings_inventory": (sum(earnings_inventory_values) / len(earnings_inventory_values)) if earnings_inventory_values else 0.0,
        "average_news_inventory": (sum(news_inventory_values) / len(news_inventory_values)) if news_inventory_values else 0.0,
        "average_mm_inventory": (sum(mm_inventory_values) / len(mm_inventory_values)) if mm_inventory_values else 0.0,
        "average_quoted_spread": (sum(quoted_spreads) / len(quoted_spreads)) if quoted_spreads else None,
        "observe_only_count": observe_only_count,
        "most_common_no_action_reasons": dict(no_action_reasons.most_common()),
        "spread_cross_count": spread_cross_count,
        "budget_shift_active_ms": budget_shift_active_ms,
        "estimated_final_mtm_pnl": latest_mtm,
        "estimated_final_mtm_basis": latest_mtm_basis,
        "fill_markouts_by_intent": _average_markouts(fill_markouts_by_intent),
        "markouts_by_action_class": _average_markouts(fill_markouts_by_action_class),
        "pnl_by_market": {key: round(value, 4) for key, value in sorted(pnl_by_market.items())},
        "pnl_by_strategy_family": {key: round(value, 4) for key, value in sorted(pnl_by_strategy_family.items())},
        "pnl_by_action_class": {key: round(value, 4) for key, value in sorted(pnl_by_action_class.items())},
        "top_losing_strategy_families": top_losing_strategy_families,
        "top_losing_signal_episodes": top_losing_signal_episodes,
        "strategy_family_stats": strategy_rows,
        "a_relevant_structured_earnings_count": a_relevant_structured_earnings_count,
        "a_irrelevant_structured_count": a_irrelevant_structured_count,
        "a_relevant_unstructured_count": a_relevant_unstructured_count,
        "a_episode_summaries": a_episode_summaries,
        "a_mm_loss_by_mode": a_mm_loss_by_mode,
        "a_strategy_breakdown": a_strategy_breakdown,
        "derived_signal_counts": dict(sorted(derived_signal_counts.items())),
        "latest_derived_signals": latest_derived_signals,
        "b_cost_adjusted_residual_stats": b_cost_adjusted_residual_stats,
        "b_strategy_block_reasons": dict(b_strategy_block_reasons.most_common()),
        "trace_volume_summary": trace_volume_summary,
        "activity_split": {
            "passive_mm": sum(count for intent, count in fills_by_intent.items() if intent in {"opening_mm", "steady_mm_passive", "multiplier_discovery_mm", "news_cautious_mm"}),
            "steady_takes": fills_by_intent.get("steady_take", 0),
            "earnings_shock_takes": fills_by_intent.get("post_earnings_shock_take", 0),
            "earnings_prejump": fills_by_intent.get("earnings_prejump", 0),
            "unwind": fills_by_intent.get("unwind", 0) + fills_by_intent.get("post_earnings_shock_unwind", 0),
        },
    }


def render_summary_markdown(summary: dict[str, Any], run_id: str, run_dir: Path) -> str:
    fills_by_intent = summary.get("fills_by_intent") or {}
    fills_by_overlay = summary.get("fills_by_overlay") or {}
    fills_by_action_class = summary.get("fills_by_action_class") or {}
    mode_durations = summary.get("mode_durations_ms") or {}
    no_action = summary.get("most_common_no_action_reasons") or {}
    pnl_by_market = summary.get("pnl_by_market") or {}
    pnl_by_strategy = summary.get("pnl_by_strategy_family") or {}
    pnl_by_action_class = summary.get("pnl_by_action_class") or {}
    losing_strategies = summary.get("top_losing_strategy_families") or []
    a_episode_summaries = summary.get("a_episode_summaries") or []
    a_mm_loss_by_mode = summary.get("a_mm_loss_by_mode") or {}
    a_strategy_breakdown = summary.get("a_strategy_breakdown") or {}
    a_news_summary = summary.get("a_news_summary") or {}
    b_cost_stats = summary.get("b_cost_adjusted_residual_stats") or {}
    b_strategy_block_reasons = summary.get("b_strategy_block_reasons") or {}
    trace_volume_summary = summary.get("trace_volume_summary") or {}
    return "\n".join(
        [
            f"# A Bot Run Summary",
            "",
            f"- Run ID: `{run_id}`",
            f"- Run folder: `{run_dir}`",
            f"- Total events: `{summary.get('total_events', 0)}`",
            f"- Estimated final MTM PnL: `{summary.get('estimated_final_mtm_pnl')}` (`{summary.get('estimated_final_mtm_basis')}`)",
            f"- Passive fills: `{summary.get('passive_fills', 0)}`",
            f"- Aggressive fills: `{summary.get('aggressive_fills', 0)}`",
            f"- Largest long inventory: `{summary.get('largest_inventory_long', 0)}`",
            f"- Largest short inventory: `{summary.get('largest_inventory_short', 0)}`",
            f"- Average inventory: `{summary.get('average_inventory', 0.0):.2f}`",
            f"- Average earnings inventory: `{summary.get('average_earnings_inventory', 0.0):.2f}`",
            f"- Average news inventory: `{summary.get('average_news_inventory', 0.0):.2f}`",
            f"- Average MM inventory: `{summary.get('average_mm_inventory', 0.0):.2f}`",
            f"- Average quoted spread: `{summary.get('average_quoted_spread')}`",
            f"- Observe-only decisions: `{summary.get('observe_only_count', 0)}`",
            f"- Budget shift active (ms): `{summary.get('budget_shift_active_ms', 0)}`",
            f"- Relevant A structured earnings: `{summary.get('a_relevant_structured_earnings_count', 0)}`",
            f"- Irrelevant structured items: `{summary.get('a_irrelevant_structured_count', 0)}`",
            f"- Relevant A unstructured news: `{summary.get('a_relevant_unstructured_count', 0)}`",
            "",
            "## PnL By Market",
            *(f"- `{market}`: `{pnl}`" for market, pnl in pnl_by_market.items()),
            "",
            "## PnL By Strategy Family",
            *(f"- `{strategy}`: `{pnl}`" for strategy, pnl in pnl_by_strategy.items()),
            "",
            "## PnL By Action Class",
            *(f"- `{action_class}`: `{pnl}`" for action_class, pnl in pnl_by_action_class.items()),
            "",
            "## Fills By Intent",
            *(f"- `{intent}`: `{count}`" for intent, count in fills_by_intent.items()),
            "",
            "## Fills By Action Class",
            *(f"- `{action_class}`: `{count}`" for action_class, count in fills_by_action_class.items()),
            "",
            "## Fills By Overlay",
            *(f"- `{overlay}`: `{count}`" for overlay, count in fills_by_overlay.items()),
            "",
            "## Mode Durations (ms)",
            *(f"- `{mode}`: `{duration}`" for mode, duration in mode_durations.items()),
            "",
            "## Top No-Action Reasons",
            *(f"- `{reason}`: `{count}`" for reason, count in no_action.items()),
            "",
            "## Top Losing Strategy Families",
            *(
                f"- `{row.get('strategy_family')}`: total=`{row.get('total_pnl')}` fills=`{row.get('fill_count')}` qty=`{row.get('fill_qty')}`"
                for row in losing_strategies
            ),
            "",
            "## A Episode Summaries",
            *(
                f"- `{row.get('signal_id')}` tick=`{row.get('tick')}` fair_jump=`{row.get('fair_jump')}` "
                f"mode_before=`{row.get('mode_before_news')}` shock_qty=`{row.get('shock_take_qty')}` "
                f"peak_total=`{row.get('peak_total_inventory')}` peak_earnings=`{row.get('peak_earnings_inventory')}` "
                f"unwind_ms=`{row.get('unwind_time_ms')}` next_during_unwind=`{row.get('next_earnings_arrived_during_unwind')}` "
                f"cycle_pnl=`{row.get('cycle_pnl_by_strategy_owner')}`"
                for row in a_episode_summaries
            ),
            "",
            "## A MM Loss By Mode",
            *(
                f"- `{mode}`: `{stats}`"
                for mode, stats in a_mm_loss_by_mode.items()
            ),
            "",
            "## A Strategy Breakdown",
            *(f"- `{key}`: `{value}`" for key, value in a_strategy_breakdown.items()),
            "",
            "## A News Summary",
            *(f"- `{key}`: `{value}`" for key, value in a_news_summary.items()),
            "",
            "## B Cost-Adjusted Residual Stats",
            f"- Composite basis: `{b_cost_stats.get('composite_basis')}`",
            *(
                f"- Strike `{strike}`: `{stats}`"
                for strike, stats in (b_cost_stats.get("by_strike") or {}).items()
            ),
            "",
            "## B Strategy Block Reasons",
            *(f"- `{reason}`: `{count}`" for reason, count in b_strategy_block_reasons.items()),
            "",
            "## Trace Volume Summary",
            f"- Event counts: `{trace_volume_summary.get('event_counts')}`",
            f"- Symbol counts: `{trace_volume_summary.get('symbol_counts')}`",
        ]
    )


class TraceRecorder:
    """Append-only analysis logger for live bot runs.

    The JSONL stream is the source of truth. CSV and summaries are derivative.
    """

    def __init__(self, trace_config: TraceConfig, *, session_prefix: str = "a_bot_run", symbol: str = "A"):
        self.trace_config = trace_config
        self.symbol = symbol
        self.run_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = (trace_config.trace_root or Path.cwd() / "analysis_runs").resolve()
        self.run_dir = root / f"{session_prefix}_{timestamp}_live_round"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "trace_events.jsonl"
        self.snapshots_path = self.run_dir / "decision_snapshots.csv"
        self.summary_json_path = self.run_dir / "session_summary.json"
        self.summary_md_path = self.run_dir / "session_summary.md"
        self.a_news_tracker_json_path = self.run_dir / "a_news_tracker.json"
        self.a_news_tracker_md_path = self.run_dir / "a_news_tracker.md"
        self.unknown_a_news_terms_json_path = self.run_dir / "unknown_a_news_terms.json"
        self.unknown_a_news_terms_md_path = self.run_dir / "unknown_a_news_terms.md"
        self._events_handle = self.events_path.open("a", encoding="utf-8")
        self._snapshot_writer = AppendSafeCsvWriter(self.snapshots_path, SNAPSHOT_FIELDNAMES)
        self._last_snapshot_ms_by_symbol: dict[str, int] = {}
        self._latest_cash: int | None = None
        self._closed = False

    @classmethod
    def create_if_enabled(cls, trace_config: TraceConfig | None, *, session_prefix: str = "a_bot_run", symbol: str = "A") -> TraceRecorder | None:
        if trace_config is None or not trace_config.trace_enabled:
            return None
        return cls(trace_config, session_prefix=session_prefix, symbol=symbol)

    def _write_event(self, event: dict[str, Any]) -> None:
        self._events_handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")
        self._events_handle.flush()

    def _base_event(
        self,
        event_type: str,
        *,
        now_ms: int,
        exchange_tick: int | None,
        mode: str | None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        wall_time_iso, wall_time_ns = _wall_clock_fields()
        return {
            "event_type": event_type,
            "run_id": self.run_id,
            "wall_time_iso": wall_time_iso,
            "wall_time_ns": wall_time_ns,
            "monotonic_ms": int(now_ms),
            "exchange_tick": exchange_tick,
            "symbol": symbol or self.symbol,
            "mode": mode,
        }

    def _state_fields(self, state: dict[str, Any], cash: int | None) -> dict[str, Any]:
        book = state.get("book") or {}
        live_orders = list(state.get("live_orders") or [])
        mtm_pnl, mtm_basis, mark_price = estimate_mtm(state, cash)
        self._latest_cash = cash if cash is not None else self._latest_cash
        return {
            "market_key": state.get("market_key"),
            "fair_value": state.get("fair_value"),
            "trusted_multiplier": state.get("trusted_multiplier"),
            "multiplier_confidence": state.get("multiplier_confidence"),
            "latest_earnings": state.get("latest_earnings"),
            "shock_direction": state.get("shock_direction"),
            "shock_threshold": state.get("shock_threshold"),
            "shock_target_fair": state.get("shock_target_fair"),
            "news_caution_active": state.get("news_caution_active"),
            "news_caution_until_ms": state.get("news_caution_until_ms"),
            "news_caution_remaining_ms": state.get("news_caution_remaining_ms"),
            "inventory": state.get("inventory"),
            "earnings_position": state.get("earnings_position"),
            "news_position": state.get("news_position"),
            "mm_position": state.get("mm_position"),
            "active_signal_kind": state.get("active_signal_kind"),
            "base_fair_value": state.get("base_fair_value"),
            "news_fair_value": state.get("news_fair_value"),
            "news_sentiment_score": state.get("news_sentiment_score"),
            "news_sentiment_bucket": state.get("news_sentiment_bucket"),
            "news_confirmation_state": state.get("news_confirmation_state"),
            "news_target_inventory": state.get("news_target_inventory"),
            "pending_news_target_inventory": state.get("pending_news_target_inventory"),
            "pending_news_signal_id": state.get("pending_news_signal_id"),
            "active_news_signal_id": state.get("active_news_signal_id"),
            "news_started_ms": state.get("news_started_ms"),
            "pe_frozen": state.get("pe_frozen"),
            "news_matched_phrases": state.get("news_matched_phrases"),
            "news_matched_unigrams": state.get("news_matched_unigrams"),
            "news_matched_bigrams": state.get("news_matched_bigrams"),
            "unknown_candidate_phrases": state.get("unknown_candidate_phrases"),
            "unknown_candidate_unigrams": state.get("unknown_candidate_unigrams"),
            "unknown_candidate_bigrams": state.get("unknown_candidate_bigrams"),
            "earnings_budget": state.get("earnings_budget"),
            "mm_budget": state.get("mm_budget"),
            "budget_shift_active": state.get("budget_shift_active"),
            "buy_exposure": state.get("buy_exposure"),
            "sell_exposure": state.get("sell_exposure"),
            "overlay_exposures": state.get("overlay_exposures"),
            "allowed_buy_size": state.get("allowed_buy_size"),
            "allowed_sell_size": state.get("allowed_sell_size"),
            "position_cap": state.get("position_cap"),
            "best_bid_px": book.get("best_bid_px"),
            "best_bid_qty": book.get("best_bid_qty"),
            "best_ask_px": book.get("best_ask_px"),
            "best_ask_qty": book.get("best_ask_qty"),
            "spread": book.get("spread"),
            "mid": book.get("mid"),
            "microprice": book.get("microprice"),
            "top_of_book_imbalance": book.get("top_of_book_imbalance"),
            "book_depth": {
                "bid_levels": list(book.get("bid_levels") or []),
                "ask_levels": list(book.get("ask_levels") or []),
            },
            "live_orders": live_orders,
            "cash": cash if cash is not None else self._latest_cash,
            "mtm_pnl_estimate": mtm_pnl,
            "mtm_basis": mtm_basis,
            "mark_price": mark_price,
            "last_trade_px": state.get("last_trade_px"),
            "last_trade_qty": state.get("last_trade_qty"),
            "last_trade_ms": state.get("last_trade_ms"),
            "discovery_window": state.get("discovery_window"),
            "current_earnings_signal_id": state.get("current_earnings_signal_id"),
            "current_news_signal_id": state.get("current_news_signal_id"),
            "last_relevant_a_earnings_ms": state.get("last_relevant_a_earnings_ms"),
            "active_earnings_cycle_id": state.get("active_earnings_cycle_id"),
        }

    @staticmethod
    def _serialize_order(order: dict[str, Any] | None) -> dict[str, Any] | None:
        if order is None:
            return None
        return {
            "side": order.get("side"),
            "px": order.get("px"),
            "qty": order.get("qty"),
            "overlay": order.get("overlay"),
            "aggressive": order.get("aggressive"),
            "reason": order.get("reason"),
            "intent": order.get("intent"),
            "mode_at_submit": order.get("mode_at_submit"),
            "evaluation_reason": order.get("evaluation_reason"),
            "market_key": order.get("market_key"),
            "strategy_family": order.get("strategy_family"),
            "action_class": order.get("action_class"),
            "pnl_owner": order.get("pnl_owner"),
            "signal_id": order.get("signal_id"),
            "trade_group_id": order.get("trade_group_id"),
            "leg_role": order.get("leg_role"),
        }

    def record_session_start(self, *, now_ms: int, config_summary: dict[str, Any], recovered_orders: list[dict[str, Any]], state: dict[str, Any], cash: int | None) -> None:
        event = self._base_event(
            "session_start",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "config_summary": config_summary,
                "recovered_orders": recovered_orders,
                "trace_root": str(self.run_dir.parent),
                "run_dir": str(self.run_dir),
            }
        )
        self._write_event(event)

    def record_recovery_state(self, *, now_ms: int, state: dict[str, Any], cash: int | None, reason: str) -> None:
        event = self._base_event(
            "recovery_state_changed",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["reason"] = reason
        self._write_event(event)

    def record_book_update(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str) -> None:
        event = self._base_event(
            "book_update",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["trigger"] = trigger
        self._write_event(event)

    def record_market_trade(self, *, now_ms: int, state: dict[str, Any], cash: int | None, price: int, qty: int) -> None:
        event = self._base_event(
            "market_trade_observed",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["price"] = int(price)
        event["qty"] = int(qty)
        self._write_event(event)

    def record_inventory_update(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str) -> None:
        event = self._base_event(
            "inventory_updated",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["trigger"] = trigger
        self._write_event(event)

    def record_news(self, *, now_ms: int, state: dict[str, Any], cash: int | None, news_payload: dict[str, Any], reaction: dict[str, Any]) -> None:
        event = self._base_event(
            "news_received",
            now_ms=now_ms,
            exchange_tick=reaction.get("tick") if reaction.get("tick") is not None else state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "raw_payload": news_payload,
                "raw_news_payload": news_payload,
                "news_kind": reaction.get("news_kind"),
                "relevant": reaction.get("relevant"),
                "mode_before_news": reaction.get("mode_before_news"),
                "mode_after_news": reaction.get("mode_after_news"),
                "old_fair_value": reaction.get("old_fair_value"),
                "new_fair_value": reaction.get("new_fair_value"),
                "earnings_value": reaction.get("earnings_value"),
                "news_sentiment_score": reaction.get("news_sentiment_score"),
                "news_sentiment_bucket": reaction.get("news_sentiment_bucket"),
                "base_fair_value": reaction.get("base_fair_value"),
                "news_fair_value": reaction.get("news_fair_value"),
                "pending_news_target_inventory": reaction.get("pending_news_target_inventory"),
                "news_confirmation_state": reaction.get("news_confirmation_state"),
                "active_news_signal_id": reaction.get("active_news_signal_id"),
                "news_matched_phrases": list(reaction.get("news_matched_phrases") or []),
                "news_matched_unigrams": list(reaction.get("news_matched_unigrams") or []),
                "news_matched_bigrams": list(reaction.get("news_matched_bigrams") or []),
                "unknown_candidate_phrases": list(reaction.get("unknown_candidate_phrases") or []),
                "unknown_candidate_unigrams": list(reaction.get("unknown_candidate_unigrams") or []),
                "unknown_candidate_bigrams": list(reaction.get("unknown_candidate_bigrams") or []),
                "shock_direction": reaction.get("shock_direction"),
                "shock_threshold": reaction.get("shock_threshold"),
                "note": reaction.get("note"),
            }
        )
        self._write_event(event)

    def record_valuation_update(self, *, now_ms: int, state: dict[str, Any], cash: int | None, source: str, details: dict[str, Any]) -> None:
        event = self._base_event(
            "valuation_updated",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["source"] = source
        event["old_fair_value"] = details.get("old_fair_value")
        event["new_fair_value"] = details.get("new_fair_value")
        event["earnings_value"] = details.get("earnings_value")
        event["valuation_status"] = details.get("status")
        event["valuation_method"] = details.get("method")
        event["valuation_reason"] = details.get("reason")
        event["estimate"] = details.get("estimate")
        event["details"] = details
        self._write_event(event)

    def record_signal(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        strategy_family: str,
        action_class: str,
        pnl_owner: str,
        signal_id: str,
        trade_group_id: str,
        leg_role: str,
        payload: dict[str, Any],
    ) -> None:
        event = self._base_event(
            "derived_signal",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "strategy_family": strategy_family,
                "action_class": action_class,
                "pnl_owner": pnl_owner,
                "signal_id": signal_id,
                "trade_group_id": trade_group_id,
                "leg_role": leg_role,
                "payload": payload,
            }
        )
        self._write_event(event)

    def record_decision(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str, plan) -> None:
        desired_bid = None if plan.bid is None else self._serialize_order(plan.bid.__dict__)
        desired_ask = None if plan.ask is None else self._serialize_order(plan.ask.__dict__)
        aggressive_actions = [self._serialize_order(action.__dict__) for action in plan.aggressive_actions]
        event = self._base_event(
            "decision_evaluated",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=plan.mode,
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "trigger": trigger,
                "observe_only": plan.observe_only,
                "reason": plan.reason,
                "desired_bid": desired_bid,
                "desired_ask": desired_ask,
                "aggressive_actions": aggressive_actions,
                "aggressive_action_count": len(aggressive_actions),
                "quoted_spread": _quoted_spread(desired_bid, desired_ask, list(state.get("live_orders") or [])),
            }
        )
        self._write_event(event)
        self._snapshot_writer.write_row(self._snapshot_row(event))

    def maybe_record_periodic_snapshot(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str) -> None:
        symbol = str(state.get("symbol") or self.symbol)
        last_snapshot_ms = self._last_snapshot_ms_by_symbol.get(symbol)
        if last_snapshot_ms is not None and now_ms - last_snapshot_ms < self.trace_config.trace_snapshot_interval_ms:
            return
        event = self._base_event(
            "session_state_snapshot",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=symbol,
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "trigger": trigger,
                "observe_only": False,
                "reason": "periodic snapshot",
                "desired_bid": None,
                "desired_ask": None,
                "aggressive_actions": [],
                "aggressive_action_count": 0,
                "quoted_spread": _quoted_spread(None, None, list(state.get("live_orders") or [])),
            }
        )
        self._write_event(event)
        self._snapshot_writer.write_row(self._snapshot_row(event))
        self._last_snapshot_ms_by_symbol[symbol] = now_ms

    def record_order_submitted(self, *, now_ms: int, state: dict[str, Any], cash: int | None, order: dict[str, Any]) -> None:
        event = self._base_event(
            "order_submitted",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "order_id": order.get("order_id"),
                "side": order.get("side"),
                "price": order.get("px"),
                "qty": order.get("qty"),
                "remaining_qty": order.get("remaining_qty"),
                "overlay": order.get("overlay"),
                "aggressive": order.get("aggressive"),
                "intent": order.get("intent"),
                "mode_at_submit": order.get("mode_at_submit"),
                "reason": order.get("evaluation_reason") or order.get("reason"),
                "market_key": order.get("market_key") or event.get("market_key"),
                "strategy_family": order.get("strategy_family"),
                "action_class": order.get("action_class"),
                "pnl_owner": order.get("pnl_owner"),
                "signal_id": order.get("signal_id"),
                "trade_group_id": order.get("trade_group_id"),
                "leg_role": order.get("leg_role"),
                "mid_at_event": event.get("mid"),
                "fair_value_at_event": event.get("fair_value"),
            }
        )
        self._write_event(event)

    def record_cancel_requested(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        order_id: str,
        side: str,
        cancel_reason: str,
        mode_at_cancel: str,
        order: dict[str, Any] | None = None,
    ) -> None:
        event = self._base_event(
            "order_cancel_requested",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "order_id": order_id,
                "side": side,
                "price": None if order is None else order.get("px"),
                "qty": None if order is None else order.get("qty"),
                "remaining_qty": None if order is None else order.get("remaining_qty"),
                "overlay": None if order is None else order.get("overlay"),
                "aggressive": None if order is None else order.get("aggressive"),
                "intent": None if order is None else order.get("intent"),
                "cancel_reason": cancel_reason,
                "mode_at_submit": mode_at_cancel,
                "market_key": None if order is None else order.get("market_key"),
                "strategy_family": None if order is None else order.get("strategy_family"),
                "action_class": None if order is None else order.get("action_class"),
                "pnl_owner": None if order is None else order.get("pnl_owner"),
                "signal_id": None if order is None else order.get("signal_id"),
                "trade_group_id": None if order is None else order.get("trade_group_id"),
                "leg_role": None if order is None else order.get("leg_role"),
            }
        )
        self._write_event(event)

    def record_cancel_response(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        order_id: str,
        success: bool,
        error: str | None,
        side: str | None = None,
        order: dict[str, Any] | None = None,
    ) -> None:
        event = self._base_event(
            "order_cancel_response",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "order_id": order_id,
                "side": side,
                "price": None if order is None else order.get("px"),
                "qty": None if order is None else order.get("qty"),
                "remaining_qty": None if order is None else order.get("remaining_qty"),
                "overlay": None if order is None else order.get("overlay"),
                "aggressive": None if order is None else order.get("aggressive"),
                "intent": None if order is None else order.get("intent"),
                "market_key": None if order is None else order.get("market_key"),
                "strategy_family": None if order is None else order.get("strategy_family"),
                "action_class": None if order is None else order.get("action_class"),
                "pnl_owner": None if order is None else order.get("pnl_owner"),
                "signal_id": None if order is None else order.get("signal_id"),
                "trade_group_id": None if order is None else order.get("trade_group_id"),
                "leg_role": None if order is None else order.get("leg_role"),
                "success": bool(success),
                "error": error,
            }
        )
        self._write_event(event)

    def record_fill(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        order: dict[str, Any] | None,
        order_id: str,
        qty: int,
        price: int,
    ) -> None:
        event = self._base_event(
            "order_filled",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "order_id": order_id,
                "side": None if order is None else order.get("side"),
                "price": int(price),
                "fill_price": int(price),
                "qty": int(qty),
                "fill_qty": int(qty),
                "remaining_qty": None if order is None else order.get("remaining_qty"),
                "overlay": None if order is None else order.get("overlay"),
                "aggressive": None if order is None else order.get("aggressive"),
                "intent": None if order is None else order.get("intent"),
                "mode_at_submit": None if order is None else order.get("mode_at_submit"),
                "reason": None if order is None else order.get("evaluation_reason"),
                "market_key": None if order is None else order.get("market_key"),
                "strategy_family": None if order is None else order.get("strategy_family"),
                "action_class": None if order is None else order.get("action_class"),
                "pnl_owner": None if order is None else order.get("pnl_owner"),
                "signal_id": None if order is None else order.get("signal_id"),
                "trade_group_id": None if order is None else order.get("trade_group_id"),
                "leg_role": None if order is None else order.get("leg_role"),
                "mid_at_event": event.get("mid"),
                "mid_price": event.get("mid"),
                "fair_value_at_event": event.get("fair_value"),
                "inventory_after_fill": state.get("inventory"),
            }
        )
        self._write_event(event)

    def record_rejection(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        order_id: str,
        reason: str,
        order: dict[str, Any] | None = None,
    ) -> None:
        event = self._base_event(
            "order_rejected",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event.update(
            {
                "order_id": order_id,
                "side": None if order is None else order.get("side"),
                "price": None if order is None else order.get("px"),
                "qty": None if order is None else order.get("qty"),
                "remaining_qty": None if order is None else order.get("remaining_qty"),
                "overlay": None if order is None else order.get("overlay"),
                "aggressive": None if order is None else order.get("aggressive"),
                "intent": None if order is None else order.get("intent"),
                "rejection_reason": reason,
                "market_key": None if order is None else order.get("market_key"),
                "strategy_family": None if order is None else order.get("strategy_family"),
                "action_class": None if order is None else order.get("action_class"),
                "pnl_owner": None if order is None else order.get("pnl_owner"),
                "signal_id": None if order is None else order.get("signal_id"),
                "trade_group_id": None if order is None else order.get("trade_group_id"),
                "leg_role": None if order is None else order.get("leg_role"),
            }
        )
        self._write_event(event)

    def _snapshot_row(self, event: dict[str, Any]) -> dict[str, Any]:
        live_orders = list(event.get("live_orders") or [])
        live_bid = next((order for order in live_orders if order.get("side") == "BUY" and not order.get("cancel_pending")), None)
        live_ask = next((order for order in live_orders if order.get("side") == "SELL" and not order.get("cancel_pending")), None)
        desired_bid = event.get("desired_bid")
        desired_ask = event.get("desired_ask")
        return {
            "event_type": event.get("event_type"),
            "run_id": event.get("run_id"),
            "wall_time_iso": event.get("wall_time_iso"),
            "monotonic_ms": event.get("monotonic_ms"),
            "exchange_tick": event.get("exchange_tick"),
            "symbol": event.get("symbol"),
            "market_key": event.get("market_key"),
            "mode": event.get("mode"),
            "trigger": event.get("trigger"),
            "observe_only": event.get("observe_only"),
            "reason": event.get("reason"),
            "best_bid_px": event.get("best_bid_px"),
            "best_bid_qty": event.get("best_bid_qty"),
            "best_ask_px": event.get("best_ask_px"),
            "best_ask_qty": event.get("best_ask_qty"),
            "spread": event.get("spread"),
            "mid": event.get("mid"),
            "microprice": event.get("microprice"),
            "fair_value": event.get("fair_value"),
            "trusted_multiplier": event.get("trusted_multiplier"),
            "multiplier_confidence": event.get("multiplier_confidence"),
            "inventory": event.get("inventory"),
            "earnings_position": event.get("earnings_position"),
            "news_position": event.get("news_position"),
            "mm_position": event.get("mm_position"),
            "buy_exposure": event.get("buy_exposure"),
            "sell_exposure": event.get("sell_exposure"),
            "earnings_budget": event.get("earnings_budget"),
            "mm_budget": event.get("mm_budget"),
            "budget_shift_active": event.get("budget_shift_active"),
            "allowed_buy_size": event.get("allowed_buy_size"),
            "allowed_sell_size": event.get("allowed_sell_size"),
            "position_cap": event.get("position_cap"),
            "live_bid_px": None if live_bid is None else live_bid.get("px"),
            "live_bid_qty": None if live_bid is None else live_bid.get("remaining_qty"),
            "live_ask_px": None if live_ask is None else live_ask.get("px"),
            "live_ask_qty": None if live_ask is None else live_ask.get("remaining_qty"),
            "desired_bid_px": None if desired_bid is None else desired_bid.get("px"),
            "desired_bid_qty": None if desired_bid is None else desired_bid.get("qty"),
            "desired_ask_px": None if desired_ask is None else desired_ask.get("px"),
            "desired_ask_qty": None if desired_ask is None else desired_ask.get("qty"),
            "aggressive_action_count": event.get("aggressive_action_count"),
            "aggressive_actions_json": json.dumps(event.get("aggressive_actions") or [], sort_keys=True, default=_json_default),
            "quoted_spread": event.get("quoted_spread"),
            "cash": event.get("cash"),
            "mtm_pnl_estimate": event.get("mtm_pnl_estimate"),
            "mtm_basis": event.get("mtm_basis"),
            "mark_price": event.get("mark_price"),
            "news_caution_remaining_ms": event.get("news_caution_remaining_ms"),
        }

    def finalize(self, *, now_ms: int, state: dict[str, Any], cash: int | None, note: str | None = None) -> None:
        if self._closed:
            return
        event = self._base_event(
            "session_end",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
            symbol=state.get("symbol"),
        )
        event.update(self._state_fields(state, cash))
        event["note"] = note
        self._write_event(event)
        self._events_handle.flush()
        self._events_handle.close()
        self._snapshot_writer.close()
        if self.trace_config.trace_write_summary_on_shutdown:
            events = load_trace_events(self.run_dir)
            summary = summarize_trace_events(events, markout_windows_ms=self.trace_config.trace_markout_windows_ms)
            tracker_report = build_a_news_tracker_report(events, config=AConfig())
            unknown_terms_report = build_unknown_news_term_report(events)
            headline_rows = tracker_report.get("headline_analyses") or []
            recommendation_rows = tracker_report.get("term_recommendations") or []
            summary["a_news_summary"] = {
                "headline_count": len(headline_rows),
                "traded_count": sum(1 for row in headline_rows if row.get("traded")),
                "missed_no_trade_count": sum(1 for row in headline_rows if row.get("verdict") == "missed_no_trade"),
                "undersized_count": sum(1 for row in headline_rows if row.get("verdict") == "undersized"),
                "wrong_direction_count": sum(1 for row in headline_rows if row.get("verdict") == "wrong_direction"),
                "a_news_pnl": round(float((summary.get("pnl_by_strategy_family") or {}).get("a_news", 0.0)), 4),
                "fill_count": int(((summary.get("strategy_family_stats") or {}).get("a_news") or {}).get("fill_count", 0)),
                "fill_qty": int(((summary.get("strategy_family_stats") or {}).get("a_news") or {}).get("fill_qty", 0)),
                "top_recommendations": [
                    {
                        "term": row.get("term"),
                        "suggested_action": row.get("suggested_action"),
                        "suggested_weight_delta": row.get("suggested_weight_delta"),
                    }
                    for row in recommendation_rows
                    if row.get("suggested_action") not in {None, "review"}
                ][:10],
            }
            with self.summary_json_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
            with self.summary_md_path.open("w", encoding="utf-8") as handle:
                handle.write(render_summary_markdown(summary, self.run_id, self.run_dir))
            with self.a_news_tracker_json_path.open("w", encoding="utf-8") as handle:
                json.dump(tracker_report, handle, indent=2, sort_keys=True, default=_json_default)
            with self.a_news_tracker_md_path.open("w", encoding="utf-8") as handle:
                handle.write(render_a_news_tracker_markdown(tracker_report, self.run_dir))
            with self.unknown_a_news_terms_json_path.open("w", encoding="utf-8") as handle:
                json.dump(unknown_terms_report, handle, indent=2, sort_keys=True, default=_json_default)
            with self.unknown_a_news_terms_md_path.open("w", encoding="utf-8") as handle:
                handle.write(render_unknown_terms_markdown(unknown_terms_report, self.run_dir))
        self._closed = True
