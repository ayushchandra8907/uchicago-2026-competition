from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable
import uuid

from a_bot_config import TraceConfig


SNAPSHOT_FIELDNAMES = [
    "event_type",
    "run_id",
    "wall_time_iso",
    "monotonic_ms",
    "exchange_tick",
    "symbol",
    "mode",
    "earnings_phase",
    "mm_phase",
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
    "discovery_contaminated",
    "inventory",
    "earnings_position",
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
    "handler_duration_ms",
    "evaluate_sync_duration_ms",
    "order_sync_duration_ms",
    "cash",
    "mtm_pnl_estimate",
    "mtm_basis",
    "mark_price",
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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


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
    snapshot_events = [
        event
        for event in event_list
        if event.get("event_type") in {"decision_evaluated", "session_state_snapshot"}
    ]
    mode_durations_ms: dict[str, int] = defaultdict(int)
    last_mode: str | None = None
    last_mode_started_ms: int | None = None

    fills_by_intent: Counter[str] = Counter()
    submits_by_intent: Counter[str] = Counter()
    cancel_reason_counts: Counter[str] = Counter()
    reject_reason_counts: Counter[str] = Counter()
    no_action_reasons: Counter[str] = Counter()
    passive_fills = 0
    aggressive_fills = 0
    spread_cross_count = 0
    observe_only_count = 0
    inventory_values: list[int] = []
    earnings_inventory_values: list[int] = []
    mm_inventory_values: list[int] = []
    quoted_spreads: list[int] = []
    handler_durations_ms: list[float] = []
    evaluate_sync_durations_ms: list[float] = []
    order_sync_durations_ms: list[float] = []
    latest_mtm: float | None = None
    latest_mtm_basis: str | None = None
    fills_by_overlay: Counter[str] = Counter()
    submits_by_overlay: Counter[str] = Counter()
    budget_shift_active_ms = 0
    last_budget_shift_state: bool | None = None
    last_budget_shift_started_ms: int | None = None

    def nearest_snapshot_mid(target_ms: int) -> float | None:
        for snapshot in snapshot_events:
            if int(snapshot.get("monotonic_ms", 0)) >= target_ms:
                mid = snapshot.get("mid")
                return None if mid is None else float(mid)
        return None

    fill_markouts_by_intent: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for event in event_list:
        mode = event.get("mode")
        now_ms = int(event.get("monotonic_ms", 0))
        if mode:
            if last_mode is None:
                last_mode = str(mode)
                last_mode_started_ms = now_ms
            elif mode != last_mode:
                if last_mode_started_ms is not None:
                    mode_durations_ms[last_mode] += max(0, now_ms - last_mode_started_ms)
                last_mode = str(mode)
                last_mode_started_ms = now_ms

        budget_shift_active = bool(event.get("budget_shift_active"))
        if last_budget_shift_state is None:
            last_budget_shift_state = budget_shift_active
            last_budget_shift_started_ms = now_ms
        elif budget_shift_active != last_budget_shift_state:
            if last_budget_shift_state and last_budget_shift_started_ms is not None:
                budget_shift_active_ms += max(0, now_ms - last_budget_shift_started_ms)
            last_budget_shift_state = budget_shift_active
            last_budget_shift_started_ms = now_ms

        if event.get("event_type") in {"decision_evaluated", "session_state_snapshot"}:
            inventory = event.get("inventory")
            if inventory is not None:
                inventory_values.append(int(inventory))
            earnings_inventory = event.get("earnings_position")
            if earnings_inventory is not None:
                earnings_inventory_values.append(int(earnings_inventory))
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
            handler_duration = event.get("handler_duration_ms")
            if handler_duration is not None:
                handler_durations_ms.append(float(handler_duration))
            evaluate_sync_duration = event.get("evaluate_sync_duration_ms")
            if evaluate_sync_duration is not None:
                evaluate_sync_durations_ms.append(float(evaluate_sync_duration))
            order_sync_duration = event.get("order_sync_duration_ms")
            if order_sync_duration is not None:
                order_sync_durations_ms.append(float(order_sync_duration))

        if event.get("event_type") == "decision_evaluated":
            if bool(event.get("observe_only")):
                observe_only_count += 1
            if int(event.get("aggressive_action_count") or 0) == 0 and event.get("desired_bid") is None and event.get("desired_ask") is None:
                no_action_reasons[str(event.get("reason") or "unknown")] += 1

        elif event.get("event_type") == "order_submitted":
            intent = str(event.get("intent") or "unknown")
            submits_by_intent[intent] += 1
            submits_by_overlay[str(event.get("overlay") or "unknown")] += 1
            if bool(event.get("aggressive")):
                spread_cross_count += 1

        elif event.get("event_type") == "order_cancel_requested":
            cancel_reason_counts[str(event.get("cancel_reason") or "unknown")] += 1

        elif event.get("event_type") == "order_rejected":
            reject_reason_counts[str(event.get("rejection_reason") or "unknown")] += 1

        elif event.get("event_type") == "order_filled":
            intent = str(event.get("intent") or "unknown")
            fills_by_intent[intent] += 1
            fills_by_overlay[str(event.get("overlay") or "unknown")] += 1
            if bool(event.get("aggressive")):
                aggressive_fills += 1
            else:
                passive_fills += 1
            event_mid = event.get("mid_at_event")
            event_px = event.get("price")
            if event_mid is not None and event_px is not None:
                signed_side = 1 if str(event.get("side")) == "BUY" else -1
                for window_ms in markout_windows_ms:
                    future_mid = nearest_snapshot_mid(int(event.get("monotonic_ms", 0)) + window_ms)
                    if future_mid is None:
                        continue
                    markout = signed_side * (future_mid - float(event_px))
                    fill_markouts_by_intent[intent][f"{window_ms}ms"].append(markout)

    if last_mode is not None and last_mode_started_ms is not None and event_list:
        final_ms = int(event_list[-1].get("monotonic_ms", 0))
        mode_durations_ms[last_mode] += max(0, final_ms - last_mode_started_ms)
        if last_budget_shift_state and last_budget_shift_started_ms is not None:
            budget_shift_active_ms += max(0, final_ms - last_budget_shift_started_ms)

    fill_markout_summary: dict[str, dict[str, float]] = {}
    for intent, windows in fill_markouts_by_intent.items():
        fill_markout_summary[intent] = {}
        for label, values in windows.items():
            if values:
                fill_markout_summary[intent][label] = sum(values) / len(values)

    return {
        "total_events": len(event_list),
        "fills_total": sum(fills_by_intent.values()),
        "cancels_total": sum(cancel_reason_counts.values()),
        "rejects_total": sum(reject_reason_counts.values()),
        "passive_fills": passive_fills,
        "aggressive_fills": aggressive_fills,
        "fills_by_intent": dict(sorted(fills_by_intent.items())),
        "fills_by_overlay": dict(sorted(fills_by_overlay.items())),
        "submits_by_intent": dict(sorted(submits_by_intent.items())),
        "submits_by_overlay": dict(sorted(submits_by_overlay.items())),
        "cancel_reasons": dict(cancel_reason_counts.most_common()),
        "reject_reasons": dict(reject_reason_counts.most_common()),
        "mode_durations_ms": dict(sorted(mode_durations_ms.items())),
        "largest_inventory_long": max(inventory_values) if inventory_values else 0,
        "largest_inventory_short": min(inventory_values) if inventory_values else 0,
        "average_inventory": (sum(inventory_values) / len(inventory_values)) if inventory_values else 0.0,
        "average_earnings_inventory": (sum(earnings_inventory_values) / len(earnings_inventory_values)) if earnings_inventory_values else 0.0,
        "average_mm_inventory": (sum(mm_inventory_values) / len(mm_inventory_values)) if mm_inventory_values else 0.0,
        "average_quoted_spread": (sum(quoted_spreads) / len(quoted_spreads)) if quoted_spreads else None,
        "observe_only_count": observe_only_count,
        "most_common_no_action_reasons": dict(no_action_reasons.most_common()),
        "spread_cross_count": spread_cross_count,
        "budget_shift_active_ms": budget_shift_active_ms,
        "estimated_final_mtm_pnl": latest_mtm,
        "estimated_final_mtm_basis": latest_mtm_basis,
        "fill_markouts_by_intent": fill_markout_summary,
        "local_processing_durations_ms": {
            "handler": {
                "p50": _percentile(handler_durations_ms, 0.50),
                "p95": _percentile(handler_durations_ms, 0.95),
                "max": max(handler_durations_ms) if handler_durations_ms else None,
            },
            "evaluate_sync": {
                "p50": _percentile(evaluate_sync_durations_ms, 0.50),
                "p95": _percentile(evaluate_sync_durations_ms, 0.95),
                "max": max(evaluate_sync_durations_ms) if evaluate_sync_durations_ms else None,
            },
            "order_sync": {
                "p50": _percentile(order_sync_durations_ms, 0.50),
                "p95": _percentile(order_sync_durations_ms, 0.95),
                "max": max(order_sync_durations_ms) if order_sync_durations_ms else None,
            },
        },
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
    mode_durations = summary.get("mode_durations_ms") or {}
    no_action = summary.get("most_common_no_action_reasons") or {}
    local_processing = summary.get("local_processing_durations_ms") or {}
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
            f"- Average MM inventory: `{summary.get('average_mm_inventory', 0.0):.2f}`",
            f"- Average quoted spread: `{summary.get('average_quoted_spread')}`",
            f"- Observe-only decisions: `{summary.get('observe_only_count', 0)}`",
            f"- Budget shift active (ms): `{summary.get('budget_shift_active_ms', 0)}`",
            "",
            "## Local Processing Durations (ms)",
            *(
                f"- `{metric}`: `p50={stats.get('p50')}` `p95={stats.get('p95')}` `max={stats.get('max')}`"
                for metric, stats in local_processing.items()
            ),
            "",
            "## Fills By Intent",
            *(f"- `{intent}`: `{count}`" for intent, count in fills_by_intent.items()),
            "",
            "## Fills By Overlay",
            *(f"- `{overlay}`: `{count}`" for overlay, count in fills_by_overlay.items()),
            "",
            "## Mode Durations (ms)",
            *(f"- `{mode}`: `{duration}`" for mode, duration in mode_durations.items()),
            "",
            "## Top No-Action Reasons",
            *(f"- `{reason}`: `{count}`" for reason, count in no_action.items()),
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
        self._events_handle = self.events_path.open("a", encoding="utf-8")
        self._snapshot_writer = AppendSafeCsvWriter(self.snapshots_path, SNAPSHOT_FIELDNAMES)
        self._last_snapshot_ms: int | None = None
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
            "earnings_phase": state.get("earnings_phase"),
            "mm_phase": state.get("mm_phase"),
            "fair_value": state.get("fair_value"),
            "trusted_multiplier": state.get("trusted_multiplier"),
            "multiplier_confidence": state.get("multiplier_confidence"),
            "latest_earnings": state.get("latest_earnings"),
            "discovery_contaminated": state.get("discovery_contaminated"),
            "shock_direction": state.get("shock_direction"),
            "shock_threshold": state.get("shock_threshold"),
            "shock_target_fair": state.get("shock_target_fair"),
            "news_caution_active": state.get("news_caution_active"),
            "inventory": state.get("inventory"),
            "earnings_position": state.get("earnings_position"),
            "mm_position": state.get("mm_position"),
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
            "ms_until_next_earnings": state.get("ms_until_next_earnings"),
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
        }

    def record_session_start(self, *, now_ms: int, config_summary: dict[str, Any], recovered_orders: list[dict[str, Any]], state: dict[str, Any], cash: int | None) -> None:
        event = self._base_event(
            "session_start",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
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
        )
        event.update(self._state_fields(state, cash))
        event["reason"] = reason
        self._write_event(event)

    def record_book_update(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str) -> None:
        if self.trace_config.trace_detail_level != "full":
            return
        event = self._base_event(
            "book_update",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
        )
        event.update(self._state_fields(state, cash))
        event["trigger"] = trigger
        self._write_event(event)

    def record_market_trade(self, *, now_ms: int, state: dict[str, Any], cash: int | None, price: int, qty: int) -> None:
        if self.trace_config.trace_detail_level != "full":
            return
        event = self._base_event(
            "market_trade_observed",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
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

    def record_decision(
        self,
        *,
        now_ms: int,
        state: dict[str, Any],
        cash: int | None,
        trigger: str,
        plan,
        handler_duration_ms: float | None = None,
        evaluate_sync_duration_ms: float | None = None,
        order_sync_duration_ms: float | None = None,
    ) -> None:
        desired_bid = None if plan.bid is None else self._serialize_order(plan.bid.__dict__)
        desired_ask = None if plan.ask is None else self._serialize_order(plan.ask.__dict__)
        aggressive_actions = [self._serialize_order(action.__dict__) for action in plan.aggressive_actions]
        event = self._base_event(
            "decision_evaluated",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=plan.mode,
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
                "handler_duration_ms": handler_duration_ms,
                "evaluate_sync_duration_ms": evaluate_sync_duration_ms,
                "order_sync_duration_ms": order_sync_duration_ms,
            }
        )
        self._write_event(event)
        self._snapshot_writer.write_row(self._snapshot_row(event))

    def maybe_record_periodic_snapshot(self, *, now_ms: int, state: dict[str, Any], cash: int | None, trigger: str) -> None:
        if self._last_snapshot_ms is not None and now_ms - self._last_snapshot_ms < self.trace_config.trace_snapshot_interval_ms:
            return
        event = self._base_event(
            "session_state_snapshot",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
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
        self._last_snapshot_ms = now_ms

    def record_order_submitted(self, *, now_ms: int, state: dict[str, Any], cash: int | None, order: dict[str, Any]) -> None:
        event = self._base_event(
            "order_submitted",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
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
            "mode": event.get("mode"),
            "earnings_phase": event.get("earnings_phase"),
            "mm_phase": event.get("mm_phase"),
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
            "discovery_contaminated": event.get("discovery_contaminated"),
            "inventory": event.get("inventory"),
            "earnings_position": event.get("earnings_position"),
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
            "handler_duration_ms": event.get("handler_duration_ms"),
            "evaluate_sync_duration_ms": event.get("evaluate_sync_duration_ms"),
            "order_sync_duration_ms": event.get("order_sync_duration_ms"),
            "cash": event.get("cash"),
            "mtm_pnl_estimate": event.get("mtm_pnl_estimate"),
            "mtm_basis": event.get("mtm_basis"),
            "mark_price": event.get("mark_price"),
        }

    def finalize(self, *, now_ms: int, state: dict[str, Any], cash: int | None, note: str | None = None) -> None:
        if self._closed:
            return
        event = self._base_event(
            "session_end",
            now_ms=now_ms,
            exchange_tick=state.get("exchange_tick"),
            mode=state.get("mode"),
        )
        event.update(self._state_fields(state, cash))
        event["note"] = note
        self._write_event(event)
        self._events_handle.flush()
        self._events_handle.close()
        self._snapshot_writer.close()
        if self.trace_config.trace_write_summary_on_shutdown:
            summary = summarize_trace_events(load_trace_events(self.run_dir), markout_windows_ms=self.trace_config.trace_markout_windows_ms)
            with self.summary_json_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
            with self.summary_md_path.open("w", encoding="utf-8") as handle:
                handle.write(render_summary_markdown(summary, self.run_id, self.run_dir))
        self._closed = True
