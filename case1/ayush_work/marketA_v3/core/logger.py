from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any
import uuid

from ..config import LoggerConfig


DECISION_FIELDNAMES = [
    "event_type",
    "run_id",
    "wall_time_iso",
    "monotonic_ms",
    "exchange_tick",
    "message_index",
    "symbol",
    "event_symbol",
    "mode",
    "trigger",
    "reason",
    "observe_only",
    "inventory",
    "target_inventory",
    "best_bid_px",
    "best_bid_qty",
    "best_ask_px",
    "best_ask_qty",
    "mid",
    "spread",
    "fair_value",
    "base_fair_value",
    "news_fair_value",
    "trusted_multiplier",
    "latest_earnings",
    "fair_change_ticks",
    "active_signal_kind",
    "pe_frozen",
    "clean_multiplier_sample_count",
    "equilibrium_reached",
    "current_edge_ticks",
    "overshoot_active",
    "overshoot_stage_index",
    "overshoot_trimmed_qty_total",
    "overshoot_trigger_ticks",
    "news_sentiment_score",
    "news_sentiment_bucket",
    "news_matched_phrases",
    "news_matched_unigrams",
    "news_matched_bigrams",
    "unknown_candidate_phrases",
    "unknown_candidate_unigrams",
    "unknown_candidate_bigrams",
    "pending_news",
    "pending_news_target_inventory",
    "news_confirmation_state",
    "news_confirmation_deadline_ms",
    "news_takeover_started_ms",
    "posterior_hike",
    "posterior_hold",
    "posterior_cut",
    "terminal_hike",
    "terminal_hold",
    "terminal_cut",
    "prior_hike",
    "prior_hold",
    "prior_cut",
    "market_prior_hike",
    "market_prior_hold",
    "market_prior_cut",
    "memory_logit_hike",
    "memory_logit_hold",
    "memory_logit_cut",
    "dominant_regime",
    "fair_value_hike",
    "fair_value_hold",
    "fair_value_cut",
    "expected_rate_delta_bp",
    "rate_macro_event_id",
    "rate_signal_source",
    "rate_no_trade_reason",
    "rate_relevance_score",
    "rate_bucket",
    "rate_target_symbol",
    "rate_target_inventory",
    "rate_chosen_edge_ticks",
    "rate_no_arb_gap_ticks",
    "rate_dominant_regime",
    "rate_contrary_signal_score",
    "rate_regime_break_score",
    "rate_long_edge_hike",
    "rate_long_edge_hold",
    "rate_long_edge_cut",
    "rate_short_edge_hike",
    "rate_short_edge_hold",
    "rate_short_edge_cut",
    "rate_hawk_score",
    "rate_hold_score",
    "rate_cut_score",
    "rate_matched_phrases",
    "rate_matched_unigrams",
    "rate_matched_bigrams",
    "rate_matched_hike_terms",
    "rate_matched_hold_terms",
    "rate_matched_cut_terms",
    "rate_unknown_candidate_phrases",
    "rate_unknown_candidate_unigrams",
    "rate_unknown_candidate_bigrams",
    "rate_baseline_targets_by_symbol",
    "rate_macro_targets_by_symbol",
    "rate_macro_pair_targets_by_symbol",
    "rate_trading_phase_targets_by_symbol",
    "rate_probe_targets_by_symbol",
    "rate_reversion_targets_by_symbol",
    "rate_pair_targets_by_symbol",
    "rate_endgame_targets_by_symbol",
    "rate_final_phase_targets_by_symbol",
    "rate_combined_targets_by_symbol",
    "rate_macro_pair_symbols",
    "rate_macro_leg_reference_mids",
    "rate_macro_leg_fairs",
    "rate_macro_leg_bucket",
    "rate_reversion_active_symbols",
    "rate_reversion_entry_px_by_symbol",
    "rate_reversion_reason_by_symbol",
    "rate_pair_active_pair",
    "rate_pair_entry_px_by_symbol",
    "rate_pair_reason_by_symbol",
    "rate_pair_move_by_symbol",
    "rate_pair_last_event_id",
    "rate_pair_last_event_kind",
    "rate_pair_last_event_pair",
    "rate_pair_last_event_reason",
    "rate_reversion_last_event_id",
    "rate_reversion_last_event_kind",
    "rate_reversion_last_event_symbol",
    "rate_reversion_last_event_reason",
    "rate_reversion_last_entry_px",
    "rate_reversion_last_exit_px",
    "desired_symbol",
    "desired_side",
    "desired_px",
    "desired_qty",
    "desired_intent",
    "cash",
    "mtm_pnl_estimate",
    "shock_pnl",
    "mm_pnl",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _wall_clock_fields() -> tuple[str, int]:
    wall_time_ns = time.time_ns()
    wall_iso = datetime.fromtimestamp(wall_time_ns / 1_000_000_000, tz=timezone.utc).isoformat()
    return wall_iso, wall_time_ns


class AppendSafeCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        needs_header = (not file_exists) or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            self._writer.writeheader()
            self._handle.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def estimate_mtm(*, inventory: int, cash: int, mark_price: float | None) -> float | None:
    if mark_price is None:
        return float(cash)
    return float(cash) + (float(inventory) * float(mark_price))


class RunLogger:
    def __init__(self, config: LoggerConfig, *, session_prefix: str = "marketA_v3", symbol: str = "A"):
        self.config = config
        self.symbol = symbol
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = uuid.uuid4().hex
        root = (config.run_root or Path.cwd() / "analysis_runs").resolve()
        self.run_dir = root / f"{session_prefix}_{timestamp}_live_round"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "trace_events.jsonl"
        self.snapshots_path = self.run_dir / "decision_snapshots.csv"
        self._events_handle = self.events_path.open("a", encoding="utf-8")
        self._snapshot_writer = AppendSafeCsvWriter(self.snapshots_path, DECISION_FIELDNAMES) if config.write_decision_snapshots else None
        self._queue: Queue[tuple[str, dict[str, Any] | None]] = Queue(maxsize=max(1, config.queue_max_events))
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, name="marketA-v3-logger", daemon=True)
        self._writer.start()
        self._closed = False

    @classmethod
    def create_if_enabled(cls, config: LoggerConfig, *, session_prefix: str = "marketA_v3", symbol: str = "A") -> "RunLogger | None":
        if not config.enabled:
            return None
        return cls(config, session_prefix=session_prefix, symbol=symbol)

    def _writer_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                kind, payload = self._queue.get(timeout=0.1)
            except Empty:
                continue
            if kind == "stop":
                self._queue.task_done()
                continue
            if kind == "event":
                self._events_handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
                self._events_handle.flush()
            elif kind == "snapshot" and self._snapshot_writer is not None:
                self._snapshot_writer.write_row(payload or {})
            self._queue.task_done()

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except Full:
            if kind == "event":
                self._events_handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
                self._events_handle.flush()
            elif kind == "snapshot" and self._snapshot_writer is not None:
                self._snapshot_writer.write_row(payload)

    def _base_event(
        self,
        event_type: str,
        *,
        now_ms: int,
        exchange_tick: int | None = None,
        message_index: int | None = None,
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
            "message_index": message_index,
            "symbol": self.symbol if symbol is None else symbol,
        }

    def record_event(
        self,
        event_type: str,
        *,
        now_ms: int,
        exchange_tick: int | None = None,
        message_index: int | None = None,
        symbol: str | None = None,
        **fields: Any,
    ) -> None:
        payload = self._base_event(
            event_type,
            now_ms=now_ms,
            exchange_tick=exchange_tick,
            message_index=message_index,
            symbol=symbol,
        )
        payload.update(fields)
        self._enqueue("event", payload)

    def record_decision_snapshot(
        self,
        *,
        now_ms: int,
        exchange_tick: int | None,
        message_index: int | None,
        row: dict[str, Any],
        symbol: str | None = None,
    ) -> None:
        event_row = self._base_event(
            "decision_evaluated",
            now_ms=now_ms,
            exchange_tick=exchange_tick,
            message_index=message_index,
            symbol=symbol,
        )
        event_row.update(row)
        self._enqueue("event", event_row)
        if self._snapshot_writer is not None:
            self._enqueue("snapshot", event_row)

    def flush(self) -> None:
        self._queue.join()
        self._events_handle.flush()
        if self._snapshot_writer is not None:
            self._snapshot_writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._stop.set()
        try:
            self._queue.put_nowait(("stop", None))
        except Full:
            pass
        self._writer.join(timeout=5.0)
        self._events_handle.flush()
        self._events_handle.close()
        if self._snapshot_writer is not None:
            self._snapshot_writer.close()


def load_trace_events(path_or_run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_run_dir).expanduser().resolve()
    run_dir = path.parent if path.is_file() else path
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
