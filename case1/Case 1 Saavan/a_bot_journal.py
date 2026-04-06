from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from a_bot_strategy import ManagedOrder


JournalStartupMode = Literal["clean_start", "crash_recovery", "finished_session_ignored"]


@dataclass(frozen=True)
class JournalReplayState:
    multiplier: float | None
    multiplier_confidence: int
    fair_value: int | None
    earnings_value: float | None
    inventory: int
    live_orders: tuple[ManagedOrder, ...]
    startup_mode: JournalStartupMode = "clean_start"
    archived_path: Path | None = None


class TradingJournal:
    """Append-only JSONL journal used to recover local bot state after a crash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._session_started = False

    def append(self, event_type: str, **payload) -> None:
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def prepare_for_startup(self) -> JournalReplayState:
        replay_state = self.load_replay_state()
        archived_path = self._archive_existing_journal()
        replay_with_archive = JournalReplayState(
            multiplier=replay_state.multiplier,
            multiplier_confidence=replay_state.multiplier_confidence,
            fair_value=replay_state.fair_value,
            earnings_value=replay_state.earnings_value,
            inventory=replay_state.inventory,
            live_orders=replay_state.live_orders,
            startup_mode=replay_state.startup_mode,
            archived_path=archived_path,
        )
        self.record_session_started(
            startup_mode=replay_with_archive.startup_mode,
            recovered_orders=len(replay_with_archive.live_orders),
            archived_path=None if archived_path is None else str(archived_path),
        )
        if replay_with_archive.live_orders:
            for order in replay_with_archive.live_orders:
                self.record_order_submitted(order)
            if replay_with_archive.multiplier is not None:
                self.record_multiplier(
                    replay_with_archive.multiplier,
                    confidence=replay_with_archive.multiplier_confidence,
                    source="recovered_journal_seed",
                )
            if replay_with_archive.fair_value is not None:
                self.record_fair_value(
                    fair_value=replay_with_archive.fair_value,
                    source="recovered_journal_seed",
                    earnings_value=replay_with_archive.earnings_value,
                )
        return replay_with_archive

    def record_session_started(
        self,
        *,
        startup_mode: JournalStartupMode,
        recovered_orders: int,
        archived_path: str | None = None,
    ) -> None:
        self._session_started = True
        self.append(
            "session_started",
            startup_mode=startup_mode,
            recovered_orders=int(recovered_orders),
            archived_path=archived_path,
        )

    def record_session_finished(self, note: str | None = None) -> None:
        if not self._session_started:
            return
        self.append("session_finished", note=note)
        self._session_started = False

    def record_order_submitted(self, order: ManagedOrder) -> None:
        self.append(
            "order_submitted",
            order_id=order.order_id,
            side=order.side,
            px=order.px,
            qty=order.qty,
            remaining_qty=order.remaining_qty,
            submitted_ms=order.submitted_ms,
            overlay=order.overlay,
            aggressive=order.aggressive,
            intent=order.intent,
            mode_at_submit=order.mode_at_submit,
            evaluation_reason=order.evaluation_reason,
        )

    def record_cancel_requested(self, order_id: str) -> None:
        self.append("cancel_requested", order_id=order_id)

    def record_cancel_response(self, order_id: str, success: bool, error: str | None = None) -> None:
        self.append("cancel_response", order_id=order_id, success=success, error=error)

    def record_fill(self, order_id: str, qty: int, price: int) -> None:
        self.append("order_fill", order_id=order_id, qty=qty, price=price)

    def record_rejection(self, order_id: str, reason: str) -> None:
        self.append("order_rejected", order_id=order_id, reason=reason)

    def record_fair_value(
        self,
        fair_value: int,
        source: str,
        earnings_value: float | None = None,
    ) -> None:
        self.append(
            "fair_value_updated",
            fair_value=int(fair_value),
            source=source,
            earnings_value=earnings_value,
        )

    def record_multiplier(
        self,
        multiplier: float,
        confidence: int,
        source: str,
        estimate: float | None = None,
        method: str | None = None,
    ) -> None:
        self.append(
            "multiplier_updated",
            multiplier=float(multiplier),
            confidence=int(confidence),
            source=source,
            estimate=estimate,
            method=method,
        )

    def record_inventory(self, inventory: int, cash: int | None = None) -> None:
        self.append("inventory_updated", inventory=int(inventory), cash=cash)

    def load_replay_state(self) -> JournalReplayState:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return JournalReplayState(
                multiplier=None,
                multiplier_confidence=0,
                fair_value=None,
                earnings_value=None,
                inventory=0,
                live_orders=(),
                startup_mode="clean_start",
            )

        records = self._load_records()
        if not records:
            return JournalReplayState(
                multiplier=None,
                multiplier_confidence=0,
                fair_value=None,
                earnings_value=None,
                inventory=0,
                live_orders=(),
                startup_mode="clean_start",
            )

        if not any(record.get("event_type") == "session_started" for record in records):
            legacy_state = self._replay_records(records)
            startup_mode: JournalStartupMode = "crash_recovery" if legacy_state.live_orders else "clean_start"
            return JournalReplayState(
                multiplier=legacy_state.multiplier,
                multiplier_confidence=legacy_state.multiplier_confidence,
                fair_value=legacy_state.fair_value,
                earnings_value=legacy_state.earnings_value,
                inventory=legacy_state.inventory,
                live_orders=legacy_state.live_orders if startup_mode == "crash_recovery" else (),
                startup_mode=startup_mode,
            )

        sessions = self._split_sessions(records)
        if not sessions:
            return JournalReplayState(
                multiplier=None,
                multiplier_confidence=0,
                fair_value=None,
                earnings_value=None,
                inventory=0,
                live_orders=(),
                startup_mode="clean_start",
            )

        latest_session = sessions[-1]
        session_state = self._replay_records(latest_session["records"])
        if latest_session["finished"]:
            return JournalReplayState(
                multiplier=None,
                multiplier_confidence=0,
                fair_value=None,
                earnings_value=None,
                inventory=0,
                live_orders=(),
                startup_mode="finished_session_ignored",
            )

        startup_mode = "crash_recovery" if session_state.live_orders else "clean_start"
        return JournalReplayState(
            multiplier=session_state.multiplier,
            multiplier_confidence=session_state.multiplier_confidence,
            fair_value=session_state.fair_value,
            earnings_value=session_state.earnings_value,
            inventory=session_state.inventory,
            live_orders=session_state.live_orders if startup_mode == "crash_recovery" else (),
            startup_mode=startup_mode,
        )

    def _load_records(self) -> list[dict]:
        records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _split_sessions(records: list[dict]) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for record in records:
            event_type = record.get("event_type")
            if event_type == "session_started":
                if current is not None:
                    sessions.append(current)
                current = {"records": [record], "finished": False}
                continue
            if current is None:
                current = {"records": [record], "finished": False}
            else:
                current["records"].append(record)
            if event_type == "session_finished":
                current["finished"] = True
                sessions.append(current)
                current = None
        if current is not None:
            sessions.append(current)
        return sessions

    def _replay_records(self, records: list[dict]) -> JournalReplayState:
        multiplier: float | None = None
        multiplier_confidence = 0
        fair_value: int | None = None
        earnings_value: float | None = None
        inventory = 0
        live_orders: dict[str, ManagedOrder] = {}

        for record in records:
            event_type = record.get("event_type")
            order_id = record.get("order_id")
            if event_type == "order_submitted" and order_id:
                live_orders[order_id] = ManagedOrder(
                    order_id=order_id,
                    side=record["side"],
                    px=int(record["px"]),
                    qty=int(record["qty"]),
                    remaining_qty=int(record.get("remaining_qty", record["qty"])),
                    submitted_ms=int(record.get("submitted_ms", 0)),
                    overlay=str(record.get("overlay", "mm")),
                    aggressive=bool(record.get("aggressive", False)),
                    restored=True,
                    intent=str(record.get("intent", "")),
                    mode_at_submit=str(record.get("mode_at_submit", "")),
                    evaluation_reason=str(record.get("evaluation_reason", "")),
                )
            elif event_type == "cancel_requested" and order_id in live_orders:
                live_orders[order_id].cancel_pending = True
            elif event_type == "cancel_response" and order_id in live_orders:
                if record.get("success"):
                    live_orders.pop(order_id, None)
                else:
                    live_orders[order_id].cancel_pending = False
            elif event_type == "order_fill" and order_id in live_orders:
                live_orders[order_id].remaining_qty = max(
                    0,
                    live_orders[order_id].remaining_qty - int(record["qty"]),
                )
                if live_orders[order_id].remaining_qty == 0:
                    live_orders.pop(order_id, None)
            elif event_type == "order_rejected" and order_id in live_orders:
                live_orders.pop(order_id, None)
            elif event_type == "multiplier_updated":
                multiplier = float(record["multiplier"])
                multiplier_confidence = int(record.get("confidence", 0))
            elif event_type == "fair_value_updated":
                fair_value = int(record["fair_value"])
                raw_earnings = record.get("earnings_value")
                earnings_value = None if raw_earnings is None else float(raw_earnings)
            elif event_type == "inventory_updated":
                inventory = int(record["inventory"])

        restored = []
        for order in live_orders.values():
            restored.append(
                ManagedOrder(
                    order_id=order.order_id,
                    side=order.side,
                    px=order.px,
                    qty=order.qty,
                    remaining_qty=order.remaining_qty,
                    submitted_ms=order.submitted_ms,
                    overlay=order.overlay,
                    aggressive=order.aggressive,
                    cancel_pending=False,
                    restored=True,
                    intent=order.intent,
                    mode_at_submit=order.mode_at_submit,
                    evaluation_reason=order.evaluation_reason,
                )
            )

        return JournalReplayState(
            multiplier=multiplier,
            multiplier_confidence=multiplier_confidence,
            fair_value=fair_value,
            earnings_value=earnings_value,
            inventory=inventory,
            live_orders=tuple(restored),
        )

    def _archive_existing_journal(self) -> Path | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.touch(exist_ok=True)
            return None
        archive_dir = self.path.parent / "journal_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive_path = archive_dir / f"{self.path.stem}_{timestamp}{self.path.suffix}"
        suffix = 1
        while archive_path.exists():
            archive_path = archive_dir / f"{self.path.stem}_{timestamp}_{suffix}{self.path.suffix}"
            suffix += 1
        self.path.replace(archive_path)
        self.path.touch()
        return archive_path


def select_recovered_pricing_state(
    replay_state: JournalReplayState,
    *,
    recover_pricing_state: bool,
) -> tuple[float | None, int, int | None, float | None]:
    if not recover_pricing_state or not replay_state.live_orders:
        return None, 0, None, None
    return (
        replay_state.multiplier,
        replay_state.multiplier_confidence,
        replay_state.fair_value,
        replay_state.earnings_value,
    )
