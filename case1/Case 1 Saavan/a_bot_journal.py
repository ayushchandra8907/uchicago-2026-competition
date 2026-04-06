from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from a_bot_strategy import ManagedOrder


@dataclass(frozen=True)
class JournalReplayState:
    multiplier: float | None
    multiplier_confidence: int
    fair_value: int | None
    earnings_value: float | None
    inventory: int
    live_orders: tuple[ManagedOrder, ...]


class TradingJournal:
    """Append-only JSONL journal used to recover local bot state after a crash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, event_type: str, **payload) -> None:
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

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
        multiplier: float | None = None
        multiplier_confidence = 0
        fair_value: int | None = None
        earnings_value: float | None = None
        inventory = 0
        live_orders: dict[str, ManagedOrder] = {}

        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

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

        # Restored orders are treated as potentially live because the exchange API
        # gives us a position snapshot but not an open-order snapshot on reconnect.
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
