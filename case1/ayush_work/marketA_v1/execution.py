from __future__ import annotations

from dataclasses import dataclass

from .models import OrderIntent, QuoteIntent, Side


@dataclass
class RestingOrder:
    side: Side
    px: int
    qty: int
    placed_time_ms: float
    queue_ahead_qty: int = 0
    remaining_qty: int | None = None
    order_id: str | None = None
    aggressive: bool = False
    mode: str = "NORMAL_MM"

    def __post_init__(self) -> None:
        if self.remaining_qty is None:
            self.remaining_qty = int(self.qty)


@dataclass(frozen=True)
class ExecutionDecision:
    cancels: tuple[RestingOrder, ...]
    placements: tuple[OrderIntent, ...]
    aggressive_actions: tuple[OrderIntent, ...]


class QuoteSynchronizer:
    def __init__(self, cooldown_ms: int) -> None:
        self.cooldown_ms = int(cooldown_ms)

    def sync(
        self,
        resting_orders: dict[Side, RestingOrder],
        intent: QuoteIntent | None,
        now_ms: float,
    ) -> ExecutionDecision:
        if intent is None:
            return ExecutionDecision(cancels=tuple(resting_orders.values()), placements=(), aggressive_actions=())

        desired = {"BUY": intent.bid, "SELL": intent.ask}
        cancels: list[RestingOrder] = []
        placements: list[OrderIntent] = []

        for side in ("BUY", "SELL"):
            current = resting_orders.get(side)
            target = desired[side]
            if current is None and target is not None:
                placements.append(target)
                continue
            if current is not None and target is None:
                cancels.append(current)
                continue
            if current is None or target is None:
                continue
            if current.px == target.px and current.remaining_qty == target.qty:
                continue
            if now_ms - current.placed_time_ms < self.cooldown_ms:
                continue
            cancels.append(current)
            placements.append(target)

        return ExecutionDecision(
            cancels=tuple(cancels),
            placements=tuple(placements),
            aggressive_actions=intent.aggressive_actions,
        )
