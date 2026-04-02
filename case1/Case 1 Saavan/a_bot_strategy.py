from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Literal

from a_bot_config import AConfig, RiskConfig


SideName = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class BookLevel:
    px: int
    qty: int


@dataclass(frozen=True)
class BookSnapshot:
    best_bid: BookLevel | None = None
    best_ask: BookLevel | None = None

    @classmethod
    def from_order_book(cls, book) -> "BookSnapshot":
        bids = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
        asks = [(int(px), int(qty)) for px, qty in book.asks.items() if qty > 0]
        best_bid_level = max(bids, key=lambda level: level[0]) if bids else None
        best_ask_level = min(asks, key=lambda level: level[0]) if asks else None
        best_bid = None if best_bid_level is None else BookLevel(px=best_bid_level[0], qty=best_bid_level[1])
        best_ask = None if best_ask_level is None else BookLevel(px=best_ask_level[0], qty=best_ask_level[1])
        return cls(best_bid=best_bid, best_ask=best_ask)


@dataclass(frozen=True)
class DesiredOrder:
    side: SideName
    px: int
    qty: int
    aggressive: bool = False
    reason: str = ""


@dataclass(frozen=True)
class QuotePlan:
    bid: DesiredOrder | None
    ask: DesiredOrder | None
    aggressive_actions: tuple[DesiredOrder, ...]
    observe_only: bool
    reason: str


@dataclass
class ManagedOrder:
    order_id: str
    side: SideName
    px: int
    qty: int
    remaining_qty: int
    submitted_ms: int
    aggressive: bool = False
    cancel_pending: bool = False
    restored: bool = False

    @property
    def is_active(self) -> bool:
        return self.remaining_qty > 0 and not self.cancel_pending


@dataclass(frozen=True)
class CancelCommand:
    order_id: str
    side: SideName


@dataclass(frozen=True)
class PlaceCommand:
    side: SideName
    px: int
    qty: int
    aggressive: bool
    reason: str


@dataclass(frozen=True)
class SyncActions:
    cancels: tuple[CancelCommand, ...]
    placements: tuple[PlaceCommand, ...]


class AValuationModel:
    """Keeps the latest fair value estimate for A and updates it from earnings news."""

    def __init__(self, pe_ratio: float, initial_fair_value: int | None = None):
        self.pe_ratio = pe_ratio
        self.fair_value = initial_fair_value
        self.last_earnings_value: float | None = None
        self.last_source = "config" if initial_fair_value is not None else "none"

    @property
    def has_fair_value(self) -> bool:
        return self.fair_value is not None

    def set_fair_value(self, fair_value: int, source: str) -> None:
        self.fair_value = int(fair_value)
        self.last_source = source

    def update_from_news(self, news_release: dict) -> bool:
        if news_release.get("kind") != "structured":
            return False
        new_data = news_release.get("new_data") or {}
        if new_data.get("structured_subtype") != "earnings":
            return False

        asset = str(new_data.get("asset") or news_release.get("symbol") or "").upper()
        if asset != "A":
            return False

        earnings_value = float(new_data["value"])
        self.last_earnings_value = earnings_value
        self.fair_value = int(round(self.pe_ratio * earnings_value))
        self.last_source = "earnings"
        return True


class QuoteEngine:
    """Converts the latest fair value, inventory, and book into target quotes."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def compute_quotes(
        self,
        fair_value: int | None,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        can_trade: bool,
        reason_if_blocked: str,
    ) -> QuotePlan:
        if not can_trade:
            return QuotePlan(
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason=reason_if_blocked,
            )

        if fair_value is None:
            return QuotePlan(
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for the first usable A fair value.",
            )

        # Treat current inventory plus same-side resting orders as committed exposure.
        allowed_buy = max(0, self.risk.max_position - (inventory + buy_exposure))
        allowed_sell = max(0, self.risk.max_position + inventory - sell_exposure)

        # Positive inventory should push the reservation price lower so the bot
        # naturally leans toward selling inventory back down, and vice versa.
        reservation = fair_value - (self.risk.inventory_skew * inventory)
        bid_px = int(floor(reservation - self.risk.min_edge))
        ask_px = int(ceil(reservation + self.risk.min_edge))

        if book.best_ask is not None:
            bid_px = min(bid_px, book.best_ask.px - 1)
        if book.best_bid is not None:
            ask_px = max(ask_px, book.best_bid.px + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        aggressive_actions: list[DesiredOrder] = []
        desired_bid: DesiredOrder | None = None
        desired_ask: DesiredOrder | None = None

        if book.best_ask is not None and allowed_buy > 0:
            if book.best_ask.px <= fair_value - self.risk.take_edge:
                aggressive_actions.append(
                    DesiredOrder(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(self.risk.quote_size, allowed_buy, book.best_ask.qty),
                        aggressive=True,
                        reason="best ask is cheap versus fair value",
                    )
                )

        if book.best_bid is not None and allowed_sell > 0:
            if book.best_bid.px >= fair_value + self.risk.take_edge:
                aggressive_actions.append(
                    DesiredOrder(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(self.risk.quote_size, allowed_sell, book.best_bid.qty),
                        aggressive=True,
                        reason="best bid is rich versus fair value",
                    )
                )

        aggressive_sides = {action.side for action in aggressive_actions}
        if allowed_buy > 0 and "BUY" not in aggressive_sides:
            desired_bid = DesiredOrder(
                side="BUY",
                px=bid_px,
                qty=min(self.risk.quote_size, allowed_buy),
                aggressive=False,
                reason="passive buy quote around reservation price",
            )
        if allowed_sell > 0 and "SELL" not in aggressive_sides:
            desired_ask = DesiredOrder(
                side="SELL",
                px=ask_px,
                qty=min(self.risk.quote_size, allowed_sell),
                aggressive=False,
                reason="passive sell quote around reservation price",
            )

        return QuotePlan(
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=tuple(aggressive_actions),
            observe_only=False,
            reason="ready",
        )


class OrderManager:
    """Tracks one managed order per side and decides when to cancel/replace."""

    def __init__(self, symbol: str, risk: RiskConfig):
        self.symbol = symbol
        self.risk = risk
        self.orders: dict[str, ManagedOrder] = {}
        self.live_by_side: dict[SideName, str | None] = {"BUY": None, "SELL": None}
        self.last_action_ms: dict[SideName, int] = {"BUY": 0, "SELL": 0}

    def restore_order(self, order: ManagedOrder) -> None:
        self.orders[order.order_id] = order
        self.live_by_side[order.side] = order.order_id

    def note_submitted(
        self,
        order_id: str,
        side: SideName,
        px: int,
        qty: int,
        now_ms: int,
        aggressive: bool = False,
        restored: bool = False,
    ) -> ManagedOrder:
        order = ManagedOrder(
            order_id=order_id,
            side=side,
            px=px,
            qty=qty,
            remaining_qty=qty,
            submitted_ms=now_ms,
            aggressive=aggressive,
            restored=restored,
        )
        self.orders[order_id] = order
        self.live_by_side[side] = order_id
        self.last_action_ms[side] = now_ms
        return order

    def live_order(self, side: SideName) -> ManagedOrder | None:
        order_id = self.live_by_side.get(side)
        if order_id is None:
            return None
        return self.orders.get(order_id)

    def buy_exposure(self) -> int:
        order = self.live_order("BUY")
        return 0 if order is None or order.cancel_pending else order.remaining_qty

    def sell_exposure(self) -> int:
        order = self.live_order("SELL")
        return 0 if order is None or order.cancel_pending else order.remaining_qty

    def mark_cancel_requested(self, order_id: str, now_ms: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.cancel_pending = True
        self.last_action_ms[order.side] = now_ms
        return order

    def handle_fill(self, order_id: str, qty: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.remaining_qty = max(0, order.remaining_qty - qty)
        if order.remaining_qty == 0:
            self._drop_order(order_id)
        return order

    def handle_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        if success:
            self._drop_order(order_id)
            return order
        order.cancel_pending = False
        return order

    def handle_rejection(self, order_id: str) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        self._drop_order(order_id)
        return order

    def forget_order(self, order_id: str) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        self._drop_order(order_id)
        return order

    def has_stale_quote(self, now_ms: int) -> bool:
        for side in ("BUY", "SELL"):
            order = self.live_order(side)
            if order is None or order.cancel_pending:
                continue
            if now_ms - order.submitted_ms >= self.risk.stale_quote_ms:
                return True
        return False

    def build_actions(self, plan: QuotePlan, now_ms: int) -> SyncActions:
        desired_by_side: dict[SideName, DesiredOrder | None] = {
            "BUY": plan.bid,
            "SELL": plan.ask,
        }
        for aggressive in plan.aggressive_actions:
            desired_by_side[aggressive.side] = aggressive

        cancels: list[CancelCommand] = []
        placements: list[PlaceCommand] = []

        for side in ("BUY", "SELL"):
            desired = desired_by_side[side]
            live = self.live_order(side)
            if live is not None and live.cancel_pending:
                continue

            needs_reprice = (
                live is not None
                and desired is not None
                and (
                    live.px != desired.px
                    or live.remaining_qty != desired.qty
                    or live.aggressive != desired.aggressive
                )
            )

            if live is not None and desired is None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    cancels.append(CancelCommand(order_id=live.order_id, side=side))
                continue

            if live is not None and needs_reprice:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    # Cancel first and let the next event loop place the replacement.
                    # This keeps the bot from doubling exposure on a single side.
                    cancels.append(CancelCommand(order_id=live.order_id, side=side))
                continue

            if live is None and desired is not None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    placements.append(
                        PlaceCommand(
                            side=side,
                            px=desired.px,
                            qty=desired.qty,
                            aggressive=desired.aggressive,
                            reason=desired.reason,
                        )
                    )

        return SyncActions(cancels=tuple(cancels), placements=tuple(placements))

    def restored_orders(self) -> Iterable[ManagedOrder]:
        return (order for order in self.orders.values() if order.restored)

    def _drop_order(self, order_id: str) -> None:
        order = self.orders.pop(order_id, None)
        if order is None:
            return
        if self.live_by_side.get(order.side) == order_id:
            self.live_by_side[order.side] = None


class MarketAStrategy:
    """Owns the valuation model, quote engine, and order manager for market A."""

    def __init__(
        self,
        a_config: AConfig,
        risk: RiskConfig,
        restored_orders: Iterable[ManagedOrder] = (),
        recovered_fair_value: int | None = None,
    ):
        starting_fair = recovered_fair_value
        if starting_fair is None:
            starting_fair = a_config.initial_fair_value

        self.valuation = AValuationModel(
            pe_ratio=a_config.pe_ratio,
            initial_fair_value=starting_fair,
        )
        if recovered_fair_value is not None:
            self.valuation.last_source = "journal"

        self.quote_engine = QuoteEngine(risk)
        self.order_manager = OrderManager(symbol="A", risk=risk)
        self.inventory = 0
        self.book = BookSnapshot()
        self.recovery_pending: set[str] = set()
        self.recovery_active = False

        for order in restored_orders:
            self.order_manager.restore_order(order)
            self.recovery_pending.add(order.order_id)

        if self.recovery_pending:
            self.recovery_active = True

    @property
    def fair_value(self) -> int | None:
        return self.valuation.fair_value

    def set_inventory(self, inventory: int) -> None:
        self.inventory = int(inventory)

    def on_book_update(self, symbol: str, book) -> bool:
        if symbol != "A":
            return False
        self.book = BookSnapshot.from_order_book(book)
        return True

    def on_news(self, news_release: dict) -> bool:
        return self.valuation.update_from_news(news_release)

    def on_fill(self, order_id: str, qty: int, price: int) -> ManagedOrder | None:
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        signed_qty = qty if order.side == "BUY" else -qty
        self.inventory += signed_qty
        if order_id in self.recovery_pending and order.remaining_qty == 0:
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.order_manager.handle_cancel_response(order_id, success)
        if order is None:
            return None
        if order_id in self.recovery_pending:
            if not success:
                # After a restart there is no open-order snapshot, so we treat a failed
                # recovery cancel as unresolved local state and stop managing it.
                self.order_manager.forget_order(order_id)
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_rejection(self, order_id: str) -> ManagedOrder | None:
        order = self.order_manager.handle_rejection(order_id)
        if order is None:
            return None
        if order_id in self.recovery_pending:
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_recovery_complete(self) -> None:
        self.recovery_active = False
        self.recovery_pending.clear()

    def recovery_orders_to_cancel(self) -> list[ManagedOrder]:
        return [
            order
            for order in self.order_manager.restored_orders()
            if order.order_id in self.recovery_pending and not order.cancel_pending
        ]

    def compute_quotes(self) -> QuotePlan:
        reason_if_blocked = "Waiting for recovered A orders to be cancelled."
        return self.quote_engine.compute_quotes(
            fair_value=self.valuation.fair_value,
            inventory=self.inventory,
            book=self.book,
            buy_exposure=self.order_manager.buy_exposure(),
            sell_exposure=self.order_manager.sell_exposure(),
            can_trade=not self.recovery_active,
            reason_if_blocked=reason_if_blocked,
        )

    def _maybe_finish_recovery(self) -> None:
        if self.recovery_active and not self.recovery_pending:
            self.on_recovery_complete()
