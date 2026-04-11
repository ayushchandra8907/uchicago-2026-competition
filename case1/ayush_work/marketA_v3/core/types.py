from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SideName = Literal["BUY", "SELL"]
ModeState = Literal["IDLE", "SHOCK", "UNWIND", "MARKET_MAKING_STUB"]


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
        bid_levels = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
        ask_levels = [(int(px), int(qty)) for px, qty in book.asks.items() if qty > 0]
        best_bid = None if not bid_levels else max(bid_levels, key=lambda item: item[0])
        best_ask = None if not ask_levels else min(ask_levels, key=lambda item: item[0])
        return cls(
            best_bid=None if best_bid is None else BookLevel(px=best_bid[0], qty=best_bid[1]),
            best_ask=None if best_ask is None else BookLevel(px=best_ask[0], qty=best_ask[1]),
        )

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.px + self.best_ask.px) / 2.0

    @property
    def spread(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask.px - self.best_bid.px


@dataclass(frozen=True)
class NewsEvent:
    now_ms: int
    tick: int | None
    kind: str
    symbol: str | None
    structured_subtype: str | None = None
    asset: str | None = None
    value: float | None = None
    content: str | None = None
    news_type: str | None = None
    forecast: float | None = None
    actual: float | None = None
    raw_payload: dict | None = None

    @property
    def is_structured_a_earnings(self) -> bool:
        return (
            self.kind == "structured"
            and self.structured_subtype == "earnings"
            and (self.asset or self.symbol or "").upper() == "A"
            and self.value is not None
        )

    @property
    def is_a_specific_unstructured(self) -> bool:
        return self.kind == "unstructured" and (self.asset or self.symbol or "").upper() == "A"

    @property
    def is_structured_cpi_print(self) -> bool:
        return self.kind == "structured" and self.structured_subtype == "cpi_print" and self.forecast is not None and self.actual is not None

    @property
    def is_fed_speak_unstructured(self) -> bool:
        return self.kind == "unstructured" and str(self.news_type or "").lower() == "fedspeak"


@dataclass(frozen=True)
class DesiredOrder:
    side: SideName
    px: int
    qty: int
    aggressive: bool = True
    intent: str = ""
    reason: str = ""
    symbol: str = "A"
    context_id: str | None = None


@dataclass(frozen=True)
class Decision:
    mode: ModeState
    target_inventory: int
    desired_order: DesiredOrder | None
    cancel_all: bool
    observe_only: bool
    reason: str
    fair_value: int | None = None
    trusted_multiplier: float | None = None
    latest_earnings: float | None = None
    equilibrium_reached: bool = False
    shock_reference_mid: float | None = None


@dataclass
class ManagedOrder:
    order_id: str
    side: SideName
    px: int
    qty: int
    remaining_qty: int
    submitted_ms: int
    aggressive: bool
    intent: str
    reason: str
    symbol: str = "A"
    context_id: str | None = None
    cancel_pending: bool = False

    @property
    def is_live(self) -> bool:
        return self.remaining_qty > 0 and not self.cancel_pending


@dataclass(frozen=True)
class StrategySnapshot:
    now_ms: int
    exchange_tick: int | None
    book: BookSnapshot
    inventory: int
    cash: int
    fair_value: int | None
    trusted_multiplier: float | None
    latest_earnings: float | None
    mode: ModeState
    open_orders: tuple[ManagedOrder, ...]
    last_trade_px: int | None = None
    message_index: int | None = None
    books_by_symbol: dict[str, BookSnapshot] = field(default_factory=dict)
    inventories_by_symbol: dict[str, int] = field(default_factory=dict)
    open_orders_by_symbol: dict[str, tuple[ManagedOrder, ...]] = field(default_factory=dict)
    last_trade_px_by_symbol: dict[str, int] = field(default_factory=dict)
    event_symbol: str | None = None

    def book_for(self, symbol: str) -> BookSnapshot:
        return self.books_by_symbol.get(symbol, BookSnapshot())

    def inventory_for(self, symbol: str) -> int:
        return int(self.inventories_by_symbol.get(symbol, 0))

    def open_orders_for(self, symbol: str) -> tuple[ManagedOrder, ...]:
        return self.open_orders_by_symbol.get(symbol, ())

    def last_trade_for(self, symbol: str) -> int | None:
        return self.last_trade_px_by_symbol.get(symbol)
