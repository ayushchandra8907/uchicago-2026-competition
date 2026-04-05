from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd


Side = Literal["BUY", "SELL"]
EventKind = Literal["book", "trade", "news"]
TradeAggressor = Literal["BUY", "SELL", "UNKNOWN"]
ModeName = Literal[
    "NORMAL_MM",
    "EARNINGS_SHOCK",
    "OVERSHOOT_FADE",
    "INVENTORY_UNWIND",
    "RISK_OFF",
]


@dataclass(frozen=True)
class BookState:
    best_bid_px: int | None = None
    best_bid_qty: int | None = None
    best_ask_px: int | None = None
    best_ask_qty: int | None = None
    bid_levels: tuple[tuple[int, int], ...] = ()
    ask_levels: tuple[tuple[int, int], ...] = ()

    @property
    def mid_px(self) -> float | None:
        if self.best_bid_px is None or self.best_ask_px is None:
            return None
        return (self.best_bid_px + self.best_ask_px) / 2.0

    @property
    def spread_px(self) -> float | None:
        if self.best_bid_px is None or self.best_ask_px is None:
            return None
        return float(self.best_ask_px - self.best_bid_px)

    def level_qty(self, side: Side, px: int) -> int | None:
        levels = self.bid_levels if side == "BUY" else self.ask_levels
        for level_px, level_qty in levels:
            if level_px == px:
                return level_qty
        return None


@dataclass(frozen=True)
class TradeState:
    price_px: int
    qty: int
    aggressor: TradeAggressor = "UNKNOWN"


@dataclass(frozen=True)
class NewsState:
    kind: str
    symbol: str | None = None
    structured_subtype: str | None = None
    earnings_asset: str | None = None
    earnings_value: float | None = None
    content: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketEvent:
    kind: EventKind
    session_id: str
    seq: int
    time_ms: float
    source_seq: int | None = None
    book: BookState | None = None
    trade: TradeState | None = None
    news: NewsState | None = None


@dataclass(frozen=True)
class SkippedRun:
    path: Path
    reason: str


@dataclass(frozen=True)
class SessionData:
    session_id: str
    path: Path
    source_layout: str
    events: tuple[MarketEvent, ...]
    book_rows: "pd.DataFrame"
    trade_rows: "pd.DataFrame"
    news_rows: "pd.DataFrame"
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RunCatalog:
    data_root: Path
    sessions: tuple[SessionData, ...]
    skipped_runs: tuple[SkippedRun, ...]


@dataclass(frozen=True)
class FeatureSnapshot:
    time_ms: float
    book: BookState | None
    last_trade_px: int | None
    mid_px: float | None
    spread_px: float | None
    microprice_px: float | None
    imbalance: float | None
    realized_vol_1s: float | None
    realized_vol_5s: float | None
    trade_count_1s: int
    trade_volume_1s: int
    trade_pressure_1s: float
    trade_pressure_5s: float
    fill_pressure_2s: float
    reference_fair_px: int | None
    quote_to_fair_px: float | None


@dataclass(frozen=True)
class OrderIntent:
    side: Side
    px: int
    qty: int
    aggressive: bool
    reason: str


@dataclass(frozen=True)
class QuoteIntent:
    mode: ModeName
    reference_fair_px: int | None
    reservation_px: int | None
    bid: OrderIntent | None
    ask: OrderIntent | None
    aggressive_actions: tuple[OrderIntent, ...]
    reason: str


@dataclass
class StrategyState:
    inventory: int = 0
    cash: float = 0.0
    latest_earnings: float | None = None
    fair_px: int | None = None
    last_earnings_time_ms: float | None = None
    last_earnings_direction: int = 0
    mode: ModeName = "NORMAL_MM"
    last_mid_px: float | None = None
    passive_buy_fills: int = 0
    passive_sell_fills: int = 0
    aggressive_buy_fills: int = 0
    aggressive_sell_fills: int = 0


@dataclass(frozen=True)
class FillRecord:
    time_ms: float
    side: Side
    px: int
    qty: int
    aggressive: bool
    mode: ModeName
    edge_px: float


@dataclass(frozen=True)
class BacktestResult:
    session_id: str
    final_inventory: int
    final_cash: float
    mark_px: float
    total_pnl: float
    max_drawdown: float
    passive_fill_count: int
    aggressive_fill_count: int
    pnl_by_mode: dict[str, float]
    inventory_path: tuple[tuple[float, int], ...]
    fills: tuple[FillRecord, ...]
