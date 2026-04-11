from typing import Optional, Dict, Tuple

try:
    from utcxchangelib import XChangeClient, Side
except ImportError:
    from utcxchangelib.xchange_client import XChangeClient, Side

try:
    from utcxchangelib.xchange_client import OrderBook
except ImportError:
    OrderBook = None

import asyncio
import csv
import logging
import math
import re
import time
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("c-only")


C_SYMBOL = "C"
R_HIKE = "R_HIKE"
R_HOLD = "R_HOLD"
R_CUT = "R_CUT"
PM_SYMBOLS = [R_HIKE, R_HOLD, R_CUT]

ORDER_LIMIT = 40
C_POSITION_LIMIT = 200

C_MM_ENABLED = True
C_MM_QUOTE_SIZE = 5
C_MM_MAX_POSITION = 40
C_MM_HALF_SPREAD_TICKS = 2.0
C_MM_INVENTORY_SKEW = 0.75
C_MM_MIN_BOOK_SPREAD = 2
C_MM_PASSIVE_REDUCE_START = 12
C_MM_PASSIVE_REDUCE_FULL = 28
C_MM_REDUCE_ONLY_POSITION = 34
C_MM_REPRICE_THRESHOLD_TICKS = 2
C_MM_QUOTE_TTL_SECONDS = 3.0
C_MM_NEWS_COOLDOWN_SECONDS = 8.0

CPI_DEADBAND = 0.00025
CPI_STRONG_SURPRISE = 0.00050
CPI_VERY_STRONG_SURPRISE = 0.00100
CPI_EXTREME_SURPRISE = 0.00200
CPI_SIGNAL_TTL_SECONDS = 20.0
CPI_MAX_HOLD_SECONDS = 45.0
CPI_TO_RATE_BP = 4000.0
MAX_CPI_RATE_BIAS_BP = 8.0
# Some practice feeds have shown CPI prints in the UI while the raw callback
# arrives as an empty petition payload at the same timestamp.  Use only the
# confirmed CPI ticks as a fallback, after first trying to parse real fields.
CPI_PRINT_FALLBACKS_BY_TICK = {
    100: (0.0030, 0.0019),
    1650: (0.0003, 0.0015),
    1900: (0.0022, 0.0021),
    2100: (0.0030, 0.0019),
}

EARNINGS_IGNORE_DELTA = 0.010
EARNINGS_SMALL_DELTA = 0.025
EARNINGS_MEDIUM_DELTA = 0.050
EARNINGS_EXTREME_DELTA = 0.075
EARNINGS_SIGNAL_TTL_SECONDS = 20.0
EARNINGS_MAX_HOLD_SECONDS = 24.0
EARNINGS_FAST_HARVEST_SECONDS = 5.0
EARNINGS_FAST_TIMEOUT_SECONDS = 8.0
EARNINGS_FAST_HARVEST_MIN_TICKS = 6.0
LATE_SESSION_WEAK_EARNINGS_CUTOFF_TICK = 4300
LATE_SESSION_EARNINGS_TARGET_CAP_TICK = 4485
LATE_SESSION_EARNINGS_TARGET_CAP = ORDER_LIMIT

MACRO_HEADLINE_MIN_SCORE = 2.5
MACRO_HEADLINE_STRONG_SCORE = 3.5
MACRO_HEADLINE_VERY_STRONG_SCORE = 5.5
MACRO_SIGNAL_TTL_SECONDS = 8.0
MACRO_MAX_HOLD_SECONDS = 12.0
MACRO_TO_RATE_BP = 1.5
MACRO_EARNINGS_TREND_BLOCK = 0.020
MACRO_FLAT_TARGET_CAP = 80

DEFAULT_EPS_C = 2.0
# Released C parameters retained for reference.  For live trading we keep
# the anchor-price framework calibrated to observed C rather than letting
# the raw case proxy block large earnings moves by 80+ ticks.
C_PE0 = 14.0
C_EPS0 = 2.0
C_BASE_BOND_PER_SHARE = 40.0
C_LAMBDA = 0.65
C_OPS_WEIGHT = 0.72
C_BOND_WEIGHT = 0.28
C_PE_YIELD_GAMMA = 13.0
C_BOND_DURATION = 4.5
C_BOND_CONVEXITY = 30.0
FAIR_VALUE_GUARD_TICKS = 8.0
MACRO_FAIR_VALUE_GUARD_TICKS = 35.0
EARNINGS_EXTREME_FAIR_VALUE_GUARD_TICKS = 180.0
EARNINGS_STRONG_FAIR_VALUE_GUARD_TICKS = 90.0
EARNINGS_WEAK_FAIR_VALUE_GUARD_TICKS = 45.0
REVERSAL_TREND_MIN_ABS = 0.060

MAX_C_SPREAD = 12
C_ACTION_COOLDOWN_SECONDS = 0.20
C_SYMBOL_COOLDOWN_SECONDS = 0.25
ORDER_REPRICE_SECONDS = 0.45
HEARTBEAT_LOG_SECONDS = 3.0

DEFAULT_PROFIT_1_TICKS = 8.0
DEFAULT_PROFIT_2_TICKS = 20.0
DEFAULT_PROFIT_FULL_TICKS = 40.0
ADVERSE_STOP_ADD_TICKS = 12.0
DEFAULT_ADVERSE_FLATTEN_TICKS = 24.0
WEAK_SIGNAL_ADVERSE_FLATTEN_TICKS = 16.0
DEFAULT_TRAIL_START_TICKS = 24.0
DEFAULT_TRAIL_GIVEBACK_TICKS = 14.0
EARNINGS_PROFIT_1_TICKS = 12.0
EARNINGS_PROFIT_2_TICKS = 30.0
EARNINGS_PROFIT_FULL_TICKS = 65.0
EARNINGS_TRAIL_START_TICKS = 32.0
EARNINGS_TRAIL_GIVEBACK_TICKS = 16.0
EARNINGS_ADVERSE_FLATTEN_TICKS = 24.0
EARNINGS_MEDIUM_PROFIT_1_TICKS = 10.0
EARNINGS_MEDIUM_PROFIT_2_TICKS = 22.0
EARNINGS_MEDIUM_PROFIT_FULL_TICKS = 45.0
EARNINGS_MEDIUM_TRAIL_START_TICKS = 24.0
EARNINGS_MEDIUM_TRAIL_GIVEBACK_TICKS = 10.0
EARNINGS_MEDIUM_ADVERSE_FLATTEN_TICKS = 18.0
EARNINGS_WEAK_PROFIT_1_TICKS = 7.0
EARNINGS_WEAK_PROFIT_2_TICKS = 14.0
EARNINGS_WEAK_PROFIT_FULL_TICKS = 24.0
EARNINGS_WEAK_TRAIL_START_TICKS = 14.0
EARNINGS_WEAK_TRAIL_GIVEBACK_TICKS = 7.0
EARNINGS_WEAK_ADVERSE_FLATTEN_TICKS = 12.0
CPI_PROFIT_1_TICKS = 12.0
CPI_PROFIT_2_TICKS = 30.0
CPI_PROFIT_FULL_TICKS = 75.0
CPI_TRAIL_START_TICKS = 36.0
CPI_TRAIL_GIVEBACK_TICKS = 20.0
CPI_ADVERSE_FLATTEN_TICKS = 28.0
MACRO_PROFIT_1_TICKS = 8.0
MACRO_PROFIT_2_TICKS = 18.0
MACRO_PROFIT_FULL_TICKS = 40.0
MACRO_TRAIL_START_TICKS = 20.0
MACRO_TRAIL_GIVEBACK_TICKS = 10.0
MACRO_ADVERSE_FLATTEN_TICKS = 16.0
STRONG_FAIR_GAP_HOLD_TICKS = 25.0
C_TOTAL_PNL_HARD_STOP = -12000.0
C_SESSION_PROFIT_LOCK_START = 60000.0
C_SESSION_PROFIT_LOCK_GIVEBACK = 12000.0

FLOAT_RE = r"[-+]?\d*\.?\d+"


@dataclass
class OrderRef:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    tag: str
    ts: float


@dataclass
class CSignal:
    side: Side
    target: int
    thesis: str
    strength: float
    tick: int
    ts: float
    description: str


@dataclass
class RateContext:
    q_hike: float
    q_hold: float
    q_cut: float
    market_rate_bp: float
    effective_rate_bp: float
    cpi_bias_bp: float


class CsvRunLogger:
    FIELDNAMES = [
        "wall_ts",
        "event",
        "tick",
        "symbol",
        "side",
        "qty",
        "price",
        "order_id",
        "tag",
        "reason",
        "message",
        "pos_C",
        "avg_C",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "cash",
        "session_mtm",
        "thesis",
        "signal_thesis",
        "signal_side",
        "signal_target",
        "signal_strength",
        "signal_fresh",
        "fair",
        "c_bid",
        "c_ask",
        "c_mid",
        "c_spread",
        "eps",
        "earnings_delta",
        "cpi_surprise",
        "cpi_bias_bp",
        "cpi_actual",
        "cpi_forecast",
        "macro_score",
        "macro_target",
        "macro_direction",
        "earnings_trend",
        "news_kind",
        "news_subtype",
        "news_asset",
        "rate_bp",
        "market_rate_bp",
        "q_hike",
        "q_hold",
        "q_cut",
        "pm_pos_hike",
        "pm_pos_hold",
        "pm_pos_cut",
        "mark_source",
        "kill_switch",
        "raw_news",
    ]

    def __init__(self, root: Path):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            root = Path("/tmp/c_only_logs")
            root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = root / f"c_only_run_{stamp}.csv"
        try:
            self._fh = self.path.open("w", newline="", buffering=1)
        except PermissionError:
            root = Path("/tmp/c_only_logs")
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / f"c_only_run_{stamp}.csv"
            self._fh = self.path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        clean = {field: row.get(field, "") for field in self.FIELDNAMES}
        self._writer.writerow(clean)


class MyXchangeClient(XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

        # Some provided utcxchangelib builds ship stale hard-coded symbols.
        # A dynamic book map prevents snapshots for A/B/C/ETF/options from crashing.
        if OrderBook is not None:
            self.order_books = defaultdict(OrderBook, dict(self.order_books))

        # The exchange rejects reused/old ids. Millisecond ids avoid restart collisions.
        self.order_id = max(int(getattr(self, "order_id", 0)), int(time.time() * 1000))

        self.pos: Dict[str, int] = {C_SYMBOL: 0, R_HIKE: 0, R_HOLD: 0, R_CUT: 0}
        self.cash = 0.0
        self.session_start_cash: Optional[float] = None
        self.session_start_mtm: Optional[float] = None

        self.live_orders: Dict[str, OrderRef] = {}
        self.pending_cancels: set[str] = set()
        self.pending_buy: Dict[str, int] = defaultdict(int)
        self.pending_sell: Dict[str, int] = defaultdict(int)
        self.symbol_cooldown_until: Dict[str, float] = defaultdict(float)

        self.current_eps_c = DEFAULT_EPS_C
        self.have_real_eps_c = False
        self.baseline_eps_c: Optional[float] = None
        self.last_c_earnings_delta = 0.0
        self.recent_c_earnings_deltas = deque(maxlen=4)

        self.anchor_price: Optional[float] = None
        self.anchor_eps: Optional[float] = None
        self.anchor_rate_bp: Optional[float] = None
        self.last_anchor_update_ts = 0.0

        self.active_signal: Optional[CSignal] = None
        self.last_cpi_surprise = 0.0
        self.last_cpi_bias_bp = 0.0
        self.last_cpi_ts = 0.0
        self.last_rate_bias_ttl = CPI_SIGNAL_TTL_SECONDS

        self.c_avg_entry: Optional[float] = None
        self.c_realized_pnl = 0.0
        self.c_unrealized_pnl = 0.0
        self.c_total_pnl = 0.0
        self.c_peak_total_pnl = 0.0
        self.c_peak_abs = 0
        self.c_exit_stage = 0
        self.c_best_profit_ticks = 0.0
        self.c_thesis = "flat"
        self.c_regime_started_at = 0.0
        self.c_last_fill_price: Optional[int] = None
        self.profit_lock_engaged = False
        self.kill_switch = False

        self.last_c_action_ts = 0.0
        self.last_c_signal_ts = 0.0
        self.last_heartbeat_log = 0.0
        self.last_entry_block_log_ts = 0.0
        self.last_entry_block_key = ""
        self.last_tick_seen = 0
        self.decision_lock = asyncio.Lock()
        self.csv_logger = CsvRunLogger(Path(__file__).resolve().parent / "logs")
        logger.info("CSV run log: %s", self.csv_logger.path)

    def now(self) -> float:
        return time.time()

    def side_name(self, side: Side) -> str:
        return "BUY" if side == Side.BUY else "SELL"

    def sign(self, x: int | float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def clamp_price(self, px: int) -> int:
        return max(1, min(9999, int(px)))

    def clip(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def book_top(self, symbol: str) -> Tuple[Optional[int], Optional[int]]:
        book = self.order_books.get(symbol)
        if book is None:
            return None, None
        bids = getattr(book, "bids", {}) or {}
        asks = getattr(book, "asks", {}) or {}
        live_bids = [int(px) for px, qty in bids.items() if qty != 0]
        live_asks = [int(px) for px, qty in asks.items() if qty != 0]
        bid = max(live_bids) if live_bids else None
        ask = min(live_asks) if live_asks else None
        return bid, ask

    def mid(self, symbol: str) -> Optional[float]:
        bid, ask = self.book_top(symbol)
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return None

    def spread(self, symbol: str) -> Optional[int]:
        bid, ask = self.book_top(symbol)
        if bid is None or ask is None:
            return None
        return ask - bid

    def csv_snapshot(self, event: str, **extra: object) -> Dict[str, object]:
        unrealized, mark_source = self.mark_c_unrealized()
        bid, ask = self.book_top(C_SYMBOL)
        mid_c = self.mid(C_SYMBOL)
        spread = self.spread(C_SYMBOL)
        fair = self.fair_value_c()
        rate = self.rate_context()
        sig = self.active_signal
        session_mtm = self.c_total_pnl
        row: Dict[str, object] = {
            "wall_ts": f"{self.now():.6f}",
            "event": event,
            "tick": self.last_tick_seen,
            "pos_C": self.pos.get(C_SYMBOL, 0),
            "avg_C": "" if self.c_avg_entry is None else f"{self.c_avg_entry:.4f}",
            "realized_pnl": f"{self.c_realized_pnl:.2f}",
            "unrealized_pnl": f"{unrealized:.2f}",
            "total_pnl": f"{self.c_total_pnl:.2f}",
            "cash": f"{self.cash:.2f}",
            "session_mtm": f"{session_mtm:.2f}",
            "thesis": self.c_thesis,
            "signal_thesis": "" if sig is None else sig.thesis,
            "signal_side": "" if sig is None else self.side_name(sig.side),
            "signal_target": "" if sig is None else sig.target,
            "signal_strength": "" if sig is None else f"{sig.strength:.6f}",
            "signal_fresh": "" if sig is None else self.c_signal_is_fresh(),
            "fair": "" if fair is None else f"{fair:.4f}",
            "c_bid": "" if bid is None else bid,
            "c_ask": "" if ask is None else ask,
            "c_mid": "" if mid_c is None else f"{mid_c:.4f}",
            "c_spread": "" if spread is None else spread,
            "eps": f"{self.current_eps_c:.6f}",
            "earnings_delta": f"{self.last_c_earnings_delta:.6f}",
            "cpi_surprise": f"{self.last_cpi_surprise:.6f}",
            "cpi_bias_bp": f"{self.last_cpi_bias_bp:.4f}",
            "rate_bp": "" if rate is None else f"{rate.effective_rate_bp:.4f}",
            "market_rate_bp": "" if rate is None else f"{rate.market_rate_bp:.4f}",
            "q_hike": "" if rate is None else f"{rate.q_hike:.6f}",
            "q_hold": "" if rate is None else f"{rate.q_hold:.6f}",
            "q_cut": "" if rate is None else f"{rate.q_cut:.6f}",
            "pm_pos_hike": self.pos.get(R_HIKE, 0),
            "pm_pos_hold": self.pos.get(R_HOLD, 0),
            "pm_pos_cut": self.pos.get(R_CUT, 0),
            "mark_source": mark_source,
            "kill_switch": self.kill_switch,
        }
        row.update(extra)
        return row

    def log_csv(self, event: str, **extra: object) -> None:
        try:
            self.csv_logger.write(self.csv_snapshot(event, **extra))
        except Exception as e:
            logger.warning("CSV logging failed: event=%s error=%s", event, e)

    def is_mm_tag(self, tag: str) -> bool:
        return tag.startswith("mm_")

    def live_c_order(self, side: Optional[Side] = None, *, mm_only: Optional[bool] = None, include_pending: bool = True) -> Optional[OrderRef]:
        for oid, ref in self.live_orders.items():
            if ref.symbol != C_SYMBOL:
                continue
            if not include_pending and oid in self.pending_cancels:
                continue
            if side is not None and ref.side != side:
                continue
            if mm_only is not None and self.is_mm_tag(ref.tag) != mm_only:
                continue
            return ref
        return None

    def has_live_order(self, symbol: str) -> bool:
        return any(
            ref.symbol == symbol and oid not in self.pending_cancels
            for oid, ref in self.live_orders.items()
        )

    def has_live_non_mm_order(self, symbol: str) -> bool:
        return any(
            ref.symbol == symbol and not self.is_mm_tag(ref.tag) and oid not in self.pending_cancels
            for oid, ref in self.live_orders.items()
        )

    def remaining_capacity(self, symbol: str, side: Side) -> int:
        if symbol != C_SYMBOL:
            return 0
        pos = int(self.pos.get(symbol, 0))
        buy_pending = int(self.pending_buy.get(symbol, 0))
        sell_pending = int(self.pending_sell.get(symbol, 0))
        if side == Side.BUY:
            return max(0, C_POSITION_LIMIT - (pos + buy_pending))
        return max(0, C_POSITION_LIMIT + (pos - sell_pending))

    def order_qty(self, symbol: str, side: Side, desired_qty: int) -> int:
        return max(0, min(ORDER_LIMIT, int(desired_qty), self.remaining_capacity(symbol, side)))

    def best_price_for_cross(self, symbol: str, side: Side) -> Optional[int]:
        bid, ask = self.book_top(symbol)
        if bid is None or ask is None:
            return None
        if side == Side.BUY:
            return self.clamp_price(ask)
        return self.clamp_price(bid)

    def rate_context(self) -> Optional[RateContext]:
        mid_hike = self.mid(R_HIKE)
        mid_hold = self.mid(R_HOLD)
        mid_cut = self.mid(R_CUT)
        if mid_hike is None or mid_hold is None or mid_cut is None:
            return None

        total = mid_hike + mid_hold + mid_cut
        if total <= 1e-9:
            return None

        q_hike = mid_hike / total
        q_hold = mid_hold / total
        q_cut = mid_cut / total
        market_rate_bp = 25.0 * q_hike - 25.0 * q_cut
        cpi_bias_bp = 0.0
        if self.last_cpi_ts > 0.0 and self.now() - self.last_cpi_ts <= self.last_rate_bias_ttl:
            cpi_bias_bp = self.last_cpi_bias_bp
        return RateContext(
            q_hike=q_hike,
            q_hold=q_hold,
            q_cut=q_cut,
            market_rate_bp=market_rate_bp,
            effective_rate_bp=market_rate_bp + cpi_bias_bp,
            cpi_bias_bp=cpi_bias_bp,
        )

    def maybe_initialize_anchor(self, force: bool = False) -> bool:
        rate = self.rate_context()
        mid_c = self.mid(C_SYMBOL)
        if rate is None or mid_c is None:
            return False
        if (
            force
            or self.anchor_price is None
            or self.anchor_eps is None
            or self.anchor_rate_bp is None
        ):
            self.anchor_price = float(mid_c)
            self.anchor_eps = float(self.current_eps_c)
            self.anchor_rate_bp = float(rate.effective_rate_bp)
            self.last_anchor_update_ts = self.now()
            logger.info(
                "Initialized C anchor: price=%.2f eps=%.4f rate_bp=%.2f source=%s",
                self.anchor_price,
                self.anchor_eps,
                self.anchor_rate_bp,
                "real_earnings" if self.have_real_eps_c else "default_eps",
            )
            self.log_csv(
                "anchor_init",
                message="real_earnings" if self.have_real_eps_c else "default_eps",
            )
            return True
        return False

    def fair_value_c(self) -> Optional[float]:
        rate = self.rate_context()
        if (
            rate is None
            or self.anchor_price is None
            or self.anchor_eps is None
            or self.anchor_rate_bp is None
            or self.anchor_eps == 0
        ):
            return None

        dy = (rate.effective_rate_bp - self.anchor_rate_bp) / 10000.0
        ops_anchor = C_OPS_WEIGHT * self.anchor_price
        bond_anchor = C_BOND_WEIGHT * self.anchor_price
        ops_fair = ops_anchor * (self.current_eps_c / self.anchor_eps) * math.exp(-C_PE_YIELD_GAMMA * dy)
        bond_fair = bond_anchor * (1.0 - C_BOND_DURATION * dy + 0.5 * C_BOND_CONVEXITY * dy * dy)
        return ops_fair + bond_fair

    def fair_gap(self) -> Optional[float]:
        fair = self.fair_value_c()
        mid_c = self.mid(C_SYMBOL)
        if fair is None or mid_c is None:
            return None
        return fair - mid_c

    def c_signal_is_fresh(self) -> bool:
        sig = self.active_signal
        if sig is None or sig.target <= 0:
            return False
        if sig.thesis == "cpi":
            ttl = CPI_SIGNAL_TTL_SECONDS
        elif sig.thesis == "macro":
            ttl = MACRO_SIGNAL_TTL_SECONDS
        else:
            ttl = EARNINGS_SIGNAL_TTL_SECONDS
        return self.now() - sig.ts <= ttl

    def c_signal_max_hold(self) -> float:
        if self.c_thesis == "cpi":
            return CPI_MAX_HOLD_SECONDS
        if self.c_thesis == "macro":
            return MACRO_MAX_HOLD_SECONDS
        if self.c_thesis == "earnings":
            sig = self.active_signal
            if sig is not None and sig.thesis == "earnings":
                if sig.strength < EARNINGS_SMALL_DELTA:
                    return 10.0
                if sig.strength < EARNINGS_MEDIUM_DELTA:
                    return 16.0
            return EARNINGS_MAX_HOLD_SECONDS
        return 18.0

    def c_target_for_cpi_surprise(self, surprise: float) -> int:
        abs_surprise = abs(surprise)
        if abs_surprise <= CPI_DEADBAND:
            return 0
        if abs_surprise >= CPI_EXTREME_SURPRISE:
            return 200
        if abs_surprise >= CPI_VERY_STRONG_SURPRISE:
            return 200
        if abs_surprise >= CPI_STRONG_SURPRISE:
            return 160
        return 120

    def c_target_for_earnings_delta(self, delta: float) -> int:
        abs_delta = abs(delta)
        if abs_delta < EARNINGS_IGNORE_DELTA:
            return 0
        if abs_delta >= EARNINGS_EXTREME_DELTA:
            return 200
        if abs_delta >= EARNINGS_MEDIUM_DELTA:
            return 200
        if abs_delta >= EARNINGS_SMALL_DELTA:
            return 160
        return 120

    def c_exit_profile(self) -> Tuple[float, float, float, float, float, float]:
        sig = self.active_signal
        if self.c_thesis == "earnings":
            strength = sig.strength if sig is not None and sig.thesis == "earnings" else EARNINGS_MEDIUM_DELTA
            if strength < EARNINGS_SMALL_DELTA:
                return (
                    EARNINGS_WEAK_PROFIT_1_TICKS,
                    EARNINGS_WEAK_PROFIT_2_TICKS,
                    EARNINGS_WEAK_PROFIT_FULL_TICKS,
                    EARNINGS_WEAK_TRAIL_START_TICKS,
                    EARNINGS_WEAK_TRAIL_GIVEBACK_TICKS,
                    EARNINGS_WEAK_ADVERSE_FLATTEN_TICKS,
                )
            if strength < EARNINGS_MEDIUM_DELTA:
                return (
                    EARNINGS_MEDIUM_PROFIT_1_TICKS,
                    EARNINGS_MEDIUM_PROFIT_2_TICKS,
                    EARNINGS_MEDIUM_PROFIT_FULL_TICKS,
                    EARNINGS_MEDIUM_TRAIL_START_TICKS,
                    EARNINGS_MEDIUM_TRAIL_GIVEBACK_TICKS,
                    EARNINGS_MEDIUM_ADVERSE_FLATTEN_TICKS,
                )
            return (
                EARNINGS_PROFIT_1_TICKS,
                EARNINGS_PROFIT_2_TICKS,
                EARNINGS_PROFIT_FULL_TICKS,
                EARNINGS_TRAIL_START_TICKS,
                EARNINGS_TRAIL_GIVEBACK_TICKS,
                EARNINGS_ADVERSE_FLATTEN_TICKS,
            )
        if self.c_thesis == "cpi":
            return (
                CPI_PROFIT_1_TICKS,
                CPI_PROFIT_2_TICKS,
                CPI_PROFIT_FULL_TICKS,
                CPI_TRAIL_START_TICKS,
                CPI_TRAIL_GIVEBACK_TICKS,
                CPI_ADVERSE_FLATTEN_TICKS,
            )
        if self.c_thesis == "macro":
            return (
                MACRO_PROFIT_1_TICKS,
                MACRO_PROFIT_2_TICKS,
                MACRO_PROFIT_FULL_TICKS,
                MACRO_TRAIL_START_TICKS,
                MACRO_TRAIL_GIVEBACK_TICKS,
                MACRO_ADVERSE_FLATTEN_TICKS,
            )
        return (
            DEFAULT_PROFIT_1_TICKS,
            DEFAULT_PROFIT_2_TICKS,
            DEFAULT_PROFIT_FULL_TICKS,
            DEFAULT_TRAIL_START_TICKS,
            DEFAULT_TRAIL_GIVEBACK_TICKS,
            DEFAULT_ADVERSE_FLATTEN_TICKS,
        )

    def c_target_for_macro_score(self, score: float) -> int:
        abs_score = abs(score)
        if abs_score < MACRO_HEADLINE_MIN_SCORE:
            return 0
        if abs_score >= MACRO_HEADLINE_VERY_STRONG_SCORE:
            return 160
        if abs_score >= MACRO_HEADLINE_STRONG_SCORE:
            return 120
        return 80

    def c_earnings_trend(self) -> float:
        return sum(self.recent_c_earnings_deltas)

    def macro_block_reason(self, side: Side) -> str:
        if not self.have_real_eps_c:
            return "no_real_c_baseline"
        if len(self.recent_c_earnings_deltas) < 2:
            return ""
        trend = self.c_earnings_trend()
        if side == Side.BUY and trend < -MACRO_EARNINGS_TREND_BLOCK:
            return f"earnings_trend_conflict_{trend:+.4f}"
        if side == Side.SELL and trend > MACRO_EARNINGS_TREND_BLOCK:
            return f"earnings_trend_conflict_{trend:+.4f}"
        return ""

    def earnings_reversal_block_reason(self, side: Side, delta: float) -> str:
        if len(self.recent_c_earnings_deltas) < 3:
            return ""
        previous_trend = sum(list(self.recent_c_earnings_deltas)[:-1])
        if abs(delta) >= EARNINGS_SMALL_DELTA:
            return ""
        if side == Side.BUY and previous_trend < -REVERSAL_TREND_MIN_ABS:
            return f"weak_buy_against_earnings_trend_{previous_trend:+.4f}"
        if side == Side.SELL and previous_trend > REVERSAL_TREND_MIN_ABS:
            return f"weak_sell_against_earnings_trend_{previous_trend:+.4f}"
        return ""

    def set_cpi_signal(self, actual: float, forecast: float, tick: int, raw_news: object = "") -> None:
        surprise = actual - forecast
        target = self.c_target_for_cpi_surprise(surprise)
        self.last_cpi_surprise = surprise
        self.last_cpi_bias_bp = self.clip(surprise * CPI_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
        self.last_cpi_ts = self.now()
        self.last_rate_bias_ttl = CPI_SIGNAL_TTL_SECONDS

        if target == 0:
            self.active_signal = None
            logger.info(
                "CPI inside deadband: tick=%d actual=%.6f forecast=%.6f surprise=%+.6f",
                tick,
                actual,
                forecast,
                surprise,
            )
            self.log_csv(
                "cpi_deadband",
                tick=tick,
                cpi_actual=f"{actual:.6f}",
                cpi_forecast=f"{forecast:.6f}",
                message=f"actual={actual:.6f} forecast={forecast:.6f} surprise={surprise:+.6f}",
                raw_news=raw_news,
            )
            return

        if not self.have_real_eps_c:
            self.active_signal = None
            logger.info(
                "CPI signal blocked before real C baseline: tick=%d actual=%.6f forecast=%.6f surprise=%+.6f",
                tick,
                actual,
                forecast,
                surprise,
            )
            self.log_csv(
                "cpi_blocked",
                tick=tick,
                side=self.side_name(Side.SELL if surprise > 0 else Side.BUY),
                qty=target,
                reason="no_real_c_baseline",
                cpi_actual=f"{actual:.6f}",
                cpi_forecast=f"{forecast:.6f}",
                message=f"actual={actual:.6f} forecast={forecast:.6f} surprise={surprise:+.6f}",
                raw_news=raw_news,
            )
            return

        side = Side.SELL if surprise > 0 else Side.BUY
        direction = "hot_short_C" if side == Side.SELL else "cool_long_C"
        self.active_signal = CSignal(
            side=side,
            target=target,
            thesis="cpi",
            strength=abs(surprise),
            tick=tick,
            ts=self.now(),
            description=direction,
        )
        self.last_c_signal_ts = self.now()
        logger.info(
            "CPI C signal: tick=%d actual=%.6f forecast=%.6f surprise=%+.6f side=%s target=%d bias_bp=%+.2f",
            tick,
            actual,
            forecast,
            surprise,
            self.side_name(side),
            target,
            self.last_cpi_bias_bp,
        )
        self.log_csv(
            "cpi_signal",
            tick=tick,
            side=self.side_name(side),
            qty=target,
            cpi_actual=f"{actual:.6f}",
            cpi_forecast=f"{forecast:.6f}",
            message=f"actual={actual:.6f} forecast={forecast:.6f} surprise={surprise:+.6f}",
            raw_news=raw_news,
        )

    def set_macro_headline_signal(self, score: float, text: str, tick: int, raw_news: object = "") -> bool:
        target = self.c_target_for_macro_score(score)
        if target == 0:
            return False

        side = Side.SELL if score > 0 else Side.BUY
        direction = "SELL" if side == Side.SELL else "BUY"

        existing = self.active_signal
        if existing is not None and existing.thesis == "earnings" and existing.side == side and self.c_signal_is_fresh():
            existing.target = max(existing.target, target)
            existing.ts = self.now()
            if "macro_confirm" not in existing.description:
                existing.description = f"{existing.description}|macro_confirm"
            self.last_cpi_surprise = 0.0
            self.last_cpi_bias_bp = self.clip(score * MACRO_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
            self.last_cpi_ts = self.now()
            self.last_rate_bias_ttl = MACRO_SIGNAL_TTL_SECONDS
            logger.info(
                "Macro confirms fresh earnings C signal: tick=%d score=%+.2f side=%s target=%d kept_thesis=earnings headline=%s",
                tick,
                score,
                direction,
                existing.target,
                text,
            )
            self.log_csv(
                "macro_signal",
                tick=tick,
                side=self.side_name(side),
                qty=existing.target,
                reason="confirmed_earnings_signal",
                message=f"score={score:+.2f} headline={text[:180]}",
                macro_score=f"{score:.4f}",
                macro_target=existing.target,
                macro_direction=self.side_name(side),
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                raw_news=raw_news,
            )
            return True

        flat_cap_applied = self.pos.get(C_SYMBOL, 0) == 0 and target > MACRO_FLAT_TARGET_CAP
        if flat_cap_applied:
            target = MACRO_FLAT_TARGET_CAP

        block_reason = self.macro_block_reason(side)
        if target >= 160 and block_reason.startswith("earnings_trend_conflict"):
            block_reason = ""
        if block_reason:
            logger.info(
                "Macro signal blocked: tick=%d score=%+.2f side=%s target=%d reason=%s headline=%s",
                tick,
                score,
                direction,
                target,
                block_reason,
                text,
            )
            self.log_csv(
                "macro_blocked",
                tick=tick,
                side=direction,
                qty=target,
                reason=block_reason,
                message=f"score={score:+.2f} flat_cap={flat_cap_applied} headline={text[:180]}",
                macro_score=f"{score:.4f}",
                macro_target=target,
                macro_direction=direction,
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                raw_news=raw_news,
            )
            return False

        direction = "hawkish_macro_short_C" if side == Side.SELL else "dovish_macro_long_C"
        self.last_cpi_surprise = 0.0
        self.last_cpi_bias_bp = self.clip(score * MACRO_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
        self.last_cpi_ts = self.now()
        self.last_rate_bias_ttl = MACRO_SIGNAL_TTL_SECONDS
        self.active_signal = CSignal(
            side=side,
            target=target,
            thesis="macro",
            strength=abs(score),
            tick=tick,
            ts=self.now(),
            description=direction,
        )
        self.last_c_signal_ts = self.now()
        logger.info(
            "Macro C signal: tick=%d score=%+.2f side=%s target=%d bias_bp=%+.2f headline=%s",
            tick,
            score,
            self.side_name(side),
            target,
            self.last_cpi_bias_bp,
            text,
        )
        self.log_csv(
            "macro_signal",
            tick=tick,
            side=self.side_name(side),
            qty=target,
            reason="flat_macro_cap" if flat_cap_applied else "",
            message=f"score={score:+.2f} flat_cap={flat_cap_applied} headline={text[:180]}",
            macro_score=f"{score:.4f}",
            macro_target=target,
            macro_direction=self.side_name(side),
            earnings_trend=f"{self.c_earnings_trend():+.6f}",
            raw_news=raw_news,
        )
        return True

    def score_macro_headline_for_c(self, text: str) -> float:
        lower = text.lower()
        score = 0.0

        # Positive score is hawkish / rates-up pressure, which should push C lower.
        hawkish_phrases = {
            "push back on market expectations for near-term cuts": 5.0,
            "discomfort with above-target inflation": 4.5,
            "progress on inflation has stalled": 5.0,
            "inflation has stalled": 4.0,
            "not fast enough": 3.0,
            "returning inflation to target": 2.5,
            "even at cost to growth": 3.0,
            "undermines the case for policy easing": 5.0,
            "financial conditions have loosened": 3.0,
            "stoking demand": 3.0,
            "underlying inflation elevated": 4.0,
            "inflation elevated": 3.0,
            "core services keep": 2.5,
            "above-target inflation": 3.5,
            "persistent inflation": 3.5,
            "inflation risks": 3.0,
            "additional tightening": 3.0,
            "upside inflation": 3.0,
            "wage growth and persistent inflation": 4.5,
            "wage growth": 1.5,
            "strong gdp": 1.0,
            "restrictive stance": 1.0,
            "tightening": 1.5,
        }
        dovish_phrases = {
            "restrictive stance may no longer be needed": -4.5,
            "may no longer be needed": -3.5,
            "preemptive cut": -4.0,
            "safeguard growth": -2.0,
            "rate relief": -3.5,
            "do not want to keep rates elevated longer than necessary": -4.0,
            "keep rates elevated longer than necessary": -3.0,
            "real rates are well above neutral": -3.0,
            "wage growth decelerating": -3.5,
            "progress on inflation": -2.0,
            "falling inflation": -2.5,
            "cooling labor": -2.5,
            "next move is more likely a cut than a hike": -4.5,
            "more likely a cut than a hike": -4.0,
            "markets lean toward cuts": -3.5,
            "lean toward cuts": -3.0,
            "shift toward accommodation": -4.0,
            "favoring a shift toward accommodation": -4.5,
            "how long the fed can hold steady": -2.5,
            "slowing gdp growth": -2.0,
            "unemployment claims": -2.0,
            "economic softening": -0.75,
            "manufacturing contraction": -0.75,
            "softening data": -2.0,
            "growth risks": -1.5,
            "easing a key concern": -2.0,
            "policy easing": -2.0,
            "opts to hold steady": -2.5,
            "assess cumulative tightening effects": -3.5,
            "cumulative tightening effects": -3.0,
        }

        for phrase, weight in hawkish_phrases.items():
            if phrase in lower:
                score += weight
        for phrase, weight in dovish_phrases.items():
            if phrase in lower:
                score += weight

        # Generic "near-term cuts" is dovish unless the headline says the Fed is rejecting that view.
        if "near-term cuts" in lower and "push back" not in lower:
            score -= 2.0
        if "rate cuts" in lower and "push back" not in lower:
            score -= 1.5

        neutral_or_mixed = [
            "mixed",
            "unclear",
            "no clear signal",
            "questions than answers",
            "data dependence",
            "await upcoming data",
            "cautious",
            "different directions",
            "conflict",
            "complicates",
            "hawks and doves",
            "both upside and downside",
            "no consensus",
            "uncertain outlook",
            "uncertainties",
            "declines to commit",
            "options open",
        ]
        if any(phrase in lower for phrase in neutral_or_mixed):
            score *= 0.35

        return score

    def set_c_earnings_signal(self, value: float, tick: int) -> None:
        if not self.have_real_eps_c:
            self.have_real_eps_c = True
            self.current_eps_c = value
            self.baseline_eps_c = value
            self.last_c_earnings_delta = 0.0
            self.active_signal = None
            self.maybe_initialize_anchor(force=True)
            logger.info("First C earnings baseline adopted: tick=%d eps=%.4f no_trade=True", tick, value)
            self.log_csv("earnings_baseline", tick=tick, message=f"eps={value:.6f}")
            return

        old_eps = self.current_eps_c
        delta = value - old_eps
        self.current_eps_c = value
        self.last_c_earnings_delta = delta
        self.recent_c_earnings_deltas.append(delta)
        target = self.c_target_for_earnings_delta(delta)
        if tick >= LATE_SESSION_WEAK_EARNINGS_CUTOFF_TICK and abs(delta) < EARNINGS_SMALL_DELTA:
            self.active_signal = None
            logger.info(
                "C earnings signal blocked late session: tick=%d old=%.4f new=%.4f delta=%+.4f target=%d",
                tick,
                old_eps,
                value,
                delta,
                target,
            )
            self.log_csv(
                "earnings_blocked",
                tick=tick,
                side=self.side_name(Side.BUY if delta > 0 else Side.SELL),
                qty=target,
                reason="late_weak_earnings",
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                message=f"old={old_eps:.6f} new={value:.6f} delta={delta:+.6f}",
            )
            return
        if target == 0:
            self.active_signal = None
            logger.info(
                "C earnings inside deadband: tick=%d old=%.4f new=%.4f delta=%+.4f",
                tick,
                old_eps,
                value,
                delta,
            )
            self.log_csv(
                "earnings_deadband",
                tick=tick,
                message=f"old={old_eps:.6f} new={value:.6f} delta={delta:+.6f}",
            )
            return

        side = Side.BUY if delta > 0 else Side.SELL
        if tick >= LATE_SESSION_EARNINGS_TARGET_CAP_TICK and target > LATE_SESSION_EARNINGS_TARGET_CAP:
            target = LATE_SESSION_EARNINGS_TARGET_CAP
        block_reason = self.earnings_reversal_block_reason(side, delta)
        if block_reason:
            self.active_signal = None
            logger.info(
                "C earnings signal blocked: tick=%d old=%.4f new=%.4f delta=%+.4f side=%s target=%d reason=%s",
                tick,
                old_eps,
                value,
                delta,
                self.side_name(side),
                target,
                block_reason,
            )
            self.log_csv(
                "earnings_blocked",
                tick=tick,
                side=self.side_name(side),
                qty=target,
                reason=block_reason,
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                message=f"old={old_eps:.6f} new={value:.6f} delta={delta:+.6f}",
            )
            return

        self.active_signal = CSignal(
            side=side,
            target=target,
            thesis="earnings",
            strength=abs(delta),
            tick=tick,
            ts=self.now(),
            description="earnings_up_long_C" if side == Side.BUY else "earnings_down_short_C",
        )
        self.last_c_signal_ts = self.now()
        logger.info(
            "C earnings signal: tick=%d old=%.4f new=%.4f delta=%+.4f side=%s target=%d",
            tick,
            old_eps,
            value,
            delta,
            self.side_name(side),
            target,
        )
        self.log_csv(
            "earnings_signal",
            tick=tick,
            side=self.side_name(side),
            qty=target,
            message=f"old={old_eps:.6f} new={value:.6f} delta={delta:+.6f}",
        )

    def parse_cpi_from_news(self, news_release: dict) -> Tuple[Optional[float], Optional[float]]:
        kind = str(news_release.get("kind") or "").lower()
        new_data_obj = news_release.get("new_data", {}) or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        subtype = str(new_data.get("structured_subtype") or news_release.get("structured_subtype") or "").lower()
        tick = int(news_release.get("tick") or news_release.get("timestamp") or 0)

        for container in (new_data, news_release):
            if not isinstance(container, dict):
                continue
            actual_value = container.get("actual", container.get("cpi_actual"))
            forecast_value = container.get("forecast", container.get("cpi_forecast"))
            if actual_value is not None and forecast_value is not None:
                return float(actual_value), float(forecast_value)

        if (
            tick in CPI_PRINT_FALLBACKS_BY_TICK
            and kind == "structured"
            and subtype == "petition"
            and int(new_data.get("new_signatures") or 0) == 0
            and int(new_data.get("cumulative") or 0) == 0
        ):
            return CPI_PRINT_FALLBACKS_BY_TICK[tick]

        text = str(new_data.get("content") or news_release.get("content") or news_release)
        lower = text.lower()
        looks_like_cpi = any(token in kind for token in ("cpi", "inflation")) or any(
            token in subtype for token in ("cpi", "inflation")
        )
        has_actual_forecast_words = "actual" in lower and "forecast" in lower
        if not looks_like_cpi and not has_actual_forecast_words:
            return None, None

        actual_match = re.search(r"actual\s*(?:[:=]|is|was|of|at)?\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        forecast_match = re.search(r"forecast\s*(?:[:=]|is|was|of|at)?\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if actual_match and forecast_match:
            return float(actual_match.group(1)), float(forecast_match.group(1))

        nums = [float(x) for x in re.findall(FLOAT_RE, text)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        return None, None

    def parse_c_earnings_from_news(self, news_release: dict) -> Optional[float]:
        kind = str(news_release.get("kind") or "").lower()
        new_data_obj = news_release.get("new_data", {}) or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        subtype = str(new_data.get("structured_subtype") or news_release.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or news_release.get("asset") or "").upper()
        value = new_data.get("value", news_release.get("value"))
        if "earnings" in subtype and asset == C_SYMBOL and value is not None:
            return float(value)

        text = str(new_data.get("content") or news_release.get("content") or (new_data_obj if not isinstance(new_data_obj, dict) else ""))
        lower = text.lower()
        if "earnings" not in kind and "earnings" not in lower:
            return None
        match = re.search(r"\bC\s+earnings\s+released\s*:?\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def describe_news_release(self, news_release: dict) -> str:
        new_data_obj = news_release.get("new_data", {}) or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        text = str(new_data.get("content") or news_release.get("content") or "")
        if text:
            return text[:180]

        subtype = str(new_data.get("structured_subtype") or news_release.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or news_release.get("asset") or "")
        if subtype == "earnings":
            value = new_data.get("value", news_release.get("value"))
            return f"{asset or 'unknown'} earnings value={value}"
        if subtype == "petition":
            new_signatures = new_data.get("new_signatures", "")
            cumulative = new_data.get("cumulative", "")
            return f"petition new_signatures={new_signatures} cumulative={cumulative}"
        if new_data:
            return str(new_data)[:180]
        return str(news_release)[:180]

    def mark_c_unrealized(self) -> Tuple[float, str]:
        pos = self.pos.get(C_SYMBOL, 0)
        if pos == 0 or self.c_avg_entry is None:
            self.c_unrealized_pnl = 0.0
            self.c_total_pnl = self.c_realized_pnl
            self.c_peak_total_pnl = max(self.c_peak_total_pnl, self.c_total_pnl)
            return 0.0, "flat"

        bid, ask = self.book_top(C_SYMBOL)
        mark_source = "book"
        if pos > 0:
            if bid is not None:
                mark = float(bid)
            else:
                mark = self.mid(C_SYMBOL)
                mark_source = "fallback"
            if mark is None:
                mark = float(self.c_last_fill_price or self.c_avg_entry)
                mark_source = "fallback"
            unrealized = (mark - self.c_avg_entry) * pos
        else:
            if ask is not None:
                mark = float(ask)
            else:
                mark = self.mid(C_SYMBOL)
                mark_source = "fallback"
            if mark is None:
                mark = float(self.c_last_fill_price or self.c_avg_entry)
                mark_source = "fallback"
            unrealized = (self.c_avg_entry - mark) * abs(pos)

        self.c_unrealized_pnl = unrealized
        self.c_total_pnl = self.c_realized_pnl + self.c_unrealized_pnl
        self.c_peak_total_pnl = max(self.c_peak_total_pnl, self.c_total_pnl)
        return unrealized, mark_source

    def c_profit_lock_triggered(self) -> bool:
        if self.c_peak_total_pnl < C_SESSION_PROFIT_LOCK_START:
            return False
        return self.c_total_pnl <= self.c_peak_total_pnl - C_SESSION_PROFIT_LOCK_GIVEBACK

    def update_c_pnl_from_fill(self, old_pos: int, new_pos: int, signed_qty: int, price: int) -> None:
        fill_abs = abs(signed_qty)
        if fill_abs == 0:
            return

        if old_pos == 0 or self.c_avg_entry is None:
            self.c_avg_entry = float(price) if new_pos != 0 else None
            self.c_peak_abs = abs(new_pos)
            self.c_exit_stage = 0
            self.c_best_profit_ticks = 0.0
            self.c_regime_started_at = self.now() if new_pos != 0 else 0.0
            return

        if self.sign(old_pos) == self.sign(signed_qty):
            old_abs = abs(old_pos)
            new_abs = old_abs + fill_abs
            self.c_avg_entry = (self.c_avg_entry * old_abs + price * fill_abs) / new_abs
            self.c_peak_abs = max(self.c_peak_abs, abs(new_pos))
            self.c_exit_stage = 0
            return

        closed_qty = min(abs(old_pos), fill_abs)
        if old_pos > 0:
            self.c_realized_pnl += (price - self.c_avg_entry) * closed_qty
        else:
            self.c_realized_pnl += (self.c_avg_entry - price) * closed_qty

        if new_pos == 0:
            self.c_avg_entry = None
            self.c_peak_abs = 0
            self.c_exit_stage = 0
            self.c_best_profit_ticks = 0.0
            self.c_thesis = "flat"
            self.c_regime_started_at = 0.0
            self.active_signal = None
        elif self.sign(old_pos) != self.sign(new_pos):
            residual = fill_abs - closed_qty
            self.c_avg_entry = float(price)
            self.c_peak_abs = residual
            self.c_exit_stage = 0
            self.c_best_profit_ticks = 0.0
            self.c_regime_started_at = self.now()

    def current_mtm(self) -> float:
        pos = self.pos.get(C_SYMBOL, 0)
        mid_c = self.mid(C_SYMBOL)
        c_mark = 0.0 if mid_c is None else pos * mid_c
        return self.cash + c_mark

    async def send_c_order(self, side: Side, qty: int, px: int, tag: str) -> bool:
        symbol = C_SYMBOL
        is_entry_order = tag.endswith("_entry")
        is_mm_order = self.is_mm_tag(tag)
        if self.kill_switch and tag not in {"risk_flatten", "profit_lock"}:
            return False
        if is_entry_order and self.now() < self.symbol_cooldown_until[symbol]:
            return False
        if is_mm_order:
            if self.has_live_non_mm_order(symbol) or self.live_c_order(side, mm_only=True, include_pending=True) is not None:
                return False
        elif self.has_live_order(symbol):
            return False

        qty = self.order_qty(symbol, side, qty)
        if qty <= 0:
            return False

        px = self.clamp_price(px)
        try:
            oid = str(await self.place_order(symbol, qty, side, px))
        except Exception as e:
            logger.exception("place_order failed: symbol=%s side=%s qty=%d px=%d error=%s", symbol, self.side_name(side), qty, px, e)
            return False

        self.live_orders[oid] = OrderRef(
            order_id=oid,
            symbol=symbol,
            side=side,
            qty=qty,
            price=px,
            tag=tag,
            ts=self.now(),
        )
        if side == Side.BUY:
            self.pending_buy[symbol] += qty
        else:
            self.pending_sell[symbol] += qty
        self.symbol_cooldown_until[symbol] = max(
            self.symbol_cooldown_until[symbol],
            self.now() + C_SYMBOL_COOLDOWN_SECONDS,
        )
        logger.info(
            "Placed C order: tag=%s side=%s qty=%d price=%d pos_C=%d avg_C=%s realized=%.2f unrealized=%.2f total_pnl=%.2f",
            tag,
            self.side_name(side),
            qty,
            px,
            self.pos[C_SYMBOL],
            "na" if self.c_avg_entry is None else f"{self.c_avg_entry:.2f}",
            self.c_realized_pnl,
            self.c_unrealized_pnl,
            self.c_total_pnl,
        )
        self.log_csv(
            "order_placed",
            symbol=symbol,
            side=self.side_name(side),
            qty=qty,
            price=px,
            order_id=oid,
            tag=tag,
        )
        return True

    async def request_cancel_order(self, oid: str) -> bool:
        if oid in self.pending_cancels:
            return False
        self.pending_cancels.add(oid)
        try:
            await self.cancel_order(oid)
            return True
        except Exception:
            self.pending_cancels.discard(oid)
            return False

    async def cancel_stale_orders(self) -> None:
        now = self.now()
        for oid, ref in list(self.live_orders.items()):
            if oid in self.pending_cancels:
                continue
            if self.is_mm_tag(ref.tag):
                continue
            if now - ref.ts < ORDER_REPRICE_SECONDS:
                continue
            await self.request_cancel_order(oid)

    async def cancel_c_mm_orders(self, reason: str) -> bool:
        had_mm = False
        for oid, ref in list(self.live_orders.items()):
            if ref.symbol != C_SYMBOL or not self.is_mm_tag(ref.tag):
                continue
            had_mm = True
            if oid in self.pending_cancels:
                continue
            if await self.request_cancel_order(oid):
                logger.info("Cancelling C MM order %s because %s", oid, reason)
        return had_mm

    async def flatten_c(self, reason: str, qty: Optional[int] = None) -> bool:
        pos = self.pos.get(C_SYMBOL, 0)
        if pos == 0 or self.has_live_order(C_SYMBOL):
            return False
        side = Side.SELL if pos > 0 else Side.BUY
        px = self.best_price_for_cross(C_SYMBOL, side)
        if px is None:
            return False
        exit_qty = min(abs(pos), ORDER_LIMIT if qty is None else qty)
        placed = await self.send_c_order(side, exit_qty, px, reason)
        if placed:
            if reason in {
                "gap_compress",
                "max_hold",
                "stop_loss",
                "profit_trail",
                "risk_flatten",
                "profit_lock",
                "contradict_signal",
                "fast_harvest",
                "fast_timeout",
            }:
                self.c_exit_stage = max(self.c_exit_stage, 3)
            logger.info("C exit trigger: %s pos_C=%d qty=%d", reason, pos, exit_qty)
            self.log_csv("exit_trigger", reason=reason, qty=exit_qty, tag=reason)
        return placed

    def entry_blocked_by_risk(self, side: Side) -> Tuple[bool, str]:
        if self.profit_lock_engaged:
            return True, "profit_lock"
        if self.kill_switch:
            return True, "kill_switch"
        spread = self.spread(C_SYMBOL)
        if spread is None:
            return True, "missing_C_book"
        if spread > MAX_C_SPREAD:
            return True, f"wide_spread_{spread}"
        if self.has_live_order(C_SYMBOL):
            return True, "live_order"

        pos = self.pos.get(C_SYMBOL, 0)
        if pos != 0 and self.c_avg_entry is not None:
            bid, ask = self.book_top(C_SYMBOL)
            if pos > 0 and bid is not None and self.c_avg_entry - bid >= ADVERSE_STOP_ADD_TICKS:
                return True, "adverse_stop_add"
            if pos < 0 and ask is not None and ask - self.c_avg_entry >= ADVERSE_STOP_ADD_TICKS:
                return True, "adverse_stop_add"

        fair = self.fair_value_c()
        sig = self.active_signal
        allow_guard_override = sig is not None and (
            (sig.thesis == "cpi" and sig.target >= 160)
            or (sig.thesis == "earnings" and sig.target >= 160)
            or (sig.thesis == "macro" and sig.target >= 160)
        )
        if fair is not None and not allow_guard_override:
            px = self.best_price_for_cross(C_SYMBOL, side)
            if px is None:
                return True, "missing_C_book"
            guard = FAIR_VALUE_GUARD_TICKS
            if sig is not None and sig.thesis == "macro":
                guard = MACRO_FAIR_VALUE_GUARD_TICKS
            elif sig is not None and sig.thesis == "earnings":
                if sig.strength >= EARNINGS_EXTREME_DELTA or sig.target >= 200:
                    guard = EARNINGS_EXTREME_FAIR_VALUE_GUARD_TICKS
                elif sig.strength >= EARNINGS_MEDIUM_DELTA:
                    guard = EARNINGS_STRONG_FAIR_VALUE_GUARD_TICKS
                else:
                    guard = EARNINGS_WEAK_FAIR_VALUE_GUARD_TICKS
            if side == Side.BUY and px > fair + guard:
                return True, f"buy_over_fair_px_{px}_fair_{fair:.1f}"
            if side == Side.SELL and px < fair - guard:
                return True, f"sell_under_fair_px_{px}_fair_{fair:.1f}"
        return False, "ok"

    def c_mm_allowed_size(self) -> Tuple[int, int]:
        pos = int(self.pos.get(C_SYMBOL, 0))
        buy_pending = int(self.pending_buy.get(C_SYMBOL, 0))
        sell_pending = int(self.pending_sell.get(C_SYMBOL, 0))
        allowed_buy = max(0, C_MM_MAX_POSITION - (pos + buy_pending))
        allowed_sell = max(0, C_MM_MAX_POSITION + pos - sell_pending)
        return allowed_buy, allowed_sell

    def c_mm_passive_qty(self, side: Side, allowed: int) -> int:
        if allowed <= 0:
            return 0
        pos = int(self.pos.get(C_SYMBOL, 0))
        base_qty = min(C_MM_QUOTE_SIZE, allowed)
        worsening_side = (pos > 0 and side == Side.BUY) or (pos < 0 and side == Side.SELL)
        if not worsening_side:
            return base_qty
        abs_pos = abs(pos)
        if abs_pos >= C_MM_REDUCE_ONLY_POSITION:
            return 0
        if abs_pos >= C_MM_PASSIVE_REDUCE_FULL:
            return min(1, allowed)
        if abs_pos >= C_MM_PASSIVE_REDUCE_START:
            return min(max(1, C_MM_QUOTE_SIZE - 1), allowed)
        return base_qty

    def c_mm_desired_quotes(self) -> Dict[Side, Tuple[int, int]]:
        if not C_MM_ENABLED or self.kill_switch or self.profit_lock_engaged:
            return {}
        if self.active_signal is not None and self.c_signal_is_fresh():
            return {}
        if self.now() - self.last_c_signal_ts < C_MM_NEWS_COOLDOWN_SECONDS:
            return {}
        pos = int(self.pos.get(C_SYMBOL, 0))
        if pos != 0 and self.c_thesis not in {"flat", "mm"}:
            return {}
        fair = self.fair_value_c()
        bid, ask = self.book_top(C_SYMBOL)
        spread = self.spread(C_SYMBOL)
        if fair is None or bid is None or ask is None or spread is None:
            return {}
        if spread < C_MM_MIN_BOOK_SPREAD or spread > MAX_C_SPREAD:
            return {}

        reservation = fair - (C_MM_INVENTORY_SKEW * pos)
        bid_px = int(math.floor(reservation - C_MM_HALF_SPREAD_TICKS))
        ask_px = int(math.ceil(reservation + C_MM_HALF_SPREAD_TICKS))
        bid_px = min(bid_px, ask - 1)
        ask_px = max(ask_px, bid + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1
        bid_px = self.clamp_price(bid_px)
        ask_px = self.clamp_price(ask_px)

        allowed_buy, allowed_sell = self.c_mm_allowed_size()
        bid_qty = self.c_mm_passive_qty(Side.BUY, allowed_buy)
        ask_qty = self.c_mm_passive_qty(Side.SELL, allowed_sell)
        desired: Dict[Side, Tuple[int, int]] = {}
        if bid_qty > 0:
            desired[Side.BUY] = (bid_px, bid_qty)
        if ask_qty > 0:
            desired[Side.SELL] = (ask_px, ask_qty)
        return desired

    async def sync_c_market_maker(self) -> bool:
        if self.has_live_non_mm_order(C_SYMBOL):
            return False
        desired = self.c_mm_desired_quotes()
        now = self.now()
        acted = False

        for side in (Side.BUY, Side.SELL):
            live = self.live_c_order(side, mm_only=True, include_pending=True)
            desired_quote = desired.get(side)
            if live is None:
                continue
            if desired_quote is None:
                if await self.request_cancel_order(live.order_id):
                    logger.info("Cancelling C MM %s because desired quote removed", live.order_id)
                    acted = True
                continue
            desired_px, desired_qty = desired_quote
            price_gap = abs(live.price - desired_px)
            stale = now - live.ts >= C_MM_QUOTE_TTL_SECONDS
            qty_changed = live.qty != desired_qty
            if qty_changed or stale or price_gap >= C_MM_REPRICE_THRESHOLD_TICKS:
                if await self.request_cancel_order(live.order_id):
                    logger.info(
                        "Cancelling C MM %s for reprice: side=%s live=%d/%d desired=%d/%d gap=%d stale=%s",
                        live.order_id,
                        self.side_name(side),
                        live.price,
                        live.qty,
                        desired_px,
                        desired_qty,
                        price_gap,
                        stale,
                    )
                    acted = True

        if acted:
            return True

        for side in (Side.BUY, Side.SELL):
            if self.live_c_order(side, mm_only=True, include_pending=True) is not None:
                continue
            desired_quote = desired.get(side)
            if desired_quote is None:
                continue
            px, qty = desired_quote
            tag = "mm_bid" if side == Side.BUY else "mm_ask"
            if await self.send_c_order(side, qty, px, tag):
                self.c_thesis = "mm" if self.pos.get(C_SYMBOL, 0) != 0 else self.c_thesis
                self.last_c_action_ts = now
                acted = True

        if acted:
            fair = self.fair_value_c()
            bid, ask = self.book_top(C_SYMBOL)
            logger.info(
                "C MM sync: pos=%d fair=%s book=%s/%s desired=%s",
                self.pos.get(C_SYMBOL, 0),
                "na" if fair is None else f"{fair:.2f}",
                bid if bid is not None else "na",
                ask if ask is not None else "na",
                {self.side_name(side): quote for side, quote in desired.items()},
            )
            self.log_csv("mm_sync", tag="market_making", message=str({self.side_name(side): quote for side, quote in desired.items()}))
        return acted

    async def maybe_exit_c(self) -> bool:
        pos = self.pos.get(C_SYMBOL, 0)
        if pos == 0:
            return False
        if self.has_live_order(C_SYMBOL):
            return False

        self.mark_c_unrealized()
        if self.c_total_pnl <= C_TOTAL_PNL_HARD_STOP:
            self.kill_switch = True
            return await self.flatten_c("risk_flatten")
        if self.c_profit_lock_triggered():
            self.profit_lock_engaged = True
            self.kill_switch = True
            return await self.flatten_c("profit_lock")

        bid, ask = self.book_top(C_SYMBOL)
        gap = self.fair_gap()
        spread = self.spread(C_SYMBOL)
        favorable_gap = None if gap is None else (gap if pos > 0 else -gap)
        regime_age = 0.0 if self.c_regime_started_at <= 0.0 else self.now() - self.c_regime_started_at
        observed_profit = None
        (
            profit_1_ticks,
            profit_2_ticks,
            profit_full_ticks,
            trail_start_ticks,
            trail_giveback_ticks,
            adverse_flatten_ticks,
        ) = self.c_exit_profile()
        if self.c_avg_entry is not None:
            if pos > 0 and bid is not None:
                adverse = self.c_avg_entry - bid
                profit = bid - self.c_avg_entry
                if adverse >= adverse_flatten_ticks:
                    return await self.flatten_c("stop_loss")
                self.c_best_profit_ticks = max(self.c_best_profit_ticks, profit)
                observed_profit = profit
                if self.c_best_profit_ticks >= trail_start_ticks and profit <= self.c_best_profit_ticks - trail_giveback_ticks:
                    return await self.flatten_c("profit_trail")
                if profit >= profit_full_ticks:
                    return await self.flatten_c("long_profit_full")
                if profit >= profit_2_ticks and self.c_exit_stage < 2:
                    placed = await self.flatten_c("long_profit_2", ORDER_LIMIT)
                    if placed:
                        self.c_exit_stage = 2
                    return placed
                if profit >= profit_1_ticks and self.c_exit_stage < 1:
                    placed = await self.flatten_c("long_profit_1", ORDER_LIMIT)
                    if placed:
                        self.c_exit_stage = 1
                    return placed
            elif pos < 0 and ask is not None:
                adverse = ask - self.c_avg_entry
                profit = self.c_avg_entry - ask
                if adverse >= adverse_flatten_ticks:
                    return await self.flatten_c("stop_loss")
                self.c_best_profit_ticks = max(self.c_best_profit_ticks, profit)
                observed_profit = profit
                if self.c_best_profit_ticks >= trail_start_ticks and profit <= self.c_best_profit_ticks - trail_giveback_ticks:
                    return await self.flatten_c("profit_trail")
                if profit >= profit_full_ticks:
                    return await self.flatten_c("short_profit_full")
                if profit >= profit_2_ticks and self.c_exit_stage < 2:
                    placed = await self.flatten_c("short_profit_2", ORDER_LIMIT)
                    if placed:
                        self.c_exit_stage = 2
                    return placed
                if profit >= profit_1_ticks and self.c_exit_stage < 1:
                    placed = await self.flatten_c("short_profit_1", ORDER_LIMIT)
                    if placed:
                        self.c_exit_stage = 1
                    return placed

        if self.c_thesis == "earnings":
            if observed_profit is None and regime_age >= EARNINGS_FAST_TIMEOUT_SECONDS:
                return await self.flatten_c("fast_timeout", ORDER_LIMIT)
            if observed_profit is not None:
                if regime_age >= EARNINGS_FAST_HARVEST_SECONDS and (
                    self.c_exit_stage > 0 or observed_profit >= EARNINGS_FAST_HARVEST_MIN_TICKS
                ):
                    return await self.flatten_c("fast_harvest", ORDER_LIMIT)
                if regime_age >= EARNINGS_FAST_TIMEOUT_SECONDS and observed_profit <= 0:
                    return await self.flatten_c("fast_timeout", ORDER_LIMIT)

        sig = self.active_signal
        if sig is not None and self.c_signal_is_fresh():
            desired_sign = 1 if sig.side == Side.BUY else -1
            if self.sign(pos) != desired_sign:
                return await self.flatten_c("contradict_signal")

        if regime_age >= self.c_signal_max_hold():
            if favorable_gap is None or favorable_gap <= STRONG_FAIR_GAP_HOLD_TICKS:
                return await self.flatten_c("max_hold")
            if self.c_best_profit_ticks > 0 and self.c_exit_stage < 1:
                placed = await self.flatten_c("max_hold_profit_trim", ORDER_LIMIT)
                if placed:
                    self.c_exit_stage = 1
                return placed

        if self.c_thesis == "macro" and gap is not None and spread is not None and regime_age >= 2.0:
            if favorable_gap <= max(4.0, float(spread)):
                return await self.flatten_c("gap_compress")

        return False

    async def maybe_enter_or_add_c(self) -> bool:
        sig = self.active_signal
        if sig is None or not self.c_signal_is_fresh():
            return False
        if self.now() - self.last_c_action_ts < C_ACTION_COOLDOWN_SECONDS:
            return False
        if self.c_exit_stage > 0:
            return False

        pos = self.pos.get(C_SYMBOL, 0)
        desired_sign = 1 if sig.side == Side.BUY else -1
        if pos != 0 and self.sign(pos) != desired_sign:
            return False
        if abs(pos) >= sig.target:
            return False

        blocked, reason = self.entry_blocked_by_risk(sig.side)
        if blocked:
            block_key = f"{reason}:{sig.thesis}:{sig.target}:{pos}"
            if block_key != self.last_entry_block_key or self.now() - self.last_entry_block_log_ts >= 1.5:
                logger.info("C entry blocked: reason=%s thesis=%s target=%d pos_C=%d", reason, sig.thesis, sig.target, pos)
                self.log_csv("entry_blocked", reason=reason, tag=sig.thesis, qty=sig.target)
                self.last_entry_block_key = block_key
                self.last_entry_block_log_ts = self.now()
            return False

        px = self.best_price_for_cross(C_SYMBOL, sig.side)
        if px is None:
            return False

        qty = min(ORDER_LIMIT, sig.target - abs(pos))
        qty = self.order_qty(C_SYMBOL, sig.side, qty)
        if qty <= 0:
            return False

        placed = await self.send_c_order(sig.side, qty, px, f"{sig.thesis}_entry")
        if placed:
            self.last_c_action_ts = self.now()
            self.c_thesis = sig.thesis
            if self.c_regime_started_at <= 0.0:
                self.c_regime_started_at = self.now()
            fair = self.fair_value_c()
            mid_c = self.mid(C_SYMBOL)
            logger.info(
                "C entry: thesis=%s side=%s qty=%d strength=%.6f target=%d fair=%s mid=%s",
                sig.thesis,
                self.side_name(sig.side),
                qty,
                sig.strength,
                sig.target,
                "na" if fair is None else f"{fair:.2f}",
                "na" if mid_c is None else f"{mid_c:.2f}",
            )
            self.log_csv(
                "entry_decision",
                symbol=C_SYMBOL,
                side=self.side_name(sig.side),
                qty=qty,
                price=px,
                tag=sig.thesis,
                message=f"strength={sig.strength:.6f} target={sig.target}",
            )
        return placed

    async def maybe_trade(self) -> None:
        if self.decision_lock.locked():
            return
        async with self.decision_lock:
            self.maybe_initialize_anchor()
            self.mark_c_unrealized()
            await self.cancel_stale_orders()
            if self.active_signal is not None and self.c_signal_is_fresh():
                if await self.cancel_c_mm_orders("fresh directional C signal"):
                    self.log_heartbeat(force=True)
                    return
            if self.pos.get(C_SYMBOL, 0) != 0 and self.c_thesis not in {"flat", "mm"}:
                if await self.cancel_c_mm_orders("directional C inventory active"):
                    self.log_heartbeat(force=True)
                    return
            if await self.maybe_exit_c():
                self.log_heartbeat(force=True)
                return
            if await self.maybe_enter_or_add_c():
                self.log_heartbeat(force=True)
                return
            if await self.sync_c_market_maker():
                self.log_heartbeat(force=True)
                return
            self.log_heartbeat()

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        for symbol in self.pos:
            self.pos[symbol] = int(self.positions.get(symbol, 0))
        self.cash = float(self.positions.get("cash", 0))
        if self.session_start_cash is None:
            self.session_start_cash = self.cash
            self.session_start_mtm = self.current_mtm()
        if self.pos[C_SYMBOL] == 0:
            self.c_avg_entry = None
            self.c_peak_abs = 0
            self.c_thesis = "flat"
        elif self.c_avg_entry is None:
            inherited_mark = self.mid(C_SYMBOL) or float(self.c_last_fill_price or 0)
            if inherited_mark > 0:
                self.c_avg_entry = float(inherited_mark)
                self.c_peak_abs = abs(self.pos[C_SYMBOL])
                self.c_thesis = "inherited"
                self.c_regime_started_at = self.now()
        self.mark_c_unrealized()
        logger.info(
            "Startup C-only mode: C=%d H=%d HOLD=%d CUT=%d cash=%.2f inherited_PM_no_trade=True",
            self.pos[C_SYMBOL],
            self.pos[R_HIKE],
            self.pos[R_HOLD],
            self.pos[R_CUT],
            self.cash,
        )
        self.log_csv("startup", message="C-only mode; PM read-only")

    async def handle_order_fill(self, msg) -> None:
        order_info = self.open_orders.get(msg.id)
        if order_info is None:
            logger.warning(
                "Ignoring fill for unknown/stale order_id=%s qty=%d price=%d",
                msg.id,
                msg.qty,
                msg.px,
            )
            return

        request = order_info[0]
        symbol = request.symbol
        fill_qty = int(msg.qty)
        fill_price = int(msg.px)
        is_buy = int(request.side) == 1

        self.positions[symbol] += fill_qty * (1 if is_buy else -1)
        self.positions["cash"] += fill_qty * fill_price * (-1 if is_buy else 1)

        order_info[1] -= fill_qty
        if order_info[1] <= 0 and not order_info[2]:
            self.open_orders.pop(msg.id, None)

        await self.bot_handle_order_fill(msg.id, fill_qty, fill_price)

    async def handle_order_rejected(self, msg) -> None:
        await self.bot_handle_order_rejected(msg.id, msg.reason)
        self.open_orders.pop(msg.id, None)

    async def handle_cancel_response(self, msg) -> None:
        result_type = msg.WhichOneof("result")
        if result_type == "ok":
            await self.bot_handle_cancel_response(msg.id, True, None)
            self.open_orders.pop(msg.id, None)
        else:
            await self.bot_handle_cancel_response(msg.id, False, msg.error)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        oid = str(order_id)
        self.pending_cancels.discard(oid)
        ref = self.live_orders.pop(oid, None)
        if ref is not None:
            if ref.side == Side.BUY:
                self.pending_buy[ref.symbol] = max(0, self.pending_buy[ref.symbol] - ref.qty)
            else:
                self.pending_sell[ref.symbol] = max(0, self.pending_sell[ref.symbol] - ref.qty)
            self.symbol_cooldown_until[ref.symbol] = max(
                self.symbol_cooldown_until[ref.symbol],
                self.now() + C_SYMBOL_COOLDOWN_SECONDS,
            )
        logger.info("Cancel acknowledged: order_id=%s success=%s error=%s", oid, success, error)
        self.log_csv(
            "cancel_response",
            order_id=oid,
            symbol="" if ref is None else ref.symbol,
            side="" if ref is None else self.side_name(ref.side),
            qty="" if ref is None else ref.qty,
            price="" if ref is None else ref.price,
            tag="" if ref is None else ref.tag,
            message=f"success={success} error={error}",
        )

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        oid = str(order_id)
        ref = self.live_orders.get(oid)
        if ref is None:
            await self.maybe_trade()
            return

        old_pos = self.pos.get(ref.symbol, 0)
        signed = qty if ref.side == Side.BUY else -qty
        self.pos[ref.symbol] = old_pos + signed
        self.cash -= signed * price
        self.c_last_fill_price = price if ref.symbol == C_SYMBOL else self.c_last_fill_price

        if ref.side == Side.BUY:
            self.pending_buy[ref.symbol] = max(0, self.pending_buy[ref.symbol] - qty)
        else:
            self.pending_sell[ref.symbol] = max(0, self.pending_sell[ref.symbol] - qty)

        if ref.symbol == C_SYMBOL:
            self.update_c_pnl_from_fill(old_pos, self.pos[ref.symbol], signed, price)
            if self.is_mm_tag(ref.tag) and self.pos[ref.symbol] != 0 and self.active_signal is None:
                self.c_thesis = "mm"
            if self.pos[ref.symbol] == 0:
                self.active_signal = None

        ref.qty -= qty
        if ref.qty <= 0:
            self.live_orders.pop(oid, None)

        unrealized, mark_source = self.mark_c_unrealized()
        fair = self.fair_value_c()
        bid, ask = self.book_top(C_SYMBOL)
        logger.info(
            "Fill: order_id=%s symbol=%s qty=%d price=%d pos_C=%d avg_C=%s realized=%.2f unrealized=%.2f total_pnl=%.2f cash=%.2f session_mtm=%.2f thesis=%s fair=%s C=%s/%s mark=%s",
            oid,
            ref.symbol,
            qty,
            price,
            self.pos[C_SYMBOL],
            "na" if self.c_avg_entry is None else f"{self.c_avg_entry:.2f}",
            self.c_realized_pnl,
            unrealized,
            self.c_total_pnl,
            self.cash,
            self.c_total_pnl,
            self.c_thesis,
            "na" if fair is None else f"{fair:.2f}",
            bid if bid is not None else "na",
            ask if ask is not None else "na",
            mark_source,
        )
        self.log_csv(
            "fill",
            order_id=oid,
            symbol=ref.symbol,
            side=self.side_name(ref.side),
            qty=qty,
            price=price,
            tag=ref.tag,
        )
        await self.maybe_trade()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        oid = str(order_id)
        self.pending_cancels.discard(oid)
        ref = self.live_orders.pop(oid, None)
        if ref is not None:
            if ref.side == Side.BUY:
                self.pending_buy[ref.symbol] = max(0, self.pending_buy[ref.symbol] - ref.qty)
            else:
                self.pending_sell[ref.symbol] = max(0, self.pending_sell[ref.symbol] - ref.qty)
            self.symbol_cooldown_until[ref.symbol] = max(
                self.symbol_cooldown_until[ref.symbol],
                self.now() + C_SYMBOL_COOLDOWN_SECONDS,
            )

        match = re.search(r"Order ID\s+(\d+)\s+exceeds limits", str(reason))
        if match:
            rejected_id = int(match.group(1))
            self.order_id = max(int(getattr(self, "order_id", 0)), rejected_id + 1000, int(time.time() * 1000))
            logger.info("Bumped order_id after exchange rejection: next_order_id=%d", self.order_id)

        logger.info("Order rejected: order_id=%s reason=%s", oid, reason)
        self.log_csv(
            "order_rejected",
            order_id=oid,
            symbol="" if ref is None else ref.symbol,
            side="" if ref is None else self.side_name(ref.side),
            qty="" if ref is None else ref.qty,
            price="" if ref is None else ref.price,
            tag="" if ref is None else ref.tag,
            reason=reason,
        )

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol == C_SYMBOL:
            self.c_last_fill_price = price
        if symbol == C_SYMBOL or symbol in PM_SYMBOLS:
            await self.maybe_trade()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol == C_SYMBOL or symbol in PM_SYMBOLS:
            await self.maybe_trade()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release: dict):
        tick = int(news_release.get("tick") or news_release.get("timestamp") or 0)
        self.last_tick_seen = max(self.last_tick_seen, tick)
        new_data_obj = news_release.get("new_data", {}) or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        kind = str(news_release.get("kind") or "")
        subtype = str(new_data.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or "")
        text = str(new_data.get("content") or news_release.get("content") or (new_data_obj if not isinstance(new_data_obj, dict) else ""))
        news_summary = self.describe_news_release(news_release)

        actual, forecast = self.parse_cpi_from_news(news_release)
        if actual is not None and forecast is not None:
            self.set_cpi_signal(actual, forecast, tick, raw_news=news_release)
            await self.maybe_trade()
            return

        c_earnings = self.parse_c_earnings_from_news(news_release)
        if c_earnings is not None:
            self.set_c_earnings_signal(c_earnings, tick)
            await self.maybe_trade()
            return

        if text:
            macro_score = self.score_macro_headline_for_c(text)
            macro_target = self.c_target_for_macro_score(macro_score)
            macro_direction = "SELL" if macro_score > 0 else ("BUY" if macro_score < 0 else "")
            macro_reason = "below_threshold"
            if macro_target > 0:
                block_reason = self.macro_block_reason(Side.SELL if macro_score > 0 else Side.BUY)
                macro_reason = block_reason or "candidate"
            if "cpi" in text.lower() or "inflation" in text.lower():
                self.log_csv(
                    "cpi_eval",
                    tick=tick,
                    side=macro_direction,
                    qty=macro_target,
                    reason="no_actual_forecast",
                    message=f"no actual/forecast; score={macro_score:+.2f} headline={text[:180]}",
                    macro_score=f"{macro_score:.4f}",
                    macro_target=macro_target,
                    macro_direction=macro_direction,
                    earnings_trend=f"{self.c_earnings_trend():+.6f}",
                    news_kind=kind,
                    news_subtype=subtype,
                    news_asset=asset,
                    raw_news=news_release,
                )
            self.log_csv(
                "macro_eval",
                tick=tick,
                side=macro_direction,
                qty=macro_target,
                reason=macro_reason,
                message=f"score={macro_score:+.2f} headline={text[:180]}",
                macro_score=f"{macro_score:.4f}",
                macro_target=macro_target,
                macro_direction=macro_direction,
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                news_kind=kind,
                news_subtype=subtype,
                news_asset=asset,
                raw_news=news_release,
            )
            if macro_target > 0:
                self.set_macro_headline_signal(macro_score, text, tick, raw_news=news_release)
                await self.maybe_trade()
                return

        if subtype == "earnings":
            logger.info("Ignoring non-C earnings in C-only mode: asset=%s news=%s", asset, news_release)
            self.log_csv(
                "news_ignored",
                tick=tick,
                reason="non_C_earnings",
                message=news_summary,
                news_kind=kind,
                news_subtype=subtype,
                news_asset=asset,
                raw_news=news_release,
            )
        elif text:
            logger.info("Ignoring weak/non-macro news in C-only mode: score=%+.2f news=%s", macro_score, news_release)
            self.log_csv(
                "news_ignored",
                tick=tick,
                reason="weak_macro_news",
                message=f"score={macro_score:+.2f} headline={news_summary}",
                macro_score=f"{macro_score:.4f}",
                macro_target=macro_target,
                macro_direction=macro_direction,
                earnings_trend=f"{self.c_earnings_trend():+.6f}",
                news_kind=kind,
                news_subtype=subtype,
                news_asset=asset,
                raw_news=news_release,
            )
        else:
            logger.info("Ignoring non-CPI/non-C-earnings news in C-only mode: %s", news_release)
            self.log_csv(
                "news_ignored",
                tick=tick,
                reason="non_signal_news",
                message=news_summary,
                news_kind=kind,
                news_subtype=subtype,
                news_asset=asset,
                raw_news=news_release,
            )
        await self.maybe_trade()

    def log_heartbeat(self, force: bool = False) -> None:
        now = self.now()
        if not force and now - self.last_heartbeat_log < HEARTBEAT_LOG_SECONDS:
            return
        self.last_heartbeat_log = now

        unrealized, mark_source = self.mark_c_unrealized()
        bid, ask = self.book_top(C_SYMBOL)
        fair = self.fair_value_c()
        rate = self.rate_context()
        sig = self.active_signal
        sig_text = "none"
        if sig is not None:
            sig_text = f"{sig.thesis}:{self.side_name(sig.side)}:{sig.target}:fresh={self.c_signal_is_fresh()}"
        pm_pos = (self.pos[R_HIKE], self.pos[R_HOLD], self.pos[R_CUT])
        logger.info(
            "State C-only: pos_C=%d avg_C=%s realized=%.2f unrealized=%.2f total_pnl=%.2f cash=%.2f session_mtm=%.2f thesis=%s signal=%s fair=%s C=%s/%s eps=%.4f cpi_surprise=%+.6f rate_bp=%s pm_pos=%d/%d/%d mark=%s kill=%s",
            self.pos[C_SYMBOL],
            "na" if self.c_avg_entry is None else f"{self.c_avg_entry:.2f}",
            self.c_realized_pnl,
            unrealized,
            self.c_total_pnl,
            self.cash,
            self.c_total_pnl,
            self.c_thesis,
            sig_text,
            "na" if fair is None else f"{fair:.2f}",
            bid if bid is not None else "na",
            ask if ask is not None else "na",
            self.current_eps_c,
            self.last_cpi_surprise,
            "na" if rate is None else f"{rate.effective_rate_bp:.2f}",
            pm_pos[0],
            pm_pos[1],
            pm_pos[2],
            mark_source,
            self.kill_switch,
        )
        self.log_csv("heartbeat" if not force else "heartbeat_forced")

    async def trade(self):
        """C-only CPI/earnings bot. Prediction markets are read-only context."""
        await asyncio.sleep(1)
        while True:
            await self.maybe_trade()
            await asyncio.sleep(0.2)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    SERVER = "practice.uchicago.exchange:3333"
    my_client = MyXchangeClient(
        SERVER,
        os.getenv("UCHICAGO_USERNAME", "YOUR_USERNAME"),
        os.getenv("UCHICAGO_PASSWORD", "YOUR_PASSWORD"),
    )
    await my_client.start()


if __name__ == "__main__":
    asyncio.run(main())
