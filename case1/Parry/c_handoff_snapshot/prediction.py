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
import logging
import math
import re
import time
import os
from collections import defaultdict
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("c-only")


C_SYMBOL = "C"
R_HIKE = "R_HIKE"
R_HOLD = "R_HOLD"
R_CUT = "R_CUT"
PM_SYMBOLS = [R_HIKE, R_HOLD, R_CUT]

ORDER_LIMIT = 40
C_POSITION_LIMIT = 200

CPI_DEADBAND = 0.00025
CPI_STRONG_SURPRISE = 0.00050
CPI_VERY_STRONG_SURPRISE = 0.00100
CPI_EXTREME_SURPRISE = 0.00200
CPI_SIGNAL_TTL_SECONDS = 12.0
CPI_MAX_HOLD_SECONDS = 24.0
CPI_TO_RATE_BP = 4000.0
MAX_CPI_RATE_BIAS_BP = 8.0

EARNINGS_IGNORE_DELTA = 0.010
EARNINGS_SMALL_DELTA = 0.025
EARNINGS_MEDIUM_DELTA = 0.050
EARNINGS_SIGNAL_TTL_SECONDS = 8.0
EARNINGS_MAX_HOLD_SECONDS = 24.0

DEFAULT_EPS_C = 2.0
C_OPS_WEIGHT = 0.72
C_BOND_WEIGHT = 0.28
C_PE_YIELD_GAMMA = 13.0
C_BOND_DURATION = 4.5
C_BOND_CONVEXITY = 30.0
FAIR_VALUE_GUARD_TICKS = 8.0

MAX_C_SPREAD = 12
C_ACTION_COOLDOWN_SECONDS = 0.35
C_SYMBOL_COOLDOWN_SECONDS = 0.45
ORDER_REPRICE_SECONDS = 0.70
HEARTBEAT_LOG_SECONDS = 3.0

LONG_PROFIT_1 = 8.0
LONG_PROFIT_2 = 20.0
LONG_PROFIT_FULL = 40.0
SHORT_PROFIT_1 = 8.0
SHORT_PROFIT_2 = 20.0
SHORT_PROFIT_FULL = 40.0
ADVERSE_STOP_ADD_TICKS = 12.0
ADVERSE_FLATTEN_TICKS = 24.0
C_TOTAL_PNL_HARD_STOP = -12000.0

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

        self.anchor_price: Optional[float] = None
        self.anchor_eps: Optional[float] = None
        self.anchor_rate_bp: Optional[float] = None
        self.last_anchor_update_ts = 0.0

        self.active_signal: Optional[CSignal] = None
        self.last_cpi_surprise = 0.0
        self.last_cpi_bias_bp = 0.0
        self.last_cpi_ts = 0.0

        self.c_avg_entry: Optional[float] = None
        self.c_realized_pnl = 0.0
        self.c_unrealized_pnl = 0.0
        self.c_total_pnl = 0.0
        self.c_peak_abs = 0
        self.c_thesis = "flat"
        self.c_regime_started_at = 0.0
        self.c_last_fill_price: Optional[int] = None
        self.kill_switch = False

        self.last_c_action_ts = 0.0
        self.last_heartbeat_log = 0.0
        self.last_tick_seen = 0
        self.decision_lock = asyncio.Lock()

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

    def has_live_order(self, symbol: str) -> bool:
        return any(
            ref.symbol == symbol and oid not in self.pending_cancels
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
        if self.last_cpi_ts > 0.0 and self.now() - self.last_cpi_ts <= CPI_SIGNAL_TTL_SECONDS:
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
        ttl = CPI_SIGNAL_TTL_SECONDS if sig.thesis == "cpi" else EARNINGS_SIGNAL_TTL_SECONDS
        return self.now() - sig.ts <= ttl

    def c_signal_max_hold(self) -> float:
        if self.c_thesis == "cpi":
            return CPI_MAX_HOLD_SECONDS
        if self.c_thesis == "earnings":
            return EARNINGS_MAX_HOLD_SECONDS
        return 18.0

    def c_target_for_cpi_surprise(self, surprise: float) -> int:
        abs_surprise = abs(surprise)
        if abs_surprise <= CPI_DEADBAND:
            return 0
        if abs_surprise >= CPI_EXTREME_SURPRISE:
            return 200
        if abs_surprise >= CPI_VERY_STRONG_SURPRISE:
            return 160
        if abs_surprise >= CPI_STRONG_SURPRISE:
            return 120
        return 80

    def c_target_for_earnings_delta(self, delta: float) -> int:
        abs_delta = abs(delta)
        if abs_delta < EARNINGS_IGNORE_DELTA:
            return 0
        if abs_delta >= EARNINGS_MEDIUM_DELTA:
            return 160
        if abs_delta >= EARNINGS_SMALL_DELTA:
            return 120
        return 80

    def set_cpi_signal(self, actual: float, forecast: float, tick: int) -> None:
        surprise = actual - forecast
        target = self.c_target_for_cpi_surprise(surprise)
        self.last_cpi_surprise = surprise
        self.last_cpi_bias_bp = self.clip(surprise * CPI_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
        self.last_cpi_ts = self.now()

        if target == 0:
            self.active_signal = None
            logger.info(
                "CPI inside deadband: tick=%d actual=%.6f forecast=%.6f surprise=%+.6f",
                tick,
                actual,
                forecast,
                surprise,
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

    def set_c_earnings_signal(self, value: float, tick: int) -> None:
        if not self.have_real_eps_c:
            self.have_real_eps_c = True
            self.current_eps_c = value
            self.baseline_eps_c = value
            self.last_c_earnings_delta = 0.0
            self.active_signal = None
            self.maybe_initialize_anchor(force=True)
            logger.info("First C earnings baseline adopted: tick=%d eps=%.4f no_trade=True", tick, value)
            return

        old_eps = self.current_eps_c
        delta = value - old_eps
        self.current_eps_c = value
        self.last_c_earnings_delta = delta
        target = self.c_target_for_earnings_delta(delta)
        if target == 0:
            self.active_signal = None
            logger.info(
                "C earnings inside deadband: tick=%d old=%.4f new=%.4f delta=%+.4f",
                tick,
                old_eps,
                value,
                delta,
            )
            return

        side = Side.BUY if delta > 0 else Side.SELL
        self.active_signal = CSignal(
            side=side,
            target=target,
            thesis="earnings",
            strength=abs(delta),
            tick=tick,
            ts=self.now(),
            description="earnings_up_long_C" if side == Side.BUY else "earnings_down_short_C",
        )
        logger.info(
            "C earnings signal: tick=%d old=%.4f new=%.4f delta=%+.4f side=%s target=%d",
            tick,
            old_eps,
            value,
            delta,
            self.side_name(side),
            target,
        )

    def parse_cpi_from_news(self, news_release: dict) -> Tuple[Optional[float], Optional[float]]:
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        subtype = str(new_data.get("structured_subtype") or "").lower()
        if kind == "structured" and subtype == "cpi_print":
            return float(new_data["actual"]), float(new_data["forecast"])

        text = str(new_data.get("content") or news_release)
        lower = text.lower()
        if "cpi" not in lower or "actual" not in lower or "forecast" not in lower:
            return None, None

        actual_match = re.search(r"actual\s*[:=]\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        forecast_match = re.search(r"forecast\s*[:=]\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if actual_match and forecast_match:
            return float(actual_match.group(1)), float(forecast_match.group(1))

        nums = [float(x) for x in re.findall(FLOAT_RE, text)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        return None, None

    def parse_c_earnings_from_news(self, news_release: dict) -> Optional[float]:
        if news_release.get("kind") != "structured":
            return None
        new_data = news_release.get("new_data", {}) or {}
        subtype = str(new_data.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or "").upper()
        if subtype == "earnings" and asset == C_SYMBOL:
            return float(new_data["value"])
        return None

    def mark_c_unrealized(self) -> Tuple[float, str]:
        pos = self.pos.get(C_SYMBOL, 0)
        if pos == 0 or self.c_avg_entry is None:
            self.c_unrealized_pnl = 0.0
            self.c_total_pnl = self.c_realized_pnl
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
        return unrealized, mark_source

    def update_c_pnl_from_fill(self, old_pos: int, new_pos: int, signed_qty: int, price: int) -> None:
        fill_abs = abs(signed_qty)
        if fill_abs == 0:
            return

        if old_pos == 0 or self.c_avg_entry is None:
            self.c_avg_entry = float(price) if new_pos != 0 else None
            self.c_peak_abs = abs(new_pos)
            self.c_regime_started_at = self.now() if new_pos != 0 else 0.0
            return

        if self.sign(old_pos) == self.sign(signed_qty):
            old_abs = abs(old_pos)
            new_abs = old_abs + fill_abs
            self.c_avg_entry = (self.c_avg_entry * old_abs + price * fill_abs) / new_abs
            self.c_peak_abs = max(self.c_peak_abs, abs(new_pos))
            return

        closed_qty = min(abs(old_pos), fill_abs)
        if old_pos > 0:
            self.c_realized_pnl += (price - self.c_avg_entry) * closed_qty
        else:
            self.c_realized_pnl += (self.c_avg_entry - price) * closed_qty

        if new_pos == 0:
            self.c_avg_entry = None
            self.c_peak_abs = 0
            self.c_thesis = "flat"
            self.c_regime_started_at = 0.0
        elif self.sign(old_pos) != self.sign(new_pos):
            residual = fill_abs - closed_qty
            self.c_avg_entry = float(price)
            self.c_peak_abs = residual
            self.c_regime_started_at = self.now()

    def current_mtm(self) -> float:
        pos = self.pos.get(C_SYMBOL, 0)
        mid_c = self.mid(C_SYMBOL)
        c_mark = 0.0 if mid_c is None else pos * mid_c
        return self.cash + c_mark

    async def send_c_order(self, side: Side, qty: int, px: int, tag: str) -> bool:
        symbol = C_SYMBOL
        if self.kill_switch and tag != "risk_flatten":
            return False
        if self.now() < self.symbol_cooldown_until[symbol]:
            return False
        if self.has_live_order(symbol):
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
        return True

    async def cancel_stale_orders(self) -> None:
        now = self.now()
        for oid, ref in list(self.live_orders.items()):
            if oid in self.pending_cancels:
                continue
            if now - ref.ts < ORDER_REPRICE_SECONDS:
                continue
            self.pending_cancels.add(oid)
            try:
                await self.cancel_order(oid)
            except Exception:
                self.pending_cancels.discard(oid)

    async def flatten_c(self, reason: str) -> bool:
        pos = self.pos.get(C_SYMBOL, 0)
        if pos == 0 or self.has_live_order(C_SYMBOL):
            return False
        side = Side.SELL if pos > 0 else Side.BUY
        px = self.best_price_for_cross(C_SYMBOL, side)
        if px is None:
            return False
        placed = await self.send_c_order(side, min(abs(pos), ORDER_LIMIT), px, reason)
        if placed:
            logger.info("C exit trigger: %s pos_C=%d", reason, pos)
        return placed

    def entry_blocked_by_risk(self, side: Side) -> Tuple[bool, str]:
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
        mid_c = self.mid(C_SYMBOL)
        sig = self.active_signal
        allow_guard_override = sig is not None and sig.thesis == "cpi" and sig.target >= 200
        if fair is not None and mid_c is not None and not allow_guard_override:
            if side == Side.BUY and mid_c > fair + FAIR_VALUE_GUARD_TICKS:
                return True, "buy_over_fair"
            if side == Side.SELL and mid_c < fair - FAIR_VALUE_GUARD_TICKS:
                return True, "sell_under_fair"
        return False, "ok"

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

        bid, ask = self.book_top(C_SYMBOL)
        if self.c_avg_entry is not None:
            if pos > 0 and bid is not None:
                adverse = self.c_avg_entry - bid
                profit = bid - self.c_avg_entry
                if adverse >= ADVERSE_FLATTEN_TICKS:
                    return await self.flatten_c("stop_loss")
                if profit >= LONG_PROFIT_FULL:
                    return await self.flatten_c("long_profit_full")
                if profit >= LONG_PROFIT_2:
                    return await self.flatten_c("long_profit_2")
                if profit >= LONG_PROFIT_1:
                    return await self.flatten_c("long_profit_1")
            elif pos < 0 and ask is not None:
                adverse = ask - self.c_avg_entry
                profit = self.c_avg_entry - ask
                if adverse >= ADVERSE_FLATTEN_TICKS:
                    return await self.flatten_c("stop_loss")
                if profit >= SHORT_PROFIT_FULL:
                    return await self.flatten_c("short_profit_full")
                if profit >= SHORT_PROFIT_2:
                    return await self.flatten_c("short_profit_2")
                if profit >= SHORT_PROFIT_1:
                    return await self.flatten_c("short_profit_1")

        sig = self.active_signal
        if sig is not None and self.c_signal_is_fresh():
            desired_sign = 1 if sig.side == Side.BUY else -1
            if self.sign(pos) != desired_sign:
                return await self.flatten_c("contradict_signal")

        regime_age = 0.0 if self.c_regime_started_at <= 0.0 else self.now() - self.c_regime_started_at
        if regime_age >= self.c_signal_max_hold():
            return await self.flatten_c("max_hold")

        gap = self.fair_gap()
        spread = self.spread(C_SYMBOL)
        if gap is not None and spread is not None:
            favorable_gap = gap if pos > 0 else -gap
            if favorable_gap <= max(4.0, float(spread)):
                return await self.flatten_c("gap_compress")

        return False

    async def maybe_enter_or_add_c(self) -> bool:
        sig = self.active_signal
        if sig is None or not self.c_signal_is_fresh():
            return False
        if self.now() - self.last_c_action_ts < C_ACTION_COOLDOWN_SECONDS:
            return False

        pos = self.pos.get(C_SYMBOL, 0)
        desired_sign = 1 if sig.side == Side.BUY else -1
        if pos != 0 and self.sign(pos) != desired_sign:
            return False
        if abs(pos) >= sig.target:
            return False

        blocked, reason = self.entry_blocked_by_risk(sig.side)
        if blocked:
            logger.info("C entry blocked: reason=%s thesis=%s target=%d pos_C=%d", reason, sig.thesis, sig.target, pos)
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
        return placed

    async def maybe_trade(self) -> None:
        if self.decision_lock.locked():
            return
        async with self.decision_lock:
            self.maybe_initialize_anchor()
            self.mark_c_unrealized()
            await self.cancel_stale_orders()
            if await self.maybe_exit_c():
                self.log_heartbeat(force=True)
                return
            if await self.maybe_enter_or_add_c():
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
        self.mark_c_unrealized()
        logger.info(
            "Startup C-only mode: C=%d H=%d HOLD=%d CUT=%d cash=%.2f inherited_PM_no_trade=True",
            self.pos[C_SYMBOL],
            self.pos[R_HIKE],
            self.pos[R_HOLD],
            self.pos[R_CUT],
            self.cash,
        )

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

        actual, forecast = self.parse_cpi_from_news(news_release)
        if actual is not None and forecast is not None:
            self.set_cpi_signal(actual, forecast, tick)
            await self.maybe_trade()
            return

        c_earnings = self.parse_c_earnings_from_news(news_release)
        if c_earnings is not None:
            self.set_c_earnings_signal(c_earnings, tick)
            await self.maybe_trade()
            return

        new_data = news_release.get("new_data", {}) or {}
        subtype = str(new_data.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or "")
        if subtype == "earnings":
            logger.info("Ignoring non-C earnings in C-only mode: asset=%s news=%s", asset, news_release)
        else:
            logger.info("Ignoring non-CPI/non-C-earnings news in C-only mode: %s", news_release)
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
