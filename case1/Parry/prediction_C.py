import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from utcxchangelib import Side, XChangeClient


# ============================================================
# Rewritten C + prediction market bot
#
# Main design changes versus the prior versions:
# 1. Uses revealed C model constants from the exchange announcement.
# 2. Trades C aggressively, but exits faster and does not sit on stale inventory.
# 3. Trades prediction markets with regime conviction, laddering toward targets.
# 4. Prevents self-created orphan repair churn loops.
# 5. Has explicit endgame winner-taking logic for prediction markets.
# 6. Removes the buggy "startup_flatten missing_signal" behavior that left stale C.
# 7. Reprices and cancels stale orders more often.
#
# This is still a framework, not a guarantee of profitability. You must tune it
# with your own practice results.
# ============================================================


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction-c")


# ============================================================
# Exchange / symbol constants
# ============================================================
C_SYMBOL = "C"
A_SYMBOL = "A"
R_HIKE = "R_HIKE"
R_HOLD = "R_HOLD"
R_CUT = "R_CUT"
PM_SYMBOLS = [R_HIKE, R_HOLD, R_CUT]


# ============================================================
# Revealed C-model parameters from exchange screenshot
# ============================================================
Y0 = 0.045
PE0 = 14.0
EPS0 = 2.00
DUR = 7.5
B0_PER_N = 40.0
CONV = 55.0
LAMBDA_C = 0.65

# This gamma was not revealed directly. It still must be estimated/tuned.
PE_SENSITIVITY_GAMMA = 9.5


# ============================================================
# Trading controls
# ============================================================
ORDER_LIMIT = 40
MAX_ABS_C = 200
MAX_ABS_PM = 200
MAX_OPEN_PER_SYMBOL = 4

# C thresholds
C_MIN_EDGE = 6.0
C_STRONG_EDGE = 12.0
C_VERY_STRONG_EDGE = 20.0
C_EXIT_EDGE = 6.5
C_HARD_STOP_EDGE = 3.0
C_TIME_STOP_SECONDS = 9.0
C_EARNINGS_HOLD_SECONDS = 6.0
C_REPRICE_SECONDS = 0.9

# PM thresholds
PM_ENTRY_MIN_STRENGTH = 1.2
PM_ADD_MIN_STRENGTH = 2.2
PM_WINNER_THRESHOLD = 0.70
PM_LOCK_THRESHOLD = 0.84
PM_EXTREME_THRESHOLD = 0.93
PM_ENDGAME_TICKS = 220
PM_REPRICE_SECONDS = 0.75
PM_PROFIT_TICKS = 8
PM_MAX_HOLD_SECONDS = 8.0
PM_NEUTRAL_FLUSH_SECONDS = 3.0

# Runtime pacing
HEARTBEAT_LOG_SECONDS = 4.0
QUOTE_STALE_SECONDS = 1.25


# ============================================================
# News / parsing helpers
# ============================================================
FLOAT_RE = r"[-+]?\d*\.?\d+"


@dataclass
class OrderRef:
    order_id: str
    symbol: str
    side: Side
    price: int
    qty: int
    ts: float
    tag: str


@dataclass
class PositionLot:
    qty: int
    avg_px: float
    opened_ts: float
    thesis: str


class MyXchangeClient(XChangeClient):
    def __init__(self, host: str, username: str, password: str):
        super().__init__(host=host, username=username, password=password)

        self.start_ts = time.time()
        self.last_heartbeat_log = 0.0

        # books
        self.best_bid: Dict[str, Optional[int]] = {}
        self.best_ask: Dict[str, Optional[int]] = {}

        # positions from fills
        self.pos: Dict[str, int] = {C_SYMBOL: 0, R_HIKE: 0, R_HOLD: 0, R_CUT: 0}
        self.cash: float = 0.0

        # working orders
        self.live_orders: Dict[str, OrderRef] = {}
        self.open_by_symbol: Dict[str, List[str]] = {C_SYMBOL: [], R_HIKE: [], R_HOLD: [], R_CUT: []}
        self.pending_cancels: set[str] = set()
        self.symbol_cooldown_until: Dict[str, float] = {C_SYMBOL: 0.0, R_HIKE: 0.0, R_HOLD: 0.0, R_CUT: 0.0}
        self.pending_buy: Dict[str, int] = defaultdict(int)
        self.pending_sell: Dict[str, int] = defaultdict(int)

        # model state
        self.c_eps = EPS0
        self.prev_c_eps = EPS0
        self.have_real_c_eps = False
        self.current_yield = Y0
        self.last_cpi_bias_bp = 0.0
        self.last_fed_bias_bp = 0.0
        self.last_macro_ts = 0.0
        self.last_macro_tick = 0
        self.last_tick_seen = 0

        # anchor state for C
        self.c_anchor_price: Optional[float] = None
        self.c_anchor_eps: Optional[float] = None
        self.c_anchor_yield: Optional[float] = None

        # c trade state
        self.c_lot: Optional[PositionLot] = None
        self.c_mode: str = "none"
        self.last_c_action_ts = 0.0

        # pm regime state
        self.regime: str = "neutral"
        self.last_regime_change_ts = 0.0
        self.last_pm_action_ts = 0.0
        self.pm_locked_side: Optional[str] = None
        self.pm_entry_ts: Dict[str, float] = {R_HIKE: 0.0, R_HOLD: 0.0, R_CUT: 0.0}

        # to prevent orphan churn loops
        self.last_pm_repair_ts = 0.0

    # --------------------------------------------------------
    # Generic helpers
    # --------------------------------------------------------
    def now(self) -> float:
        return time.time()

    def seconds_running(self) -> float:
        return self.now() - self.start_ts

    def mid(self, symbol: str) -> Optional[float]:
        bid = self.best_bid.get(symbol)
        ask = self.best_ask.get(symbol)
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def spread(self, symbol: str) -> Optional[int]:
        bid = self.best_bid.get(symbol)
        ask = self.best_ask.get(symbol)
        if bid is None or ask is None:
            return None
        return ask - bid

    def clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def sign(self, x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def working_order_imbalance(self, symbol: str) -> int:
        return int(self.pending_buy.get(symbol, 0) - self.pending_sell.get(symbol, 0))

    def remaining_capacity(self, symbol: str, side: Side) -> int:
        pos = int(self.pos.get(symbol, 0))
        buy_pending = int(self.pending_buy.get(symbol, 0))
        sell_pending = int(self.pending_sell.get(symbol, 0))
        limit = MAX_ABS_C if symbol == C_SYMBOL else MAX_ABS_PM
        if side == Side.BUY:
            return max(0, limit - (pos + buy_pending))
        return max(0, limit + (pos - sell_pending))

    def top_qty(self, symbol: str, side: Side) -> int:
        cap = self.remaining_capacity(symbol, side)
        return max(0, min(ORDER_LIMIT, cap))

    def best_price_for_passive(self, symbol: str, side: Side, improve: int = 0) -> Optional[int]:
        bid = self.best_bid.get(symbol)
        ask = self.best_ask.get(symbol)
        if bid is None or ask is None:
            return None
        if side == Side.BUY:
            return max(1, min(999, bid + improve))
        return max(1, min(999, ask - improve))

    def best_price_for_cross(self, symbol: str, side: Side, offset: int = 0) -> Optional[int]:
        bid = self.best_bid.get(symbol)
        ask = self.best_ask.get(symbol)
        if bid is None or ask is None:
            return None
        if side == Side.BUY:
            return max(1, min(999, ask + offset))
        return max(1, min(999, bid - offset))

    def c_exp_bp_from_probs(self) -> float:
        qh, qo, qc = self.pm_probs()
        return 25.0 * qh + 0.0 * qo - 25.0 * qc

    def pm_probs(self) -> Tuple[float, float, float]:
        mh = self.mid(R_HIKE)
        mo = self.mid(R_HOLD)
        mc = self.mid(R_CUT)
        if mh is None or mo is None or mc is None:
            return (1 / 3, 1 / 3, 1 / 3)
        total = max(1.0, mh + mo + mc)
        return (mh / total, mo / total, mc / total)

    def last_minute_mode(self) -> bool:
        # approximate endgame based on ticks observed
        return self.last_tick_seen >= PM_ENDGAME_TICKS

    # --------------------------------------------------------
    # C fair value model
    # --------------------------------------------------------
    def update_anchor_if_needed(self) -> None:
        m = self.mid(C_SYMBOL)
        if m is None:
            return
        if self.c_anchor_price is None:
            self.c_anchor_price = m
            self.c_anchor_eps = self.c_eps
            self.c_anchor_yield = self.current_yield
            logger.info(
                "Initialized C anchor: price=%.2f eps=%.4f y=%.4f",
                self.c_anchor_price,
                self.c_eps,
                self.current_yield,
            )

        if self.have_real_c_eps and self.c_anchor_eps == EPS0:
            self.c_anchor_price = m
            self.c_anchor_eps = self.c_eps
            self.c_anchor_yield = self.current_yield
            logger.info(
                "Adopted first real C EPS baseline anchor: price=%.2f eps=%.4f",
                self.c_anchor_price,
                self.c_eps,
            )

    def fair_c(self) -> Optional[float]:
        self.update_anchor_if_needed()
        if self.c_anchor_price is None or self.c_anchor_eps is None or self.c_anchor_yield is None:
            return None

        dy = self.current_yield - self.c_anchor_yield

        pe_anchor = PE0 * math.exp(-PE_SENSITIVITY_GAMMA * (self.c_anchor_yield - Y0))
        pe_now = PE0 * math.exp(-PE_SENSITIVITY_GAMMA * (self.current_yield - Y0))

        ops_anchor = self.c_anchor_eps * pe_anchor
        ops_now = self.c_eps * pe_now

        bond_delta = B0_PER_N * (-DUR * dy + 0.5 * CONV * dy * dy)
        fair = self.c_anchor_price + (ops_now - ops_anchor) + LAMBDA_C * bond_delta
        return fair

    # --------------------------------------------------------
    # News handling -> bias / yield updates
    # --------------------------------------------------------
    def apply_macro_bias_bp(self, bp: float, source: str) -> None:
        self.last_macro_ts = self.now()
        if source == "cpi":
            self.last_cpi_bias_bp = bp
        elif source == "fed":
            self.last_fed_bias_bp = bp

        total_bp = self.last_cpi_bias_bp + self.last_fed_bias_bp
        self.current_yield = Y0 + total_bp / 10000.0
        self.current_yield = self.clamp(self.current_yield, 0.0, 0.12)

    def decay_macro_bias(self) -> None:
        if self.last_macro_ts <= 0:
            return
        age = self.now() - self.last_macro_ts
        if age <= 0:
            return
        decay = math.exp(-age / 14.0)
        self.current_yield = Y0 + (self.last_cpi_bias_bp + self.last_fed_bias_bp) * decay / 10000.0
        self.current_yield = self.clamp(self.current_yield, 0.0, 0.12)

    def parse_text_bias_bp(self, text: str) -> float:
        t = text.lower()

        strong_hawk = [
            "persistent inflation",
            "restrictive for longer",
            "sticky prices",
            "wage growth",
            "inflation risks",
            "strong demand",
            "reassess path of cuts",
            "concerned about wage growth",
        ]
        mild_hawk = [
            "higher for longer",
            "hawkish",
            "inflation remains",
            "upside inflation",
            "policy stay restrictive",
        ]
        strong_dove = [
            "moving back to target",
            "confidence inflation is moving back",
            "policy easing",
            "softening data",
            "balanced risks",
            "easing path",
        ]
        mild_dove = [
            "dovish",
            "cooling inflation",
            "slowing data",
            "lower inflation",
        ]

        score = 0.0
        for s in strong_hawk:
            if s in t:
                score += 4.0
        for s in mild_hawk:
            if s in t:
                score += 2.0
        for s in strong_dove:
            if s in t:
                score -= 4.0
        for s in mild_dove:
            if s in t:
                score -= 2.0

        return self.clamp(score, -5.0, 5.0)

    def update_regime(self) -> None:
        qh, qo, qc = self.pm_probs()
        exp_bp = self.c_exp_bp_from_probs()
        total_bp = 25.0 * (qh - qc)
        strength = abs(total_bp)

        if qc >= PM_LOCK_THRESHOLD:
            regime = "lock_cut"
            self.pm_locked_side = R_CUT
        elif qh >= PM_LOCK_THRESHOLD:
            regime = "lock_hike"
            self.pm_locked_side = R_HIKE
        elif qc >= PM_WINNER_THRESHOLD:
            regime = "winner_cut"
            self.pm_locked_side = R_CUT
        elif qh >= PM_WINNER_THRESHOLD:
            regime = "winner_hike"
            self.pm_locked_side = R_HIKE
        elif exp_bp <= -PM_ENTRY_MIN_STRENGTH:
            regime = "dovish"
            self.pm_locked_side = None
        elif exp_bp >= PM_ENTRY_MIN_STRENGTH:
            regime = "hawkish"
            self.pm_locked_side = None
        else:
            regime = "neutral"
            self.pm_locked_side = None

        if regime != self.regime:
            self.regime = regime
            self.last_regime_change_ts = self.now()
            logger.info(
                "Regime change: %s q_hike=%.3f q_hold=%.3f q_cut=%.3f exp_bp=%.2f strength=%.2f",
                regime, qh, qo, qc, exp_bp, strength,
            )

    # --------------------------------------------------------
    # Order send / cancel helpers
    # --------------------------------------------------------
    async def send_limit(self, symbol: str, side: Side, qty: int, px: int, tag: str) -> None:
        if px is None:
            return
        if self.now() < self.symbol_cooldown_until[symbol]:
            return

        qty = int(max(0, min(qty, ORDER_LIMIT)))
        if qty <= 0:
            return

        active_open = [oid for oid in self.open_by_symbol[symbol] if oid in self.live_orders and oid not in self.pending_cancels]
        if len(active_open) >= MAX_OPEN_PER_SYMBOL:
            return

        qty = int(max(0, min(qty, self.remaining_capacity(symbol, side))))
        if qty <= 0:
            return

        try:
            oid = await self.place_order(symbol, qty, side, px)
            if oid is None:
                return
            oid = str(oid)
            self.live_orders[oid] = OrderRef(
                order_id=oid,
                symbol=symbol,
                side=side,
                price=px,
                qty=qty,
                ts=self.now(),
                tag=tag,
            )
            if oid not in self.open_by_symbol[symbol]:
                self.open_by_symbol[symbol].append(oid)
            if side == Side.BUY:
                self.pending_buy[symbol] += qty
            else:
                self.pending_sell[symbol] += qty
            logger.info(
                "Placed %s order: symbol=%s side=%s qty=%d price=%d",
                tag, symbol, "BUY" if side == Side.BUY else "SELL", qty, px,
            )
        except Exception as e:
            logger.exception("place_order failed for %s: %s", symbol, e)

    async def cancel_order_safe(self, oid: str) -> None:
        ref = self.live_orders.get(oid)
        if ref is None or oid in self.pending_cancels:
            return
        self.pending_cancels.add(oid)
        try:
            await self.cancel_order(oid)
        except Exception:
            self.pending_cancels.discard(oid)
            pass

    async def cancel_symbol_orders(self, symbol: str, older_than: float = 0.0) -> None:
        now = self.now()
        seen = set()
        for oid in list(self.open_by_symbol.get(symbol, [])):
            if oid in seen:
                continue
            seen.add(oid)
            ref = self.live_orders.get(oid)
            if ref is None:
                continue
            if older_than > 0.0 and now - ref.ts < older_than:
                continue
            await self.cancel_order_safe(oid)

    async def refresh_stale_orders(self) -> None:
        now = self.now()
        for symbol in [C_SYMBOL, R_HIKE, R_CUT, R_HOLD]:
            for oid in list(self.open_by_symbol[symbol]):
                if oid in self.pending_cancels:
                    continue
                ref = self.live_orders.get(oid)
                if ref is None:
                    continue
                stale = C_REPRICE_SECONDS if symbol == C_SYMBOL else PM_REPRICE_SECONDS
                if now - ref.ts > stale:
                    await self.cancel_order_safe(oid)
                    self.symbol_cooldown_until[symbol] = max(self.symbol_cooldown_until[symbol], now + 0.15)

    # --------------------------------------------------------
    # Position / fill bookkeeping
    # --------------------------------------------------------
    def update_position_from_fill(self, symbol: str, side: Side, qty: int, price: int) -> None:
        signed = qty if side == Side.BUY else -qty
        self.pos[symbol] = self.pos.get(symbol, 0) + signed
        self.cash -= signed * price

        if symbol == C_SYMBOL:
            p = self.pos[C_SYMBOL]
            if p == 0:
                self.c_lot = None
                self.c_mode = "none"
            else:
                if self.c_lot is None or self.sign(self.c_lot.qty) != self.sign(p):
                    self.c_lot = PositionLot(qty=p, avg_px=price, opened_ts=self.now(), thesis=self.c_mode)
                else:
                    old_qty = abs(self.c_lot.qty)
                    new_qty = abs(p)
                    if new_qty > old_qty:
                        self.c_lot.avg_px = (self.c_lot.avg_px * old_qty + price * (new_qty - old_qty)) / new_qty
                        self.c_lot.opened_ts = self.now()
                    self.c_lot.qty = p

    # --------------------------------------------------------
    # C logic
    # --------------------------------------------------------
    async def maybe_trade_c(self) -> None:
        fair = self.fair_c()
        bid = self.best_bid.get(C_SYMBOL)
        ask = self.best_ask.get(C_SYMBOL)
        mid = self.mid(C_SYMBOL)
        if fair is None or bid is None or ask is None or mid is None:
            return

        pos = self.pos[C_SYMBOL]
        gap = fair - mid
        abs_gap = abs(gap)
        now = self.now()

        # exit stale or compressed positions much faster
        if pos != 0:
            held = 0.0 if self.c_lot is None else now - self.c_lot.opened_ts
            same_sign = self.sign(gap) == self.sign(pos)
            trim_qty = min(abs(pos), ORDER_LIMIT)

            if not same_sign or held >= C_TIME_STOP_SECONDS or abs_gap <= C_EXIT_EDGE:
                side = Side.SELL if pos > 0 else Side.BUY
                px = self.best_price_for_cross(C_SYMBOL, side)
                if px is not None and trim_qty > 0:
                    await self.cancel_symbol_orders(C_SYMBOL)
                    self.symbol_cooldown_until[C_SYMBOL] = max(self.symbol_cooldown_until[C_SYMBOL], self.now() + 0.12)
                    await self.send_limit(C_SYMBOL, side, trim_qty, px, "c_exit")
                    logger.info(
                        "C exit trigger: pos=%d gap=%.2f held=%.2f fair=%.2f mid=%.2f",
                        pos, gap, held, fair, mid,
                    )
                    self.last_c_action_ts = now
                    return

            # if the trade is old, keep bleeding inventory out even if edge still points our way
            if held >= 4.5 and abs(pos) > 0:
                side = Side.SELL if pos > 0 else Side.BUY
                px = self.best_price_for_cross(C_SYMBOL, side)
                if px is not None and trim_qty > 0:
                    await self.send_limit(C_SYMBOL, side, trim_qty, px, "c_age_trim")
                    self.last_c_action_ts = now
                    return

        # hard stop on too-small edge after entry
        if pos != 0 and abs_gap <= C_HARD_STOP_EDGE:
            side = Side.SELL if pos > 0 else Side.BUY
            px = self.best_price_for_cross(C_SYMBOL, side)
            qty = min(abs(pos), ORDER_LIMIT)
            if px is not None and qty > 0:
                await self.cancel_symbol_orders(C_SYMBOL)
                self.symbol_cooldown_until[C_SYMBOL] = max(self.symbol_cooldown_until[C_SYMBOL], self.now() + 0.12)
                await self.send_limit(C_SYMBOL, side, qty, px, "c_hard_stop")
                self.last_c_action_ts = now
                return

        # new entry / add logic
        if now - self.last_c_action_ts < 0.35:
            return

        side: Optional[Side] = None
        if gap >= C_MIN_EDGE:
            side = Side.BUY
        elif gap <= -C_MIN_EDGE:
            side = Side.SELL

        if side is None:
            return

        desired = 0
        if abs_gap >= C_VERY_STRONG_EDGE:
            desired = 160
        elif abs_gap >= C_STRONG_EDGE:
            desired = 80
        else:
            desired = 40

        desired *= 1 if side == Side.BUY else -1

        # do not pyramid forever into stale moves
        if pos != 0 and self.sign(pos) == self.sign(desired):
            held = 0.0 if self.c_lot is None else now - self.c_lot.opened_ts
            if held >= 3.5:
                return
            if abs(pos) >= 80 and abs_gap < C_VERY_STRONG_EDGE:
                return

        delta = desired - pos
        if delta == 0:
            return

        order_side = Side.BUY if delta > 0 else Side.SELL
        qty = min(abs(delta), self.top_qty(C_SYMBOL, order_side))
        if qty <= 0:
            return

        # entry price policy: stronger edges cross more aggressively
        if abs_gap >= C_VERY_STRONG_EDGE:
            px = self.best_price_for_cross(C_SYMBOL, order_side, offset=1)
        elif abs_gap >= C_STRONG_EDGE:
            px = self.best_price_for_cross(C_SYMBOL, order_side)
        else:
            px = self.best_price_for_passive(C_SYMBOL, order_side, improve=1)
        if px is None:
            return

        await self.send_limit(C_SYMBOL, order_side, qty, px, "c_entry")
        self.last_c_action_ts = now

        if self.have_real_c_eps and abs(self.c_eps - self.prev_c_eps) >= 0.03:
            self.c_mode = "earnings"
        elif abs(self.current_yield - Y0) >= 0.002:
            self.c_mode = "rates_lead_lag"
        else:
            self.c_mode = "none"

        logger.info(
            "C entry: side=%s qty=%d gap=%.2f fair=%.2f mid=%.2f pos=%d mode=%s",
            "BUY" if order_side == Side.BUY else "SELL", qty, gap, fair, mid, pos, self.c_mode,
        )

    # --------------------------------------------------------
    # Prediction market logic
    # --------------------------------------------------------
    def winner_target(self, side: str, q: float) -> int:
        if self.last_minute_mode() and q >= PM_WINNER_THRESHOLD:
            if q >= PM_EXTREME_THRESHOLD:
                return 200
            if q >= PM_LOCK_THRESHOLD:
                return 160
            return 120
        if q >= PM_EXTREME_THRESHOLD:
            return 120
        if q >= PM_LOCK_THRESHOLD:
            return 80
        if q >= PM_WINNER_THRESHOLD:
            return 40
        return 0

    async def flatten_pm_if_wrong_way(self, winner: str) -> None:
        losers = [s for s in [R_HIKE, R_CUT] if s != winner]
        for s in losers:
            p = self.pos[s]
            if p == 0:
                continue
            side = Side.BUY if p < 0 else Side.SELL
            px = self.best_price_for_cross(s, side)
            qty = min(abs(p), ORDER_LIMIT)
            if px is not None and qty > 0:
                await self.send_limit(s, side, qty, px, "pm_flatten_loser")

    async def maybe_trade_prediction_markets(self) -> None:
        self.update_regime()
        qh, qo, qc = self.pm_probs()
        exp_bp = self.c_exp_bp_from_probs()
        now = self.now()

        if now - self.last_pm_action_ts < 0.25:
            return

        # Endgame winner accumulation
        if self.regime in ("winner_cut", "lock_cut"):
            target = self.winner_target(R_CUT, qc)
            await self.flatten_pm_if_wrong_way(R_CUT)
            await self.pm_accumulate_single(R_CUT, target, qc, exp_bp)
            self.last_pm_action_ts = now
            return

        if self.regime in ("winner_hike", "lock_hike"):
            target = self.winner_target(R_HIKE, qh)
            await self.flatten_pm_if_wrong_way(R_HIKE)
            await self.pm_accumulate_single(R_HIKE, target, qh, exp_bp)
            self.last_pm_action_ts = now
            return

        # Regime spread trades, but no churn when the odds are already extreme
        if self.regime == "hawkish":
            await self.pm_pair_trade("hawkish", strength=exp_bp)
            self.last_pm_action_ts = now
            return

        if self.regime == "dovish":
            await self.pm_pair_trade("dovish", strength=-exp_bp)
            self.last_pm_action_ts = now
            return

        # Neutral regime: flatten residual PM inventory slowly
        for s in [R_HIKE, R_CUT]:
            p = self.pos[s]
            if p == 0:
                continue
            if abs(p) >= 20 or now - self.last_regime_change_ts > 2.5:
                side = Side.BUY if p < 0 else Side.SELL
                px = self.best_price_for_cross(s, side)
                qty = min(abs(p), ORDER_LIMIT)
                if px is not None and qty > 0:
                    await self.send_limit(s, side, qty, px, "pm_neutral_flatten")
                    self.last_pm_action_ts = now

    async def pm_accumulate_single(self, winner: str, target: int, q: float, exp_bp: float) -> None:
        pos = self.pos[winner]

        # take profits faster once the winner trade has been held for a bit
        if pos > 0 and self.pm_entry_ts[winner] > 0.0:
            held = self.now() - self.pm_entry_ts[winner]
            if held >= PM_MAX_HOLD_SECONDS or q < PM_WINNER_THRESHOLD - 0.06:
                side = Side.SELL
                px = self.best_price_for_cross(winner, side)
                qty = min(abs(pos), ORDER_LIMIT)
                if px is not None and qty > 0:
                    await self.send_limit(winner, side, qty, px, "pm_winner_exit")
                    return

        delta = target - pos
        if delta <= 0:
            return

        side = Side.BUY
        qty = min(abs(delta), self.top_qty(winner, side))
        if qty <= 0:
            return

        ask = self.best_ask.get(winner)
        bid = self.best_bid.get(winner)
        if ask is None or bid is None:
            return

        # Early: patient. Late / high conviction: cross.
        if self.last_minute_mode() or q >= PM_LOCK_THRESHOLD:
            px = ask
        elif q >= PM_WINNER_THRESHOLD:
            px = min(999, bid + 1)
        else:
            px = min(999, bid + 1)

        await self.send_limit(winner, side, qty, px, f"pm_winner_{winner.lower()}")
        self.pm_entry_ts[winner] = self.now()
        logger.info(
            "Rates winner entry: symbol=%s qty=%d q=%.3f exp_bp=%.2f target=%d",
            winner, qty, q, exp_bp, target,
        )

    async def pm_pair_trade(self, regime: str, strength: float) -> None:
        if strength < PM_ENTRY_MIN_STRENGTH:
            return

        # do not let pair trades sit around too long
        for sym in [R_HIKE, R_CUT]:
            p = self.pos[sym]
            if p != 0 and self.pm_entry_ts[sym] > 0.0 and self.now() - self.pm_entry_ts[sym] >= PM_MAX_HOLD_SECONDS:
                side = Side.BUY if p < 0 else Side.SELL
                px = self.best_price_for_cross(sym, side)
                qty = min(abs(p), ORDER_LIMIT)
                if px is not None and qty > 0:
                    await self.send_limit(sym, side, qty, px, "pm_time_exit")
                    return

        if regime == "hawkish":
            long_sym = R_HIKE
            short_sym = R_CUT
        else:
            long_sym = R_CUT
            short_sym = R_HIKE

        # stronger conviction => larger target
        if strength >= 5.0:
            target = 160
        elif strength >= 3.0:
            target = 120
        elif strength >= PM_ADD_MIN_STRENGTH:
            target = 80
        else:
            target = 40

        # avoid nonsense pair trades if one side is already near 0/1000 type winner zone
        long_mid = self.mid(long_sym)
        short_mid = self.mid(short_sym)
        if long_mid is None or short_mid is None:
            return

        # Don't short something near zero or buy something near max as a "pair trade".
        # In those cases, we should switch to winner logic instead.
        if long_mid >= 850 or short_mid <= 120:
            return

        long_delta = target - self.pos[long_sym]
        short_delta = -target - self.pos[short_sym]

        tasks = []
        if long_delta > 0:
            qty = min(abs(long_delta), self.top_qty(long_sym, Side.BUY))
            px = self.best_price_for_passive(long_sym, Side.BUY, improve=1)
            if qty > 0 and px is not None:
                self.pm_entry_ts[long_sym] = self.now()
                tasks.append(self.send_limit(long_sym, Side.BUY, qty, px, regime))

        if short_delta < 0:
            qty = min(abs(short_delta), self.top_qty(short_sym, Side.SELL))
            px = self.best_price_for_passive(short_sym, Side.SELL, improve=1)
            if qty > 0 and px is not None:
                self.pm_entry_ts[short_sym] = self.now()
                tasks.append(self.send_limit(short_sym, Side.SELL, qty, px, regime))

        if tasks:
            for t in tasks:
                await t
            logger.info("Rates entry: direction=%s strength=%.2f target=%d", regime, strength, target)

    # --------------------------------------------------------
    # Risk / cleanup
    # --------------------------------------------------------
    async def fix_stuck_c(self) -> None:
        # This specifically fixes the old issue where C sat forever after no signal.
        fair = self.fair_c()
        mid = self.mid(C_SYMBOL)
        if fair is None or mid is None:
            return
        pos = self.pos[C_SYMBOL]
        if pos == 0:
            return
        gap = fair - mid
        if self.sign(pos) != self.sign(gap) or abs(gap) <= C_HARD_STOP_EDGE:
            side = Side.SELL if pos > 0 else Side.BUY
            px = self.best_price_for_cross(C_SYMBOL, side)
            qty = min(abs(pos), ORDER_LIMIT)
            if px is not None and qty > 0:
                await self.cancel_symbol_orders(C_SYMBOL)
                self.symbol_cooldown_until[C_SYMBOL] = max(self.symbol_cooldown_until[C_SYMBOL], self.now() + 0.12)
                await self.send_limit(C_SYMBOL, side, qty, px, "c_unstuck")

    async def pm_orphan_repair(self) -> None:
        now = self.now()
        if now - self.last_pm_repair_ts < 1.5:
            return

        # only repair genuine one-sided broken exposures, not immediately every tick
        for sym in [R_HIKE, R_CUT]:
            p = self.pos[sym]
            if p == 0:
                continue
            other = R_CUT if sym == R_HIKE else R_HIKE
            other_p = self.pos[other]

            # repair if opposite side vanished and regime no longer justifies exposure
            if (abs(p) >= 40 and abs(other_p) == 0 and self.regime == "neutral") or (self.pm_entry_ts[sym] > 0.0 and now - self.pm_entry_ts[sym] >= PM_NEUTRAL_FLUSH_SECONDS and self.regime == "neutral"):
                side = Side.BUY if p < 0 else Side.SELL
                px = self.best_price_for_cross(sym, side)
                qty = min(abs(p), ORDER_LIMIT)
                if px is not None and qty > 0:
                    await self.send_limit(sym, side, qty, px, "pm_orphan_repair")
                    self.last_pm_repair_ts = now
                    logger.info("PM orphan repair triggered for %s pos=%d", sym, p)
                    return

    async def risk_loop(self) -> None:
        await self.refresh_stale_orders()
        await self.fix_stuck_c()
        await self.pm_orphan_repair()

    # --------------------------------------------------------
    # Main decision loop
    # --------------------------------------------------------
    async def maybe_trade(self) -> None:
        self.decay_macro_bias()
        await self.risk_loop()
        await self.maybe_trade_prediction_markets()
        await self.maybe_trade_c()
        self.log_heartbeat()

    def log_heartbeat(self) -> None:
        now = self.now()
        if now - self.last_heartbeat_log < HEARTBEAT_LOG_SECONDS:
            return
        self.last_heartbeat_log = now

        fair = self.fair_c()
        m = self.mid(C_SYMBOL)
        gap = None if fair is None or m is None else fair - m
        exp_bp = self.c_exp_bp_from_probs()
        qh, qo, qc = self.pm_probs()
        logger.info(
            "State: exp_bp=%.2f qh=%.3f qhld=%.3f qc=%.3f fair=%s mid=%s gap=%s pos_C=%d rates=%d/%d/%d cash=%.2f c_mode=%s regime=%s",
            exp_bp,
            qh,
            qo,
            qc,
            "na" if fair is None else f"{fair:.2f}",
            "na" if m is None else f"{m:.2f}",
            "na" if gap is None else f"{gap:.2f}",
            self.pos[C_SYMBOL],
            self.pos[R_HIKE],
            self.pos[R_HOLD],
            self.pos[R_CUT],
            self.cash,
            self.c_mode,
            self.regime,
        )

    # --------------------------------------------------------
    # Exchange callbacks
    # --------------------------------------------------------
    async def bot_handle_book_update(self, symbol: str) -> None:
        try:
            book = self.order_books[symbol]
        except Exception:
            return

        try:
            bids = book.bids
            asks = book.asks
        except Exception:
            return

        self.best_bid[symbol] = max(bids.keys()) if bids else None
        self.best_ask[symbol] = min(asks.keys()) if asks else None

        await self.maybe_trade()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        await self.maybe_trade()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        oid = str(order_id)
        self.pending_cancels.discard(oid)
        ref = self.live_orders.pop(oid, None)
        if ref is not None:
            if ref.side == Side.BUY:
                self.pending_buy[ref.symbol] = max(0, self.pending_buy[ref.symbol] - ref.qty)
            else:
                self.pending_sell[ref.symbol] = max(0, self.pending_sell[ref.symbol] - ref.qty)
            if oid in self.open_by_symbol[ref.symbol]:
                self.open_by_symbol[ref.symbol].remove(oid)
            self.symbol_cooldown_until[ref.symbol] = max(self.symbol_cooldown_until[ref.symbol], self.now() + 0.05)
        logger.info("Cancel acknowledged: order_id=%s success=%s error=%s", oid, success, error)

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        oid = str(order_id)
        self.pending_cancels.discard(oid)
        ref = self.live_orders.get(oid)
        if ref is None:
            return

        self.update_position_from_fill(ref.symbol, ref.side, qty, price)
        if ref.side == Side.BUY:
            self.pending_buy[ref.symbol] = max(0, self.pending_buy[ref.symbol] - qty)
        else:
            self.pending_sell[ref.symbol] = max(0, self.pending_sell[ref.symbol] - qty)
        logger.info(
            "Fill: order_id=%s symbol=%s qty=%d price=%d pos_C=%d pos_rates=%d/%d/%d cash=%.2f",
            oid,
            ref.symbol,
            qty,
            price,
            self.pos[C_SYMBOL],
            self.pos[R_HIKE],
            self.pos[R_HOLD],
            self.pos[R_CUT],
            self.cash,
        )

        ref.qty -= qty
        if ref.qty <= 0:
            self.live_orders.pop(oid, None)
            if oid in self.open_by_symbol[ref.symbol]:
                self.open_by_symbol[ref.symbol].remove(oid)

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
            if oid in self.open_by_symbol[ref.symbol]:
                self.open_by_symbol[ref.symbol].remove(oid)
            self.symbol_cooldown_until[ref.symbol] = max(self.symbol_cooldown_until[ref.symbol], self.now() + 0.25)
        logger.info("Order rejected: order_id=%s reason=%s", oid, reason)

    async def bot_handle_position_snapshot(self, positions: Dict[str, int], cash: float) -> None:
        for s, p in positions.items():
            if s in self.pos:
                self.pos[s] = p
        self.cash = cash
        logger.info(
            "Startup position snapshot: C=%d H=%d O=%d CUT=%d cash=%.2f",
            self.pos[C_SYMBOL], self.pos[R_HIKE], self.pos[R_HOLD], self.pos[R_CUT], self.cash,
        )

    async def bot_handle_news(self, news_id: str, text: str, tick: int) -> None:
        self.last_tick_seen = max(self.last_tick_seen, tick)
        lower = text.lower()

        # CPI: parse actual and forecast, convert surprise into bp bias
        if "cpi" in lower and "forecast" in lower and "actual" in lower:
            nums = [float(x) for x in re.findall(FLOAT_RE, text)]
            if len(nums) >= 2:
                actual = nums[0]
                forecast = nums[1]
                surprise = actual - forecast
                # scale into rate-bias bp, capped
                bias_bp = self.clamp(4200.0 * surprise, -5.0, 5.0)
                self.apply_macro_bias_bp(bias_bp, "cpi")
                logger.info(
                    "CPI surprise at tick %d: actual=%.6f forecast=%.6f bias_bp=%.2f",
                    tick, actual, forecast, bias_bp,
                )
                await self.maybe_trade()
                return

        # C earnings
        if "c earnings" in lower or ("earnings" in lower and "c" in lower):
            nums = [float(x) for x in re.findall(FLOAT_RE, text)]
            if nums:
                new_eps = nums[-1]
                old_eps = self.c_eps
                self.prev_c_eps = self.c_eps
                self.c_eps = new_eps
                if not self.have_real_c_eps:
                    self.have_real_c_eps = True
                logger.info(
                    "C earnings update at tick %d: %.4f -> %.4f delta=%+.4f initial=%s",
                    tick, old_eps, new_eps, new_eps - old_eps, str(not self.have_real_c_eps),
                )
                await self.maybe_trade()
                return

        # Ignore A earnings here on purpose
        if "a earnings" in lower or ("earnings" in lower and " a " in f" {lower} "):
            logger.info("Ignoring A earnings in C-only mode at tick %d: %s", tick, text)
            return

        # Fed / unstructured macro text
        fedish_words = ["fed", "chair", "officials", "policy", "inflation", "rates"]
        if any(w in lower for w in fedish_words):
            bias_bp = self.parse_text_bias_bp(text)
            if abs(bias_bp) > 0:
                self.apply_macro_bias_bp(bias_bp, "fed")
                logger.info("Fed headline at tick %d: bias_bp=%.2f content=%s", tick, bias_bp, text)
                await self.maybe_trade()
                return

        await self.maybe_trade()

    # --------------------------------------------------------
    # Start / reconnect
    # --------------------------------------------------------
    async def start(self) -> None:
        while True:
            try:
                logger.info("Connecting to exchange...")
                await self.connect()
                logger.info("Connected.")
                await self.trade_loop()
            except Exception as e:
                logger.exception("Disconnected / error: %s", e)
                await asyncio.sleep(1.5)

    async def trade_loop(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            await self.maybe_trade()


async def run_bot() -> None:
    client = MyXchangeClient(
        host="practice.uchicago.exchange:3333",
        username="uiuc",
        password="mesa-lynx-octopus",
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(run_bot())
