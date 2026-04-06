from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError:
    class Side(Enum):
        BUY = "BUY"
        SELL = "SELL"

    class XChangeClient:  # pragma: no cover - fallback for offline replay analysis mode
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "utcxchangelib is required to run the live bot. Replay analysis mode can run without it."
            )


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("prediction-c")


@dataclass(frozen=True)
class BotConfig:
    symbol_c: str = "C"
    fed_hike: str = "R_HIKE"
    fed_hold: str = "R_HOLD"
    fed_cut: str = "R_CUT"

    payout_scale: float = 100.0
    default_eps_c: float = 2.0

    c_ops_weight: float = 0.72
    c_bond_weight: float = 0.28
    c_pe_yield_gamma: float = 13.0
    c_bond_duration: float = 4.5
    c_bond_convexity: float = 30.0

    cpi_to_rate_bp: float = 4000.0
    max_temp_rate_bias_bp: float = 8.0
    cpi_bias_ttl_secs: float = 2.5
    headline_bias_ttl_secs: float = 2.0

    c_earnings_ignore_delta: float = 0.010
    c_earnings_small_delta: float = 0.025
    c_earnings_medium_delta: float = 0.045
    c_earnings_fresh_secs: float = 4.0
    c_earnings_hold_secs: float = 2.5
    c_initial_shock_window_secs: float = 4.0
    c_initial_shock_gap_ticks: float = 14.0
    c_initial_shock_initial_size: int = 60
    c_initial_shock_add_size: int = 60
    c_initial_shock_cap: int = 120
    c_initial_shock_compress_ticks: float = 10.0
    c_initial_shock_flatten_ticks: float = 6.0
    c_initial_shock_max_hold_secs: float = 12.0
    c_leadlag_max_hold_secs: float = 25.0

    post_baseline_entry_secs: float = 10.0
    post_baseline_gap_ticks: float = 14.0
    post_baseline_add_ticks: float = 20.0

    lead_lag_bp_trigger: float = 3.0
    lead_lag_entry_ticks: float = 16.0
    lead_lag_add_ticks: float = 22.0

    c_hard_position_limit: int = 200
    rate_hard_position_limit: int = 150
    c_max_order_size: int = 40
    rate_max_order_size: int = 40

    rate_normal_size: int = 50
    rate_strong_size: int = 100
    rate_extreme_size: int = 150
    rate_entry_edge_bp: float = 2.5
    rate_exit_edge_bp: float = 1.0
    rate_add_edge_step_bp: float = 1.5
    rate_add_edge_frac: float = 0.65
    rate_reentry_block_secs: float = 1.0

    c_tier1_initial_size: int = 60
    c_tier1_add_size: int = 40
    c_tier1_cap: int = 100
    c_tier2_initial_size: int = 80
    c_tier2_add_size: int = 70
    c_tier2_cap: int = 150
    c_tier3_initial_size: int = 100
    c_tier3_add_size: int = 100
    c_tier3_cap: int = 200
    c_baseline_initial_size: int = 60
    c_baseline_add_size: int = 60
    c_baseline_cap: int = 120
    c_leadlag_initial_size: int = 50
    c_leadlag_add_size: int = 50
    c_leadlag_cap: int = 100
    c_event_base_budget: int = 120
    c_background_base_budget: int = 100
    c_event_shift_budget: int = 200
    c_background_shift_budget: int = 0
    c_unwind_entry_position: int = 80
    c_unwind_exit_position: int = 20
    c_unwind_aggressive_entry: int = 140
    c_unwind_aggressive_exit: int = 80

    c_add_edge_step_ticks: float = 6.0
    c_news_add_edge_frac: float = 0.60
    c_hard_flip_min_ticks: float = 6.0
    c_compression_min_ticks: float = 6.0
    c_compression_frac: float = 0.35
    c_rate_reversal_bp: float = 1.5
    c_reentry_block_secs: float = 1.5
    c_flat_entry_cooldown_secs: float = 0.8
    c_add_cooldown_secs: float = 0.35

    max_active_orders_per_symbol: int = 1
    order_stale_secs: float = 0.50
    anchor_reprice_secs: float = 2.0
    loop_sleep_secs: float = 0.20
    status_log_interval_secs: float = 3.0

    startup_flatten_chunk_c: int = 25
    startup_flatten_chunk_rate: int = 40
    startup_flatten_sleep_secs: float = 0.25

    @property
    def tracked_symbols(self) -> tuple[str, str, str, str]:
        return (self.symbol_c, self.fed_hike, self.fed_hold, self.fed_cut)

    @property
    def rate_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


@dataclass
class TopOfBook:
    bid: Optional[int] = None
    bid_qty: int = 0
    ask: Optional[int] = None
    ask_qty: int = 0
    updated_ts: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        if self.bid is not None:
            return float(self.bid)
        if self.ask is not None:
            return float(self.ask)
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return float(self.ask - self.bid)

    @property
    def usable_spread(self) -> float:
        spread = self.spread
        if spread is None:
            return 0.0
        return max(1.0, abs(spread))


@dataclass
class TrackedOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    role: str
    engine: str
    reason: str
    thesis: Optional[str]
    signal_strength: float
    created_at: float = field(default_factory=time.time)


@dataclass
class RateSnapshot:
    q_hike: float
    q_hold: float
    q_cut: float
    market_expected_rate_bp: float
    effective_expected_rate_bp: float
    delta_market_bp: float
    delta_effective_bp: float
    bias_bp: float
    urgent: bool
    fresh_macro_event: bool
    macro_source: Optional[str]


@dataclass
class EarningsContext:
    delta: float
    abs_delta: float
    age: float
    tier: int
    is_initial: bool
    side: Optional[Side]
    hold_active: bool


@dataclass
class CSignal:
    bid: int
    bid_qty: int
    ask: int
    ask_qty: int
    mid: float
    spread: float
    fair: float
    gap: float
    gap_abs: float
    bp_dislocation: float
    fair_change: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class CEntryDecision:
    side: Side
    thesis: str
    edge_ticks: float
    initial_size: int
    add_size: int
    thesis_cap: int


@dataclass
class RatesEntryDecision:
    direction: str
    edge_bp: float
    target_size: int


@dataclass
class MarketState:
    books: Dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: Dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)
    last_trade_price: Dict[str, int] = field(default_factory=dict)

    current_eps_c: float = 2.0
    have_real_eps_c: bool = False
    last_c_earnings_delta: float = 0.0
    last_c_earnings_ts: float = 0.0
    last_c_earnings_is_initial: bool = False
    last_c_initial_baseline_ts: float = 0.0
    c_initial_shock_consumed: bool = False

    anchor_price: Optional[float] = None
    anchor_eps: Optional[float] = None
    anchor_yield_bp: Optional[float] = None
    anchor_has_real_eps: bool = False
    last_anchor_update_ts: float = 0.0

    temp_rate_bias_bp: float = 0.0
    temp_rate_bias_started_at: float = 0.0
    temp_rate_bias_expires_at: float = 0.0
    last_macro_event_ts: float = 0.0
    last_macro_source: Optional[str] = None
    last_macro_bias_bp: float = 0.0
    news_urgency_until: float = 0.0

    last_market_expected_rate_bp: Optional[float] = None
    last_effective_expected_rate_bp: Optional[float] = None
    last_fair_c: Optional[float] = None

    startup_flatten_complete: bool = False
    session_start_cash: Optional[float] = None
    session_start_mtm: Optional[float] = None

    last_status_log_ts: float = 0.0

    c_regime_side: Optional[Side] = None
    c_regime_thesis: Optional[str] = None
    c_entry_stage: int = 0
    c_last_entry_edge: float = 0.0
    c_last_add_ts: float = 0.0
    c_regime_started_at: float = 0.0
    c_blocked_side: Optional[Side] = None
    c_blocked_until: float = 0.0
    c_flat_entry_cooldown_until: float = 0.0
    c_unwind_active: bool = False
    c_unwind_aggressive_active: bool = False

    rates_regime_direction: Optional[str] = None
    rates_entry_stage: int = 0
    rates_last_entry_edge: float = 0.0
    rates_last_add_ts: float = 0.0
    rates_blocked_direction: Optional[str] = None
    rates_blocked_until: float = 0.0

    def clear_c_regime(self) -> None:
        self.c_regime_side = None
        self.c_regime_thesis = None
        self.c_entry_stage = 0
        self.c_last_entry_edge = 0.0
        self.c_last_add_ts = 0.0
        self.c_regime_started_at = 0.0
        self.c_unwind_active = False
        self.c_unwind_aggressive_active = False

    def clear_rates_regime(self) -> None:
        self.rates_regime_direction = None
        self.rates_entry_stage = 0
        self.rates_last_entry_edge = 0.0
        self.rates_last_add_ts = 0.0


class OrderManager:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def has_live_order(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        engine: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> bool:
        for order in self.state.live_orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if engine is not None and order.engine != engine:
                continue
            if side is not None and order.side != side:
                continue
            return True
        return False

    def pending_qty(self, symbol: str, side: Side) -> int:
        total = 0
        for order in self.state.live_orders.values():
            if order.symbol == symbol and order.side == side:
                total += int(order.qty)
        return total

    async def cancel_order_if_present(self, order_id: str) -> None:
        order_key = str(order_id)
        if order_key in self.state.pending_cancels:
            return

        tracked = self.state.live_orders.get(order_key)
        self.state.pending_cancels.add(order_key)
        try:
            await self.client.cancel_order(order_id)
        except Exception as exc:
            message = str(exc)
            if "No such order" in message:
                self.state.live_orders.pop(order_key, None)
            else:
                LOGGER.warning("Cancel failed for %s: %s", order_id, exc)
                if tracked is not None:
                    self.state.live_orders[order_key] = tracked
        finally:
            self.state.pending_cancels.discard(order_key)

    async def cancel_live_orders(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        engine: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> None:
        order_ids: list[str] = []
        for order_id, order in self.state.live_orders.items():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if engine is not None and order.engine != engine:
                continue
            if side is not None and order.side != side:
                continue
            order_ids.append(order_id)

        for order_id in order_ids:
            await self.cancel_order_if_present(order_id)

    async def cancel_stale_orders(self) -> None:
        now = time.time()
        stale = [
            order_id
            for order_id, order in self.state.live_orders.items()
            if now - order.created_at >= self.cfg.order_stale_secs
        ]
        for order_id in stale:
            await self.cancel_order_if_present(order_id)

    async def place_tracked_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: Side,
        price: int,
        role: str,
        engine: str,
        reason: str,
        thesis: Optional[str] = None,
        signal_strength: float = 0.0,
    ) -> bool:
        if qty <= 0:
            return False

        if self.has_live_order(symbol=symbol):
            return False

        same_symbol_orders = [order for order in self.state.live_orders.values() if order.symbol == symbol]
        if len(same_symbol_orders) >= self.cfg.max_active_orders_per_symbol:
            await self.cancel_order_if_present(same_symbol_orders[0].order_id)
            return False

        order_id = await self.client.place_order(symbol, int(qty), side, int(price))
        if order_id is None:
            return False

        tracked = TrackedOrder(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            qty=int(qty),
            price=int(price),
            role=role,
            engine=engine,
            reason=reason,
            thesis=thesis,
            signal_strength=float(signal_strength),
        )
        self.state.live_orders[str(order_id)] = tracked
        LOGGER.info(
            "Placed %s order: symbol=%s side=%s qty=%s price=%s thesis=%s strength=%.2f pos=%s",
            reason,
            symbol,
            side.name,
            qty,
            price,
            thesis or "none",
            signal_strength,
            self.client.get_position(symbol),
        )
        return True

    def sync_fill(self, order_id: str) -> Optional[TrackedOrder]:
        order_key = str(order_id)
        tracked = self.state.live_orders.get(order_key)
        if tracked is None:
            return None

        if order_key in self.client.open_orders:
            remaining_qty = int(self.client.open_orders[order_key][1])
            tracked.qty = remaining_qty
            if remaining_qty <= 0:
                self.state.live_orders.pop(order_key, None)
        else:
            self.state.live_orders.pop(order_key, None)
        return tracked

    def sync_rejected(self, order_id: str) -> Optional[TrackedOrder]:
        return self.state.live_orders.pop(str(order_id), None)

    def sync_cancel_response(self, order_id: str, success: bool) -> Optional[TrackedOrder]:
        if success:
            return self.state.live_orders.pop(str(order_id), None)
        return self.state.live_orders.get(str(order_id))


class RiskManager:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState, orders: OrderManager):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders

    def clip_c_qty(self, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(self.cfg.symbol_c)
        pending = self.orders.pending_qty(self.cfg.symbol_c, side)

        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.c_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.c_max_order_size))

    def clip_rate_qty(self, symbol: str, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(symbol)
        pending = self.orders.pending_qty(symbol, side)

        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.rate_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.rate_max_order_size))

    def arm_session_baseline_if_ready(self) -> bool:
        if self.state.session_start_cash is not None and self.state.session_start_mtm is not None:
            return False

        for symbol in self.cfg.tracked_symbols:
            if self.client.get_position(symbol) != 0:
                return False
        if self.client.open_orders:
            return False
        if self.state.live_orders:
            return False

        cash, mtm = self.client.cash_and_total_mtm()
        self.state.session_start_cash = cash
        self.state.session_start_mtm = mtm
        LOGGER.info(
            "Session baseline armed: cash=%.2f mtm=%.2f session_pnl=0.00",
            cash,
            mtm,
        )
        return True

    async def startup_flatten_step(self) -> bool:
        inherited_order_ids = [str(order_id) for order_id in self.client.open_orders.keys() if str(order_id) not in self.state.live_orders]
        for order_id in inherited_order_ids:
            await self.orders.cancel_order_if_present(order_id)

        for symbol in self.cfg.tracked_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue

            book = self.client.top(symbol)
            if book.bid is None or book.ask is None:
                continue

            side = Side.SELL if pos > 0 else Side.BUY
            chunk = self.cfg.startup_flatten_chunk_c if symbol == self.cfg.symbol_c else self.cfg.startup_flatten_chunk_rate
            max_order_size = self.cfg.c_max_order_size if symbol == self.cfg.symbol_c else self.cfg.rate_max_order_size
            qty = min(abs(pos), chunk, max_order_size)
            price = int(book.bid if side == Side.SELL else book.ask)
            if self.orders.has_live_order(symbol=symbol):
                continue
            await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="flatten",
                engine="risk",
                reason="startup_flatten",
                signal_strength=float(abs(pos)),
            )

        all_flat = all(self.client.get_position(symbol) == 0 for symbol in self.cfg.tracked_symbols)
        no_orders = not self.client.open_orders and not self.state.live_orders
        if all_flat and no_orders:
            self.state.startup_flatten_complete = True
            self.arm_session_baseline_if_ready()
            LOGGER.info("Startup flatten complete; all tracked positions and inherited orders are flat.")
            return True
        return False


class RatesSignalEngine:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def current_temp_bias(self) -> float:
        now = time.time()
        if now >= self.state.temp_rate_bias_expires_at or self.state.temp_rate_bias_expires_at <= self.state.temp_rate_bias_started_at:
            self.state.temp_rate_bias_bp = 0.0
            return 0.0

        duration = self.state.temp_rate_bias_expires_at - self.state.temp_rate_bias_started_at
        remaining = max(0.0, self.state.temp_rate_bias_expires_at - now)
        if duration <= 0:
            return 0.0
        return self.state.temp_rate_bias_bp * (remaining / duration)

    def mark_news_urgent(self, ttl_secs: float) -> None:
        self.state.news_urgency_until = max(self.state.news_urgency_until, time.time() + ttl_secs)

    def is_news_urgent(self) -> bool:
        return time.time() < self.state.news_urgency_until

    def apply_temp_rate_bias(self, delta_bp: float, ttl_secs: float, source: str) -> None:
        current_bias = self.current_temp_bias()
        next_bias = self.client.clip(current_bias + delta_bp, -self.cfg.max_temp_rate_bias_bp, self.cfg.max_temp_rate_bias_bp)
        now = time.time()
        self.state.temp_rate_bias_bp = next_bias
        self.state.temp_rate_bias_started_at = now
        self.state.temp_rate_bias_expires_at = now + ttl_secs
        self.state.last_macro_event_ts = now
        self.state.last_macro_source = source
        self.state.last_macro_bias_bp = next_bias
        self.mark_news_urgent(ttl_secs)

    def headline_rate_bias_bp(self, content: str) -> float:
        text = content.lower()
        score = 0.0

        hawkish = {
            "inflation risks": 1.5,
            "reassess path of cuts": 1.5,
            "higher for longer": 2.0,
            "keeps options open": 1.0,
            "keep options open": 1.0,
            "strong demand and sticky prices": 2.0,
            "sticky prices": 1.5,
            "stay restrictive for longer": 2.0,
            "policy may stay restrictive": 2.0,
            "persistent inflation": 1.5,
            "concerned about wage growth": 1.5,
            "pressure on the fed": 1.5,
            "emphasizes inflation risks": 2.0,
        }
        dovish = {
            "moving back to target": -1.5,
            "cooling inflation": -1.0,
            "softer inflation": -1.0,
            "disinflation": -1.0,
            "confidence inflation is moving back to target": -2.0,
            "softening data": -1.5,
            "policy easing": -1.5,
            "expectations of policy easing": -2.0,
            "cooling labor market": -1.25,
            "easing inflation pressures": -1.5,
            "cooling labor market and easing inflation pressures": -2.0,
            "increasing confidence inflation is moving back to target": -2.0,
        }

        for phrase, value in hawkish.items():
            if phrase in text:
                score += value
        for phrase, value in dovish.items():
            if phrase in text:
                score += value

        if (
            "balanced risks" in text
            or "mixed economic indicators" in text
            or "communication remains cautious" in text
            or "await upcoming data" in text
        ):
            score *= 0.5

        return self.client.clip(score, -2.0, 2.0)

    def fed_probs(self) -> Optional[tuple[float, float, float]]:
        mid_hike = self.client.mid(self.cfg.fed_hike)
        mid_hold = self.client.mid(self.cfg.fed_hold)
        mid_cut = self.client.mid(self.cfg.fed_cut)
        if mid_hike is None or mid_hold is None or mid_cut is None:
            return None

        q_hike = mid_hike / self.cfg.payout_scale
        q_hold = mid_hold / self.cfg.payout_scale
        q_cut = mid_cut / self.cfg.payout_scale
        total = q_hike + q_hold + q_cut
        if total <= 1e-9:
            return None
        return q_hike / total, q_hold / total, q_cut / total

    def expected_rate_bp(self) -> Optional[float]:
        probs = self.fed_probs()
        if probs is None:
            return None
        q_hike, _, q_cut = probs
        return 25.0 * q_hike - 25.0 * q_cut

    def snapshot(self) -> Optional[RateSnapshot]:
        probs = self.fed_probs()
        if probs is None:
            return None

        market_expected_rate_bp = self.expected_rate_bp()
        if market_expected_rate_bp is None:
            return None

        bias_bp = self.current_temp_bias()
        effective_expected_rate_bp = market_expected_rate_bp + bias_bp
        delta_market_bp = (
            0.0
            if self.state.last_market_expected_rate_bp is None
            else market_expected_rate_bp - self.state.last_market_expected_rate_bp
        )
        delta_effective_bp = (
            0.0
            if self.state.last_effective_expected_rate_bp is None
            else effective_expected_rate_bp - self.state.last_effective_expected_rate_bp
        )
        fresh_macro = False
        if self.state.last_macro_event_ts > 0.0:
            ttl = self.cfg.cpi_bias_ttl_secs if self.state.last_macro_source == "cpi_print" else self.cfg.headline_bias_ttl_secs
            fresh_macro = time.time() - self.state.last_macro_event_ts <= ttl

        return RateSnapshot(
            q_hike=probs[0],
            q_hold=probs[1],
            q_cut=probs[2],
            market_expected_rate_bp=market_expected_rate_bp,
            effective_expected_rate_bp=effective_expected_rate_bp,
            delta_market_bp=delta_market_bp,
            delta_effective_bp=delta_effective_bp,
            bias_bp=bias_bp,
            urgent=self.is_news_urgent(),
            fresh_macro_event=fresh_macro,
            macro_source=self.state.last_macro_source,
        )


class CFairValueEngine:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState, rates: RatesSignalEngine):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.rates = rates

    def earnings_context(self) -> EarningsContext:
        if self.state.last_c_earnings_ts <= 0.0:
            return EarningsContext(0.0, 0.0, float("inf"), 0, False, None, False)

        age = max(0.0, time.time() - self.state.last_c_earnings_ts)
        delta = self.state.last_c_earnings_delta
        abs_delta = abs(delta)
        side: Optional[Side] = None
        if delta > 0:
            side = Side.BUY
        elif delta < 0:
            side = Side.SELL

        tier = 0
        if not self.state.last_c_earnings_is_initial:
            if abs_delta >= self.cfg.c_earnings_medium_delta:
                tier = 3
            elif abs_delta >= self.cfg.c_earnings_small_delta:
                tier = 2
            elif abs_delta >= self.cfg.c_earnings_ignore_delta:
                tier = 1

        hold_secs = {
            0: 0.0,
            1: 1.50,
            2: self.cfg.c_earnings_hold_secs,
            3: self.cfg.c_earnings_hold_secs + 0.75,
        }[tier]

        return EarningsContext(
            delta=delta,
            abs_delta=abs_delta,
            age=age,
            tier=tier,
            is_initial=self.state.last_c_earnings_is_initial,
            side=side,
            hold_active=tier > 0 and age <= hold_secs,
        )

    def initialize_anchor(self, rate_snapshot: Optional[RateSnapshot], *, force: bool = False) -> bool:
        book = self.client.top(self.cfg.symbol_c)
        mid_c = book.mid
        if mid_c is None or rate_snapshot is None:
            return False

        if (
            self.state.anchor_price is None
            or self.state.anchor_eps is None
            or self.state.anchor_yield_bp is None
            or force
        ):
            self.state.anchor_price = mid_c
            self.state.anchor_eps = self.state.current_eps_c
            self.state.anchor_yield_bp = rate_snapshot.effective_expected_rate_bp
            self.state.anchor_has_real_eps = self.state.have_real_eps_c
            self.state.last_anchor_update_ts = time.time()
            LOGGER.info(
                "Initialized C anchor: price=%.2f eps=%.4f exp_bp=%.2f source=%s",
                self.state.anchor_price,
                self.state.anchor_eps,
                self.state.anchor_yield_bp,
                "real_earnings" if self.state.anchor_has_real_eps else "default_eps",
            )
            return True
        return False

    def maybe_refresh_anchor_after_first_real_eps(self, rate_snapshot: Optional[RateSnapshot]) -> None:
        if not self.state.have_real_eps_c:
            return
        if self.state.anchor_has_real_eps:
            return
        if self.initialize_anchor(rate_snapshot, force=True):
            now = time.time()
            self.state.last_c_initial_baseline_ts = now
            self.state.c_initial_shock_consumed = False
            LOGGER.info(
                "Adopted first real C EPS as new baseline anchor: price=%.2f eps=%.4f exp_bp=%.2f",
                self.state.anchor_price or 0.0,
                self.state.anchor_eps or 0.0,
                self.state.anchor_yield_bp or 0.0,
            )

    def maybe_reanchor(self, signal: Optional[CSignal], rate_snapshot: Optional[RateSnapshot]) -> None:
        if signal is None or rate_snapshot is None:
            return
        if self.client.get_position(self.cfg.symbol_c) != 0:
            return
        if self.orders_live_for_c():
            return
        if self.rates.is_news_urgent():
            return
        if time.time() - self.state.last_anchor_update_ts < self.cfg.anchor_reprice_secs:
            return
        if signal.gap_abs > max(signal.spread, signal.exit_threshold):
            return

        self.state.anchor_price = 0.98 * float(self.state.anchor_price) + 0.02 * signal.mid
        self.state.anchor_yield_bp = (
            0.98 * float(self.state.anchor_yield_bp) + 0.02 * rate_snapshot.effective_expected_rate_bp
        )
        self.state.last_anchor_update_ts = time.time()

    def orders_live_for_c(self) -> bool:
        return any(order.symbol == self.cfg.symbol_c for order in self.state.live_orders.values())

    def fair_value(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[float]:
        if (
            self.state.anchor_price is None
            or self.state.anchor_eps is None
            or self.state.anchor_yield_bp is None
            or rate_snapshot is None
        ):
            return None

        if self.state.anchor_eps == 0:
            return None

        anchor_price = float(self.state.anchor_price)
        anchor_eps = float(self.state.anchor_eps)
        anchor_yield_bp = float(self.state.anchor_yield_bp)
        yield_now = float(rate_snapshot.effective_expected_rate_bp)
        dy = (yield_now - anchor_yield_bp) / 10000.0

        ops_anchor = self.cfg.c_ops_weight * anchor_price
        bond_anchor = self.cfg.c_bond_weight * anchor_price

        ops_fair = ops_anchor * (self.state.current_eps_c / anchor_eps) * math.exp(-self.cfg.c_pe_yield_gamma * dy)
        bond_fair = bond_anchor * (1.0 - self.cfg.c_bond_duration * dy + 0.5 * self.cfg.c_bond_convexity * dy * dy)
        return ops_fair + bond_fair

    def snapshot(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[CSignal]:
        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return None

        fair = self.fair_value(rate_snapshot)
        if fair is None:
            return None
        if self.state.anchor_yield_bp is None or rate_snapshot is None:
            return None

        mid = book.mid
        if mid is None:
            return None

        fair_change = 0.0 if self.state.last_fair_c is None else fair - self.state.last_fair_c
        spread = book.usable_spread
        return CSignal(
            bid=int(book.bid),
            bid_qty=int(book.bid_qty),
            ask=int(book.ask),
            ask_qty=int(book.ask_qty),
            mid=float(mid),
            spread=float(spread),
            fair=float(fair),
            gap=float(fair - mid),
            gap_abs=float(abs(fair - mid)),
            bp_dislocation=float(rate_snapshot.effective_expected_rate_bp - self.state.anchor_yield_bp),
            fair_change=float(fair_change),
            entry_threshold=max(8.0, 1.25 * spread),
            exit_threshold=max(6.0, 0.75 * spread),
        )


class RatesTradingEngine:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk

    def current_pair_direction(self) -> Optional[str]:
        pos_hike = self.client.get_position(self.cfg.fed_hike)
        pos_cut = self.client.get_position(self.cfg.fed_cut)
        if pos_hike > 0 and pos_cut < 0:
            return "hawkish"
        if pos_hike < 0 and pos_cut > 0:
            return "dovish"
        return None

    def current_pair_abs(self) -> int:
        pos_hike = abs(self.client.get_position(self.cfg.fed_hike))
        pos_cut = abs(self.client.get_position(self.cfg.fed_cut))
        return max(pos_hike, pos_cut)

    def max_entry_stages(self, target_size: int) -> int:
        return max(1, math.ceil(max(0, target_size) / max(1, self.cfg.rate_max_order_size)))

    async def flatten_all_rates(self, reason: str) -> bool:
        acted = False
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None:
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.rate_max_order_size)
            price = int(book.bid if side == Side.SELL else book.ask)
            if self.orders.has_live_order(symbol=symbol):
                continue
            placed = await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="exit",
                engine="rates",
                reason=reason,
                thesis=self.state.rates_regime_direction,
                signal_strength=float(abs(pos)),
            )
            acted = acted or placed
        if acted:
            self.state.rates_blocked_direction = self.state.rates_regime_direction
            self.state.rates_blocked_until = time.time() + self.cfg.rate_reentry_block_secs
            self.state.clear_rates_regime()
            LOGGER.info("Rates exit trigger: %s", reason)
        return acted

    async def handle_missing_signal_exit(self) -> bool:
        if any(self.client.get_position(symbol) != 0 for symbol in self.cfg.rate_symbols):
            return await self.flatten_all_rates("signal_lost")
        return False

    async def maybe_exit(self, snapshot: RateSnapshot) -> bool:
        direction = self.current_pair_direction()
        if direction is None:
            if self.client.get_position(self.cfg.fed_hold) != 0:
                return await self.flatten_all_rates("signal_lost")
            return False

        edge_bp = abs(snapshot.bias_bp)
        compressed = edge_bp <= max(self.cfg.rate_exit_edge_bp, self.state.rates_last_entry_edge * 0.30)
        opposite = (direction == "hawkish" and snapshot.bias_bp <= -self.cfg.rate_exit_edge_bp) or (
            direction == "dovish" and snapshot.bias_bp >= self.cfg.rate_exit_edge_bp
        )
        stale = edge_bp < self.cfg.rate_exit_edge_bp

        if stale or opposite or compressed:
            return await self.flatten_all_rates("bias_decay" if stale or compressed else "macro_reversal")
        return False

    def compute_entry_decision(self, snapshot: RateSnapshot) -> Optional[RatesEntryDecision]:
        if not snapshot.fresh_macro_event:
            return None
        if snapshot.bias_bp >= self.cfg.rate_entry_edge_bp:
            direction = "hawkish"
        elif snapshot.bias_bp <= -self.cfg.rate_entry_edge_bp:
            direction = "dovish"
        else:
            return None

        edge_bp = abs(snapshot.bias_bp)
        if snapshot.macro_source == "cpi_print" and edge_bp >= 5.0:
            target_size = self.cfg.rate_extreme_size
        elif edge_bp >= 3.5 or snapshot.urgent:
            target_size = self.cfg.rate_strong_size
        else:
            target_size = self.cfg.rate_normal_size
        return RatesEntryDecision(direction=direction, edge_bp=edge_bp, target_size=target_size)

    async def maybe_enter(self, snapshot: RateSnapshot) -> bool:
        decision = self.compute_entry_decision(snapshot)
        if decision is None:
            return False

        now = time.time()
        if self.state.rates_blocked_direction == decision.direction and now < self.state.rates_blocked_until:
            return False

        current_direction = self.current_pair_direction()
        if current_direction is not None and current_direction != decision.direction:
            return False

        if current_direction is not None:
            if self.state.rates_entry_stage >= self.max_entry_stages(decision.target_size):
                return False
            if decision.edge_bp < self.state.rates_last_entry_edge + self.cfg.rate_add_edge_step_bp:
                required_edge = max(
                    self.cfg.rate_entry_edge_bp,
                    self.state.rates_last_entry_edge * self.cfg.rate_add_edge_frac,
                )
                if decision.edge_bp < required_edge:
                    return False
            if now - self.state.rates_last_add_ts < self.cfg.c_add_cooldown_secs:
                return False

        if decision.direction == "hawkish":
            buy_symbol = self.cfg.fed_hike
            sell_symbol = self.cfg.fed_cut
        else:
            buy_symbol = self.cfg.fed_cut
            sell_symbol = self.cfg.fed_hike

        buy_book = self.client.top(buy_symbol)
        sell_book = self.client.top(sell_symbol)
        if buy_book.ask is None or sell_book.bid is None:
            return False

        current_leg_abs = self.current_pair_abs()
        if current_leg_abs >= decision.target_size:
            return False
        desired_leg = decision.target_size - current_leg_abs

        buy_qty = self.risk.clip_rate_qty(buy_symbol, Side.BUY, desired_leg, decision.target_size)
        sell_qty = self.risk.clip_rate_qty(sell_symbol, Side.SELL, desired_leg, decision.target_size)
        qty = min(buy_qty, sell_qty)
        if qty <= 0:
            return False

        if self.orders.has_live_order(symbol=buy_symbol) or self.orders.has_live_order(symbol=sell_symbol):
            return False

        placed_buy = await self.orders.place_tracked_order(
            symbol=buy_symbol,
            qty=qty,
            side=Side.BUY,
            price=int(buy_book.ask),
            role="entry",
            engine="rates",
            reason="rates_entry",
            thesis=decision.direction,
            signal_strength=decision.edge_bp,
        )
        placed_sell = await self.orders.place_tracked_order(
            symbol=sell_symbol,
            qty=qty,
            side=Side.SELL,
            price=int(sell_book.bid),
            role="entry",
            engine="rates",
            reason="rates_entry",
            thesis=decision.direction,
            signal_strength=decision.edge_bp,
        )
        placed = placed_buy or placed_sell
        if placed:
            self.state.rates_regime_direction = decision.direction
            self.state.rates_entry_stage = 1 if current_direction is None else self.state.rates_entry_stage + 1
            self.state.rates_last_entry_edge = decision.edge_bp
            self.state.rates_last_add_ts = now
            LOGGER.info(
                "Rates entry: direction=%s qty=%s edge_bp=%.2f target=%s",
                decision.direction,
                qty,
                decision.edge_bp,
                decision.target_size,
            )
        return placed


class CTradingEngine:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        fair_engine: CFairValueEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.fair_engine = fair_engine

    def initial_shock_window_remaining(self, now: Optional[float] = None) -> float:
        if self.state.last_c_initial_baseline_ts <= 0.0:
            return 0.0
        if now is None:
            now = time.time()
        elapsed = max(0.0, now - self.state.last_c_initial_baseline_ts)
        return max(0.0, self.cfg.c_initial_shock_window_secs - elapsed)

    def initial_shock_window_active(self, now: Optional[float] = None) -> bool:
        return self.initial_shock_window_remaining(now) > 0.0

    def is_news_thesis(self, thesis: str) -> bool:
        return thesis in {"earnings", "initial_earnings_shock"}

    def max_entry_stages(self, thesis_cap: int) -> int:
        return max(1, math.ceil(max(0, thesis_cap) / max(1, self.cfg.c_max_order_size)))

    def rates_positions_or_orders_live(self) -> bool:
        if any(self.client.get_position(symbol) != 0 for symbol in self.cfg.rate_symbols):
            return True
        return any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values())

    def leadlag_catalyst_live(self, rate_snapshot: RateSnapshot) -> bool:
        if self.rates_positions_or_orders_live():
            return True
        if rate_snapshot.fresh_macro_event:
            return True
        return abs(rate_snapshot.bias_bp) >= self.cfg.rate_entry_edge_bp

    def refresh_unwind_state(self) -> None:
        pos_abs = abs(self.client.get_position(self.cfg.symbol_c))
        active_news_regime = self.is_news_thesis(self.state.c_regime_thesis or "")
        if not active_news_regime or pos_abs == 0:
            self.state.c_unwind_active = False
            self.state.c_unwind_aggressive_active = False
            return

        if self.state.c_unwind_active:
            if pos_abs <= self.cfg.c_unwind_exit_position:
                self.state.c_unwind_active = False
                self.state.c_unwind_aggressive_active = False
                return
        elif pos_abs >= self.cfg.c_unwind_entry_position:
            self.state.c_unwind_active = True

        if not self.state.c_unwind_active:
            self.state.c_unwind_aggressive_active = False
            return

        if self.state.c_unwind_aggressive_active:
            if pos_abs <= self.cfg.c_unwind_aggressive_exit:
                self.state.c_unwind_aggressive_active = False
            return

        if pos_abs >= self.cfg.c_unwind_aggressive_entry:
            self.state.c_unwind_aggressive_active = True

    def c_mode_name(self, earnings: EarningsContext, now: float) -> str:
        if self.initial_shock_window_active(now):
            return "INITIAL_EARNINGS_SHOCK"
        if self.state.c_unwind_active:
            return "EARNINGS_UNWIND"
        if earnings.tier >= 1 and earnings.age <= self.cfg.c_earnings_fresh_secs:
            return "EARNINGS_ACTIVE"
        return "NORMAL"

    def c_overlay_budgets(self, earnings: EarningsContext, now: float) -> tuple[int, int, bool]:
        mode = self.c_mode_name(earnings, now)
        if mode in {"INITIAL_EARNINGS_SHOCK", "EARNINGS_ACTIVE", "EARNINGS_UNWIND"}:
            return self.cfg.c_event_shift_budget, self.cfg.c_background_shift_budget, True
        return self.cfg.c_event_base_budget, self.cfg.c_background_base_budget, False

    def budgeted_decision(
        self,
        decision: CEntryDecision,
        *,
        event_budget: int,
        background_budget: int,
    ) -> Optional[CEntryDecision]:
        if self.is_news_thesis(decision.thesis):
            cap = min(decision.thesis_cap, event_budget)
        else:
            cap = min(decision.thesis_cap, background_budget)
        if cap <= 0:
            return None
        if cap == decision.thesis_cap:
            return decision
        return CEntryDecision(
            side=decision.side,
            thesis=decision.thesis,
            edge_ticks=decision.edge_ticks,
            initial_size=decision.initial_size,
            add_size=decision.add_size,
            thesis_cap=cap,
        )

    def compute_entry_decision(self, signal: CSignal, rate_snapshot: RateSnapshot) -> Optional[CEntryDecision]:
        if not self.state.have_real_eps_c:
            return None

        now = time.time()
        earnings = self.fair_engine.earnings_context()
        fresh_earnings = earnings.tier >= 1 and not earnings.is_initial and earnings.age <= self.cfg.c_earnings_fresh_secs
        initial_shock_window = self.initial_shock_window_active(now)
        self.refresh_unwind_state()
        event_budget, background_budget, _ = self.c_overlay_budgets(earnings, now)

        rate_tailwind = (
            (signal.gap > 0 and signal.bp_dislocation <= -self.cfg.lead_lag_bp_trigger)
            or (signal.gap < 0 and signal.bp_dislocation >= self.cfg.lead_lag_bp_trigger)
        )

        if fresh_earnings and earnings.side is not None:
            if earnings.side == Side.BUY and signal.gap <= 0:
                return None
            if earnings.side == Side.SELL and signal.gap >= 0:
                return None

            if earnings.tier == 1:
                threshold = max(self.cfg.lead_lag_entry_ticks - 2.0, 1.25 * signal.spread, 14.0)
                if signal.gap_abs >= threshold:
                    return self.budgeted_decision(CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=self.cfg.c_tier1_initial_size,
                        add_size=self.cfg.c_tier1_add_size,
                        thesis_cap=self.cfg.c_tier1_cap,
                    ), event_budget=event_budget, background_budget=background_budget)
            elif earnings.tier == 2:
                initial_size = self.cfg.c_tier2_initial_size
                add_size = self.cfg.c_tier2_add_size
                cap = self.cfg.c_tier2_cap
                if rate_tailwind:
                    initial_size = self.cfg.c_tier3_initial_size
                    add_size = self.cfg.c_tier3_add_size
                    cap = self.cfg.c_tier3_cap
                threshold = max(10.0, signal.entry_threshold)
                if signal.gap_abs >= threshold:
                    return self.budgeted_decision(CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=initial_size,
                        add_size=add_size,
                        thesis_cap=cap,
                    ), event_budget=event_budget, background_budget=background_budget)
            else:
                threshold = max(8.0, signal.entry_threshold)
                if signal.gap_abs >= threshold:
                    return self.budgeted_decision(CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=self.cfg.c_tier3_initial_size,
                        add_size=self.cfg.c_tier3_add_size,
                        thesis_cap=self.cfg.c_tier3_cap,
                    ), event_budget=event_budget, background_budget=background_budget)

        if not fresh_earnings and initial_shock_window:
            threshold = max(self.cfg.c_initial_shock_gap_ticks, 1.25 * signal.spread)
            if signal.gap_abs >= threshold and (
                not self.state.c_initial_shock_consumed or self.state.c_regime_thesis == "initial_earnings_shock"
            ):
                side = Side.BUY if signal.gap > 0 else Side.SELL
                return self.budgeted_decision(CEntryDecision(
                    side=side,
                    thesis="initial_earnings_shock",
                    edge_ticks=signal.gap_abs,
                    initial_size=self.cfg.c_initial_shock_initial_size,
                    add_size=self.cfg.c_initial_shock_add_size,
                    thesis_cap=self.cfg.c_initial_shock_cap,
                ), event_budget=event_budget, background_budget=background_budget)
            return None

        if background_budget <= 0:
            return None

        if (
            not fresh_earnings
            and abs(signal.bp_dislocation) >= self.cfg.lead_lag_bp_trigger
            and signal.gap_abs >= max(self.cfg.lead_lag_entry_ticks, 1.5 * signal.spread)
        ):
            side = Side.BUY if signal.gap > 0 else Side.SELL
            return self.budgeted_decision(CEntryDecision(
                side=side,
                thesis="rates_lead_lag",
                edge_ticks=signal.gap_abs,
                initial_size=self.cfg.c_leadlag_initial_size,
                add_size=self.cfg.c_leadlag_add_size,
                thesis_cap=self.cfg.c_leadlag_cap,
            ), event_budget=event_budget, background_budget=background_budget)

        return None

    async def handle_missing_signal_exit(self) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False

        side = Side.SELL if pos > 0 else Side.BUY
        price = int(book.bid if side == Side.SELL else book.ask)
        qty = min(abs(pos), self.cfg.c_max_order_size)
        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=side,
            price=price,
            role="exit",
            engine="c",
            reason="signal_lost",
            thesis=self.state.c_regime_thesis,
            signal_strength=float(abs(pos)),
        )
        if placed:
            self.state.c_blocked_side = self.state.c_regime_side
            self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
            self.state.clear_c_regime()
            LOGGER.info("C exit trigger: signal_lost")
        return placed

    async def maybe_exit(self, signal: CSignal, rate_snapshot: RateSnapshot) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            self.state.clear_c_regime()
            return False

        now = time.time()
        earnings = self.fair_engine.earnings_context()
        side = None
        reason = None
        qty = 0
        long_pos = pos > 0
        favorable_gap = signal.gap if long_pos else -signal.gap
        adverse_gap = -signal.gap if long_pos else signal.gap
        entry_edge = max(self.state.c_last_entry_edge, signal.entry_threshold)
        compression_band = max(self.cfg.c_compression_min_ticks, self.cfg.c_compression_frac * entry_edge)
        initial_shock_active = self.initial_shock_window_active()
        thesis = self.state.c_regime_thesis or ""
        regime_age = 0.0 if self.state.c_regime_started_at <= 0.0 else max(0.0, now - self.state.c_regime_started_at)
        self.refresh_unwind_state()
        if thesis in {"earnings", "initial_earnings_shock"} and self.state.c_unwind_active:
            unwind_band = entry_edge if self.state.c_unwind_aggressive_active else max(self.cfg.c_initial_shock_compress_ticks, 0.60 * entry_edge)
            compression_band = max(compression_band, unwind_band)

        earnings_opposite = (
            earnings.tier >= 1
            and earnings.side is not None
            and earnings.side != (Side.BUY if long_pos else Side.SELL)
            and earnings.age <= self.cfg.c_earnings_fresh_secs
        )

        if earnings_opposite and earnings.tier >= 2:
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "earnings_reversal"
        elif adverse_gap >= max(self.cfg.c_hard_flip_min_ticks, signal.exit_threshold):
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "gap_flip"
        elif self.state.c_regime_thesis == "initial_earnings_shock":
            compress_threshold = max(self.cfg.c_initial_shock_compress_ticks, signal.exit_threshold)
            if regime_age >= self.cfg.c_initial_shock_max_hold_secs:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "initial_shock_timeout"
            elif not initial_shock_active:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "initial_shock_unwind"
            elif favorable_gap <= compress_threshold:
                side = Side.SELL if long_pos else Side.BUY
                if favorable_gap <= self.cfg.c_initial_shock_flatten_ticks:
                    qty = abs(pos)
                    reason = "initial_shock_compress_full"
                else:
                    qty = min(abs(pos), max(10, abs(pos) // 2))
                    reason = "initial_shock_compress_trim"
        elif thesis == "earnings" and self.state.c_unwind_active and favorable_gap <= compression_band:
            side = Side.SELL if long_pos else Side.BUY
            qty = min(abs(pos), max(20 if self.state.c_unwind_aggressive_active else 10, abs(pos) // 2))
            reason = "earnings_unwind_aggressive" if self.state.c_unwind_aggressive_active else "earnings_unwind_passive"
        elif thesis == "rates_lead_lag":
            if regime_age >= self.cfg.c_leadlag_max_hold_secs:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "leadlag_timeout"
            elif not self.leadlag_catalyst_live(rate_snapshot):
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "leadlag_catalyst_gone"
            else:
                rate_reversal = (
                    (long_pos and rate_snapshot.delta_effective_bp >= self.cfg.c_rate_reversal_bp)
                    or ((not long_pos) and rate_snapshot.delta_effective_bp <= -self.cfg.c_rate_reversal_bp)
                )
                if rate_reversal and favorable_gap <= max(entry_edge * 0.75, compression_band):
                    side = Side.SELL if long_pos else Side.BUY
                    qty = abs(pos)
                    reason = "rate_reversal"
        elif thesis == "post_baseline_gap":
            rate_reversal = (
                (long_pos and rate_snapshot.delta_effective_bp >= self.cfg.c_rate_reversal_bp)
                or ((not long_pos) and rate_snapshot.delta_effective_bp <= -self.cfg.c_rate_reversal_bp)
            )
            if rate_reversal and favorable_gap <= max(entry_edge * 0.75, compression_band):
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "rate_reversal"

        if side is None and favorable_gap <= compression_band:
            side = Side.SELL if long_pos else Side.BUY
            if favorable_gap <= 0:
                qty = abs(pos)
                reason = "gap_compress_full"
            else:
                qty = min(abs(pos), max(10, abs(pos) // 2))
                reason = "gap_compress_trim"

        qty = min(int(qty), self.cfg.c_max_order_size)

        if side is None or qty <= 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False

        await self.orders.cancel_live_orders(symbol=self.cfg.symbol_c, role="entry", engine="c")

        price = int(book.bid if side == Side.SELL else book.ask)
        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=side,
            price=price,
            role="exit",
            engine="c",
            reason=reason,
            thesis=self.state.c_regime_thesis,
            signal_strength=signal.gap_abs,
        )
        if placed:
            if qty >= abs(pos):
                self.state.c_blocked_side = self.state.c_regime_side
                self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
                self.state.clear_c_regime()
            LOGGER.info(
                "C exit trigger: %s pos=%s gap=%.2f delta_effective_bp=%.2f",
                reason,
                pos,
                signal.gap,
                rate_snapshot.delta_effective_bp,
            )
        return placed

    async def maybe_enter(self, decision: Optional[CEntryDecision], signal: Optional[CSignal]) -> bool:
        if decision is None or signal is None:
            return False

        now = time.time()
        pos = self.client.get_position(self.cfg.symbol_c)
        same_direction = (pos > 0 and decision.side == Side.BUY) or (pos < 0 and decision.side == Side.SELL)

        if self.state.c_blocked_side == decision.side and now < self.state.c_blocked_until:
            return False
        if pos == 0 and now < self.state.c_flat_entry_cooldown_until:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False
        if pos != 0 and not same_direction:
            return False
        self.refresh_unwind_state()

        if same_direction:
            if self.state.c_regime_thesis != decision.thesis:
                return False
            if self.is_news_thesis(decision.thesis) and self.state.c_unwind_active:
                return False
            max_stages = self.max_entry_stages(decision.thesis_cap)
            if self.state.c_entry_stage >= max_stages:
                return False
            if now - self.state.c_last_add_ts < self.cfg.c_add_cooldown_secs:
                return False
            if self.is_news_thesis(decision.thesis):
                required_edge = max(signal.entry_threshold, self.state.c_last_entry_edge * self.cfg.c_news_add_edge_frac)
                if decision.edge_ticks < required_edge:
                    return False
            elif decision.edge_ticks < self.state.c_last_entry_edge + self.cfg.c_add_edge_step_ticks:
                return False
            raw_qty = decision.add_size
        else:
            raw_qty = decision.initial_size

        qty = self.risk.clip_c_qty(decision.side, raw_qty, decision.thesis_cap)
        if qty <= 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        price = int(book.ask if decision.side == Side.BUY else book.bid)

        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=decision.side,
            price=price,
            role="entry",
            engine="c",
            reason="add_entry" if same_direction else "initial_entry",
            thesis=decision.thesis,
            signal_strength=decision.edge_ticks,
        )
        if placed:
            self.state.c_regime_side = decision.side
            self.state.c_regime_thesis = decision.thesis
            self.state.c_entry_stage = 1 if not same_direction else self.state.c_entry_stage + 1
            self.state.c_last_entry_edge = decision.edge_ticks
            self.state.c_last_add_ts = now
            if not same_direction:
                self.state.c_regime_started_at = now
            if decision.thesis == "initial_earnings_shock":
                self.state.c_initial_shock_consumed = True
            if pos == 0:
                self.state.c_flat_entry_cooldown_until = now + self.cfg.c_flat_entry_cooldown_secs
            LOGGER.info(
                "C entry: thesis=%s side=%s qty=%s gap=%.2f fair=%.2f mid=%.2f",
                decision.thesis,
                decision.side.name,
                qty,
                signal.gap,
                signal.fair,
                signal.mid,
            )
        return placed


class Coordinator:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        rates_signals: RatesSignalEngine,
        c_fair: CFairValueEngine,
        rates_trading: RatesTradingEngine,
        c_trading: CTradingEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.rates_signals = rates_signals
        self.c_fair = c_fair
        self.rates_trading = rates_trading
        self.c_trading = c_trading

    def sync_regimes_to_positions(self) -> None:
        pos_c = self.client.get_position(self.cfg.symbol_c)
        if pos_c == 0 and not self.orders.has_live_order(symbol=self.cfg.symbol_c):
            self.state.clear_c_regime()
        if all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                self.state.clear_rates_regime()

    async def evaluate(self) -> None:
        self.client.refresh_all_books()
        await self.orders.cancel_stale_orders()
        self.sync_regimes_to_positions()

        if not self.state.startup_flatten_complete:
            await self.risk.startup_flatten_step()
            self.log_status("startup_flatten")
            return

        self.risk.arm_session_baseline_if_ready()

        rate_snapshot = self.rates_signals.snapshot()
        self.c_fair.initialize_anchor(rate_snapshot)
        self.c_fair.maybe_refresh_anchor_after_first_real_eps(rate_snapshot)
        c_signal = self.c_fair.snapshot(rate_snapshot)
        self.c_fair.maybe_reanchor(c_signal, rate_snapshot)

        if rate_snapshot is None:
            rates_acted = await self.rates_trading.handle_missing_signal_exit()
            c_acted = await self.c_trading.handle_missing_signal_exit()
            self.log_status("signal_not_ready" if rates_acted or c_acted else "signal_not_ready")
            return

        if c_signal is None:
            c_acted = await self.c_trading.handle_missing_signal_exit()
            self.log_status("waiting_real_eps" if not self.state.have_real_eps_c else "signal_not_ready")
            self.state.last_market_expected_rate_bp = rate_snapshot.market_expected_rate_bp
            self.state.last_effective_expected_rate_bp = rate_snapshot.effective_expected_rate_bp
            return

        c_entry_decision = self.c_trading.compute_entry_decision(c_signal, rate_snapshot)

        if await self.rates_trading.maybe_exit(rate_snapshot):
            self.update_last_signals(rate_snapshot, c_signal)
            return
        if await self.c_trading.maybe_exit(c_signal, rate_snapshot):
            self.update_last_signals(rate_snapshot, c_signal)
            return

        rates_entered = await self.rates_trading.maybe_enter(rate_snapshot)

        c_is_high_priority = c_entry_decision is not None and c_entry_decision.thesis in {
            "earnings",
            "initial_earnings_shock",
            "post_baseline_gap",
        }
        if (not rates_entered) or c_is_high_priority:
            if await self.c_trading.maybe_enter(c_entry_decision, c_signal):
                self.update_last_signals(rate_snapshot, c_signal)
                return

        self.log_status("no_trade", rate_snapshot, c_signal)
        self.update_last_signals(rate_snapshot, c_signal)

    def update_last_signals(self, rate_snapshot: RateSnapshot, c_signal: CSignal) -> None:
        self.state.last_market_expected_rate_bp = rate_snapshot.market_expected_rate_bp
        self.state.last_effective_expected_rate_bp = rate_snapshot.effective_expected_rate_bp
        self.state.last_fair_c = c_signal.fair

    def log_status(
        self,
        reason: str,
        rate_snapshot: Optional[RateSnapshot] = None,
        c_signal: Optional[CSignal] = None,
    ) -> None:
        now = time.time()
        if now - self.state.last_status_log_ts < self.cfg.status_log_interval_secs:
            return
        self.state.last_status_log_ts = now

        cash, mtm = self.client.cash_and_total_mtm()
        session_cash, session_mtm = self.client.session_pnl_snapshot(cash, mtm)

        if rate_snapshot is None or c_signal is None:
            LOGGER.info(
                "Idle: %s missing=%s pos_C=%s pos_rates=%s/%s/%s eps=%.4f cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
                reason,
                self.client.missing_prereqs(),
                self.client.get_position(self.cfg.symbol_c),
                self.client.get_position(self.cfg.fed_hike),
                self.client.get_position(self.cfg.fed_hold),
                self.client.get_position(self.cfg.fed_cut),
                self.state.current_eps_c,
                cash,
                mtm,
                session_cash,
                session_mtm,
            )
            return

        earnings = self.c_fair.earnings_context()
        LOGGER.info(
            "Idle: %s exp_bp=%.2f market_bp=%.2f bias=%.2f eps=%.4f eps_delta=%+.4f eps_tier=%s fair=%.2f C=%s/%s gap=%.2f bp_disloc=%.2f pos_C=%s rates=%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            reason,
            rate_snapshot.effective_expected_rate_bp,
            rate_snapshot.market_expected_rate_bp,
            rate_snapshot.bias_bp,
            self.state.current_eps_c,
            earnings.delta,
            earnings.tier,
            c_signal.fair,
            c_signal.bid,
            c_signal.ask,
            c_signal.gap,
            c_signal.bp_dislocation,
            self.client.get_position(self.cfg.symbol_c),
            self.client.get_position(self.cfg.fed_hike),
            self.client.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )


class MyXchangeClient(XChangeClient):
    def __init__(self, host: str, username: str, password: str):
        self.cfg = BotConfig()
        super().__init__(host, username, password, silent=True, symbols=list(self.cfg.tracked_symbols))

        self.state = MarketState(current_eps_c=self.cfg.default_eps_c)
        self._decision_lock = asyncio.Lock()

        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)
        self.rates_signal_engine = RatesSignalEngine(self, self.cfg, self.state)
        self.c_fair_engine = CFairValueEngine(self, self.cfg, self.state, self.rates_signal_engine)
        self.rates_trading_engine = RatesTradingEngine(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
        )
        self.c_trading_engine = CTradingEngine(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.c_fair_engine,
        )
        self.coordinator = Coordinator(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.rates_signal_engine,
            self.c_fair_engine,
            self.rates_trading_engine,
            self.c_trading_engine,
        )

    @staticmethod
    def clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def refresh_book(self, symbol: str) -> TopOfBook:
        book = self.order_books.get(symbol)
        bids = []
        asks = []
        if book is not None:
            bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
            asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]

        best_bid = max(bids, key=lambda level: level[0]) if bids else None
        best_ask = min(asks, key=lambda level: level[0]) if asks else None
        snapshot = TopOfBook(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.state.books[symbol] = snapshot
        return snapshot

    def refresh_all_books(self) -> None:
        for symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)

    def top(self, symbol: str) -> TopOfBook:
        if symbol not in self.state.books:
            return self.refresh_book(symbol)
        return self.state.books[symbol]

    def mid(self, symbol: str) -> Optional[float]:
        return self.top(symbol).mid

    def mark_price(self, symbol: str) -> Optional[float]:
        return self.mid(symbol)

    def cash_and_total_mtm(self) -> tuple[float, float]:
        cash = float(self.positions.get("cash", 0))
        mtm = cash
        for symbol in self.cfg.tracked_symbols:
            pos = self.get_position(symbol)
            if pos == 0:
                continue
            mark = self.mark_price(symbol)
            if mark is not None:
                mtm += pos * mark
        return cash, mtm

    def session_pnl_snapshot(self, cash: float, mtm: float) -> tuple[float, float]:
        if self.state.session_start_cash is None or self.state.session_start_mtm is None:
            return 0.0, 0.0
        return cash - self.state.session_start_cash, mtm - self.state.session_start_mtm

    def missing_prereqs(self) -> list[str]:
        missing: list[str] = []
        for symbol in self.cfg.tracked_symbols:
            book = self.top(symbol)
            if book.bid is None or book.ask is None:
                missing.append(f"{symbol}_book")
        if self.state.anchor_price is None or self.state.anchor_eps is None or self.state.anchor_yield_bp is None:
            missing.append("anchor")
        return missing

    async def evaluate(self) -> None:
        async with self._decision_lock:
            await self.coordinator.evaluate()

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        tracked = self.order_manager.sync_cancel_response(order_id, success)
        self.state.pending_cancels.discard(str(order_id))
        if success:
            LOGGER.info("Cancel acknowledged: order_id=%s", order_id)
        else:
            LOGGER.warning("Cancel failed: order_id=%s error=%s", order_id, error)
            if tracked is not None:
                self.state.live_orders[str(order_id)] = tracked

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        tracked = self.order_manager.sync_fill(order_id)
        pos_c = self.get_position(self.cfg.symbol_c)
        if pos_c == 0 and not self.order_manager.has_live_order(symbol=self.cfg.symbol_c):
            self.state.clear_c_regime()

        if self.get_position(self.cfg.fed_hike) == 0 and self.get_position(self.cfg.fed_cut) == 0 and self.get_position(self.cfg.fed_hold) == 0:
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                self.state.clear_rates_regime()

        cash, mtm = self.cash_and_total_mtm()
        session_cash, session_mtm = self.session_pnl_snapshot(cash, mtm)
        LOGGER.info(
            "Fill: order_id=%s symbol=%s qty=%s price=%s pos_C=%s pos_rates=%s/%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            order_id,
            tracked.symbol if tracked is not None else "unknown",
            qty,
            price,
            self.get_position(self.cfg.symbol_c),
            self.get_position(self.cfg.fed_hike),
            self.get_position(self.cfg.fed_hold),
            self.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.state.pending_cancels.discard(str(order_id))
        reason_text = (reason or "").lower()
        limit_rejection = "exceeds limits" in reason_text or "limit" in reason_text
        if tracked is not None:
            if tracked.engine == "c" and tracked.role == "entry":
                self.state.c_blocked_side = tracked.side
                cooldown = self.cfg.c_reentry_block_secs * (2.0 if limit_rejection else 1.0)
                self.state.c_blocked_until = time.time() + cooldown
            if tracked.engine == "rates" and tracked.role == "entry":
                self.state.rates_blocked_direction = tracked.thesis
                cooldown = self.cfg.rate_reentry_block_secs * (2.0 if limit_rejection else 1.0)
                self.state.rates_blocked_until = time.time() + cooldown
        LOGGER.warning("Order rejected: order_id=%s reason=%s", order_id, reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.tracked_symbols:
            self.state.last_trade_price[symbol] = int(price)
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)
            await self.evaluate()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        LOGGER.info("Ignoring swap response: swap=%s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        tick = news_release.get("tick")

        rate_snapshot = self.rates_signal_engine.snapshot()

        if kind == "structured":
            subtype = new_data.get("structured_subtype")

            if subtype == "earnings":
                asset = str(new_data.get("asset", "")).upper()
                value = float(new_data["value"])
                if asset == self.cfg.symbol_c:
                    previous_eps = self.state.current_eps_c
                    had_real_eps = self.state.have_real_eps_c
                    self.state.current_eps_c = value
                    self.state.have_real_eps_c = True
                    self.state.last_c_earnings_delta = 0.0 if not had_real_eps else value - previous_eps
                    self.state.last_c_earnings_ts = time.time()
                    self.state.last_c_earnings_is_initial = not had_real_eps

                    earnings_delta = abs(self.state.last_c_earnings_delta)
                    if not self.state.last_c_earnings_is_initial:
                        if earnings_delta >= self.cfg.c_earnings_medium_delta:
                            self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs + 0.75)
                        elif earnings_delta >= self.cfg.c_earnings_small_delta:
                            self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs)
                        elif earnings_delta >= self.cfg.c_earnings_ignore_delta:
                            self.rates_signal_engine.mark_news_urgent(1.50)
                    else:
                        self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs)

                    self.c_fair_engine.maybe_refresh_anchor_after_first_real_eps(rate_snapshot)
                    LOGGER.info(
                        "C earnings update at tick %s: %.4f -> %.4f delta=%+.4f initial=%s",
                        tick,
                        previous_eps,
                        value,
                        self.state.last_c_earnings_delta,
                        self.state.last_c_earnings_is_initial,
                    )
                else:
                    LOGGER.info("Ignoring %s earnings in C-only mode at tick %s: value=%.4f", asset, tick, value)

            elif subtype == "cpi_print":
                forecast = float(new_data["forecast"])
                actual = float(new_data["actual"])
                surprise = actual - forecast
                bias_bp = self.clip(
                    surprise * self.cfg.cpi_to_rate_bp,
                    -self.cfg.max_temp_rate_bias_bp,
                    self.cfg.max_temp_rate_bias_bp,
                )
                if abs(bias_bp) >= 0.25:
                    self.rates_signal_engine.apply_temp_rate_bias(bias_bp, self.cfg.cpi_bias_ttl_secs, "cpi_print")
                    LOGGER.info(
                        "CPI surprise at tick %s: actual=%.6f forecast=%.6f bias_bp=%.2f",
                        tick,
                        actual,
                        forecast,
                        bias_bp,
                    )

        elif kind == "unstructured":
            content = str(new_data.get("content", ""))
            bias_bp = self.rates_signal_engine.headline_rate_bias_bp(content)
            if abs(bias_bp) >= 0.25:
                self.rates_signal_engine.apply_temp_rate_bias(bias_bp, self.cfg.headline_bias_ttl_secs, "headline")
                LOGGER.info("Fed headline at tick %s: bias_bp=%.2f content=%s", tick, bias_bp, content)

        await self.evaluate()

    async def trade(self):
        await asyncio.sleep(2.0)
        while True:
            try:
                await self.evaluate()
            except Exception as exc:
                LOGGER.exception("trade loop error: %s", exc)
            await asyncio.sleep(self.cfg.loop_sleep_secs)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def run_bot(host: str, username: str, password: str) -> None:
    client = MyXchangeClient(host, username, password)
    await client.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C + Fed event bot.")
    parser.add_argument("--host", default="practice.uchicago.exchange:3333")
    parser.add_argument("--username", default="uiuc")
    parser.add_argument("--password", default="mesa-lynx-octopus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_bot(args.host, args.username, args.password))


if __name__ == "__main__":
    main()
