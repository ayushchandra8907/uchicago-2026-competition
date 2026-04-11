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

    class XChangeClient:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("utcxchangelib is required to run the live bot.")


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("prediction-c-pm-only")


@dataclass(frozen=True)
class BotConfig:
    fed_hike: str = "R_HIKE"
    fed_hold: str = "R_HOLD"
    fed_cut: str = "R_CUT"

    payout_scale: float = 100.0
    cpi_to_rate_bp: float = 4000.0
    max_temp_rate_bias_bp: float = 8.0
    cpi_bias_ttl_secs: float = 2.5
    headline_bias_ttl_secs: float = 2.0

    dead_tail_mid_frac: float = 0.12
    dead_tail_gap_frac: float = 0.18

    rate_hard_position_limit: int = 200
    rate_max_order_size: int = 40
    rate_normal_size: int = 80
    rate_strong_size: int = 120
    rate_extreme_size: int = 160
    rate_cpi_entry_edge_bp: float = 2.5
    rate_headline_entry_edge_bp: float = 1.5
    rate_headline_strong_edge_bp: float = 1.75
    rate_exit_edge_bp: float = 1.0
    rate_add_edge_step_bp: float = 1.5
    rate_add_edge_frac: float = 0.65
    rate_reentry_block_secs: float = 1.0
    add_cooldown_secs: float = 0.35

    max_active_orders_per_symbol: int = 1
    order_stale_secs: float = 0.50
    urgent_order_stale_secs: float = 0.10
    hedge_followup_secs: float = 0.10
    entry_pair_grace_secs: float = 0.25
    repair_aggressive_ticks: int = 1
    cleanup_aggressive_ticks: int = 1
    loop_sleep_secs: float = 0.20
    status_log_interval_secs: float = 2.0

    startup_flatten_chunk_rate: int = 40

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


@dataclass
class TrackedOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    role: str
    reason: str
    thesis: Optional[str]
    signal_strength: float
    event_id: int
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
class RatesEntryDecision:
    direction: str
    edge_bp: float
    target_size: int
    buy_symbol: str
    sell_symbol: str


@dataclass
class MarketState:
    books: Dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: Dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)

    temp_rate_bias_bp: float = 0.0
    temp_rate_bias_started_at: float = 0.0
    temp_rate_bias_expires_at: float = 0.0
    last_macro_event_ts: float = 0.0
    last_macro_source: Optional[str] = None
    last_macro_bias_bp: float = 0.0
    last_macro_event_id: int = 0
    news_urgency_until: float = 0.0

    last_market_expected_rate_bp: Optional[float] = None
    last_effective_expected_rate_bp: Optional[float] = None

    startup_flatten_complete: bool = False
    session_start_cash: Optional[float] = None
    session_start_mtm: Optional[float] = None
    last_status_log_ts: float = 0.0

    rates_regime_direction: Optional[str] = None
    rates_buy_symbol: Optional[str] = None
    rates_sell_symbol: Optional[str] = None
    rates_active_event_id: int = 0
    rates_last_closed_event_id: int = 0
    rates_entry_stage: int = 0
    rates_last_entry_edge: float = 0.0
    rates_last_add_ts: float = 0.0
    rates_blocked_direction: Optional[str] = None
    rates_blocked_until: float = 0.0
    rates_unwind_active: bool = False
    rates_pairing_until: float = 0.0

    def clear_rates_regime(self) -> None:
        self.rates_regime_direction = None
        self.rates_buy_symbol = None
        self.rates_sell_symbol = None
        self.rates_active_event_id = 0
        self.rates_entry_stage = 0
        self.rates_last_entry_edge = 0.0
        self.rates_last_add_ts = 0.0
        self.rates_unwind_active = False
        self.rates_pairing_until = 0.0


class OrderManager:
    def __init__(self, client: "PredictionCPMOnlyClient", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def has_live_order(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> bool:
        for order in self.state.live_orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if side is not None and order.side != side:
                continue
            return True
        return False

    def pending_qty(self, symbol: str, side: Side) -> int:
        return sum(order.qty for order in self.state.live_orders.values() if order.symbol == symbol and order.side == side)

    async def cancel_order_if_present(self, order_id: str) -> None:
        order_key = str(order_id)
        if order_key in self.state.pending_cancels:
            return
        self.state.pending_cancels.add(order_key)
        try:
            await self.client.cancel_order(order_id)
        finally:
            self.state.pending_cancels.discard(order_key)

    async def cancel_stale_orders(self) -> None:
        now = time.time()
        stale_secs = self.cfg.urgent_order_stale_secs if now < self.state.news_urgency_until else self.cfg.order_stale_secs
        stale = [order_id for order_id, order in self.state.live_orders.items() if now - order.created_at >= stale_secs]
        for order_id in stale:
            await self.cancel_order_if_present(order_id)

    async def cancel_counterpart_after_fill(self, tracked: Optional[TrackedOrder]) -> None:
        if tracked is None or tracked.role not in {"entry", "exit"}:
            return
        now = time.time()
        for other in list(self.state.live_orders.values()):
            if other.order_id == tracked.order_id:
                continue
            if other.role != tracked.role or other.event_id != tracked.event_id:
                continue
            if tracked.role == "entry" and other.thesis != tracked.thesis:
                continue
            if now - other.created_at < self.cfg.hedge_followup_secs:
                continue
            await self.cancel_order_if_present(other.order_id)

    async def place_tracked_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: Side,
        price: int,
        role: str,
        reason: str,
        thesis: Optional[str],
        signal_strength: float,
        event_id: int,
    ) -> bool:
        if qty <= 0 or self.has_live_order(symbol=symbol):
            return False
        same_symbol_orders = [order for order in self.state.live_orders.values() if order.symbol == symbol]
        if len(same_symbol_orders) >= self.cfg.max_active_orders_per_symbol:
            await self.cancel_order_if_present(same_symbol_orders[0].order_id)
            return False
        order_id = await self.client.place_order(symbol, int(qty), side, int(price))
        if order_id is None:
            return False
        self.state.live_orders[str(order_id)] = TrackedOrder(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            qty=int(qty),
            price=int(price),
            role=role,
            reason=reason,
            thesis=thesis,
            signal_strength=float(signal_strength),
            event_id=int(event_id),
        )
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
    def __init__(self, client: "PredictionCPMOnlyClient", cfg: BotConfig, state: MarketState, orders: OrderManager):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders

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
        for symbol in self.cfg.rate_symbols:
            if self.client.get_position(symbol) != 0:
                return False
        if self.client.open_orders or self.state.live_orders:
            return False
        cash, mtm = self.client.cash_and_total_mtm()
        self.state.session_start_cash = cash
        self.state.session_start_mtm = mtm
        LOGGER.info("Session baseline armed: cash=%.2f mtm=%.2f session_pnl=0.00", cash, mtm)
        return True

    async def startup_flatten_step(self) -> bool:
        inherited_order_ids = [str(order_id) for order_id in self.client.open_orders.keys() if str(order_id) not in self.state.live_orders]
        for order_id in inherited_order_ids:
            await self.orders.cancel_order_if_present(order_id)
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=symbol):
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.startup_flatten_chunk_rate, self.cfg.rate_max_order_size)
            price = int(book.bid if side == Side.SELL else book.ask)
            await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="flatten",
                reason="startup_flatten",
                thesis=None,
                signal_strength=float(abs(pos)),
                event_id=0,
            )
        all_flat = all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols)
        no_orders = not self.client.open_orders and not self.state.live_orders
        if all_flat and no_orders:
            self.state.startup_flatten_complete = True
            self.arm_session_baseline_if_ready()
            LOGGER.info("Startup flatten complete; prediction-market book is flat.")
            return True
        return False


class RatesSignalEngine:
    def __init__(self, client: "PredictionCPMOnlyClient", cfg: BotConfig, state: MarketState):
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
        self.state.last_macro_event_id += 1
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
        delta_market_bp = 0.0 if self.state.last_market_expected_rate_bp is None else market_expected_rate_bp - self.state.last_market_expected_rate_bp
        delta_effective_bp = 0.0 if self.state.last_effective_expected_rate_bp is None else effective_expected_rate_bp - self.state.last_effective_expected_rate_bp
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


class RatesTradingEngine:
    def __init__(
        self,
        client: "PredictionCPMOnlyClient",
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
        if self.state.rates_buy_symbol is None or self.state.rates_sell_symbol is None:
            return None
        buy_pos = self.client.get_position(self.state.rates_buy_symbol)
        sell_pos = self.client.get_position(self.state.rates_sell_symbol)
        if buy_pos > 0 and sell_pos < 0:
            return self.state.rates_regime_direction
        return None

    def any_rate_inventory(self) -> bool:
        return any(self.client.get_position(symbol) != 0 for symbol in self.cfg.rate_symbols)

    def current_pair_abs(self) -> int:
        if self.state.rates_buy_symbol is None or self.state.rates_sell_symbol is None:
            return 0
        return max(abs(self.client.get_position(self.state.rates_buy_symbol)), abs(self.client.get_position(self.state.rates_sell_symbol)))

    def has_orphaned_inventory(self) -> bool:
        if not self.any_rate_inventory():
            return False
        if self.state.rates_buy_symbol is None or self.state.rates_sell_symbol is None:
            return True
        buy_pos = self.client.get_position(self.state.rates_buy_symbol)
        sell_pos = self.client.get_position(self.state.rates_sell_symbol)
        other_symbols = set(self.cfg.rate_symbols) - {self.state.rates_buy_symbol, self.state.rates_sell_symbol}
        if any(self.client.get_position(symbol) != 0 for symbol in other_symbols):
            return True
        if buy_pos == 0 and sell_pos == 0:
            return False
        if buy_pos <= 0 or sell_pos >= 0:
            return True
        return abs(buy_pos) != abs(sell_pos)

    def marketable_price(self, book: TopOfBook, side: Side, aggressive_ticks: int = 0) -> Optional[int]:
        if side == Side.BUY:
            if book.ask is None:
                return None
            return int(book.ask + max(0, aggressive_ticks))
        if book.bid is None:
            return None
        return int(max(0, book.bid - max(0, aggressive_ticks)))

    def max_entry_stages(self, target_size: int) -> int:
        return max(1, math.ceil(max(0, target_size) / max(1, self.cfg.rate_max_order_size)))

    def select_pair(self, direction: str) -> tuple[str, str]:
        mids = {}
        for symbol in self.cfg.rate_symbols:
            mid = self.client.mid(symbol)
            mids[symbol] = -1.0 if mid is None else float(mid)
        ranked = sorted(mids.items(), key=lambda item: item[1], reverse=True)
        winner, challenger, dead = ranked[0][0], ranked[1][0], ranked[2][0]
        dead_mid = ranked[2][1]
        challenger_mid = ranked[1][1]
        dead_tail = dead_mid <= self.cfg.payout_scale * self.cfg.dead_tail_mid_frac or (challenger_mid - dead_mid) >= self.cfg.payout_scale * self.cfg.dead_tail_gap_frac
        if direction == "hawkish":
            if dead_tail and dead == self.cfg.fed_cut:
                return self.cfg.fed_hike, self.cfg.fed_hold
            return self.cfg.fed_hike, self.cfg.fed_cut
        if dead_tail and dead == self.cfg.fed_hike:
            return self.cfg.fed_cut, self.cfg.fed_hold
        return self.cfg.fed_cut, self.cfg.fed_hike

    async def flatten_all_rates(self, reason: str) -> bool:
        acted = False
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=symbol):
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.rate_max_order_size)
            price = self.marketable_price(book, side, self.cfg.cleanup_aggressive_ticks)
            if price is None:
                continue
            placed = await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="exit",
                reason=reason,
                thesis=self.state.rates_regime_direction,
                signal_strength=float(abs(pos)),
                event_id=self.state.rates_active_event_id,
            )
            acted = acted or placed
        if acted:
            self.state.rates_blocked_direction = self.state.rates_regime_direction
            self.state.rates_blocked_until = max(
                self.state.temp_rate_bias_expires_at if reason == "orphan_cleanup" else 0.0,
                time.time() + self.cfg.rate_reentry_block_secs,
            )
            self.state.rates_unwind_active = True
            self.state.rates_pairing_until = 0.0
            LOGGER.info("Rates exit trigger: %s", reason)
        return acted

    async def handle_missing_signal_exit(self) -> bool:
        if self.any_rate_inventory():
            return await self.flatten_all_rates("signal_lost")
        return False

    async def maybe_exit(self, snapshot: RateSnapshot) -> bool:
        direction = self.current_pair_direction()
        if direction is None:
            if self.any_rate_inventory():
                if time.time() < self.state.rates_pairing_until or self.orders.has_live_order(role="entry"):
                    return False
                return await self.flatten_all_rates("orphan_cleanup")
            return False
        edge_bp = abs(snapshot.bias_bp)
        compressed = edge_bp <= max(self.cfg.rate_exit_edge_bp, self.state.rates_last_entry_edge * 0.30)
        opposite = (direction == "hawkish" and snapshot.bias_bp <= -self.cfg.rate_exit_edge_bp) or (direction == "dovish" and snapshot.bias_bp >= self.cfg.rate_exit_edge_bp)
        stale = edge_bp < self.cfg.rate_exit_edge_bp
        if stale or opposite or compressed:
            return await self.flatten_all_rates("bias_decay" if stale or compressed else "macro_reversal")
        return False

    def compute_entry_decision(self, snapshot: RateSnapshot) -> Optional[RatesEntryDecision]:
        if not snapshot.fresh_macro_event:
            return None
        source = snapshot.macro_source or ""
        entry_edge = self.cfg.rate_cpi_entry_edge_bp if source == "cpi_print" else self.cfg.rate_headline_entry_edge_bp
        if snapshot.bias_bp >= entry_edge:
            direction = "hawkish"
        elif snapshot.bias_bp <= -entry_edge:
            direction = "dovish"
        else:
            return None
        edge_bp = abs(snapshot.bias_bp)
        if source == "cpi_print" and edge_bp >= 5.0:
            target_size = self.cfg.rate_extreme_size
        elif source == "cpi_print" and edge_bp >= 3.5:
            target_size = self.cfg.rate_strong_size
        elif source == "headline" and edge_bp >= self.cfg.rate_headline_strong_edge_bp:
            target_size = self.cfg.rate_strong_size
        else:
            target_size = self.cfg.rate_normal_size
        buy_symbol, sell_symbol = self.select_pair(direction)
        return RatesEntryDecision(direction=direction, edge_bp=edge_bp, target_size=target_size, buy_symbol=buy_symbol, sell_symbol=sell_symbol)

    async def maybe_enter(self, snapshot: RateSnapshot) -> bool:
        decision = self.compute_entry_decision(snapshot)
        if decision is None:
            return False
        now = time.time()
        if self.state.rates_unwind_active:
            return False
        if self.state.rates_last_closed_event_id == self.state.last_macro_event_id and not self.any_rate_inventory():
            return False
        if self.state.rates_blocked_direction == decision.direction and now < self.state.rates_blocked_until:
            return False
        if self.orders.has_live_order(role="exit"):
            return False

        current_direction = self.current_pair_direction()
        same_regime = (
            self.state.rates_regime_direction == decision.direction
            and self.state.rates_buy_symbol == decision.buy_symbol
            and self.state.rates_sell_symbol == decision.sell_symbol
            and self.any_rate_inventory()
        )

        if self.has_orphaned_inventory() and not same_regime:
            if now < self.state.rates_pairing_until or self.orders.has_live_order(role="entry"):
                return False
            return False

        if current_direction is not None and not same_regime:
            return False

        if same_regime:
            if self.state.rates_entry_stage >= self.max_entry_stages(decision.target_size):
                return False
            if decision.edge_bp < self.state.rates_last_entry_edge + self.cfg.rate_add_edge_step_bp:
                required_edge = max(
                    self.cfg.rate_cpi_entry_edge_bp if snapshot.macro_source == "cpi_print" else self.cfg.rate_headline_entry_edge_bp,
                    self.state.rates_last_entry_edge * self.cfg.rate_add_edge_frac,
                )
                if decision.edge_bp < required_edge:
                    return False
            if now - self.state.rates_last_add_ts < self.cfg.add_cooldown_secs:
                return False

        buy_book = self.client.top(decision.buy_symbol)
        sell_book = self.client.top(decision.sell_symbol)
        if buy_book.ask is None or sell_book.bid is None:
            return False

        buy_pos = self.client.get_position(decision.buy_symbol)
        sell_pos = self.client.get_position(decision.sell_symbol)
        buy_filled_abs = max(0, buy_pos)
        sell_filled_abs = max(0, -sell_pos)
        if buy_filled_abs >= decision.target_size and sell_filled_abs >= decision.target_size:
            return False

        buy_needed = max(0, decision.target_size - buy_filled_abs)
        sell_needed = max(0, decision.target_size - sell_filled_abs)
        if same_regime and buy_filled_abs != sell_filled_abs:
            if buy_filled_abs < sell_filled_abs:
                repair_qty = min(buy_needed, sell_filled_abs - buy_filled_abs)
                buy_qty = self.risk.clip_rate_qty(decision.buy_symbol, Side.BUY, repair_qty, decision.target_size)
                sell_qty = 0
            else:
                repair_qty = min(sell_needed, buy_filled_abs - sell_filled_abs)
                buy_qty = 0
                sell_qty = self.risk.clip_rate_qty(decision.sell_symbol, Side.SELL, repair_qty, decision.target_size)
        else:
            buy_qty = self.risk.clip_rate_qty(decision.buy_symbol, Side.BUY, buy_needed, decision.target_size)
            sell_qty = self.risk.clip_rate_qty(decision.sell_symbol, Side.SELL, sell_needed, decision.target_size)
            paired_qty = min(buy_qty, sell_qty)
            buy_qty = paired_qty
            sell_qty = paired_qty

        if buy_qty <= 0 and sell_qty <= 0:
            return False
        if (buy_qty > 0 and self.orders.has_live_order(symbol=decision.buy_symbol)) or (sell_qty > 0 and self.orders.has_live_order(symbol=decision.sell_symbol)):
            return False

        buy_price = self.marketable_price(buy_book, Side.BUY, self.cfg.repair_aggressive_ticks if same_regime and buy_qty > 0 and sell_qty == 0 else 0)
        sell_price = self.marketable_price(sell_book, Side.SELL, self.cfg.repair_aggressive_ticks if same_regime and sell_qty > 0 and buy_qty == 0 else 0)
        if (buy_qty > 0 and buy_price is None) or (sell_qty > 0 and sell_price is None):
            return False

        placed_buy = False
        placed_sell = False
        if buy_qty > 0:
            placed_buy = await self.orders.place_tracked_order(
                symbol=decision.buy_symbol,
                qty=buy_qty,
                side=Side.BUY,
                price=buy_price,
                role="entry",
                reason="rates_entry",
                thesis=decision.direction,
                signal_strength=decision.edge_bp,
                event_id=self.state.last_macro_event_id,
            )
        if sell_qty > 0:
            placed_sell = await self.orders.place_tracked_order(
                symbol=decision.sell_symbol,
                qty=sell_qty,
                side=Side.SELL,
                price=sell_price,
                role="entry",
                reason="rates_entry",
                thesis=decision.direction,
                signal_strength=decision.edge_bp,
                event_id=self.state.last_macro_event_id,
            )
        placed = placed_buy or placed_sell
        if placed:
            self.state.rates_regime_direction = decision.direction
            self.state.rates_buy_symbol = decision.buy_symbol
            self.state.rates_sell_symbol = decision.sell_symbol
            self.state.rates_active_event_id = self.state.last_macro_event_id
            self.state.rates_entry_stage = 1 if not same_regime else self.state.rates_entry_stage + 1
            self.state.rates_last_entry_edge = decision.edge_bp
            self.state.rates_last_add_ts = now
            self.state.rates_pairing_until = now + self.cfg.entry_pair_grace_secs
            LOGGER.info(
                "Rates entry: direction=%s pair=%s/%s edge_bp=%.2f target=%s buy_qty=%s sell_qty=%s",
                decision.direction,
                decision.buy_symbol,
                decision.sell_symbol,
                decision.edge_bp,
                decision.target_size,
                buy_qty,
                sell_qty,
            )
        return placed


class Coordinator:
    def __init__(
        self,
        client: "PredictionCPMOnlyClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        rates_signals: RatesSignalEngine,
        rates_trading: RatesTradingEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.rates_signals = rates_signals
        self.rates_trading = rates_trading

    def sync_regimes_to_positions(self) -> None:
        if all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
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

        snapshot = self.rates_signals.snapshot()
        if snapshot is None:
            acted = await self.rates_trading.handle_missing_signal_exit()
            self.log_status("signal_not_ready" if acted else "signal_not_ready")
            return

        if await self.rates_trading.maybe_exit(snapshot):
            self.update_last_signals(snapshot)
            return

        if await self.rates_trading.maybe_enter(snapshot):
            self.update_last_signals(snapshot)
            return

        self.log_status("no_trade", snapshot)
        self.update_last_signals(snapshot)

    def update_last_signals(self, snapshot: RateSnapshot) -> None:
        self.state.last_market_expected_rate_bp = snapshot.market_expected_rate_bp
        self.state.last_effective_expected_rate_bp = snapshot.effective_expected_rate_bp

    def log_status(self, reason: str, snapshot: Optional[RateSnapshot] = None) -> None:
        now = time.time()
        if now - self.state.last_status_log_ts < self.cfg.status_log_interval_secs:
            return
        self.state.last_status_log_ts = now
        cash, mtm = self.client.cash_and_total_mtm()
        session_cash, session_mtm = self.client.session_pnl_snapshot(cash, mtm)
        if snapshot is None:
            LOGGER.info(
                "Idle: %s pos_rates=%s/%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
                reason,
                self.client.get_position(self.cfg.fed_hike),
                self.client.get_position(self.cfg.fed_hold),
                self.client.get_position(self.cfg.fed_cut),
                cash,
                mtm,
                session_cash,
                session_mtm,
            )
            return
        LOGGER.info(
            "Idle: %s exp_bp=%.2f market_bp=%.2f bias=%.2f q=(%.3f, %.3f, %.3f) pos=%s/%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            reason,
            snapshot.effective_expected_rate_bp,
            snapshot.market_expected_rate_bp,
            snapshot.bias_bp,
            snapshot.q_hike,
            snapshot.q_hold,
            snapshot.q_cut,
            self.client.get_position(self.cfg.fed_hike),
            self.client.get_position(self.cfg.fed_hold),
            self.client.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )


class PredictionCPMOnlyClient(XChangeClient):
    def __init__(self, host: str, username: str, password: str):
        self.cfg = BotConfig()
        super().__init__(host, username, password, silent=True, symbols=list(self.cfg.rate_symbols))
        self.state = MarketState()
        self._decision_lock = asyncio.Lock()
        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)
        self.rates_signal_engine = RatesSignalEngine(self, self.cfg, self.state)
        self.rates_trading_engine = RatesTradingEngine(self, self.cfg, self.state, self.order_manager, self.risk_manager)
        self.coordinator = Coordinator(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.rates_signal_engine,
            self.rates_trading_engine,
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
        for symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)

    def top(self, symbol: str) -> TopOfBook:
        if symbol not in self.state.books:
            return self.refresh_book(symbol)
        return self.state.books[symbol]

    def mid(self, symbol: str) -> Optional[float]:
        return self.top(symbol).mid

    def cash_and_total_mtm(self) -> tuple[float, float]:
        cash = float(self.positions.get("cash", 0))
        mtm = cash
        for symbol in self.cfg.rate_symbols:
            pos = self.get_position(symbol)
            if pos == 0:
                continue
            mark = self.mid(symbol)
            if mark is not None:
                mtm += pos * mark
        return cash, mtm

    def session_pnl_snapshot(self, cash: float, mtm: float) -> tuple[float, float]:
        if self.state.session_start_cash is None or self.state.session_start_mtm is None:
            return 0.0, 0.0
        return cash - self.state.session_start_cash, mtm - self.state.session_start_mtm

    async def evaluate(self) -> None:
        async with self._decision_lock:
            await self.coordinator.evaluate()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
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
        if all(self.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
                self.state.clear_rates_regime()
        cash, mtm = self.cash_and_total_mtm()
        session_cash, session_mtm = self.session_pnl_snapshot(cash, mtm)
        LOGGER.info(
            "Fill: order_id=%s symbol=%s qty=%s price=%s pos_rates=%s/%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            order_id,
            tracked.symbol if tracked is not None else "unknown",
            qty,
            price,
            self.get_position(self.cfg.fed_hike),
            self.get_position(self.cfg.fed_hold),
            self.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )
        await self.order_manager.cancel_counterpart_after_fill(tracked)
        await self.evaluate()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.state.pending_cancels.discard(str(order_id))
        reason_text = (reason or "").lower()
        limit_rejection = "exceeds limits" in reason_text or "limit" in reason_text
        if tracked is not None and tracked.role == "entry":
            self.state.rates_blocked_direction = tracked.thesis
            cooldown = self.cfg.rate_reentry_block_secs * (2.0 if limit_rejection else 1.0)
            self.state.rates_blocked_until = time.time() + cooldown
        LOGGER.warning("Order rejected: order_id=%s reason=%s", order_id, reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.rate_symbols:
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            await self.evaluate()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        LOGGER.info("Ignoring swap response: swap=%s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        tick = news_release.get("tick")
        if kind == "structured":
            subtype = new_data.get("structured_subtype")
            if subtype == "cpi_print":
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
    client = PredictionCPMOnlyClient(host, username, password)
    await client.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction-market-only rates bot extracted from prediction_C.py.")
    parser.add_argument("--host", default="practice.uchicago.exchange:3333")
    parser.add_argument("--username", default="uiuc")
    parser.add_argument("--password", default="mesa-lynx-octopus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_bot(args.host, args.username, args.password))


if __name__ == "__main__":
    main()
