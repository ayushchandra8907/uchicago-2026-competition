from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case1.caleb_work import new_PM1 as pm1


Side = pm1.Side
TopOfBook = pm1.TopOfBook
TrackedOrder = pm1.TrackedOrder
RateSnapshot = pm1.RateSnapshot
OrderManager = pm1.OrderManager


@dataclass(frozen=True)
class BotConfig(pm1.BotConfig):
    symbol_c: str
    default_eps_c: float

    c_y0: float
    c_pe0: float
    c_b0_per_share: float
    c_duration: float
    c_convexity: float
    c_lambda: float
    c_beta_y: float
    c_gamma: float

    c_formula_residual_alpha: float
    c_formula_reanchor_gap_ticks: float
    c_formula_reanchor_secs: float

    c_hard_position_limit: int
    c_max_order_size: int
    c_entry_base_ticks: float
    c_exit_base_ticks: float
    c_add_edge_step_ticks: float
    c_compression_frac: float
    c_hard_flip_ticks: float
    c_rate_reversal_bp: float

    c_reentry_block_secs: float
    c_flat_entry_cooldown_secs: float
    c_add_cooldown_secs: float

    c_earnings_ignore_delta: float
    c_earnings_small_delta: float
    c_earnings_medium_delta: float
    c_earnings_hold_secs: float
    c_earnings_max_hold_secs: float
    c_rates_max_hold_secs: float
    c_background_max_hold_secs: float

    c_rates_trigger_bias_bp: float
    c_rates_gap_ticks: float
    c_background_gap_ticks: float

    c_tier1_initial_size: int
    c_tier1_add_size: int
    c_tier1_cap: int
    c_tier2_initial_size: int
    c_tier2_add_size: int
    c_tier2_cap: int
    c_tier3_initial_size: int
    c_tier3_add_size: int
    c_tier3_cap: int
    c_rates_initial_size: int
    c_rates_add_size: int
    c_rates_cap: int
    c_background_initial_size: int
    c_background_add_size: int
    c_background_cap: int

    startup_flatten_chunk_c: int

    @property
    def tracked_symbols(self) -> tuple[str, str, str, str]:
        return (self.symbol_c, self.fed_hike, self.fed_hold, self.fed_cut)


def load_config() -> BotConfig:
    data = dict(vars(pm1.load_config()))
    data.update(
        fed_hike=pm1.env_str("C1_FED_HIKE_SYMBOL", data["fed_hike"]),
        fed_hold=pm1.env_str("C1_FED_HOLD_SYMBOL", data["fed_hold"]),
        fed_cut=pm1.env_str("C1_FED_CUT_SYMBOL", data["fed_cut"]),
        payout_scale=pm1.env_float("C1_PAYOUT_SCALE", data["payout_scale"]),
        cpi_to_rate_bp=pm1.env_float("C1_CPI_TO_RATE_BP", data["cpi_to_rate_bp"]),
        max_temp_rate_bias_bp=pm1.env_float("C1_MAX_TEMP_RATE_BIAS_BP", data["max_temp_rate_bias_bp"]),
        cpi_bias_ttl_secs=pm1.env_float("C1_CPI_BIAS_TTL_SECS", data["cpi_bias_ttl_secs"]),
        headline_bias_ttl_secs=pm1.env_float("C1_HEADLINE_BIAS_TTL_SECS", data["headline_bias_ttl_secs"]),
        headline_apply_min_abs_bias_bp=pm1.env_float(
            "C1_HEADLINE_MIN_ABS_BIAS_BP",
            data["headline_apply_min_abs_bias_bp"],
        ),
        dead_tail_mid_frac=pm1.env_float("C1_DEAD_TAIL_MID_FRAC", data["dead_tail_mid_frac"]),
        dead_tail_gap_frac=pm1.env_float("C1_DEAD_TAIL_GAP_FRAC", data["dead_tail_gap_frac"]),
        rate_hard_position_limit=pm1.env_int("C1_RATE_HARD_POSITION_LIMIT", data["rate_hard_position_limit"]),
        rate_max_order_size=pm1.env_int("C1_RATE_MAX_ORDER_SIZE", data["rate_max_order_size"]),
        rate_normal_size=pm1.env_int("C1_RATE_NORMAL_SIZE", data["rate_normal_size"]),
        rate_strong_size=pm1.env_int("C1_RATE_STRONG_SIZE", data["rate_strong_size"]),
        rate_extreme_size=pm1.env_int("C1_RATE_EXTREME_SIZE", data["rate_extreme_size"]),
        rate_cpi_entry_edge_bp=pm1.env_float("C1_RATE_CPI_ENTRY_EDGE_BP", data["rate_cpi_entry_edge_bp"]),
        rate_headline_entry_edge_bp=pm1.env_float(
            "C1_RATE_HEADLINE_ENTRY_EDGE_BP",
            data["rate_headline_entry_edge_bp"],
        ),
        rate_headline_strong_edge_bp=pm1.env_float(
            "C1_RATE_HEADLINE_STRONG_EDGE_BP",
            data["rate_headline_strong_edge_bp"],
        ),
        rate_exit_edge_bp=pm1.env_float("C1_RATE_EXIT_EDGE_BP", data["rate_exit_edge_bp"]),
        rate_add_edge_step_bp=pm1.env_float("C1_RATE_ADD_EDGE_STEP_BP", data["rate_add_edge_step_bp"]),
        rate_add_edge_frac=pm1.env_float("C1_RATE_ADD_EDGE_FRAC", data["rate_add_edge_frac"]),
        rate_reentry_block_secs=pm1.env_float("C1_RATE_REENTRY_BLOCK_SECS", data["rate_reentry_block_secs"]),
        add_cooldown_secs=pm1.env_float("C1_ADD_COOLDOWN_SECS", data["add_cooldown_secs"]),
        max_active_orders_per_symbol=pm1.env_int(
            "C1_MAX_ACTIVE_ORDERS_PER_SYMBOL",
            data["max_active_orders_per_symbol"],
        ),
        order_stale_secs=pm1.env_float("C1_ORDER_STALE_SECS", data["order_stale_secs"]),
        urgent_order_stale_secs=pm1.env_float("C1_URGENT_ORDER_STALE_SECS", data["urgent_order_stale_secs"]),
        hedge_followup_secs=pm1.env_float("C1_HEDGE_FOLLOWUP_SECS", data["hedge_followup_secs"]),
        entry_pair_grace_secs=pm1.env_float("C1_ENTRY_PAIR_GRACE_SECS", data["entry_pair_grace_secs"]),
        repair_aggressive_ticks=pm1.env_int("C1_REPAIR_AGGRESSIVE_TICKS", data["repair_aggressive_ticks"]),
        cleanup_aggressive_ticks=pm1.env_int("C1_CLEANUP_AGGRESSIVE_TICKS", data["cleanup_aggressive_ticks"]),
        loop_sleep_secs=pm1.env_float("C1_LOOP_SLEEP_SECS", data["loop_sleep_secs"]),
        status_log_interval_secs=pm1.env_float(
            "C1_STATUS_LOG_INTERVAL_SECS",
            data["status_log_interval_secs"],
        ),
        startup_flatten_chunk_rate=pm1.env_int(
            "C1_STARTUP_FLATTEN_CHUNK_RATE",
            data["startup_flatten_chunk_rate"],
        ),
        trace_enabled=pm1.env_bool("C1_TRACE_ENABLED", data["trace_enabled"]),
        trace_dir=pm1.env_str("C1_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
        symbol_c=pm1.env_str("C1_SYMBOL_C", "C"),
        default_eps_c=pm1.env_float("C1_DEFAULT_EPS_C", 2.0),
        c_y0=pm1.env_float("C1_Y0", 0.045),
        c_pe0=pm1.env_float("C1_PE0", 14.0),
        c_b0_per_share=pm1.env_float("C1_B0_PER_SHARE", 40.0),
        c_duration=pm1.env_float("C1_DURATION", 7.5),
        c_convexity=pm1.env_float("C1_CONVEXITY", 55.0),
        c_lambda=pm1.env_float("C1_LAMBDA", 0.65),
        c_beta_y=pm1.env_float("C1_BETA_Y", 0.00010),
        c_gamma=pm1.env_float("C1_GAMMA", 13.0),
        c_formula_residual_alpha=pm1.env_float("C1_FORMULA_RESIDUAL_ALPHA", 0.12),
        c_formula_reanchor_gap_ticks=pm1.env_float("C1_FORMULA_REANCHOR_GAP_TICKS", 5.0),
        c_formula_reanchor_secs=pm1.env_float("C1_FORMULA_REANCHOR_SECS", 2.0),
        c_hard_position_limit=pm1.env_int("C1_HARD_POSITION_LIMIT", 200),
        c_max_order_size=pm1.env_int("C1_MAX_ORDER_SIZE", 40),
        c_entry_base_ticks=pm1.env_float("C1_ENTRY_BASE_TICKS", 10.0),
        c_exit_base_ticks=pm1.env_float("C1_EXIT_BASE_TICKS", 6.0),
        c_add_edge_step_ticks=pm1.env_float("C1_ADD_EDGE_STEP_TICKS", 5.0),
        c_compression_frac=pm1.env_float("C1_COMPRESSION_FRAC", 0.35),
        c_hard_flip_ticks=pm1.env_float("C1_HARD_FLIP_TICKS", 8.0),
        c_rate_reversal_bp=pm1.env_float("C1_RATE_REVERSAL_BP", 1.75),
        c_reentry_block_secs=pm1.env_float("C1_REENTRY_BLOCK_SECS", 1.25),
        c_flat_entry_cooldown_secs=pm1.env_float("C1_FLAT_ENTRY_COOLDOWN_SECS", 0.60),
        c_add_cooldown_secs=pm1.env_float("C1_ADD_COOLDOWN_SECS", 0.35),
        c_earnings_ignore_delta=pm1.env_float("C1_EARNINGS_IGNORE_DELTA", 0.010),
        c_earnings_small_delta=pm1.env_float("C1_EARNINGS_SMALL_DELTA", 0.025),
        c_earnings_medium_delta=pm1.env_float("C1_EARNINGS_MEDIUM_DELTA", 0.045),
        c_earnings_hold_secs=pm1.env_float("C1_EARNINGS_HOLD_SECS", 2.75),
        c_earnings_max_hold_secs=pm1.env_float("C1_EARNINGS_MAX_HOLD_SECS", 12.0),
        c_rates_max_hold_secs=pm1.env_float("C1_RATES_MAX_HOLD_SECS", 18.0),
        c_background_max_hold_secs=pm1.env_float("C1_BACKGROUND_MAX_HOLD_SECS", 25.0),
        c_rates_trigger_bias_bp=pm1.env_float("C1_RATES_TRIGGER_BIAS_BP", 1.50),
        c_rates_gap_ticks=pm1.env_float("C1_RATES_GAP_TICKS", 12.0),
        c_background_gap_ticks=pm1.env_float("C1_BACKGROUND_GAP_TICKS", 16.0),
        c_tier1_initial_size=pm1.env_int("C1_TIER1_INITIAL_SIZE", 40),
        c_tier1_add_size=pm1.env_int("C1_TIER1_ADD_SIZE", 40),
        c_tier1_cap=pm1.env_int("C1_TIER1_CAP", 80),
        c_tier2_initial_size=pm1.env_int("C1_TIER2_INITIAL_SIZE", 60),
        c_tier2_add_size=pm1.env_int("C1_TIER2_ADD_SIZE", 40),
        c_tier2_cap=pm1.env_int("C1_TIER2_CAP", 120),
        c_tier3_initial_size=pm1.env_int("C1_TIER3_INITIAL_SIZE", 80),
        c_tier3_add_size=pm1.env_int("C1_TIER3_ADD_SIZE", 60),
        c_tier3_cap=pm1.env_int("C1_TIER3_CAP", 160),
        c_rates_initial_size=pm1.env_int("C1_RATES_INITIAL_SIZE", 40),
        c_rates_add_size=pm1.env_int("C1_RATES_ADD_SIZE", 40),
        c_rates_cap=pm1.env_int("C1_RATES_CAP", 100),
        c_background_initial_size=pm1.env_int("C1_BACKGROUND_INITIAL_SIZE", 25),
        c_background_add_size=pm1.env_int("C1_BACKGROUND_ADD_SIZE", 25),
        c_background_cap=pm1.env_int("C1_BACKGROUND_CAP", 60),
        startup_flatten_chunk_c=pm1.env_int("C1_STARTUP_FLATTEN_CHUNK_C", 25),
    )
    return BotConfig(**data)


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
    core_fair: float
    gap: float
    gap_abs: float
    fair_change: float
    entry_threshold: float
    exit_threshold: float
    expected_rate_bp: float
    yield_level: float
    pe_t: float
    equity_component: float
    bond_component: float
    residual: float


@dataclass
class CEntryDecision:
    side: Side
    thesis: str
    edge_ticks: float
    initial_size: int
    add_size: int
    thesis_cap: int


@dataclass
class MarketState(pm1.MarketState):
    current_eps_c: float = 2.0
    have_real_eps_c: bool = False
    last_c_earnings_delta: float = 0.0
    last_c_earnings_ts: float = 0.0
    last_c_earnings_is_initial: bool = False

    c_formula_residual: Optional[float] = None
    c_formula_anchor_ts: float = 0.0
    last_fair_c: Optional[float] = None

    c_event_seq: int = 0
    c_active_event_id: int = 0
    c_regime_side: Optional[Side] = None
    c_regime_thesis: Optional[str] = None
    c_entry_stage: int = 0
    c_last_entry_edge: float = 0.0
    c_last_add_ts: float = 0.0
    c_regime_started_at: float = 0.0
    c_blocked_side: Optional[Side] = None
    c_blocked_until: float = 0.0
    c_flat_entry_cooldown_until: float = 0.0

    def clear_c_regime(self) -> None:
        self.c_active_event_id = 0
        self.c_regime_side = None
        self.c_regime_thesis = None
        self.c_entry_stage = 0
        self.c_last_entry_edge = 0.0
        self.c_last_add_ts = 0.0
        self.c_regime_started_at = 0.0


class RiskManager(pm1.RiskManager):
    def __init__(self, client: "NewC1Client", cfg: BotConfig, state: MarketState, orders: OrderManager):
        super().__init__(client, cfg, state, orders)

    def clip_c_qty(self, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(self.cfg.symbol_c)
        pending = self.orders.pending_qty(self.cfg.symbol_c, side)
        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.c_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.c_max_order_size))

    def arm_session_baseline_if_ready(self) -> bool:
        if self.state.session_start_cash is not None and self.state.session_start_mtm is not None:
            return False
        for symbol in self.cfg.tracked_symbols:
            if self.client.get_position(symbol) != 0:
                return False
        if self.client.open_orders or self.state.live_orders:
            return False
        cash, mtm = self.client.cash_and_total_mtm()
        self.state.session_start_cash = cash
        self.state.session_start_mtm = mtm
        self.client._trace(
            "session_baseline",
            tick=self.client.current_tick,
            mtm=mtm,
            **self.client.positions_payload(),
        )
        return True

    async def startup_flatten_step(self) -> bool:
        inherited_order_ids = [
            str(order_id)
            for order_id in self.client.open_orders.keys()
            if str(order_id) not in self.state.live_orders
        ]
        for order_id in inherited_order_ids:
            await self.orders.cancel_order_if_present(order_id)

        for symbol in self.cfg.tracked_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=symbol):
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            if symbol == self.cfg.symbol_c:
                qty = min(abs(pos), self.cfg.startup_flatten_chunk_c, self.cfg.c_max_order_size)
            else:
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

        all_flat = all(self.client.get_position(symbol) == 0 for symbol in self.cfg.tracked_symbols)
        no_orders = not self.client.open_orders and not self.state.live_orders
        if all_flat and no_orders:
            self.state.startup_flatten_complete = True
            self.arm_session_baseline_if_ready()
            self.client._trace(
                "startup_flatten_complete",
                tick=self.client.current_tick,
                **self.client.positions_payload(),
            )
            return True
        return False


class CFairValueEngine:
    def __init__(
        self,
        client: "NewC1Client",
        cfg: BotConfig,
        state: MarketState,
        rates: pm1.RatesSignalEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.rates = rates

    def orders_live_for_c(self) -> bool:
        return any(order.symbol == self.cfg.symbol_c for order in self.state.live_orders.values())

    def earnings_context(self) -> EarningsContext:
        if self.state.last_c_earnings_ts <= 0.0:
            return EarningsContext(0.0, 0.0, float("inf"), 0, False, None, False)

        age = max(0.0, time.time() - self.state.last_c_earnings_ts)
        delta = self.state.last_c_earnings_delta
        abs_delta = abs(delta)
        side: Optional[Side]
        if delta > 0:
            side = Side.BUY
        elif delta < 0:
            side = Side.SELL
        else:
            side = None

        tier = 0
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
        if self.state.last_c_earnings_is_initial:
            hold_secs = max(hold_secs, self.cfg.c_earnings_hold_secs)

        return EarningsContext(
            delta=delta,
            abs_delta=abs_delta,
            age=age,
            tier=tier,
            is_initial=self.state.last_c_earnings_is_initial,
            side=side,
            hold_active=tier > 0 and age <= hold_secs,
        )

    def formula_components(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[dict[str, float]]:
        if rate_snapshot is None:
            return None
        expected_rate_bp = float(rate_snapshot.effective_expected_rate_bp)
        yield_level = self.cfg.c_y0 + self.cfg.c_beta_y * expected_rate_bp
        pe_t = self.cfg.c_pe0 * math.exp(-self.cfg.c_gamma * (yield_level - self.cfg.c_y0))
        delta_y = yield_level - self.cfg.c_y0
        bond_delta_per_share = self.cfg.c_b0_per_share * (
            -self.cfg.c_duration * delta_y + 0.5 * self.cfg.c_convexity * delta_y * delta_y
        )
        bond_component = self.cfg.c_lambda * bond_delta_per_share
        equity_component = self.state.current_eps_c * pe_t
        return {
            "expected_rate_bp": expected_rate_bp,
            "yield_level": yield_level,
            "pe_t": pe_t,
            "equity_component": equity_component,
            "bond_component": bond_component,
            "core_fair": equity_component + bond_component,
        }

    def ensure_anchor(self, rate_snapshot: Optional[RateSnapshot], *, force: bool = False) -> bool:
        components = self.formula_components(rate_snapshot)
        if components is None:
            return False
        mid_c = self.client.mid(self.cfg.symbol_c)
        if mid_c is None:
            return False
        if self.state.c_formula_residual is None or force:
            self.state.c_formula_residual = float(mid_c) - components["core_fair"]
            self.state.c_formula_anchor_ts = time.time()
            self.client._trace(
                "c_anchor",
                tick=self.client.current_tick,
                action="initialize" if not force else "force_reset",
                core_fair=components["core_fair"],
                residual=self.state.c_formula_residual,
                fair=components["core_fair"] + self.state.c_formula_residual,
                expected_rate_bp=components["expected_rate_bp"],
                eps=self.state.current_eps_c,
                **self.client.positions_payload(),
            )
            return True
        return False

    def maybe_reanchor(self, signal: Optional[CSignal]) -> None:
        if signal is None or self.state.c_formula_residual is None:
            return
        if self.client.get_position(self.cfg.symbol_c) != 0:
            return
        if self.orders_live_for_c():
            return
        if self.rates.is_news_urgent():
            return
        now = time.time()
        if now - self.state.c_formula_anchor_ts < self.cfg.c_formula_reanchor_secs:
            return
        if signal.gap_abs > self.cfg.c_formula_reanchor_gap_ticks:
            return

        target_residual = signal.mid - signal.core_fair
        alpha = self.cfg.c_formula_residual_alpha
        self.state.c_formula_residual = (
            (1.0 - alpha) * float(self.state.c_formula_residual) + alpha * float(target_residual)
        )
        self.state.c_formula_anchor_ts = now
        self.client._trace(
            "c_anchor",
            tick=self.client.current_tick,
            action="reanchor",
            core_fair=signal.core_fair,
            residual=self.state.c_formula_residual,
            fair=signal.core_fair + self.state.c_formula_residual,
            mid=signal.mid,
            gap=signal.gap,
            **self.client.positions_payload(),
        )

    def snapshot(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[CSignal]:
        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return None
        components = self.formula_components(rate_snapshot)
        if components is None or self.state.c_formula_residual is None:
            return None
        mid = book.mid
        if mid is None:
            return None
        fair = components["core_fair"] + float(self.state.c_formula_residual)
        spread = max(1.0, float(book.ask - book.bid))
        fair_change = 0.0 if self.state.last_fair_c is None else fair - self.state.last_fair_c
        return CSignal(
            bid=int(book.bid),
            bid_qty=int(book.bid_qty),
            ask=int(book.ask),
            ask_qty=int(book.ask_qty),
            mid=float(mid),
            spread=spread,
            fair=float(fair),
            core_fair=float(components["core_fair"]),
            gap=float(fair - mid),
            gap_abs=float(abs(fair - mid)),
            fair_change=float(fair_change),
            entry_threshold=max(self.cfg.c_entry_base_ticks, 1.25 * spread),
            exit_threshold=max(self.cfg.c_exit_base_ticks, 0.75 * spread),
            expected_rate_bp=float(components["expected_rate_bp"]),
            yield_level=float(components["yield_level"]),
            pe_t=float(components["pe_t"]),
            equity_component=float(components["equity_component"]),
            bond_component=float(components["bond_component"]),
            residual=float(self.state.c_formula_residual),
        )


class CTradingEngine:
    def __init__(
        self,
        client: "NewC1Client",
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

    def max_entry_stages(self, thesis_cap: int) -> int:
        return max(
            1,
            (max(0, thesis_cap) + max(1, self.cfg.c_max_order_size) - 1) // max(1, self.cfg.c_max_order_size),
        )

    def handle_missing_signal_exit_needed(self) -> bool:
        return self.client.get_position(self.cfg.symbol_c) != 0

    async def handle_missing_signal_exit(self) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            return False
        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=self.cfg.symbol_c):
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
            reason="signal_lost",
            thesis=self.state.c_regime_thesis,
            signal_strength=float(abs(pos)),
            event_id=self.state.c_active_event_id,
        )
        if placed and qty >= abs(pos):
            self.state.c_blocked_side = self.state.c_regime_side
            self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
            self.state.clear_c_regime()
        return placed

    def compute_entry_decision(
        self,
        signal: Optional[CSignal],
        rate_snapshot: Optional[RateSnapshot],
    ) -> Optional[CEntryDecision]:
        if signal is None or rate_snapshot is None:
            return None

        side = Side.BUY if signal.gap > 0 else Side.SELL
        earnings = self.fair_engine.earnings_context()

        if earnings.hold_active and earnings.side is not None:
            if earnings.side != side:
                return None
            if earnings.tier == 1 and signal.gap_abs >= max(signal.entry_threshold, 12.0):
                return CEntryDecision(side, "earnings", signal.gap_abs, self.cfg.c_tier1_initial_size, self.cfg.c_tier1_add_size, self.cfg.c_tier1_cap)
            if earnings.tier == 2 and signal.gap_abs >= max(signal.entry_threshold, 10.0):
                return CEntryDecision(side, "earnings", signal.gap_abs, self.cfg.c_tier2_initial_size, self.cfg.c_tier2_add_size, self.cfg.c_tier2_cap)
            if earnings.tier >= 3 and signal.gap_abs >= max(signal.entry_threshold, 8.0):
                return CEntryDecision(side, "earnings", signal.gap_abs, self.cfg.c_tier3_initial_size, self.cfg.c_tier3_add_size, self.cfg.c_tier3_cap)

        if (
            rate_snapshot.fresh_macro_event
            and abs(rate_snapshot.bias_bp) >= self.cfg.c_rates_trigger_bias_bp
            and signal.gap_abs >= max(signal.entry_threshold, self.cfg.c_rates_gap_ticks)
        ):
            return CEntryDecision(
                side=side,
                thesis="rates_shock",
                edge_ticks=signal.gap_abs,
                initial_size=self.cfg.c_rates_initial_size,
                add_size=self.cfg.c_rates_add_size,
                thesis_cap=self.cfg.c_rates_cap,
            )

        if signal.gap_abs >= max(signal.entry_threshold, self.cfg.c_background_gap_ticks) and not rate_snapshot.urgent:
            return CEntryDecision(
                side=side,
                thesis="formula_gap",
                edge_ticks=signal.gap_abs,
                initial_size=self.cfg.c_background_initial_size,
                add_size=self.cfg.c_background_add_size,
                thesis_cap=self.cfg.c_background_cap,
            )

        return None

    async def maybe_exit(self, signal: Optional[CSignal], rate_snapshot: Optional[RateSnapshot]) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            self.state.clear_c_regime()
            return False
        if signal is None or rate_snapshot is None:
            return await self.handle_missing_signal_exit()

        long_pos = pos > 0
        regime_side = Side.BUY if long_pos else Side.SELL
        favorable_gap = signal.gap if long_pos else -signal.gap
        adverse_gap = -signal.gap if long_pos else signal.gap
        compression_band = max(signal.exit_threshold, self.state.c_last_entry_edge * self.cfg.c_compression_frac)
        regime_age = 0.0
        if self.state.c_regime_started_at > 0.0:
            regime_age = max(0.0, time.time() - self.state.c_regime_started_at)
        thesis = self.state.c_regime_thesis or "formula_gap"
        earnings = self.fair_engine.earnings_context()

        side: Optional[Side] = None
        qty = 0
        reason: Optional[str] = None

        if earnings.hold_active and earnings.side is not None and earnings.side != regime_side:
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "earnings_flip"
        elif adverse_gap >= max(self.cfg.c_hard_flip_ticks, signal.entry_threshold):
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "hard_flip"
        elif thesis == "earnings":
            if regime_age >= self.cfg.c_earnings_max_hold_secs:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "earnings_timeout"
            elif favorable_gap <= compression_band:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos) if favorable_gap <= 0 else min(abs(pos), max(10, abs(pos) // 2))
                reason = "earnings_compress"
        elif thesis == "rates_shock":
            macro_reversal = (
                (long_pos and rate_snapshot.bias_bp <= -self.cfg.c_rate_reversal_bp)
                or ((not long_pos) and rate_snapshot.bias_bp >= self.cfg.c_rate_reversal_bp)
            )
            if macro_reversal:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "macro_reversal"
            elif regime_age >= self.cfg.c_rates_max_hold_secs:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "rates_timeout"
            elif favorable_gap <= compression_band and (
                not rate_snapshot.fresh_macro_event or abs(rate_snapshot.bias_bp) < self.cfg.c_rates_trigger_bias_bp
            ):
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos) if favorable_gap <= 0 else min(abs(pos), max(10, abs(pos) // 2))
                reason = "rates_compress"
        else:
            if regime_age >= self.cfg.c_background_max_hold_secs:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "gap_timeout"
            elif favorable_gap <= compression_band:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos) if favorable_gap <= 0 else min(abs(pos), max(10, abs(pos) // 2))
                reason = "gap_compress"

        if side is None or qty <= 0:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False

        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=min(int(qty), self.cfg.c_max_order_size),
            side=side,
            price=int(book.bid if side == Side.SELL else book.ask),
            role="exit",
            reason=reason,
            thesis=self.state.c_regime_thesis,
            signal_strength=signal.gap_abs,
            event_id=self.state.c_active_event_id,
        )
        if placed and qty >= abs(pos):
            self.state.c_blocked_side = self.state.c_regime_side
            self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
            self.state.clear_c_regime()
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

        if same_direction:
            if self.state.c_regime_thesis != decision.thesis:
                return False
            if self.state.c_entry_stage >= self.max_entry_stages(decision.thesis_cap):
                return False
            if now - self.state.c_last_add_ts < self.cfg.c_add_cooldown_secs:
                return False
            required_edge = max(signal.entry_threshold, self.state.c_last_entry_edge + self.cfg.c_add_edge_step_ticks)
            if decision.edge_ticks < required_edge:
                return False
            raw_qty = decision.add_size
            event_id = self.state.c_active_event_id
        else:
            raw_qty = decision.initial_size
            self.state.c_event_seq += 1
            event_id = self.state.c_event_seq

        qty = self.risk.clip_c_qty(decision.side, raw_qty, decision.thesis_cap)
        if qty <= 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False

        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=decision.side,
            price=int(book.ask if decision.side == Side.BUY else book.bid),
            role="entry",
            reason="c_entry",
            thesis=decision.thesis,
            signal_strength=decision.edge_ticks,
            event_id=event_id,
        )
        if placed:
            self.state.c_active_event_id = event_id
            self.state.c_regime_side = decision.side
            self.state.c_regime_thesis = decision.thesis
            self.state.c_entry_stage = 1 if not same_direction else self.state.c_entry_stage + 1
            self.state.c_last_entry_edge = decision.edge_ticks
            self.state.c_last_add_ts = now
            if not same_direction:
                self.state.c_regime_started_at = now
            if pos == 0:
                self.state.c_flat_entry_cooldown_until = now + self.cfg.c_flat_entry_cooldown_secs
        return placed


class Coordinator:
    def __init__(
        self,
        client: "NewC1Client",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        rates_signals: pm1.RatesSignalEngine,
        c_fair: CFairValueEngine,
        rates_trading: pm1.RatesTradingEngine,
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
        if self.client.get_position(self.cfg.symbol_c) == 0:
            if not any(order.symbol == self.cfg.symbol_c for order in self.state.live_orders.values()):
                self.state.clear_c_regime()
        if all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
                self.state.clear_rates_regime()

    def update_last_signals(self, rate_snapshot: Optional[RateSnapshot], c_signal: Optional[CSignal]) -> None:
        if rate_snapshot is not None:
            self.state.last_market_expected_rate_bp = rate_snapshot.market_expected_rate_bp
            self.state.last_effective_expected_rate_bp = rate_snapshot.effective_expected_rate_bp
        if c_signal is not None:
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
        self.client._trace(
            "status",
            tick=self.client.current_tick,
            reason=reason,
            rate_snapshot=self.client.current_snapshot_payload(rate_snapshot),
            c_signal=self.client.current_c_signal_payload(c_signal),
            mtm=mtm,
            session_cash=session_cash,
            session_mtm=session_mtm,
            **self.client.positions_payload(),
        )

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
        if rate_snapshot is not None:
            self.c_fair.ensure_anchor(rate_snapshot)
        c_signal = self.c_fair.snapshot(rate_snapshot)
        self.c_fair.maybe_reanchor(c_signal)

        if rate_snapshot is None:
            rates_acted = await self.rates_trading.handle_missing_signal_exit()
            c_acted = await self.c_trading.handle_missing_signal_exit()
            self.client._trace(
                "decision",
                tick=self.client.current_tick,
                reason="signal_not_ready",
                rates_acted=rates_acted,
                c_acted=c_acted,
                rate_snapshot=None,
                c_signal=None,
                books={symbol: vars(self.client.top(symbol)) for symbol in self.cfg.tracked_symbols},
                **self.client.positions_payload(),
            )
            self.log_status("signal_not_ready")
            return

        rates_exited = await self.rates_trading.maybe_exit(rate_snapshot)
        c_exited = await self.c_trading.maybe_exit(c_signal, rate_snapshot)
        if rates_exited or c_exited:
            self.update_last_signals(rate_snapshot, c_signal)
            self.client._trace(
                "decision",
                tick=self.client.current_tick,
                reason="exit",
                rates_exited=rates_exited,
                c_exited=c_exited,
                rate_snapshot=self.client.current_snapshot_payload(rate_snapshot),
                c_signal=self.client.current_c_signal_payload(c_signal),
                books={symbol: vars(self.client.top(symbol)) for symbol in self.cfg.tracked_symbols},
                **self.client.positions_payload(),
            )
            return

        c_decision = self.c_trading.compute_entry_decision(c_signal, rate_snapshot)
        c_first = c_decision is not None and c_decision.thesis == "earnings"

        rates_entered = False
        c_entered = False
        if c_first:
            c_entered = await self.c_trading.maybe_enter(c_decision, c_signal)
            rates_entered = await self.rates_trading.maybe_enter(rate_snapshot)
        else:
            rates_entered = await self.rates_trading.maybe_enter(rate_snapshot)
            c_entered = await self.c_trading.maybe_enter(c_decision, c_signal)

        if rates_entered or c_entered:
            self.update_last_signals(rate_snapshot, c_signal)
            self.client._trace(
                "decision",
                tick=self.client.current_tick,
                reason="entry",
                rates_entered=rates_entered,
                c_entered=c_entered,
                c_decision=None if c_decision is None else vars(c_decision),
                rate_snapshot=self.client.current_snapshot_payload(rate_snapshot),
                c_signal=self.client.current_c_signal_payload(c_signal),
                books={symbol: vars(self.client.top(symbol)) for symbol in self.cfg.tracked_symbols},
                **self.client.positions_payload(),
            )
            return

        self.update_last_signals(rate_snapshot, c_signal)
        self.client._trace(
            "decision",
            tick=self.client.current_tick,
            reason="no_trade",
            rate_snapshot=self.client.current_snapshot_payload(rate_snapshot),
            c_signal=self.client.current_c_signal_payload(c_signal),
            books={symbol: vars(self.client.top(symbol)) for symbol in self.cfg.tracked_symbols},
            **self.client.positions_payload(),
        )
        self.log_status("no_trade", rate_snapshot, c_signal)


class NewC1Client(pm1.NewPM1Client):
    def __init__(self, host: str, username: str, password: str, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or load_config()
        pm1.XChangeClient.__init__(self, host, username, password, silent=True, symbols=list(self.cfg.tracked_symbols))
        self.state = MarketState(current_eps_c=self.cfg.default_eps_c)
        self.current_tick: Optional[int] = None
        self._decision_lock = asyncio.Lock()
        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)
        self.rates_signal_engine = pm1.RatesSignalEngine(self, self.cfg, self.state)
        self.c_fair_engine = CFairValueEngine(self, self.cfg, self.state, self.rates_signal_engine)
        self.rates_trading_engine = pm1.RatesTradingEngine(
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
        self._trace_file = None
        self._trace_path: Optional[Path] = None

    def _trace(self, event_type: str, **kwargs) -> None:
        if not self.cfg.trace_enabled:
            return
        if self._trace_file is None:
            trace_dir = Path(self.cfg.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = trace_dir / f"new_C1_{int(time.time())}.jsonl"
            self._trace_file = self._trace_path.open("a", encoding="utf-8")
        payload = {"event_type": event_type, "timestamp": time.time(), **kwargs}
        self._trace_file.write(json.dumps(self._trace_jsonable(payload), ensure_ascii=True) + "\n")
        self._trace_file.flush()

    def positions_payload(self) -> dict[str, Any]:
        return {
            "positions": {symbol: self.get_position(symbol) for symbol in self.cfg.tracked_symbols},
            "cash": float(self.positions.get("cash", 0)),
        }

    def refresh_all_books(self) -> None:
        for symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)

    def cash_and_total_mtm(self) -> tuple[float, float]:
        cash = float(self.positions.get("cash", 0))
        mtm = cash
        for symbol in self.cfg.tracked_symbols:
            pos = self.get_position(symbol)
            if pos == 0:
                continue
            mark = self.mid(symbol)
            if mark is not None:
                mtm += pos * mark
        return cash, mtm

    def current_c_signal_payload(self, signal: Optional[CSignal]) -> Optional[dict[str, Any]]:
        if signal is None:
            return None
        return {
            "mid": signal.mid,
            "fair": signal.fair,
            "core_fair": signal.core_fair,
            "gap": signal.gap,
            "gap_abs": signal.gap_abs,
            "entry_threshold": signal.entry_threshold,
            "exit_threshold": signal.exit_threshold,
            "expected_rate_bp": signal.expected_rate_bp,
            "yield_level": signal.yield_level,
            "pe_t": signal.pe_t,
            "equity_component": signal.equity_component,
            "bond_component": signal.bond_component,
            "residual": signal.residual,
        }

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        tracked = self.order_manager.sync_fill(order_id)
        if self.get_position(self.cfg.symbol_c) == 0:
            if not any(order.symbol == self.cfg.symbol_c for order in self.state.live_orders.values()):
                self.state.clear_c_regime()
        if all(self.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
                self.state.clear_rates_regime()

        cash, mtm = self.cash_and_total_mtm()
        session_cash, session_mtm = self.session_pnl_snapshot(cash, mtm)
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="fill",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            qty=qty,
            price=price,
            mtm=mtm,
            session_cash=session_cash,
            session_mtm=session_mtm,
            **self.positions_payload(),
        )
        await self.order_manager.cancel_counterpart_after_fill(tracked)
        await self.evaluate()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.state.pending_cancels.discard(str(order_id))
        reason_text = (reason or "").lower()
        limit_rejection = "exceeds limits" in reason_text or "limit" in reason_text
        if tracked is not None and tracked.role == "entry":
            cooldown_multiplier = 2.0 if limit_rejection else 1.0
            if tracked.symbol == self.cfg.symbol_c:
                self.state.c_blocked_side = tracked.side
                self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs * cooldown_multiplier
            elif tracked.thesis is not None:
                self.state.rates_blocked_direction = tracked.thesis
                self.state.rates_blocked_until = time.time() + self.cfg.rate_reentry_block_secs * cooldown_multiplier
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="reject",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            reason=reason,
            **self.positions_payload(),
        )

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.tracked_symbols:
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)
            await self.evaluate()

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")

        c_update: Optional[dict[str, Any]] = None
        if kind == "structured" and str(new_data.get("structured_subtype") or "") == "earnings":
            asset = str(new_data.get("asset", "")).upper()
            if asset == self.cfg.symbol_c:
                value = float(new_data["value"])
                previous_eps = self.state.current_eps_c
                had_real_eps = self.state.have_real_eps_c
                delta = value - previous_eps
                self.state.current_eps_c = value
                self.state.have_real_eps_c = True
                self.state.last_c_earnings_delta = delta
                self.state.last_c_earnings_ts = time.time()
                self.state.last_c_earnings_is_initial = not had_real_eps

                abs_delta = abs(delta)
                urgent_ttl = 0.0
                if abs_delta >= self.cfg.c_earnings_medium_delta:
                    urgent_ttl = self.cfg.c_earnings_hold_secs + 0.75
                elif abs_delta >= self.cfg.c_earnings_small_delta:
                    urgent_ttl = self.cfg.c_earnings_hold_secs
                elif abs_delta >= self.cfg.c_earnings_ignore_delta or self.state.last_c_earnings_is_initial:
                    urgent_ttl = 1.50
                if urgent_ttl > 0.0:
                    self.rates_signal_engine.mark_news_urgent(urgent_ttl)

                c_update = {
                    "previous_eps": previous_eps,
                    "new_eps": value,
                    "delta": delta,
                    "is_initial": self.state.last_c_earnings_is_initial,
                }

        event = self.rates_signal_engine.classify_macro_event(kind, new_data)
        applied_bias_bp = 0.0
        parsed_values: dict[str, Any] = {}
        parsed_type = "ignored"

        if event is not None:
            parsed_type = event.parsed_type
            parsed_values = {
                "source": event.source,
                "bias_bp": event.bias_bp,
                "label": event.label,
                "actual": event.actual,
                "forecast": event.forecast,
                "surprise": event.surprise,
                "message_type": event.message_type,
            }
            if event.sentiment is not None:
                parsed_values.update(
                    {
                        "score": event.sentiment.score,
                        "bucket": event.sentiment.bucket,
                        "matched_phrases": event.sentiment.matched_phrases,
                        "unknown_candidate_phrases": event.sentiment.unknown_candidate_phrases,
                    }
                )
            if abs(event.bias_bp) >= 0.25:
                applied_bias_bp = event.bias_bp
                self.rates_signal_engine.apply_temp_rate_bias(
                    event.bias_bp,
                    event.ttl_secs,
                    event.source,
                    bucket=None if event.sentiment is None else event.sentiment.bucket,
                    score=0.0 if event.sentiment is None else event.sentiment.score,
                    label=event.label,
                )

        self._trace(
            "news",
            tick=self.current_tick,
            raw_kind=kind,
            raw_new_data=dict(new_data),
            parsed_type=parsed_type,
            parsed_values=parsed_values,
            applied_bias_bp=applied_bias_bp,
            c_update=c_update,
            **self.positions_payload(),
        )
        await self.evaluate()


async def main():
    client = NewC1Client(
        pm1.env_str("UTC_HOST", "34.197.188.76:3333"),
        pm1.env_str("UTC_USERNAME", "uiuc"),
        pm1.env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
