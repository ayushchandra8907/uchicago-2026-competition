from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import re
import time
from statistics import median
from typing import Any

from a_bot_config import MarketCConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


PM_HIKE = "R_HIKE"
PM_HOLD = "R_HOLD"
PM_CUT = "R_CUT"

DEFAULT_EPS_C = 2.0
C_POSITION_LIMIT = 200
C_ORDER_LIMIT = 40

CPI_DEADBAND = 0.00025
CPI_STRONG_SURPRISE = 0.00050
CPI_VERY_STRONG_SURPRISE = 0.00100
CPI_EXTREME_SURPRISE = 0.00200
CPI_SIGNAL_TTL_MS = 20_000
CPI_TO_RATE_BP = 4000.0
MAX_CPI_RATE_BIAS_BP = 8.0
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
LATE_SESSION_WEAK_EARNINGS_CUTOFF_TICK = 4300
LATE_SESSION_EARNINGS_TARGET_CAP_TICK = 4485
LATE_SESSION_EARNINGS_TARGET_CAP = C_ORDER_LIMIT
REVERSAL_TREND_MIN_ABS = 0.060

MACRO_HEADLINE_MIN_SCORE = 2.5
MACRO_HEADLINE_STRONG_SCORE = 3.5
MACRO_HEADLINE_VERY_STRONG_SCORE = 5.5
MACRO_SIGNAL_TTL_MS = 8_000
MACRO_TO_RATE_BP = 1.5
MACRO_EARNINGS_TREND_BLOCK = 0.020

C_OPS_WEIGHT = 0.72
C_BOND_WEIGHT = 0.28
C_PE_YIELD_GAMMA = 13.0
C_BOND_DURATION = 4.5
C_BOND_CONVEXITY = 30.0

C_SHOCK_POSITION_CAP = C_POSITION_LIMIT
C_SHOCK_MIN_EDGE_TICKS = 10.0
C_SHOCK_FULL_CONFIDENCE_EDGE_TICKS = 80.0
C_SHOCK_POSITION_SCALE = 1.20
C_SHOCK_FULL_CONFIDENCE_CHANGE_TICKS = 40.0
C_SHOCK_CHANGE_POSITION_SCALE = 0.75
C_SHOCK_MIN_POSITION = C_ORDER_LIMIT
C_SHOCK_SLICE_TARGET_QTY = 20
C_SHOCK_SLICE_MIN_QTY = 7
C_SHOCK_SLICE_MAX_QTY = 24
C_SHOCK_MAX_HOLD_MS = 12_500
C_SHOCK_EMERGENCY_DUMP_MIN_MS = 250
C_SHOCK_EMERGENCY_DUMP_TICKS = 40.0
C_SHOCK_EMERGENCY_DUMP_FRACTION = 0.20
C_SHOCK_EMERGENCY_DUMP_MIN_INVENTORY = 12
C_SHOCK_DECAY_START_MS = 5_000
C_SHOCK_DECAY_INTERVAL_MS = 500
C_SHOCK_DECAY_FRACTION = 0.08
C_SHOCK_DECAY_MIN_QTY = 6
C_SHOCK_DECAY_MAX_QTY = 10
C_SHOCK_DECAY_MIN_INVENTORY = 40
C_SHOCK_DECAY_MIN_RESIDUAL_FRACTION = 0.10
C_SHOCK_DECAY_STALL_WINDOW_MS = 1_200
C_SHOCK_DECAY_STALL_THRESHOLD_TICKS = 12.0
C_SHOCK_OVERSHOOT_HOLD_MS = 225
C_SHOCK_OVERSHOOT_MAX_WAIT_MS = 600
C_SHOCK_OVERSHOOT_BAND_TICKS = 10.0
C_SHOCK_OVERSHOOT_REVERSAL_TICKS = 2.0
C_SHOCK_OVERSHOOT_STAGE_FRACTIONS = (0.30, 0.25, 0.20)
C_SHOCK_OVERSHOOT_STAGE_MIN_QTY = 4
C_SHOCK_OVERSHOOT_STAGE_MAX_QTY = 16
C_SHOCK_OVERSHOOT_MIN_RESIDUAL_FRACTION = 0.30
C_SHOCK_OVERSHOOT_LARGE_POSITION_THRESHOLD = 100
C_SHOCK_OVERSHOOT_LARGE_STAGE1_FRACTION = 0.50
C_SHOCK_OVERSHOOT_LARGE_RESIDUAL_FRACTION = 0.50
C_SHOCK_EQUILIBRIUM_BAND_TICKS = 8.0
C_SHOCK_EQUILIBRIUM_HOLD_MS = 1_000
C_SHOCK_EQUILIBRIUM_MIN_SAMPLES = 6
C_SHOCK_EQUILIBRIUM_MIN_MS = 1_000
C_SHOCK_EQUILIBRIUM_RESIDUAL_EDGE_TICKS = 40.0
C_SHOCK_EQUILIBRIUM_MIN_CAPTURE_FRACTION = 0.55

FLOAT_RE = r"[-+]?\\d*\\.?\\d+"


@dataclass(frozen=True)
class RateContext:
    q_hike: float
    q_hold: float
    q_cut: float
    market_rate_bp: float
    effective_rate_bp: float
    cpi_bias_bp: float


@dataclass(frozen=True)
class CParsedSignal:
    strategy_family: str
    action_class: str
    signal_id: str
    payload: dict[str, Any]


@dataclass
class CSignal:
    signal_id: str
    side: str
    target: int
    thesis: str
    strength: float
    tick: int
    started_ms: int
    description: str
    blocked_reason: str | None = None
    live_trading_enabled: bool = False


@dataclass
class CEarningsShockState:
    mode: str = "IDLE"
    signal_id: str | None = None
    direction: int = 0
    target_inventory: int = 0
    original_target_inventory: int = 0
    peak_inventory_abs: int = 0
    started_ms: int = 0
    tick: int = 0
    reference_mid: float | None = None
    fair_before: float | None = None
    fair_after: float | None = None
    fair_change_ticks: float | None = None
    initial_edge: float = 0.0
    equilibrium_reached_ms: int | None = None
    overshoot_stage_index: int = 0
    overshoot_trimmed_qty_total: int = 0
    overshoot_active: bool = False
    overshoot_trigger_ticks: float | None = None
    overshoot_crossed_ms: int | None = None
    decay_steps_applied: int = 0
    decay_trimmed_qty_total: int = 0
    post_event_mids: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=160))


class MarketCStrategy:
    def __init__(self, config: MarketCConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.config = config
        self.risk = risk
        self.symbol = config.symbol
        self.pm_symbols = tuple(config.pm_symbols)
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.order_manager = OrderManager(symbol=self.symbol, risk=risk)
        self.books: dict[str, BookSnapshot] = {
            self.symbol: BookSnapshot(),
            **{pm_symbol: BookSnapshot() for pm_symbol in self.pm_symbols},
        }
        self.inventory = 0
        self.pm_positions: dict[str, int] = {pm_symbol: 0 for pm_symbol in self.pm_symbols}
        self.mode = "C_OBSERVE_ONLY"
        self.last_block_reason: str | None = None
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.last_news_ms: int | None = None
        self.last_tick_seen = 0

        self.current_eps_c = DEFAULT_EPS_C
        self.have_real_eps_c = False
        self.baseline_eps_c: float | None = None
        self.last_c_earnings_delta = 0.0
        self.recent_c_earnings_deltas: deque[float] = deque(maxlen=4)

        self.anchor_price: float | None = None
        self.anchor_eps: float | None = None
        self.anchor_rate_bp: float | None = None
        self.last_rate_context: RateContext | None = None

        self.last_cpi_surprise = 0.0
        self.last_cpi_bias_bp = 0.0
        self.last_cpi_ms = 0
        self.last_rate_bias_ttl_ms = CPI_SIGNAL_TTL_MS

        self.active_signal: CSignal | None = None
        self.last_shadow_signal: CSignal | None = None
        self.c_earnings_shock = CEarningsShockState()
        self.signal_seq = 0
        self.c_unwind_wait_reason: str | None = None

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol not in self.books:
            return False
        self.books[symbol] = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        self.last_trade_ms = int(now_ms) if symbol == self.symbol and self.last_trade_ms is None else self.last_trade_ms
        self.maybe_initialize_anchor()
        return True

    def on_market_trade(self, symbol: str, price: int, qty: int, now_ms: int) -> None:
        if symbol == self.symbol:
            self.last_trade_px = int(price)
            self.last_trade_qty = int(qty)
            self.last_trade_ms = int(now_ms)

    def sync_inventory_from_exchange(self, symbol: str, inventory: int, now_ms: int | None = None) -> None:
        if symbol == self.symbol:
            self.inventory = int(inventory)
        elif symbol in self.pm_positions:
            self.pm_positions[symbol] = int(inventory)

    def on_fill(
        self,
        order_id: str,
        qty: int,
        price: int,
        *,
        authoritative_inventory: int | None = None,
    ):
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        if authoritative_inventory is None:
            signed_qty = int(qty) if order.side == "BUY" else -int(qty)
            self.inventory += signed_qty
        else:
            self.inventory = int(authoritative_inventory)
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms()
        if self.c_earnings_shock.mode == "SHOCK" and self._sign(self.inventory) == self.c_earnings_shock.direction:
            self.c_earnings_shock.peak_inventory_abs = max(
                self.c_earnings_shock.peak_inventory_abs,
                abs(int(self.inventory)),
            )
        return order

    def on_cancel_response(self, order_id: str, success: bool):
        return self.order_manager.handle_cancel_response(order_id, success)

    def on_rejection(self, order_id: str):
        return self.order_manager.handle_rejection(order_id)

    def on_news(self, news_release: dict, now_ms: int) -> dict[str, Any]:
        self.last_news_ms = int(now_ms)
        self.last_tick_seen = int(news_release.get("tick") or news_release.get("timestamp") or self.last_tick_seen or 0)
        signals: list[CParsedSignal] = []
        note = None

        earnings_value = self.parse_c_earnings_from_news(news_release)
        if earnings_value is not None:
            earnings_signal, note = self._handle_earnings_signal(earnings_value, self.last_tick_seen, int(now_ms))
            if earnings_signal is not None:
                signals.append(earnings_signal)

        actual_cpi, forecast_cpi = self.parse_cpi_from_news(news_release)
        if actual_cpi is not None and forecast_cpi is not None:
            signals.append(self._build_cpi_shadow_signal(actual_cpi, forecast_cpi, self.last_tick_seen, int(now_ms)))

        macro_signal = self._build_macro_shadow_signal(news_release, int(now_ms))
        if macro_signal is not None:
            signals.append(macro_signal)

        return {
            "tick": self.last_tick_seen,
            "signals": signals,
            "note": note,
            "earnings_value": earnings_value,
            "cpi_actual": actual_cpi,
            "cpi_forecast": forecast_cpi,
            "macro_score": None if macro_signal is None else macro_signal.payload.get("score"),
        }

    def compute_quotes(self, now_ms: int) -> QuotePlan:
        if not self.config.enabled:
            self.mode = "C_DISABLED"
            return QuotePlan(self.mode, None, None, (), True, "market_c_disabled")
        self.maybe_initialize_anchor()
        self._record_c_shock_mid(int(now_ms))
        if self.c_earnings_shock.mode == "IDLE":
            self.c_unwind_wait_reason = None
            self.mode = "C_OBSERVE_ONLY"
            return QuotePlan(self.mode, None, None, (), True, self.last_block_reason or "waiting_for_c_earnings_signal")

        if self.c_earnings_shock.mode == "UNWIND":
            self.mode = "C_EARNINGS_UNWIND"
            if self.inventory == 0:
                if self._has_live_orders():
                    self.c_unwind_wait_reason = "waiting_for_c_cancel_before_flatten"
                    self.last_block_reason = self.c_unwind_wait_reason
                    return QuotePlan(self.mode, None, None, (), True, self.c_unwind_wait_reason)
                self.reset_c_earnings_shock()
                self.active_signal = None
                self.c_unwind_wait_reason = None
                self.mode = "C_OBSERVE_ONLY"
                return QuotePlan(self.mode, None, None, (), True, "c_earnings_flat")
            return self._target_plan(
                target_inventory=0,
                reason="c_earnings_unwind",
                action_class="shock_unwind",
                mode="C_EARNINGS_UNWIND",
                signal_id=self.c_earnings_shock.signal_id,
            )

        self.mode = "C_EARNINGS_SHOCK"
        if self._c_shock_should_emergency_dump(int(now_ms)):
            self.c_earnings_shock.mode = "UNWIND"
            self.c_earnings_shock.target_inventory = 0
            return self._target_plan(
                target_inventory=0,
                reason="c_earnings_emergency_dump",
                action_class="risk_flatten",
                mode="C_EARNINGS_UNWIND",
                signal_id=self.c_earnings_shock.signal_id,
            )
        if self._c_shock_should_force_time_flatten(int(now_ms)):
            self.c_earnings_shock.mode = "UNWIND"
            self.c_earnings_shock.target_inventory = 0
            return self._target_plan(
                target_inventory=0,
                reason="c_earnings_max_hold",
                action_class="risk_flatten",
                mode="C_EARNINGS_UNWIND",
                signal_id=self.c_earnings_shock.signal_id,
            )

        overshoot_plan = self._maybe_build_c_overshoot_trim(int(now_ms))
        if overshoot_plan is not None:
            return overshoot_plan

        if self._c_shock_equilibrium_reached(int(now_ms)):
            self.c_earnings_shock.mode = "UNWIND"
            self.c_earnings_shock.target_inventory = 0
            self.c_earnings_shock.equilibrium_reached_ms = int(now_ms)
            return self._target_plan(
                target_inventory=0,
                reason="c_earnings_equilibrium",
                action_class="shock_unwind",
                mode="C_EARNINGS_UNWIND",
                signal_id=self.c_earnings_shock.signal_id,
            )

        if self._maybe_apply_c_shock_decay(int(now_ms)):
            return self._target_plan(
                target_inventory=self.c_earnings_shock.target_inventory,
                reason="c_earnings_decay_trim",
                action_class="decay_trim",
                mode="C_EARNINGS_SHOCK",
                signal_id=self.c_earnings_shock.signal_id,
            )

        return self._target_plan(
            target_inventory=self.c_earnings_shock.target_inventory,
            reason="c_earnings_shock",
            action_class="shock_take",
            mode="C_EARNINGS_SHOCK",
            signal_id=self.c_earnings_shock.signal_id,
        )

    def trace_state(self, now_ms: int) -> dict[str, Any]:
        book = self.books[self.symbol]
        fair_value = self.fair_value_c()
        rate = self.rate_context() or self.last_rate_context
        shock = self.c_earnings_shock
        signal = self.active_signal
        return {
            "symbol": self.symbol,
            "market_key": "C",
            "mode": self.mode,
            "fair_value": None if fair_value is None else round(fair_value, 4),
            "inventory": self.inventory,
            "earnings_position": self.inventory if shock.mode in {"SHOCK", "UNWIND"} else 0,
            "news_position": 0,
            "mm_position": 0,
            "buy_exposure": self.order_manager.buy_exposure(),
            "sell_exposure": self.order_manager.sell_exposure(),
            "allowed_buy_size": max(0, C_POSITION_LIMIT - self.inventory - self.order_manager.buy_exposure()),
            "allowed_sell_size": max(0, C_POSITION_LIMIT + self.inventory - self.order_manager.sell_exposure()),
            "position_cap": C_POSITION_LIMIT,
            "live_orders": self.order_manager.live_orders_snapshot(),
            "book": {
                "best_bid_px": None if book.best_bid is None else book.best_bid.px,
                "best_bid_qty": None if book.best_bid is None else book.best_bid.qty,
                "best_ask_px": None if book.best_ask is None else book.best_ask.px,
                "best_ask_qty": None if book.best_ask is None else book.best_ask.qty,
                "spread": book.spread,
                "mid": book.mid,
                "microprice": book.microprice,
                "top_of_book_imbalance": book.top_of_book_imbalance,
                "bid_levels": [{"px": level.px, "qty": level.qty} for level in book.bid_levels],
                "ask_levels": [{"px": level.px, "qty": level.qty} for level in book.ask_levels],
            },
            "last_trade_px": self.last_trade_px,
            "last_trade_qty": self.last_trade_qty,
            "last_trade_ms": self.last_trade_ms,
            "block_reason": self.last_block_reason,
            "c_fair_value": None if fair_value is None else round(fair_value, 4),
            "c_fair_gap": None if fair_value is None or book.mid is None else round(float(fair_value) - float(book.mid), 4),
            "c_anchor_price": self.anchor_price,
            "c_anchor_eps": self.anchor_eps,
            "c_anchor_rate_bp": self.anchor_rate_bp,
            "c_current_eps": self.current_eps_c,
            "c_last_earnings_delta": self.last_c_earnings_delta,
            "c_signal_id": None if signal is None else signal.signal_id,
            "c_signal_thesis": None if signal is None else signal.thesis,
            "c_signal_strength": None if signal is None else signal.strength,
            "c_signal_target": None if signal is None else signal.target,
            "c_signal_fresh": False if signal is None else self.c_signal_is_fresh(int(now_ms)),
            "c_market_rate_bp": None if rate is None else rate.market_rate_bp,
            "c_effective_rate_bp": None if rate is None else rate.effective_rate_bp,
            "c_cpi_bias_bp": None if rate is None else rate.cpi_bias_bp,
            "c_last_cpi_surprise": self.last_cpi_surprise,
            "c_q_hike": None if rate is None else rate.q_hike,
            "c_q_hold": None if rate is None else rate.q_hold,
            "c_q_cut": None if rate is None else rate.q_cut,
            "c_shock_mode": shock.mode,
            "c_shock_target_inventory": shock.target_inventory if shock.mode != "IDLE" else None,
            "c_shock_original_target_inventory": shock.original_target_inventory if shock.mode != "IDLE" else None,
            "c_shock_reference_mid": shock.reference_mid,
            "c_shock_fair_before": shock.fair_before,
            "c_shock_fair_after": shock.fair_after,
            "c_shock_edge": shock.initial_edge if shock.mode != "IDLE" else None,
            "c_unwind_wait_reason": self.c_unwind_wait_reason,
            "c_live_order_count": len(self.order_manager.live_orders_snapshot()),
            "c_live_order_side": self._live_order_side_tag(),
            "c_target_side": self._target_side_for_inventory(shock.target_inventory),
            "exchange_tick": self.last_tick_seen,
        }

    def trace_state_at(self, symbol: str, now_ms: int) -> dict[str, Any]:
        if symbol == self.symbol:
            return self.trace_state(now_ms)
        if symbol not in self.pm_symbols:
            raise ValueError(f"Unsupported Market C trace symbol: {symbol}")
        book = self.books.get(symbol) or BookSnapshot()
        rate = self.rate_context() or self.last_rate_context
        return {
            "symbol": symbol,
            "market_key": "C_PM",
            "mode": "C_PM_OBSERVE_ONLY",
            "fair_value": None,
            "inventory": int(self.pm_positions.get(symbol, 0)),
            "earnings_position": 0,
            "news_position": 0,
            "mm_position": 0,
            "buy_exposure": 0,
            "sell_exposure": 0,
            "allowed_buy_size": 0,
            "allowed_sell_size": 0,
            "position_cap": 0,
            "live_orders": [],
            "book": {
                "best_bid_px": None if book.best_bid is None else book.best_bid.px,
                "best_bid_qty": None if book.best_bid is None else book.best_bid.qty,
                "best_ask_px": None if book.best_ask is None else book.best_ask.px,
                "best_ask_qty": None if book.best_ask is None else book.best_ask.qty,
                "spread": book.spread,
                "mid": book.mid,
                "microprice": book.microprice,
                "top_of_book_imbalance": book.top_of_book_imbalance,
                "bid_levels": [{"px": level.px, "qty": level.qty} for level in book.bid_levels],
                "ask_levels": [{"px": level.px, "qty": level.qty} for level in book.ask_levels],
            },
            "last_trade_px": None,
            "last_trade_qty": None,
            "last_trade_ms": None,
            "block_reason": None,
            "c_market_rate_bp": None if rate is None else rate.market_rate_bp,
            "c_effective_rate_bp": None if rate is None else rate.effective_rate_bp,
            "c_cpi_bias_bp": None if rate is None else rate.cpi_bias_bp,
            "c_q_hike": None if rate is None else rate.q_hike,
            "c_q_hold": None if rate is None else rate.q_hold,
            "c_q_cut": None if rate is None else rate.q_cut,
            "exchange_tick": self.last_tick_seen,
        }

    def active_etf_projection(self) -> dict[str, Any] | None:
        shock = self.c_earnings_shock
        if shock.mode not in {"SHOCK", "UNWIND"} or shock.fair_before is None or shock.fair_after is None:
            return None
        return {
            "source_market": "C",
            "source_kind": "structured_earnings",
            "source_signal_id": shock.signal_id,
            "fair_shift_ticks": float(shock.fair_after) - float(shock.fair_before),
            "source_target_inventory": shock.original_target_inventory,
            "source_direction": shock.direction,
        }

    def rate_context(self) -> RateContext | None:
        hike_mid = self.mid(PM_HIKE)
        hold_mid = self.mid(PM_HOLD)
        cut_mid = self.mid(PM_CUT)
        if hike_mid is None or hold_mid is None or cut_mid is None:
            return None
        total = hike_mid + hold_mid + cut_mid
        if total <= 1e-9:
            return None
        q_hike = hike_mid / total
        q_hold = hold_mid / total
        q_cut = cut_mid / total
        market_rate_bp = 25.0 * q_hike - 25.0 * q_cut
        cpi_bias_bp = 0.0
        if self.last_cpi_ms > 0 and self._now_ms() - self.last_cpi_ms <= self.last_rate_bias_ttl_ms:
            cpi_bias_bp = self.last_cpi_bias_bp
        context = RateContext(
            q_hike=q_hike,
            q_hold=q_hold,
            q_cut=q_cut,
            market_rate_bp=market_rate_bp,
            effective_rate_bp=market_rate_bp + cpi_bias_bp,
            cpi_bias_bp=cpi_bias_bp,
        )
        self.last_rate_context = context
        return context

    def maybe_initialize_anchor(self, force: bool = False) -> bool:
        rate = self.rate_context() or self.last_rate_context
        mid_c = self.mid(self.symbol)
        if rate is None or mid_c is None:
            return False
        if force or self.anchor_price is None or self.anchor_eps is None or self.anchor_rate_bp is None:
            self.anchor_price = float(mid_c)
            self.anchor_eps = float(self.current_eps_c)
            self.anchor_rate_bp = float(rate.effective_rate_bp)
            return True
        return False

    def fair_value_c(self) -> float | None:
        rate = self.rate_context() or self.last_rate_context
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

    def fair_gap(self) -> float | None:
        fair = self.fair_value_c()
        mid_c = self.mid(self.symbol)
        if fair is None or mid_c is None:
            return None
        return float(fair) - float(mid_c)

    def c_target_for_cpi_surprise(self, surprise: float) -> int:
        abs_surprise = abs(float(surprise))
        if abs_surprise <= CPI_DEADBAND:
            return 0
        if abs_surprise >= CPI_VERY_STRONG_SURPRISE:
            return 200
        if abs_surprise >= CPI_STRONG_SURPRISE:
            return 160
        return 120

    def c_target_for_macro_score(self, score: float) -> int:
        abs_score = abs(float(score))
        if abs_score < MACRO_HEADLINE_MIN_SCORE:
            return 0
        if abs_score >= MACRO_HEADLINE_VERY_STRONG_SCORE:
            return 160
        if abs_score >= MACRO_HEADLINE_STRONG_SCORE:
            return 120
        return 80

    def parse_cpi_from_news(self, news_release: dict) -> tuple[float | None, float | None]:
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
        if not looks_like_cpi and not ("actual" in lower and "forecast" in lower):
            return None, None
        actual_match = re.search(r"actual\\s*(?:[:=]|is|was|of|at)?\\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        forecast_match = re.search(r"forecast\\s*(?:[:=]|is|was|of|at)?\\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if actual_match and forecast_match:
            return float(actual_match.group(1)), float(forecast_match.group(1))
        nums = [float(value) for value in re.findall(FLOAT_RE, text)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        return None, None

    def parse_c_earnings_from_news(self, news_release: dict) -> float | None:
        kind = str(news_release.get("kind") or "").lower()
        new_data_obj = news_release.get("new_data", {}) or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        subtype = str(new_data.get("structured_subtype") or news_release.get("structured_subtype") or "").lower()
        asset = str(new_data.get("asset") or news_release.get("asset") or "").upper()
        value = new_data.get("value", news_release.get("value"))
        if "earnings" in subtype and asset == self.symbol and value is not None:
            return float(value)
        text = str(new_data.get("content") or news_release.get("content") or "")
        if "earnings" not in kind and "earnings" not in text.lower():
            return None
        match = re.search(r"\\bC\\s+earnings\\s+released\\s*:?\\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def score_macro_headline_for_c(self, text: str) -> float:
        lower = text.lower()
        score = 0.0
        hawkish_phrases = {
            "push back on market expectations for near-term cuts": 5.0,
            "discomfort with above-target inflation": 4.5,
            "progress on inflation has stalled": 5.0,
            "inflation has stalled": 4.0,
            "not fast enough": 3.0,
            "returning inflation to target": 2.5,
            "even at cost to growth": 3.0,
            "underlying inflation elevated": 4.0,
            "persistent inflation": 3.5,
            "inflation risks": 3.0,
            "additional tightening": 3.0,
            "wage growth": 1.5,
        }
        dovish_phrases = {
            "restrictive stance may no longer be needed": -4.5,
            "preemptive cut": -4.0,
            "rate relief": -3.5,
            "real rates are well above neutral": -3.0,
            "wage growth decelerating": -3.5,
            "falling inflation": -2.5,
            "cooling labor": -2.5,
            "more likely a cut than a hike": -4.0,
            "lean toward cuts": -3.0,
            "shift toward accommodation": -4.0,
            "slowing gdp growth": -2.0,
            "softening data": -2.0,
            "policy easing": -2.0,
            "assess cumulative tightening effects": -3.5,
        }
        neutral_or_mixed = (
            "mixed",
            "unclear",
            "no clear signal",
            "data dependence",
            "cautious",
            "conflict",
            "uncertain outlook",
            "declines to commit",
        )
        for phrase, weight in hawkish_phrases.items():
            if phrase in lower:
                score += weight
        for phrase, weight in dovish_phrases.items():
            if phrase in lower:
                score += weight
        if "near-term cuts" in lower and "push back" not in lower:
            score -= 2.0
        if "rate cuts" in lower and "push back" not in lower:
            score -= 1.5
        if any(phrase in lower for phrase in neutral_or_mixed):
            score *= 0.35
        return score

    def c_earnings_trend(self) -> float:
        return sum(self.recent_c_earnings_deltas)

    def c_signal_is_fresh(self, now_ms: int) -> bool:
        signal = self.active_signal
        if signal is None:
            return False
        ttl_ms = {
            "earnings": 20_000,
            "cpi": CPI_SIGNAL_TTL_MS,
            "macro": MACRO_SIGNAL_TTL_MS,
        }.get(signal.thesis, 20_000)
        return int(now_ms) - int(signal.started_ms) <= int(ttl_ms)

    def _handle_earnings_signal(self, value: float, tick: int, now_ms: int) -> tuple[CParsedSignal | None, str | None]:
        if not self.have_real_eps_c:
            self.have_real_eps_c = True
            self.current_eps_c = float(value)
            self.baseline_eps_c = float(value)
            self.last_c_earnings_delta = 0.0
            self.active_signal = None
            self.maybe_initialize_anchor(force=True)
            signal_id = self._next_signal_id("c_earnings")
            return (
                CParsedSignal(
                    strategy_family="c_earnings",
                    action_class="baseline_adopted",
                    signal_id=signal_id,
                    payload={
                        "signal_id": signal_id,
                        "tick": tick,
                        "eps": float(value),
                        "live_trading_enabled": False,
                        "reason": "first_real_c_baseline",
                    },
                ),
                f"First C earnings baseline adopted: tick={tick} eps={value:.4f}",
            )

        self.maybe_initialize_anchor()
        old_eps = float(self.current_eps_c)
        fair_before = self.fair_value_c()
        reference_mid = self.mid(self.symbol)
        delta = float(value) - old_eps
        self.current_eps_c = float(value)
        self.last_c_earnings_delta = float(delta)
        self.recent_c_earnings_deltas.append(float(delta))
        fair_after = self.fair_value_c()

        side = "BUY" if delta > 0 else "SELL"
        block_reason = None
        if tick >= LATE_SESSION_WEAK_EARNINGS_CUTOFF_TICK and abs(delta) < EARNINGS_SMALL_DELTA:
            block_reason = "late_session_weak_earnings"
        previous_trend = sum(list(self.recent_c_earnings_deltas)[:-1]) if len(self.recent_c_earnings_deltas) >= 3 else 0.0
        if abs(delta) < EARNINGS_SMALL_DELTA:
            if side == "BUY" and previous_trend < -REVERSAL_TREND_MIN_ABS:
                block_reason = f"weak_buy_against_earnings_trend_{previous_trend:+.4f}"
            if side == "SELL" and previous_trend > REVERSAL_TREND_MIN_ABS:
                block_reason = f"weak_sell_against_earnings_trend_{previous_trend:+.4f}"

        if fair_after is None or reference_mid is None:
            block_reason = block_reason or "missing_shock_fair"

        edge = None if fair_after is None or reference_mid is None else float(fair_after) - float(reference_mid)
        fair_change = None if fair_before is None or fair_after is None else float(fair_after) - float(fair_before)
        target = 0 if edge is None else self._c_shock_scaled_target(float(edge), fair_change)
        if block_reason is None and target == 0:
            block_reason = "earnings_deadband"

        signal_id = self._next_signal_id("c_earnings")
        payload = {
            "signal_id": signal_id,
            "tick": tick,
            "old_eps": old_eps,
            "new_eps": float(value),
            "earnings_delta": float(delta),
            "fair_before": fair_before,
            "fair_after": fair_after,
            "reference_mid": reference_mid,
            "fair_shift_ticks": fair_change,
            "edge": edge,
            "target_inventory": target,
            "blocked_reason": block_reason,
            "live_trading_enabled": bool(self.config.live_earnings_enabled and block_reason is None),
        }
        if block_reason is not None or not self.config.live_earnings_enabled:
            self.reset_c_earnings_shock()
            self.active_signal = CSignal(
                signal_id=signal_id,
                side=side,
                target=max(0, abs(int(target))),
                thesis="earnings",
                strength=abs(float(delta)),
                tick=int(tick),
                started_ms=int(now_ms),
                description="c_earnings_shadow",
                blocked_reason=block_reason,
                live_trading_enabled=False,
            )
            action_class = "shadow_blocked" if block_reason is not None else "shadow_signal"
            return CParsedSignal("c_earnings", action_class, signal_id, payload), None

        if tick >= LATE_SESSION_EARNINGS_TARGET_CAP_TICK:
            target = self._sign(target) * min(abs(int(target)), int(LATE_SESSION_EARNINGS_TARGET_CAP))
            payload["target_inventory"] = target
        self._start_c_earnings_shock(
            signal_id=signal_id,
            tick=int(tick),
            delta=float(delta),
            fair_before=fair_before,
            fair_after=fair_after,
            fair_change_ticks=fair_change,
            reference_mid=float(reference_mid),
            target_inventory=int(target),
            started_ms=int(now_ms),
        )
        return CParsedSignal("c_earnings", "earnings_signal", signal_id, payload), None

    def _build_cpi_shadow_signal(self, actual: float, forecast: float, tick: int, now_ms: int) -> CParsedSignal:
        surprise = float(actual) - float(forecast)
        target = self.c_target_for_cpi_surprise(surprise)
        self.last_cpi_surprise = surprise
        self.last_cpi_bias_bp = self._clip(surprise * CPI_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
        self.last_cpi_ms = int(now_ms)
        self.last_rate_bias_ttl_ms = CPI_SIGNAL_TTL_MS
        signal_id = self._next_signal_id("c_cpi")
        blocked_reason = None
        if target == 0:
            blocked_reason = "cpi_deadband"
        elif not self.have_real_eps_c:
            blocked_reason = "no_real_c_baseline"
        self.last_shadow_signal = CSignal(
            signal_id=signal_id,
            side="SELL" if surprise > 0 else "BUY",
            target=target,
            thesis="cpi",
            strength=abs(surprise),
            tick=tick,
            started_ms=now_ms,
            description="cpi_shadow",
            blocked_reason=blocked_reason,
            live_trading_enabled=False,
        )
        return CParsedSignal(
            strategy_family="c_cpi_shadow",
            action_class="shadow_blocked" if blocked_reason else "shadow_signal",
            signal_id=signal_id,
            payload={
                "signal_id": signal_id,
                "tick": tick,
                "actual": float(actual),
                "forecast": float(forecast),
                "surprise": surprise,
                "target_inventory": target,
                "bias_bp": self.last_cpi_bias_bp,
                "blocked_reason": blocked_reason,
                "live_trading_enabled": False,
            },
        )

    def _build_macro_shadow_signal(self, news_release: dict, now_ms: int) -> CParsedSignal | None:
        new_data = news_release.get("new_data", {}) or {}
        if isinstance(new_data, dict):
            text = str(new_data.get("content") or news_release.get("content") or "")
        else:
            text = str(news_release.get("content") or "")
        if not text:
            return None
        score = self.score_macro_headline_for_c(text)
        target = self.c_target_for_macro_score(score)
        if target == 0:
            return None
        side = "SELL" if score > 0 else "BUY"
        blocked_reason = None
        trend = self.c_earnings_trend()
        if self.have_real_eps_c and len(self.recent_c_earnings_deltas) >= 2:
            if side == "BUY" and trend < -MACRO_EARNINGS_TREND_BLOCK:
                blocked_reason = f"earnings_trend_conflict_{trend:+.4f}"
            if side == "SELL" and trend > MACRO_EARNINGS_TREND_BLOCK:
                blocked_reason = f"earnings_trend_conflict_{trend:+.4f}"
        self.last_cpi_surprise = 0.0
        self.last_cpi_bias_bp = self._clip(score * MACRO_TO_RATE_BP, -MAX_CPI_RATE_BIAS_BP, MAX_CPI_RATE_BIAS_BP)
        self.last_cpi_ms = int(now_ms)
        self.last_rate_bias_ttl_ms = MACRO_SIGNAL_TTL_MS
        signal_id = self._next_signal_id("c_macro")
        self.last_shadow_signal = CSignal(
            signal_id=signal_id,
            side=side,
            target=target,
            thesis="macro",
            strength=abs(score),
            tick=self.last_tick_seen,
            started_ms=now_ms,
            description="macro_shadow",
            blocked_reason=blocked_reason,
            live_trading_enabled=False,
        )
        return CParsedSignal(
            strategy_family="c_macro_shadow",
            action_class="shadow_blocked" if blocked_reason else "shadow_signal",
            signal_id=signal_id,
            payload={
                "signal_id": signal_id,
                "tick": self.last_tick_seen,
                "headline": text[:180],
                "score": round(score, 4),
                "target_inventory": target,
                "blocked_reason": blocked_reason,
                "live_trading_enabled": False,
            },
        )

    def _start_c_earnings_shock(
        self,
        *,
        signal_id: str,
        tick: int,
        delta: float,
        fair_before: float | None,
        fair_after: float | None,
        fair_change_ticks: float | None,
        reference_mid: float,
        target_inventory: int,
        started_ms: int,
    ) -> None:
        direction = self._sign(target_inventory)
        shock = CEarningsShockState(
            mode="SHOCK",
            signal_id=signal_id,
            direction=direction,
            target_inventory=int(target_inventory),
            original_target_inventory=int(target_inventory),
            started_ms=int(started_ms),
            tick=int(tick),
            reference_mid=float(reference_mid),
            fair_before=fair_before,
            fair_after=fair_after,
            fair_change_ticks=fair_change_ticks,
            initial_edge=0.0 if fair_after is None else float(fair_after) - float(reference_mid),
        )
        shock.post_event_mids.append((int(started_ms), float(reference_mid)))
        self.c_earnings_shock = shock
        self.active_signal = CSignal(
            signal_id=signal_id,
            side="BUY" if target_inventory > 0 else "SELL",
            target=abs(int(target_inventory)),
            thesis="earnings",
            strength=abs(float(delta)),
            tick=int(tick),
            started_ms=int(started_ms),
            description="earnings_shock",
            live_trading_enabled=True,
        )
        self.mode = "C_EARNINGS_SHOCK"
        self.last_block_reason = None

    def reset_c_earnings_shock(self) -> None:
        self.c_earnings_shock = CEarningsShockState()

    def _c_shock_scaled_target(self, edge: float, fair_change_ticks: float | None) -> int:
        edge_abs = abs(float(edge))
        if edge_abs < C_SHOCK_MIN_EDGE_TICKS:
            return 0
        confidence_span = max(1.0, C_SHOCK_FULL_CONFIDENCE_EDGE_TICKS - C_SHOCK_MIN_EDGE_TICKS)
        confidence = self._clip((edge_abs - C_SHOCK_MIN_EDGE_TICKS) / confidence_span, 0.0, 1.0)
        base_target = max(C_SHOCK_MIN_POSITION, round(C_SHOCK_POSITION_CAP * confidence))
        scaled_target = max(base_target, round(edge_abs * C_SHOCK_POSITION_SCALE))
        target_abs = min(C_SHOCK_POSITION_CAP, scaled_target)
        if fair_change_ticks is not None:
            change_abs = abs(float(fair_change_ticks))
            change_span = max(1.0, C_SHOCK_FULL_CONFIDENCE_CHANGE_TICKS - C_SHOCK_MIN_EDGE_TICKS)
            change_conf = self._clip((change_abs - C_SHOCK_MIN_EDGE_TICKS) / change_span, 0.0, 1.0)
            change_base = max(C_SHOCK_MIN_POSITION, round(C_SHOCK_POSITION_CAP * change_conf))
            change_scaled = max(change_base, round(change_abs * C_SHOCK_CHANGE_POSITION_SCALE))
            target_abs = max(target_abs, min(C_SHOCK_POSITION_CAP, change_scaled))
        return self._sign(edge) * int(target_abs)

    def _c_shock_next_slice_qty(self, remaining_target_qty: int) -> int:
        remaining = max(0, int(remaining_target_qty))
        max_legal = min(C_ORDER_LIMIT, C_SHOCK_SLICE_MAX_QTY)
        if remaining <= max_legal:
            return remaining
        slice_qty = min(max_legal, C_SHOCK_SLICE_TARGET_QTY)
        remainder = remaining - slice_qty
        if 0 < remainder < C_SHOCK_SLICE_MIN_QTY:
            slice_qty = min(max_legal, slice_qty + (C_SHOCK_SLICE_MIN_QTY - remainder))
        return max(1, slice_qty)

    def _record_c_shock_mid(self, now_ms: int) -> None:
        if self.c_earnings_shock.mode == "IDLE":
            return
        mid_c = self.mid(self.symbol)
        if mid_c is not None:
            self.c_earnings_shock.post_event_mids.append((int(now_ms), float(mid_c)))

    def _c_shock_should_emergency_dump(self, now_ms: int) -> bool:
        shock = self.c_earnings_shock
        mid_c = self.mid(self.symbol)
        if (
            shock.mode != "SHOCK"
            or shock.reference_mid is None
            or mid_c is None
            or self.inventory == 0
            or abs(int(self.inventory)) < C_SHOCK_EMERGENCY_DUMP_MIN_INVENTORY
        ):
            return False
        if int(now_ms) - int(shock.started_ms) < C_SHOCK_EMERGENCY_DUMP_MIN_MS:
            return False
        move_from_reference = float(mid_c) - float(shock.reference_mid)
        adverse_move = -move_from_reference if self.inventory > 0 else move_from_reference
        threshold = max(C_SHOCK_EMERGENCY_DUMP_TICKS, abs(float(shock.initial_edge)) * C_SHOCK_EMERGENCY_DUMP_FRACTION)
        return adverse_move >= threshold

    def _c_shock_should_force_time_flatten(self, now_ms: int) -> bool:
        shock = self.c_earnings_shock
        return shock.mode == "SHOCK" and self.inventory != 0 and int(now_ms) - int(shock.started_ms) >= C_SHOCK_MAX_HOLD_MS

    def _c_shock_stall_confirmed(self, now_ms: int) -> bool:
        cutoff = int(now_ms) - C_SHOCK_DECAY_STALL_WINDOW_MS
        recent = [mid for ts, mid in self.c_earnings_shock.post_event_mids if ts >= cutoff]
        if len(recent) < 3:
            return True
        return max(recent) - min(recent) <= C_SHOCK_DECAY_STALL_THRESHOLD_TICKS

    def _maybe_apply_c_shock_decay(self, now_ms: int) -> bool:
        shock = self.c_earnings_shock
        if shock.mode != "SHOCK" or shock.direction == 0 or shock.target_inventory == 0:
            return False
        original_abs = abs(int(shock.original_target_inventory))
        if original_abs < C_SHOCK_DECAY_MIN_INVENTORY:
            return False
        elapsed = int(now_ms) - int(shock.started_ms)
        if elapsed < C_SHOCK_DECAY_START_MS or not self._c_shock_stall_confirmed(int(now_ms)):
            return False
        steps_due = 1 + int((elapsed - C_SHOCK_DECAY_START_MS) // C_SHOCK_DECAY_INTERVAL_MS)
        if steps_due <= shock.decay_steps_applied:
            return False
        step_qty = round(original_abs * C_SHOCK_DECAY_FRACTION)
        step_qty = max(C_SHOCK_DECAY_MIN_QTY, min(C_SHOCK_DECAY_MAX_QTY, step_qty))
        residual_floor = max(C_SHOCK_MIN_POSITION, round(original_abs * C_SHOCK_DECAY_MIN_RESIDUAL_FRACTION))
        current_abs = abs(int(shock.target_inventory))
        allowed_trim = max(0, current_abs - residual_floor)
        trim_qty = min(allowed_trim, (steps_due - shock.decay_steps_applied) * step_qty)
        shock.decay_steps_applied = steps_due
        if trim_qty <= 0:
            return False
        shock.target_inventory = shock.direction * (current_abs - trim_qty)
        shock.decay_trimmed_qty_total += int(trim_qty)
        return True

    def _maybe_build_c_overshoot_trim(self, now_ms: int) -> QuotePlan | None:
        shock = self.c_earnings_shock
        mid_c = self.mid(self.symbol)
        if (
            shock.mode != "SHOCK"
            or shock.fair_after is None
            or mid_c is None
            or self.inventory == 0
            or shock.target_inventory == 0
            or shock.overshoot_stage_index >= 3
            or self.inventory * shock.direction <= 0
        ):
            return None
        threshold = self._c_shock_stage_thresholds()[shock.overshoot_stage_index]
        crossed_fair = float(mid_c) >= float(shock.fair_after) if shock.direction > 0 else float(mid_c) <= float(shock.fair_after)
        if crossed_fair and shock.overshoot_crossed_ms is None:
            shock.overshoot_crossed_ms = int(now_ms)
        beyond_fair = (
            float(mid_c) >= float(shock.fair_after) + threshold
            if shock.direction > 0
            else float(mid_c) <= float(shock.fair_after) - threshold
        )
        if not beyond_fair:
            return None
        cutoff = int(now_ms) - C_SHOCK_OVERSHOOT_HOLD_MS
        window = [(ts, mid) for ts, mid in shock.post_event_mids if ts >= cutoff]
        if len(window) < 3:
            return None
        mids = [mid for _ts, mid in window]
        if max(mids) - min(mids) > C_SHOCK_OVERSHOOT_BAND_TICKS:
            return None
        reversal_ticks = max(mids) - float(mid_c) if shock.direction > 0 else float(mid_c) - min(mids)
        if reversal_ticks < C_SHOCK_OVERSHOOT_REVERSAL_TICKS:
            return None

        original_abs = abs(int(shock.original_target_inventory))
        stage_fractions = list(C_SHOCK_OVERSHOOT_STAGE_FRACTIONS)
        residual_fraction = C_SHOCK_OVERSHOOT_MIN_RESIDUAL_FRACTION
        stage_max_qty = C_SHOCK_OVERSHOOT_STAGE_MAX_QTY
        if original_abs >= C_SHOCK_OVERSHOOT_LARGE_POSITION_THRESHOLD:
            residual_fraction = max(residual_fraction, C_SHOCK_OVERSHOOT_LARGE_RESIDUAL_FRACTION)
            if shock.overshoot_stage_index == 0:
                stage_fractions[0] = max(stage_fractions[0], C_SHOCK_OVERSHOOT_LARGE_STAGE1_FRACTION)
                stage_max_qty = max(stage_max_qty, round(original_abs * C_SHOCK_OVERSHOOT_LARGE_STAGE1_FRACTION))
        residual_floor = max(C_SHOCK_MIN_POSITION, round(original_abs * residual_fraction))
        remaining_abs = abs(int(shock.target_inventory))
        allowed_trim = max(0, remaining_abs - residual_floor)
        if allowed_trim <= 0:
            return None
        trim_qty = round(original_abs * stage_fractions[shock.overshoot_stage_index])
        trim_qty = max(C_SHOCK_OVERSHOOT_STAGE_MIN_QTY, min(stage_max_qty, trim_qty, allowed_trim))
        if trim_qty <= 0:
            return None
        new_target_abs = remaining_abs - trim_qty
        new_target = shock.direction * new_target_abs
        shock.target_inventory = new_target
        shock.overshoot_active = True
        shock.overshoot_trigger_ticks = float(threshold)
        shock.overshoot_trimmed_qty_total += abs(self.inventory - new_target)
        shock.overshoot_stage_index += 1
        return self._target_plan(
            target_inventory=new_target,
            reason="c_earnings_overshoot_trim",
            action_class="overshoot_trim",
            mode="C_EARNINGS_SHOCK",
            signal_id=shock.signal_id,
        )

    def _c_shock_stage_thresholds(self) -> tuple[int, int, int]:
        edge_abs = abs(float(self.c_earnings_shock.initial_edge))
        return (
            max(12, round(0.20 * edge_abs)),
            max(20, round(0.35 * edge_abs)),
            max(28, round(0.50 * edge_abs)),
        )

    def _c_shock_equilibrium_reached(self, now_ms: int) -> bool:
        shock = self.c_earnings_shock
        if shock.mode != "SHOCK":
            return False
        if int(now_ms) - int(shock.started_ms) < C_SHOCK_EQUILIBRIUM_MIN_MS:
            return False
        cutoff = int(now_ms) - C_SHOCK_EQUILIBRIUM_HOLD_MS
        window = [(ts, mid) for ts, mid in shock.post_event_mids if ts >= cutoff]
        if len(window) < C_SHOCK_EQUILIBRIUM_MIN_SAMPLES:
            return False
        if window[-1][0] - window[0][0] < C_SHOCK_EQUILIBRIUM_HOLD_MS:
            return False
        mids = [mid for _ts, mid in window]
        if max(mids) - min(mids) > C_SHOCK_EQUILIBRIUM_BAND_TICKS:
            return False
        if shock.fair_after is None:
            return True
        settled_mid = float(median(mids))
        current_edge = float(shock.fair_after) - settled_mid
        current_direction = self._sign(current_edge)
        residual_edge = abs(float(current_edge))
        if current_direction != shock.direction or residual_edge <= C_SHOCK_EQUILIBRIUM_RESIDUAL_EDGE_TICKS:
            return True
        if shock.reference_mid is None:
            return False
        if (
            shock.overshoot_crossed_ms is not None
            and int(now_ms) - int(shock.overshoot_crossed_ms) >= C_SHOCK_OVERSHOOT_MAX_WAIT_MS
            and shock.overshoot_stage_index == 0
        ):
            return True
        initial_edge_abs = max(1e-9, abs(float(shock.initial_edge)))
        captured_ticks = abs(float(settled_mid) - float(shock.reference_mid))
        captured_fraction = min(1.0, captured_ticks / initial_edge_abs)
        return captured_fraction >= C_SHOCK_EQUILIBRIUM_MIN_CAPTURE_FRACTION

    def _target_plan(
        self,
        *,
        target_inventory: int,
        reason: str,
        action_class: str,
        mode: str,
        signal_id: str | None,
    ) -> QuotePlan:
        wait_reason = self._c_wait_reason_before_target(int(target_inventory))
        if wait_reason is not None:
            self.c_unwind_wait_reason = wait_reason
            self.last_block_reason = wait_reason
            return QuotePlan(mode, None, None, (), True, wait_reason)
        self.c_unwind_wait_reason = None
        current_inventory = int(self.inventory)
        desired_delta = int(target_inventory) - current_inventory
        if desired_delta == 0:
            self.last_block_reason = "already_at_c_target"
            return QuotePlan(mode, None, None, (), True, "already_at_c_target")
        side = "BUY" if desired_delta > 0 else "SELL"
        px = self.best_price_for_cross(self.symbol, side)
        if px is None:
            self.last_block_reason = "missing_c_book"
            return QuotePlan(mode, None, None, (), True, "missing_c_book")
        qty = self._c_shock_next_slice_qty(abs(desired_delta))
        qty = min(qty, C_ORDER_LIMIT)
        order = DesiredOrder(
            side=side,
            px=int(px),
            qty=int(qty),
            overlay="earnings",
            aggressive=True,
            reason=reason,
            intent="c_earnings_shock",
            mode_at_submit=mode,
            evaluation_reason=reason,
            market_key="C",
            strategy_family="c_earnings",
            action_class=action_class,
            pnl_owner="c_earnings",
            signal_id=str(signal_id or ""),
            trade_group_id=str(signal_id or ""),
            leg_role="single",
        )
        self.last_block_reason = None
        return QuotePlan(mode, None, None, (order,), False, reason)

    def _c_wait_reason_before_target(self, target_inventory: int) -> str | None:
        live_buy = self.order_manager.live_order("BUY")
        live_sell = self.order_manager.live_order("SELL")
        if target_inventory == 0:
            if live_buy is not None and not live_buy.cancel_pending:
                return "waiting_for_c_cancel_before_flatten"
            if live_sell is not None and not live_sell.cancel_pending:
                return "waiting_for_c_cancel_before_flatten"
            return None
        desired_delta = int(target_inventory) - int(self.inventory)
        if desired_delta == 0:
            return None
        desired_side = "BUY" if desired_delta > 0 else "SELL"
        opposite_order = self.order_manager.live_order("SELL" if desired_side == "BUY" else "BUY")
        if opposite_order is not None and not opposite_order.cancel_pending:
            return "waiting_for_c_cancel_before_reversal"
        return None

    def _live_order_side_tag(self) -> str:
        sides = [
            side
            for side in ("BUY", "SELL")
            if (order := self.order_manager.live_order(side)) is not None and int(order.remaining_qty) > 0
        ]
        if not sides:
            return "NONE"
        if len(sides) == 2:
            return "BOTH"
        return sides[0]

    @staticmethod
    def _target_side_for_inventory(target_inventory: int | None) -> str:
        if target_inventory is None:
            return "NONE"
        if int(target_inventory) > 0:
            return "BUY"
        if int(target_inventory) < 0:
            return "SELL"
        return "FLAT"

    def best_price_for_cross(self, symbol: str, side: str) -> int | None:
        book = self.books.get(symbol) or BookSnapshot()
        if book.best_bid is None or book.best_ask is None:
            return None
        return int(book.best_ask.px) if side == "BUY" else int(book.best_bid.px)

    def book_top(self, symbol: str) -> tuple[int | None, int | None]:
        book = self.books.get(symbol) or BookSnapshot()
        return (
            None if book.best_bid is None else int(book.best_bid.px),
            None if book.best_ask is None else int(book.best_ask.px),
        )

    def mid(self, symbol: str) -> float | None:
        book = self.books.get(symbol) or BookSnapshot()
        return book.mid

    def _has_live_orders(self) -> bool:
        return any(order.get("remaining_qty", 0) for order in self.order_manager.live_orders_snapshot())

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _sign(value: float | int) -> int:
        return 1 if value > 0 else (-1 if value < 0 else 0)

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def _next_signal_id(self, prefix: str) -> str:
        self.signal_seq += 1
        return f"{prefix}_{self.signal_seq}"
