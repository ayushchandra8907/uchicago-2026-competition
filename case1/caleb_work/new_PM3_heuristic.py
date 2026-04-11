from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient

LOGGER = logging.getLogger("new-pm3-heuristic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class BotConfig:
    fed_hike: str
    fed_hold: str
    fed_cut: str

    cpi_trigger_abs: float
    cpi_medium_abs: float
    cpi_strong_abs: float
    cpi_event_cooldown_sec: float
    cpi_confirm_delay_sec: float
    cpi_micro_lookback_points: int
    cpi_model_weight: float
    cpi_micro_weight: float
    cpi_min_signal_gap: float
    cpi_relabel_margin: float
    cpi_learn_delay_sec: float
    cpi_learn_rate: float
    cpi_prior_hike: float
    cpi_prior_hold: float
    cpi_prior_cut: float
    cpi_cutover_cut_min: float
    cpi_cutover_hold_saturated: float
    cpi_strong_dovish_abs: float
    cpi_strong_dovish_hike_min: float
    cpi_max_target_abs: int

    target_light: int
    target_medium: int
    target_strong: int
    entry_initial_clip: int
    entry_reinforce_clip: int
    unwind_clip: int
    near_flat_threshold: int

    shock_max_hold_sec: float
    emergency_wrong_way_ticks: float
    emergency_min_built_qty: int
    spread_retrace_trigger_ticks: float

    max_abs_position: int
    max_order_size: int
    max_open_orders: int
    max_outstanding_volume: int

    corr_lookback_points: int
    min_corr_points: int
    min_pair_corr: float

    max_book_age_sec: float
    order_stale_sec: float
    order_action_cooldown_sec: float
    status_interval_sec: float
    loop_sleep_sec: float
    trace_enabled: bool
    trace_dir: str

    @property
    def rate_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


def load_config() -> BotConfig:
    return BotConfig(
        fed_hike=env_str("PM3_FED_HIKE_SYMBOL", "R_HIKE"),
        fed_hold=env_str("PM3_FED_HOLD_SYMBOL", "R_HOLD"),
        fed_cut=env_str("PM3_FED_CUT_SYMBOL", "R_CUT"),
        cpi_trigger_abs=env_float("PM3_CPI_TRIGGER_ABS", 0.0003),
        cpi_medium_abs=env_float("PM3_CPI_MEDIUM_ABS", 0.00055),
        cpi_strong_abs=env_float("PM3_CPI_STRONG_ABS", 0.0010),
        cpi_event_cooldown_sec=env_float("PM3_CPI_EVENT_COOLDOWN_SEC", 0.75),
        cpi_confirm_delay_sec=env_float("PM3_CPI_CONFIRM_DELAY_SEC", 0.20),
        cpi_micro_lookback_points=env_int("PM3_CPI_MICRO_LOOKBACK_POINTS", 6),
        cpi_model_weight=env_float("PM3_CPI_MODEL_WEIGHT", 1.0),
        cpi_micro_weight=env_float("PM3_CPI_MICRO_WEIGHT", 0.70),
        cpi_min_signal_gap=env_float("PM3_CPI_MIN_SIGNAL_GAP", 0.20),
        cpi_relabel_margin=env_float("PM3_CPI_RELABEL_MARGIN", 0.15),
        cpi_learn_delay_sec=env_float("PM3_CPI_LEARN_DELAY_SEC", 1.0),
        cpi_learn_rate=env_float("PM3_CPI_LEARN_RATE", 0.20),
        cpi_prior_hike=env_float("PM3_CPI_PRIOR_HIKE", 1.00),
        cpi_prior_hold=env_float("PM3_CPI_PRIOR_HOLD", -0.75),
        cpi_prior_cut=env_float("PM3_CPI_PRIOR_CUT", -0.30),
        cpi_cutover_cut_min=env_float("PM3_CPI_CUTOVER_CUT_MIN", 0.07),
        cpi_cutover_hold_saturated=env_float("PM3_CPI_CUTOVER_HOLD_SATURATED", 0.70),
        cpi_strong_dovish_abs=env_float("PM3_CPI_STRONG_DOVISH_ABS", 0.0005),
        cpi_strong_dovish_hike_min=env_float("PM3_CPI_STRONG_DOVISH_HIKE_MIN", 0.60),
        cpi_max_target_abs=env_int("PM3_CPI_MAX_TARGET_ABS", 40),
        target_light=env_int("PM3_TARGET_LIGHT", 40),
        target_medium=env_int("PM3_TARGET_MEDIUM", 40),
        target_strong=env_int("PM3_TARGET_STRONG", 40),
        entry_initial_clip=env_int("PM3_ENTRY_INITIAL_CLIP", 40),
        entry_reinforce_clip=env_int("PM3_ENTRY_REINFORCE_CLIP", 0),
        unwind_clip=env_int("PM3_UNWIND_CLIP", 40),
        near_flat_threshold=env_int("PM3_NEAR_FLAT_THRESHOLD", 2),
        shock_max_hold_sec=env_float("PM3_SHOCK_MAX_HOLD_SEC", 3.0),
        emergency_wrong_way_ticks=env_float("PM3_EMERGENCY_WRONG_WAY_TICKS", 10.0),
        emergency_min_built_qty=env_int("PM3_EMERGENCY_MIN_BUILT_QTY", 20),
        spread_retrace_trigger_ticks=env_float("PM3_SPREAD_RETRACE_TRIGGER_TICKS", 6.0),
        max_abs_position=env_int("PM3_MAX_ABS_POSITION", 200),
        max_order_size=env_int("PM3_MAX_ORDER_SIZE", 40),
        max_open_orders=env_int("PM3_MAX_OPEN_ORDERS", 50),
        max_outstanding_volume=env_int("PM3_MAX_OUTSTANDING_VOLUME", 120),
        corr_lookback_points=env_int("PM3_CORR_LOOKBACK_POINTS", 80),
        min_corr_points=env_int("PM3_MIN_CORR_POINTS", 12),
        min_pair_corr=env_float("PM3_MIN_PAIR_CORR", 0.55),
        max_book_age_sec=env_float("PM3_MAX_BOOK_AGE_SEC", 1.0),
        order_stale_sec=env_float("PM3_ORDER_STALE_SEC", 0.9),
        order_action_cooldown_sec=env_float("PM3_ORDER_ACTION_COOLDOWN_SEC", 0.20),
        status_interval_sec=env_float("PM3_STATUS_INTERVAL_SEC", 3.0),
        loop_sleep_sec=env_float("PM3_LOOP_SLEEP_SEC", 0.20),
        trace_enabled=env_bool("PM3_TRACE_ENABLED", True),
        trace_dir=env_str("PM3_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
    )


def parse_cpi(kind: Any, new_data: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
    if kind == "structured":
        subtype = str(new_data.get("structured_subtype") or "")
        if subtype == "cpi_print" and "actual" in new_data and "forecast" in new_data:
            return float(new_data["actual"]), float(new_data["forecast"]), "structured"
        return None, None, "none"

    if kind != "unstructured":
        return None, None, "none"

    content = str(new_data.get("content", "") or "")
    patterns = [
        r"cpi\s*actual\s*([+-]?\d*\.?\d+)\s*vs\.?\s*forecast\s*([+-]?\d*\.?\d+)",
        r"actual\s*cpi\s*([+-]?\d*\.?\d+)\s*vs\.?\s*forecast(?:ed)?\s*cpi\s*([+-]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2)), "unstructured_text"
    return None, None, "none"


@dataclass
class Top:
    bid: Optional[int] = None
    bid_qty: int = 0
    ask: Optional[int] = None
    ask_qty: int = 0
    updated_ts: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass
class MarketLabels:
    important_1: str
    important_2: str
    irrelevant: str
    pair_corr: float = 0.0
    updated_ts: float = field(default_factory=time.time)


@dataclass
class ShockState:
    event_id: int = 0
    winner: Optional[str] = None
    loser: Optional[str] = None
    surprise: float = 0.0
    started_ts: float = 0.0
    target_abs: int = 0
    reference_spread: Optional[float] = None
    peak_favorable_spread: Optional[float] = None
    initial_edge: float = 0.0
    baseline_mids: dict[str, float] = field(default_factory=dict)
    learned_applied: bool = False


@dataclass
class PendingSignal:
    winner: str
    loser: str
    surprise: float
    target_abs: int
    reason: str
    source: str
    score_winner: float = 0.0
    score_loser: float = 0.0
    score_constant: float = 0.0
    q_hike: float = 0.0
    q_hold: float = 0.0
    q_cut: float = 0.0
    queued_ts: float = field(default_factory=time.time)


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    side: str  # BUY | SELL
    px: int
    qty: int
    remaining_qty: int
    submitted_ts: float
    aggressive: bool = False
    cancel_pending: bool = False
    reason: str = ""


class SymbolOrderManager:
    """Saavan-style one-live-order-per-side manager for a single symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.orders: dict[str, ManagedOrder] = {}
        self.live_by_side: dict[str, Optional[str]] = {"BUY": None, "SELL": None}
        self.last_action_ts: dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

    def live_order(self, side: str) -> Optional[ManagedOrder]:
        order_id = self.live_by_side.get(side)
        if order_id is None:
            return None
        return self.orders.get(order_id)

    def has_active(self, side: Optional[str] = None) -> bool:
        if side is not None:
            order = self.live_order(side)
            return bool(order and not order.cancel_pending)
        for side_name in ("BUY", "SELL"):
            order = self.live_order(side_name)
            if order and not order.cancel_pending:
                return True
        return False

    def can_take_action(self, side: str, now_ts: float, cooldown_sec: float) -> bool:
        return (now_ts - self.last_action_ts.get(side, 0.0)) >= cooldown_sec

    def note_submitted(
        self,
        *,
        order_id: str,
        side: str,
        px: int,
        qty: int,
        now_ts: float,
        aggressive: bool,
        reason: str,
    ) -> ManagedOrder:
        order = ManagedOrder(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            px=int(px),
            qty=int(qty),
            remaining_qty=int(qty),
            submitted_ts=float(now_ts),
            aggressive=bool(aggressive),
            cancel_pending=False,
            reason=reason,
        )
        self.orders[order_id] = order
        self.live_by_side[side] = order_id
        self.last_action_ts[side] = now_ts
        return order

    def mark_cancel_requested(self, order_id: str, now_ts: float) -> Optional[ManagedOrder]:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.cancel_pending = True
        self.last_action_ts[order.side] = now_ts
        return order

    def handle_fill(self, order_id: str, qty: int) -> Optional[ManagedOrder]:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.remaining_qty = max(0, order.remaining_qty - int(qty))
        if order.remaining_qty <= 0:
            self._drop_order(order_id)
        return order

    def handle_cancel_response(self, order_id: str, success: bool) -> Optional[ManagedOrder]:
        order = self.orders.get(order_id)
        if order is None:
            return None
        if success:
            self._drop_order(order_id)
            return order
        order.cancel_pending = False
        return order

    def handle_rejection(self, order_id: str) -> Optional[ManagedOrder]:
        order = self.orders.get(order_id)
        if order is None:
            return None
        self._drop_order(order_id)
        return order

    def stale_order_ids(self, now_ts: float, stale_sec: float) -> list[str]:
        stale: list[str] = []
        for side_name in ("BUY", "SELL"):
            order = self.live_order(side_name)
            if order is None or order.cancel_pending:
                continue
            if (now_ts - order.submitted_ts) >= stale_sec:
                stale.append(order.order_id)
        return stale

    def _drop_order(self, order_id: str) -> None:
        order = self.orders.pop(order_id, None)
        if order is None:
            return
        if self.live_by_side.get(order.side) == order_id:
            self.live_by_side[order.side] = None

class NewPM3HeuristicClient(XChangeClient):
    def __init__(self, host: str, username: str, password: str, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or load_config()
        super().__init__(host, username, password, silent=False, symbols=list(self.cfg.rate_symbols))

        self.mode: str = "IDLE"  # IDLE | SHOCK | UNWIND
        self.startup_flatten_complete = False
        self.current_tick: Optional[int] = None
        self.last_status_ts = 0.0
        self.last_event_ts = 0.0
        self.event_counter = 0

        self.books: dict[str, Top] = {sym: Top() for sym in self.cfg.rate_symbols}
        self.mid_hist: dict[str, deque[float]] = {
            sym: deque(maxlen=max(16, self.cfg.corr_lookback_points)) for sym in self.cfg.rate_symbols
        }
        self.labels = MarketLabels(
            important_1=self.cfg.fed_hike,
            important_2=self.cfg.fed_hold,
            irrelevant=self.cfg.fed_cut,
            pair_corr=0.0,
        )
        self.response_beta: dict[str, float] = {
            self.cfg.fed_hike: float(self.cfg.cpi_prior_hike),
            self.cfg.fed_hold: float(self.cfg.cpi_prior_hold),
            self.cfg.fed_cut: float(self.cfg.cpi_prior_cut),
        }

        self.order_managers: dict[str, SymbolOrderManager] = {
            sym: SymbolOrderManager(sym) for sym in self.cfg.rate_symbols
        }
        self.order_to_symbol: dict[str, str] = {}
        self.shock = ShockState()
        self.pending_signal: Optional[PendingSignal] = None

        self.entry_count = 0
        self.unwind_count = 0
        self.reject_count = 0
        self.recovery_cancel_count = 0

        self._trace_fp = None
        self._trace_path: Optional[Path] = None
        self._position_snapshot_seen = asyncio.Event()
        self._quote_lock = asyncio.Lock()

    @staticmethod
    def _sign(value: float) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    @staticmethod
    def side_name(side: Side) -> str:
        return "BUY" if side == Side.BUY else "SELL"

    def side_from_name(self, side_name: str) -> Side:
        return Side.BUY if side_name == "BUY" else Side.SELL

    def manager_for_order(self, order_id: str) -> Optional[SymbolOrderManager]:
        symbol = self.order_to_symbol.get(str(order_id))
        if symbol is None:
            return None
        return self.order_managers.get(symbol)

    def _trace(self, event_type: str, **payload: Any) -> None:
        if not self.cfg.trace_enabled:
            return
        if self._trace_fp is None:
            trace_dir = Path(self.cfg.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = trace_dir / f"new_PM3_heuristic_{int(time.time())}.jsonl"
            self._trace_fp = self._trace_path.open("a", encoding="utf-8")
        row = {"event_type": event_type, "ts": time.time(), **payload}
        self._trace_fp.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._trace_fp.flush()

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def total_open_orders(self) -> int:
        return len(self.open_orders)

    def outstanding_volume(self) -> int:
        total = 0
        for _, order_data in self.open_orders.items():
            if order_data:
                total += int(order_data[1])
        return total

    def refresh_book(self, symbol: str) -> Top:
        book = self.order_books.get(symbol)
        bids = []
        asks = []
        if book is not None:
            bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
            asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]
        best_bid = max(bids, key=lambda lvl: lvl[0]) if bids else None
        best_ask = min(asks, key=lambda lvl: lvl[0]) if asks else None
        top = Top(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.books[symbol] = top
        if top.mid is not None:
            self.mid_hist[symbol].append(top.mid)
        return top

    def top(self, symbol: str) -> Top:
        top = self.books.get(symbol)
        if top is None or (time.time() - top.updated_ts) > self.cfg.max_book_age_sec:
            return self.refresh_book(symbol)
        return top

    def has_open_order_on(self, symbol: str) -> bool:
        manager = self.order_managers.get(symbol)
        if manager is not None and manager.has_active():
            return True
        for _, order_data in self.open_orders.items():
            if order_data and str(order_data[0]) == symbol:
                return True
        return False

    def returns(self, symbol: str) -> list[float]:
        mids = list(self.mid_hist[symbol])
        if len(mids) < 3:
            return []
        rets = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        return rets[-self.cfg.corr_lookback_points :]

    def pearson(self, a: list[float], b: list[float]) -> Optional[float]:
        n = min(len(a), len(b))
        if n < self.cfg.min_corr_points:
            return None
        a = a[-n:]
        b = b[-n:]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((x - mean_b) ** 2 for x in b)
        if var_a <= 1e-9 or var_b <= 1e-9:
            return None
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        return cov / ((var_a * var_b) ** 0.5)

    def update_labels(self) -> None:
        syms = list(self.cfg.rate_symbols)
        pairs = [(syms[0], syms[1]), (syms[0], syms[2]), (syms[1], syms[2])]
        best_pair: Optional[tuple[str, str]] = None
        best_corr = -2.0
        for a, b in pairs:
            corr = self.pearson(self.returns(a), self.returns(b))
            if corr is None:
                continue
            if corr > best_corr:
                best_corr = corr
                best_pair = (a, b)
        if best_pair is None or best_corr < self.cfg.min_pair_corr:
            return
        a, b = best_pair
        c = next(sym for sym in syms if sym not in {a, b})
        self.labels = MarketLabels(important_1=a, important_2=b, irrelevant=c, pair_corr=best_corr, updated_ts=time.time())

    def choose_counterpart(self, winner_symbol: str) -> str:
        imp_a = self.labels.important_1
        imp_b = self.labels.important_2
        if winner_symbol == imp_a:
            return imp_b
        if winner_symbol == imp_b:
            return imp_a
        corr_a = self.pearson(self.returns(winner_symbol), self.returns(imp_a))
        corr_b = self.pearson(self.returns(winner_symbol), self.returns(imp_b))
        score_a = -2.0 if corr_a is None else corr_a
        score_b = -2.0 if corr_b is None else corr_b
        return imp_a if score_a >= score_b else imp_b

    def recent_mid_move(self, symbol: str, points: int) -> float:
        mids = list(self.mid_hist[symbol])
        if len(mids) < 2:
            return 0.0
        lookback = max(1, min(points, len(mids) - 1))
        return float(mids[-1] - mids[-1 - lookback])

    def current_probabilities(self) -> Optional[dict[str, float]]:
        mids: dict[str, float] = {}
        for sym in self.cfg.rate_symbols:
            mid = self.top(sym).mid
            if mid is None:
                return None
            mids[sym] = float(mid)
        total = sum(mids.values())
        if total <= 1e-6:
            return None
        return {sym: mids[sym] / total for sym in self.cfg.rate_symbols}

    def choose_cpi_pair_math(self, surprise: float, probs: dict[str, float]) -> tuple[str, str, str, float, float, float]:
        q_hike = float(probs.get(self.cfg.fed_hike, 0.0))
        q_hold = float(probs.get(self.cfg.fed_hold, 0.0))
        q_cut = float(probs.get(self.cfg.fed_cut, 0.0))

        if surprise > 0:
            # Empirical + case-consistent: hotter CPI pushes hike odds up.
            winner = self.cfg.fed_hike
            loser = self.cfg.fed_hold if q_hold >= q_cut else self.cfg.fed_cut
            constant = self.cfg.fed_cut if loser == self.cfg.fed_hold else self.cfg.fed_hold
            return winner, loser, constant, q_hike, q_hold, q_cut

        # Dovish CPI edge-case map fitted to historical PM logs:
        # CUT tends to lead in 3 regimes:
        # 1) CUT already meaningful
        # 2) HOLD saturated
        # 3) strong dovish surprise while HIKE is still dominant (tail repricing into CUT)
        cutover = (
            (q_cut >= self.cfg.cpi_cutover_cut_min)
            or (q_hold >= self.cfg.cpi_cutover_hold_saturated)
            or (
                abs(surprise) >= self.cfg.cpi_strong_dovish_abs
                and q_hike >= self.cfg.cpi_strong_dovish_hike_min
            )
        )
        if cutover:
            winner = self.cfg.fed_cut
            loser = self.cfg.fed_hold
            constant = self.cfg.fed_hike
            return winner, loser, constant, q_hike, q_hold, q_cut

        winner = self.cfg.fed_hold
        loser = self.cfg.fed_hike
        constant = self.cfg.fed_cut
        return winner, loser, constant, q_hike, q_hold, q_cut

    def build_signal_from_cpi(self, surprise: float, source: str) -> Optional[PendingSignal]:
        probs = self.current_probabilities()
        if probs is None:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="cpi_skip_missing_probabilities",
                surprise=surprise,
            )
            return None

        winner, loser, constant, q_hike, q_hold, q_cut = self.choose_cpi_pair_math(surprise, probs)

        # Confidence score: larger surprise + stronger regime separation.
        direction_score = max(0.10, min(1.0, abs(surprise) / max(self.cfg.cpi_trigger_abs, 1e-9)))
        regime_score = abs((q_hike - q_hold) if surprise < 0 else (q_hike - max(q_hold, q_cut)))
        score_gap = direction_score + regime_score
        if score_gap < self.cfg.cpi_min_signal_gap:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="cpi_skip_low_confidence",
                surprise=surprise,
                score_gap=score_gap,
                probs=probs,
            )
            return None

        self.labels = MarketLabels(
            important_1=winner,
            important_2=loser,
            irrelevant=constant,
            pair_corr=self.labels.pair_corr,
            updated_ts=time.time(),
        )

        raw_target = self.desired_target_abs(abs(surprise))
        capped_target = min(raw_target, self.cfg.cpi_max_target_abs)
        confidence_scale = max(0.40, min(1.0, score_gap))
        target_abs = max(self.cfg.entry_initial_clip, int(round(raw_target * confidence_scale)))
        target_abs = min(target_abs, capped_target)
        direction_name = "positive" if surprise > 0 else "negative"
        return PendingSignal(
            winner=winner,
            loser=loser,
            surprise=surprise,
            target_abs=target_abs,
            reason=f"cpi_math_{direction_name}_{winner}_over_{loser}",
            source=source,
            score_winner=score_gap,
            score_loser=-score_gap,
            score_constant=0.0,
            q_hike=q_hike,
            q_hold=q_hold,
            q_cut=q_cut,
        )

    def desired_target_abs(self, abs_surprise: float) -> int:
        if abs_surprise >= self.cfg.cpi_strong_abs:
            return self.cfg.target_strong
        if abs_surprise >= self.cfg.cpi_medium_abs:
            return self.cfg.target_medium
        return self.cfg.target_light

    def pair_spread(self, winner: str, loser: str) -> Optional[float]:
        win_mid = self.top(winner).mid
        lose_mid = self.top(loser).mid
        if win_mid is None or lose_mid is None:
            return None
        return float(win_mid - lose_mid)

    def signed_progress(self) -> Optional[float]:
        if self.shock.winner is None or self.shock.loser is None:
            return None
        spread = self.pair_spread(self.shock.winner, self.shock.loser)
        if spread is None or self.shock.reference_spread is None:
            return None
        return float(spread - self.shock.reference_spread)

    def update_peak_progress(self) -> None:
        progress = self.signed_progress()
        if progress is None:
            return
        if self.shock.peak_favorable_spread is None:
            self.shock.peak_favorable_spread = progress
            return
        self.shock.peak_favorable_spread = max(self.shock.peak_favorable_spread, progress)

    def clip_qty(self, symbol: str, side: Side, requested: int) -> int:
        requested = max(0, min(int(requested), self.cfg.max_order_size))
        if requested <= 0:
            return 0
        if self.total_open_orders() >= self.cfg.max_open_orders:
            return 0
        if self.outstanding_volume() >= self.cfg.max_outstanding_volume:
            return 0
        pos = self.get_position(symbol)
        if side == Side.BUY:
            room = self.cfg.max_abs_position - max(0, pos)
        else:
            room = self.cfg.max_abs_position - max(0, -pos)
        if room <= 0:
            return 0
        qty = min(requested, room)
        if self.outstanding_volume() + qty > self.cfg.max_outstanding_volume:
            qty = max(0, self.cfg.max_outstanding_volume - self.outstanding_volume())
        return qty

    def compute_order_price(self, symbol: str, side: Side, aggressive: bool) -> Optional[int]:
        top = self.top(symbol)
        if side == Side.BUY:
            if top.ask is None:
                return None
            if aggressive:
                return int(top.ask)
            return int(max(top.bid + 1 if top.bid is not None else top.ask, top.ask - 1))
        if top.bid is None:
            return None
        if aggressive:
            return int(top.bid)
        return int(min(top.ask - 1 if top.ask is not None else top.bid, top.bid + 1))

    async def submit_order(self, symbol: str, side: Side, qty: int, aggressive: bool, reason: str = "") -> bool:
        if qty <= 0:
            return False
        side_name = self.side_name(side)
        manager = self.order_managers[symbol]
        now_ts = time.time()
        if not manager.can_take_action(side_name, now_ts, self.cfg.order_action_cooldown_sec):
            return False
        if manager.has_active(side_name):
            return False
        px = self.compute_order_price(symbol, side, aggressive)
        if px is None:
            return False
        order_id = await self.place_order(symbol, int(qty), side, int(px))
        if order_id is None:
            return False
        oid = str(order_id)
        manager.note_submitted(
            order_id=oid,
            side=side_name,
            px=int(px),
            qty=int(qty),
            now_ts=now_ts,
            aggressive=aggressive,
            reason=reason,
        )
        self.order_to_symbol[oid] = symbol
        self._trace(
            "order_submitted",
            tick=self.current_tick,
            mode=self.mode,
            order_id=oid,
            symbol=symbol,
            side=side_name,
            px=int(px),
            qty=int(qty),
            aggressive=aggressive,
            reason=reason,
        )
        return True

    async def place_pair_toward_target(self, target_abs: int, clip: int, reason: str) -> bool:
        winner = self.shock.winner
        loser = self.shock.loser
        if winner is None or loser is None:
            return False
        if self.total_open_orders() > 0:
            return False

        win_pos = self.get_position(winner)
        lose_pos = self.get_position(loser)
        win_delta = max(0, target_abs - max(0, win_pos))
        lose_delta = max(0, target_abs - max(0, -lose_pos))
        qty = min(win_delta, lose_delta, clip)
        if qty <= 0:
            return False

        buy_qty = self.clip_qty(winner, Side.BUY, qty)
        sell_qty = self.clip_qty(loser, Side.SELL, qty)
        pair_qty = min(buy_qty, sell_qty)
        if pair_qty <= 0:
            self._trace("decision", tick=self.current_tick, mode=self.mode, reason="blocked_risk_or_limits", target_abs=target_abs)
            return False

        placed_buy = await self.submit_order(winner, Side.BUY, pair_qty, aggressive=True, reason=reason)
        placed_sell = await self.submit_order(loser, Side.SELL, pair_qty, aggressive=True, reason=reason)

        if placed_buy != placed_sell:
            orphan_symbol = winner if placed_buy else loser
            manager = self.order_managers[orphan_symbol]
            side_name = "BUY" if placed_buy else "SELL"
            live = manager.live_order(side_name)
            if live is not None and not live.cancel_pending:
                tracked = manager.mark_cancel_requested(live.order_id, time.time())
                if tracked is not None:
                    try:
                        await self.cancel_order(tracked.order_id)
                        self._trace(
                            "decision",
                            tick=self.current_tick,
                            mode=self.mode,
                            reason="pair_leg_submit_mismatch_cancel_orphan",
                            orphan_order_id=tracked.order_id,
                            orphan_symbol=orphan_symbol,
                            orphan_side=tracked.side,
                        )
                    except Exception:
                        pass
        acted = placed_buy or placed_sell
        if acted:
            self.entry_count += 1
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason=reason,
                winner=winner,
                loser=loser,
                qty=pair_qty,
                target_abs=target_abs,
                positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
            )
        return acted

    async def unwind_all(self, reason: str) -> bool:
        if self.total_open_orders() > 0:
            return False
        acted = False
        for symbol in self.cfg.rate_symbols:
            pos = self.get_position(symbol)
            if abs(pos) <= self.cfg.near_flat_threshold:
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = self.clip_qty(symbol, side, min(abs(pos), self.cfg.unwind_clip))
            if qty <= 0:
                continue
            placed = await self.submit_order(symbol, side, qty, aggressive=True, reason=reason)
            acted = acted or placed
        if acted:
            self.unwind_count += 1
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason=reason,
                positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
            )
        return acted

    async def maybe_cancel_stale_orders(self) -> None:
        now_ts = time.time()
        for symbol in self.cfg.rate_symbols:
            manager = self.order_managers[symbol]
            for side_name in ("BUY", "SELL"):
                order = manager.live_order(side_name)
                if order is None or order.cancel_pending:
                    continue
                is_stale = (now_ts - order.submitted_ts) >= self.cfg.order_stale_sec
                if not is_stale:
                    continue
                if not manager.can_take_action(side_name, now_ts, self.cfg.order_action_cooldown_sec):
                    continue
                tracked = manager.mark_cancel_requested(order.order_id, now_ts)
                if tracked is None:
                    continue
                try:
                    await self.cancel_order(order.order_id)
                    self._trace(
                        "decision",
                        tick=self.current_tick,
                        mode=self.mode,
                        reason="stale_cancel",
                        order_id=order.order_id,
                        symbol=symbol,
                        side=tracked.side,
                    )
                except Exception:
                    pass

    async def cancel_unknown_open_orders(self) -> None:
        now_ts = time.time()
        for order_id in list(self.open_orders.keys()):
            oid = str(order_id)
            if oid in self.order_to_symbol:
                continue
            try:
                await self.cancel_order(order_id)
                self.recovery_cancel_count += 1
                self._trace(
                    "decision",
                    tick=self.current_tick,
                    mode=self.mode,
                    reason="startup_recovery_cancel_unknown_order",
                    order_id=oid,
                    open_orders=len(self.open_orders),
                    ts=now_ts,
                )
            except Exception:
                pass

    def is_flat(self) -> bool:
        return all(abs(self.get_position(s)) <= self.cfg.near_flat_threshold for s in self.cfg.rate_symbols)

    def start_shock(self, signal: PendingSignal) -> None:
        self.event_counter += 1
        self.mode = "SHOCK"
        baseline_mids = {}
        for symbol in self.cfg.rate_symbols:
            mid = self.top(symbol).mid
            if mid is not None:
                baseline_mids[symbol] = float(mid)
        self.shock = ShockState(
            event_id=self.event_counter,
            winner=signal.winner,
            loser=signal.loser,
            surprise=signal.surprise,
            started_ts=time.time(),
            target_abs=signal.target_abs,
            reference_spread=self.pair_spread(signal.winner, signal.loser),
            peak_favorable_spread=0.0,
            initial_edge=0.0,
            baseline_mids=baseline_mids,
            learned_applied=False,
        )
        self.pending_signal = None
        self.last_event_ts = time.time()
        self._trace(
            "shock_start",
            tick=self.current_tick,
            event_id=self.shock.event_id,
            winner=self.shock.winner,
            loser=self.shock.loser,
            surprise=self.shock.surprise,
            target_abs=self.shock.target_abs,
            reference_spread=self.shock.reference_spread,
            signal_reason=signal.reason,
            score_winner=signal.score_winner,
            score_loser=signal.score_loser,
            score_constant=signal.score_constant,
            q_hike=signal.q_hike,
            q_hold=signal.q_hold,
            q_cut=signal.q_cut,
            response_beta=self.response_beta,
            labels=vars(self.labels),
        )

    def should_emergency_unwind(self) -> bool:
        if self.mode != "SHOCK":
            return False
        winner = self.shock.winner
        loser = self.shock.loser
        if winner is None or loser is None:
            return False
        built_qty = min(max(0, self.get_position(winner)), max(0, -self.get_position(loser)))
        if built_qty < self.cfg.emergency_min_built_qty:
            return False
        progress = self.signed_progress()
        if progress is None:
            return False
        return progress <= -self.cfg.emergency_wrong_way_ticks

    def should_retrace_unwind(self) -> bool:
        if self.mode != "SHOCK":
            return False
        progress = self.signed_progress()
        if progress is None:
            return False
        peak = self.shock.peak_favorable_spread
        if peak is None:
            return False
        retrace = peak - progress
        return peak > 0 and retrace >= self.cfg.spread_retrace_trigger_ticks

    def should_time_unwind(self) -> bool:
        if self.mode != "SHOCK":
            return False
        return (time.time() - self.shock.started_ts) >= self.cfg.shock_max_hold_sec

    def maybe_apply_cpi_learning(self) -> None:
        if self.mode != "SHOCK":
            return
        if self.shock.learned_applied:
            return
        if (time.time() - self.shock.started_ts) < self.cfg.cpi_learn_delay_sec:
            return
        direction = self._sign(self.shock.surprise)
        if direction == 0:
            self.shock.learned_applied = True
            return

        updates: dict[str, float] = {}
        for symbol in self.cfg.rate_symbols:
            base_mid = self.shock.baseline_mids.get(symbol)
            now_mid = self.top(symbol).mid
            if base_mid is None or now_mid is None:
                continue
            observed = direction * ((float(now_mid) - float(base_mid)) / 100.0)
            prior = float(self.response_beta.get(symbol, 0.0))
            learned = (1.0 - self.cfg.cpi_learn_rate) * prior + (self.cfg.cpi_learn_rate * observed)
            self.response_beta[symbol] = learned
            updates[symbol] = learned
        self.shock.learned_applied = True
        self._trace(
            "cpi_model_update",
            tick=self.current_tick,
            event_id=self.shock.event_id,
            surprise=self.shock.surprise,
            updated_betas=updates,
            response_beta=self.response_beta,
        )

    async def _evaluate_and_sync(self, reason: str) -> None:
        if not self.connected or not self._position_snapshot_seen.is_set():
            return
        async with self._quote_lock:
            await self.evaluate_state()
            self._trace(
                "cycle",
                tick=self.current_tick,
                reason=reason,
                mode=self.mode,
                startup_flatten_complete=self.startup_flatten_complete,
                pending_signal=None if self.pending_signal is None else vars(self.pending_signal),
                labels=vars(self.labels),
                positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
                open_orders=len(self.open_orders),
            )

    async def evaluate_state(self) -> None:
        for symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
        self.update_labels()
        await self.maybe_cancel_stale_orders()

        if not self.startup_flatten_complete:
            if self.total_open_orders() > 0:
                await self.cancel_unknown_open_orders()
            if self.is_flat() and self.total_open_orders() == 0:
                self.startup_flatten_complete = True
                self.mode = "IDLE"
                self._trace("transition", tick=self.current_tick, reason="startup_flatten_complete")
                return
            self.mode = "UNWIND"
            await self.unwind_all("startup_flatten")
            return

        if self.mode == "IDLE":
            if self.pending_signal is not None:
                if (time.time() - self.pending_signal.queued_ts) < self.cfg.cpi_confirm_delay_sec:
                    return
                if self.total_open_orders() > 0 or not self.is_flat():
                    self.mode = "UNWIND"
                    return
                self.start_shock(self.pending_signal)
                await self.place_pair_toward_target(self.shock.target_abs, self.cfg.entry_initial_clip, "shock_initial_entry")
            return

        if self.mode == "SHOCK":
            self.update_peak_progress()
            self.maybe_apply_cpi_learning()
            if self.should_emergency_unwind():
                self.mode = "UNWIND"
                self._trace("transition", tick=self.current_tick, reason="emergency_wrong_way", event_id=self.shock.event_id)
                return
            if self.should_retrace_unwind():
                self.mode = "UNWIND"
                self._trace("transition", tick=self.current_tick, reason="spread_retrace", event_id=self.shock.event_id)
                return
            if self.should_time_unwind():
                self.mode = "UNWIND"
                self._trace("transition", tick=self.current_tick, reason="max_hold", event_id=self.shock.event_id)
                return
            if self.total_open_orders() > 0:
                await self.cancel_unknown_open_orders()
                return
            await self.place_pair_toward_target(self.shock.target_abs, self.cfg.entry_reinforce_clip, "shock_reinforce")
            return

        # UNWIND
        if self.is_flat() and self.total_open_orders() == 0:
            self.mode = "IDLE"
            self._trace("transition", tick=self.current_tick, reason="unwind_complete", event_id=self.shock.event_id)
            self.shock = ShockState()
            if self.pending_signal is not None:
                if (time.time() - self.pending_signal.queued_ts) < self.cfg.cpi_confirm_delay_sec:
                    return
                self.start_shock(self.pending_signal)
                await self.place_pair_toward_target(self.shock.target_abs, self.cfg.entry_initial_clip, "shock_takeover_initial_entry")
            return
        if self.total_open_orders() > 0:
            await self.cancel_unknown_open_orders()
            return
        await self.unwind_all("unwind_to_flat")

    def print_status(self) -> None:
        now = time.time()
        if now - self.last_status_ts < self.cfg.status_interval_sec:
            return
        self.last_status_ts = now
        spread = None
        if self.shock.winner and self.shock.loser:
            spread = self.pair_spread(self.shock.winner, self.shock.loser)
        print(
            "[PM3] "
            f"mode={self.mode} startup_flat={self.startup_flatten_complete} labels=({self.labels.important_1},{self.labels.important_2},{self.labels.irrelevant}) corr={self.labels.pair_corr:.3f} "
            f"shock=({self.shock.winner},{self.shock.loser}) target={self.shock.target_abs} spread={spread} peak={self.shock.peak_favorable_spread} "
            f"beta=({self.response_beta.get(self.cfg.fed_hike,0.0):.2f},{self.response_beta.get(self.cfg.fed_hold,0.0):.2f},{self.response_beta.get(self.cfg.fed_cut,0.0):.2f}) "
            f"entries={self.entry_count} unwinds={self.unwind_count} rejects={self.reject_count} startup_cancels={self.recovery_cancel_count} "
            f"open_orders={len(self.open_orders)} "
            f"pos={{'{self.cfg.fed_hike}':{self.get_position(self.cfg.fed_hike)},'{self.cfg.fed_hold}':{self.get_position(self.cfg.fed_hold)},'{self.cfg.fed_cut}':{self.get_position(self.cfg.fed_cut)}}}"
        )

    async def on_cpi_signal(self, surprise: float, source: str) -> None:
        if abs(surprise) <= self.cfg.cpi_trigger_abs:
            return
        now = time.time()
        if now - self.last_event_ts < self.cfg.cpi_event_cooldown_sec:
            self._trace("decision", tick=self.current_tick, mode=self.mode, reason="event_cooldown_skip", surprise=surprise)
            return

        signal = self.build_signal_from_cpi(surprise, source)
        if signal is None:
            return

        self.pending_signal = signal
        if self.mode == "IDLE" and self.is_flat() and self.total_open_orders() == 0 and self.startup_flatten_complete:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="queued_signal_for_confirm",
                pending=vars(signal),
            )
            return

        if self.mode != "UNWIND":
            self.mode = "UNWIND"
        self._trace(
            "transition",
            tick=self.current_tick,
            mode=self.mode,
            reason="new_event_takeover_queue",
            surprise=surprise,
            pending=vars(signal),
        )

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        if not self._position_snapshot_seen.is_set():
            self._position_snapshot_seen.set()
            LOGGER.info("PM3 received initial position snapshot; enabling strategy loop.")
            self._trace(
                "position_snapshot",
                tick=self.current_tick,
                positions={sym: self.get_position(sym) for sym in self.cfg.rate_symbols},
                cash=int(self.positions.get("cash", 0)),
            )

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            self.update_labels()
            await self._evaluate_and_sync(f"book_update:{symbol}")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            self.update_labels()
            await self._evaluate_and_sync(f"trade:{symbol}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        oid = str(order_id)
        manager = self.manager_for_order(oid)
        if manager is not None:
            tracked = manager.handle_fill(oid, int(qty))
            if tracked is not None and tracked.remaining_qty <= 0:
                self.order_to_symbol.pop(oid, None)
        self._trace("fill", tick=self.current_tick, order_id=oid, qty=qty, price=price)
        await self._evaluate_and_sync("order_fill")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        oid = str(order_id)
        manager = self.manager_for_order(oid)
        if manager is not None:
            manager.handle_rejection(oid)
            self.order_to_symbol.pop(oid, None)
        self.reject_count += 1
        self._trace("reject", tick=self.current_tick, order_id=oid, reason=reason)
        await self._evaluate_and_sync("order_rejected")

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        oid = str(order_id)
        manager = self.manager_for_order(oid)
        if manager is not None:
            manager.handle_cancel_response(oid, bool(success))
            if success:
                self.order_to_symbol.pop(oid, None)
        self._trace("cancel", tick=self.current_tick, order_id=oid, success=success, error=error)
        await self._evaluate_and_sync("cancel_response")

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")
        actual, forecast, source = parse_cpi(kind, new_data)
        if actual is None or forecast is None:
            return
        surprise = actual - forecast
        self._trace(
            "news",
            tick=self.current_tick,
            kind=kind,
            source=source,
            actual=actual,
            forecast=forecast,
            surprise=surprise,
            mode=self.mode,
            labels=vars(self.labels),
        )
        await self.on_cpi_signal(surprise, source)
        await self._evaluate_and_sync("news")

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        self.current_tick = tick
        self._trace(
            "round_end",
            tick=tick,
            winning_symbol=winning_symbol,
            positions={sym: self.get_position(sym) for sym in self.cfg.rate_symbols},
        )

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        self.current_tick = tick
        self._trace("payout", tick=tick, user=user, amount=amount)

    async def trade(self):
        await self._position_snapshot_seen.wait()
        while True:
            try:
                await self._evaluate_and_sync("timer")
                self.print_status()
            except Exception as exc:
                self._trace("loop_error", tick=self.current_tick, error=repr(exc))
            await asyncio.sleep(self.cfg.loop_sleep_sec)

    async def start(self):
        asyncio.create_task(self.trade())
        while True:
            try:
                await self.connect()
                break
            except EOFError:
                LOGGER.warning("PM3 stream ended with EOF; reconnecting in 1.0s")
                await asyncio.sleep(1.0)


async def main():
    client = NewPM3HeuristicClient(
        env_str("UTC_HOST", "34.197.188.76:3333"),
        env_str("UTC_USERNAME", "uiuc"),
        env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
