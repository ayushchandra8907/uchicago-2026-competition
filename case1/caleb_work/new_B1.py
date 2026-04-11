from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient


OPTION_RE = re.compile(r"^B_(C|P)_(\d+)$")
DEFAULT_STRIKES = (950, 1000, 1050)


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


def live_bid(book) -> tuple[Optional[int], int]:
    if book is None:
        return None, 0
    levels = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
    if not levels:
        return None, 0
    return max(levels, key=lambda level: level[0])


def live_ask(book) -> tuple[Optional[int], int]:
    if book is None:
        return None, 0
    levels = [(int(px), int(qty)) for px, qty in book.asks.items() if qty > 0]
    if not levels:
        return None, 0
    return min(levels, key=lambda level: level[0])


def mid_from_book(book, fallback: Optional[float] = None) -> Optional[float]:
    bid, _ = live_bid(book)
    ask, _ = live_ask(book)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return fallback


@dataclass(frozen=True)
class BotConfig:
    host: str
    username: str
    password: str
    hard_max_order_size: int
    hard_max_open_orders: int
    hard_max_outstanding_volume: int
    hard_max_absolute_position: int
    soft_b_inventory_cap: int
    soft_option_inventory_cap: int
    soft_outstanding_volume: int
    soft_open_orders: int
    min_band_leg_qty: int
    outlier_ticks: int
    enter_edge_ticks: int
    exit_buffer_ticks: int
    edge_decay_exit_ticks: int
    strong_cross_ticks: int
    entry_ttl_sec: float
    exit_ttl_sec: float
    entry_cooldown_sec: float
    max_hold_sec: float
    conv_rev_min_edge_ticks: int
    conv_rev_size: int
    conv_rev_cooldown_sec: float
    post_fill_keep_edge: int
    poll_sec: float
    status_sec: float
    capture_enabled: bool
    capture_dir: str
    rebound_enabled: bool
    rebound_lookback_sec: float
    rebound_moderate_b_mid: float
    rebound_strong_b_mid: float
    rebound_moderate_shock_ticks: float
    rebound_strong_shock_ticks: float
    rebound_symbol_moderate: str
    rebound_symbol_strong: str
    rebound_max_qty: int
    rebound_entry_cooldown_sec: float
    rebound_max_attempts_per_round: int
    rebound_base_tp_ticks: float
    rebound_base_sl_ticks: float
    rebound_tp_spread_mult: float
    rebound_sl_spread_mult: float
    rebound_max_spread: int
    rebound_iv_proxy_max: float
    rebound_max_hold_sec: float
    rebound_force_exit_after_sec: float
    rebound_min_spread_for_passive: int


@dataclass(frozen=True)
class OptionPair:
    strike: int
    call_symbol: str
    put_symbol: str


@dataclass
class OrderMeta:
    symbol: str
    side: Side
    price: int
    qty: int
    purpose: str
    strategy: str
    created_at: float
    strike: Optional[int] = None
    role: Optional[str] = None
    bundle_id: Optional[str] = None
    cancel_pending: bool = False


@dataclass(frozen=True)
class StrikeAnalysis:
    strike: int
    call_symbol: str
    put_symbol: str
    call_bid: Optional[int]
    call_bid_qty: int
    call_ask: Optional[int]
    call_ask_qty: int
    put_bid: Optional[int]
    put_bid_qty: int
    put_ask: Optional[int]
    put_ask_qty: int
    call_mid: Optional[float]
    put_mid: Optional[float]
    midpoint_fair: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    valid_for_band: bool
    rejected_reason: Optional[str]
    conv_edge: Optional[float]
    conv_qty: int
    rev_edge: Optional[float]
    rev_qty: int


@dataclass(frozen=True)
class ChainAnalysis:
    spot_bid: Optional[int]
    spot_bid_qty: int
    spot_ask: Optional[int]
    spot_ask_qty: int
    spot_mid: Optional[float]
    analyses: list[StrikeAnalysis]
    filtered_strikes: list[int]
    dropped_outlier: Optional[int]
    band_lower: Optional[float]
    band_upper: Optional[float]
    band_width: Optional[float]
    regime: str
    consistent: bool
    primary_enabled: bool
    disable_reason: str
    midpoint_fair_median: Optional[float]
    best_box_edge: Optional[float]
    best_box_desc: Optional[str]
    best_struct_violation: Optional[float]
    best_struct_desc: Optional[str]


@dataclass
class BCampaign:
    state: str
    side: Side
    regime: str
    desired_qty: int
    entry_edge_initial: float
    entry_reason: str
    started_at: float
    stage_started_at: float
    current_order_id: Optional[str] = None
    current_order_role: Optional[str] = None
    filled_qty: int = 0
    fill_notional: float = 0.0
    ttl_expiries: int = 0
    cooldown_until: float = 0.0
    last_reason: str = ""

    def avg_fill_price(self) -> Optional[float]:
        if self.filled_qty <= 0:
            return None
        return self.fill_notional / self.filled_qty


@dataclass(frozen=True)
class BundleLeg:
    symbol: str
    side: Side
    price: int
    qty: int
    role: str


@dataclass
class ConvRevBundle:
    bundle_id: str
    state: str
    kind: str
    strike: int
    created_at: float
    initial_edge: float
    expected_positions: dict[str, int]
    legs: dict[str, BundleLeg]
    current_edge: Optional[float] = None
    order_ids: set[str] = field(default_factory=set)
    fills_by_symbol: dict[str, int] = field(default_factory=dict)
    fill_notional_by_symbol: dict[str, float] = field(default_factory=dict)
    first_fill_seen: bool = False
    post_fill_checked: bool = False
    last_reason: str = ""


@dataclass
class ReboundState:
    active: bool = False
    symbol: str = ""
    tier: str = ""
    entry_px: Optional[float] = None
    filled_qty: int = 0
    entry_time: float = 0.0
    current_order_id: Optional[str] = None
    current_order_role: Optional[str] = None
    cooldown_until: float = 0.0
    attempts_used: int = 0
    last_reason: str = ""
    stage: str = "IDLE"
    tp_ticks: float = 0.0
    sl_ticks: float = 0.0
    entry_spread: int = 0
    entry_notional: float = 0.0
    round_index: int = 0

    def avg_entry_price(self) -> Optional[float]:
        if self.filled_qty <= 0:
            return None
        return self.entry_notional / self.filled_qty


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    block_code: Optional[str] = None
    details: dict[str, object] = field(default_factory=dict)


def load_config() -> BotConfig:
    return BotConfig(
        host=env_str("UTC_HOST", "34.197.188.76:3333"),
        username=env_str("UTC_USERNAME", "uiuc"),
        password=env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
        hard_max_order_size=env_int("B1_MAX_ORDER_SIZE", 40),
        hard_max_open_orders=env_int("B1_MAX_OPEN_ORDERS", 50),
        hard_max_outstanding_volume=env_int("B1_MAX_OUTSTANDING_VOLUME", 120),
        hard_max_absolute_position=env_int("B1_MAX_ABSOLUTE_POSITION", 200),
        soft_b_inventory_cap=env_int("B1_B_SOFT_CAP", 80),
        soft_option_inventory_cap=env_int("B1_OPTION_SOFT_CAP", 20),
        soft_outstanding_volume=env_int("B1_SOFT_OUTSTANDING", 80),
        soft_open_orders=env_int("B1_SOFT_OPEN_ORDERS", 12),
        min_band_leg_qty=max(1, env_int("B1_MIN_BAND_LEG_QTY", 5)),
        outlier_ticks=max(1, env_int("B1_OUTLIER_TICKS", 8)),
        enter_edge_ticks=max(1, env_int("B1_ENTER_EDGE_TICKS", 6)),
        exit_buffer_ticks=max(1, env_int("B1_EXIT_BUFFER_TICKS", 2)),
        edge_decay_exit_ticks=max(1, env_int("B1_EDGE_DECAY_EXIT_TICKS", 4)),
        strong_cross_ticks=max(1, env_int("B1_STRONG_CROSS_TICKS", 12)),
        entry_ttl_sec=env_float("B1_ENTRY_TTL_SEC", 0.75),
        exit_ttl_sec=env_float("B1_EXIT_TTL_SEC", 0.75),
        entry_cooldown_sec=env_float("B1_ENTRY_COOLDOWN_SEC", 1.0),
        max_hold_sec=env_float("B1_MAX_HOLD_SEC", 8.0),
        conv_rev_min_edge_ticks=max(1, env_int("B1_CONV_REV_MIN_EDGE", 10)),
        conv_rev_size=max(1, env_int("B1_CONV_REV_SIZE", 5)),
        conv_rev_cooldown_sec=env_float("B1_CONV_REV_COOLDOWN_SEC", 1.5),
        post_fill_keep_edge=max(1, env_int("B1_POST_FILL_KEEP_EDGE", 4)),
        poll_sec=env_float("B1_POLL_SEC", 0.25),
        status_sec=env_float("B1_STATUS_INTERVAL", 3.0),
        capture_enabled=env_bool("B1_TRACE_ENABLED", True),
        capture_dir=env_str("B1_CAPTURE_DIR", str(Path(__file__).resolve().parent / "run_logs")),
        rebound_enabled=env_bool("B1_REBOUND_ENABLED", True),
        rebound_lookback_sec=max(0.5, env_float("B1_REBOUND_LOOKBACK_SEC", 2.0)),
        rebound_moderate_b_mid=env_float("B1_REBOUND_MODERATE_B_MID", 995.0),
        rebound_strong_b_mid=env_float("B1_REBOUND_STRONG_B_MID", 980.0),
        rebound_moderate_shock_ticks=env_float("B1_REBOUND_MODERATE_SHOCK_TICKS", 4.0),
        rebound_strong_shock_ticks=env_float("B1_REBOUND_STRONG_SHOCK_TICKS", 8.0),
        rebound_symbol_moderate=env_str("B1_REBOUND_SYMBOL_MODERATE", "B_C_1000"),
        rebound_symbol_strong=env_str("B1_REBOUND_SYMBOL_STRONG", "B_C_950"),
        rebound_max_qty=max(1, env_int("B1_REBOUND_MAX_QTY", 5)),
        rebound_entry_cooldown_sec=max(0.0, env_float("B1_REBOUND_ENTRY_COOLDOWN_SEC", 45.0)),
        rebound_max_attempts_per_round=max(1, env_int("B1_REBOUND_MAX_ATTEMPTS_PER_ROUND", 2)),
        rebound_base_tp_ticks=max(1.0, env_float("B1_REBOUND_BASE_TP_TICKS", 8.0)),
        rebound_base_sl_ticks=max(1.0, env_float("B1_REBOUND_BASE_SL_TICKS", 4.0)),
        rebound_tp_spread_mult=max(0.0, env_float("B1_REBOUND_TP_SPREAD_MULT", 1.5)),
        rebound_sl_spread_mult=max(0.0, env_float("B1_REBOUND_SL_SPREAD_MULT", 1.0)),
        rebound_max_spread=max(1, env_int("B1_REBOUND_MAX_SPREAD", 12)),
        rebound_iv_proxy_max=env_float("B1_REBOUND_IV_PROXY_MAX", 2.5),
        rebound_max_hold_sec=max(1.0, env_float("B1_REBOUND_MAX_HOLD_SEC", 120.0)),
        rebound_force_exit_after_sec=max(1.0, env_float("B1_REBOUND_FORCE_EXIT_AFTER_SEC", 780.0)),
        rebound_min_spread_for_passive=max(1, env_int("B1_REBOUND_MIN_SPREAD_FOR_PASSIVE", 2)),
    )


class MarketBExecutableBandBot(XChangeClient):
    def __init__(self, config: BotConfig):
        super().__init__(config.host, config.username, config.password, silent=False)
        self.cfg = config
        self.option_pairs = [
            OptionPair(strike=strike, call_symbol=f"B_C_{strike}", put_symbol=f"B_P_{strike}")
            for strike in DEFAULT_STRIKES
        ]
        self.last_trade: dict[str, int] = {}
        self.order_meta: dict[str, OrderMeta] = {}
        self.local_positions: dict[str, int] = {symbol: 0 for symbol in self.tracked_symbols()}
        self.local_positions_seeded = False
        self.last_local_fill_at: dict[str, float] = {}
        self.last_gate_log_at: dict[str, float] = {}
        self.bundle_cooldown_until: dict[int, float] = {}
        self.active_campaign: Optional[BCampaign] = None
        self.active_bundle: Optional[ConvRevBundle] = None
        self.rebound = ReboundState()
        self.round_duration_sec = 900.0
        self.history_horizon_sec = max(30.0, self.cfg.rebound_lookback_sec * 12.0)
        self.mid_history: dict[str, deque[tuple[float, float]]] = {
            symbol: deque(maxlen=5000) for symbol in self.tracked_symbols()
        }
        self.startup_flatten_complete = False
        self.capture_handle = None
        self.capture_file_path: Optional[Path] = None
        self.last_decision = Decision(action="init", reason="startup")
        self.last_chain: Optional[ChainAnalysis] = None
        self.last_status = 0.0
        self.cycle_count = 0
        self.connection_started_at = self.now()
        self._book_event = asyncio.Event()
        self._strategy_lock = asyncio.Lock()
        self.action_counts = {
            "entries": 0,
            "exits": 0,
            "repairs": 0,
            "rejections": 0,
            "blocks": 0,
        }
        self.setup_capture()

    def now(self) -> float:
        return time.monotonic()

    def tracked_symbols(self) -> list[str]:
        symbols = ["B"]
        for pair in self.option_pairs:
            symbols.extend([pair.call_symbol, pair.put_symbol])
        return symbols

    def option_symbols(self) -> list[str]:
        symbols: list[str] = []
        for pair in self.option_pairs:
            symbols.extend([pair.call_symbol, pair.put_symbol])
        return symbols

    def setup_capture(self) -> None:
        if not self.cfg.capture_enabled:
            return
        capture_dir = Path(self.cfg.capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        session_name = time.strftime("new_b1_%Y%m%d_%H%M%S")
        self.capture_file_path = capture_dir / f"{session_name}.jsonl"
        self.capture_handle = self.capture_file_path.open("a", encoding="utf-8", buffering=1)
        print(f"[CAPTURE] file={self.capture_file_path}")
        self.record_event(
            "session_start",
            config={
                "host": self.cfg.host,
                "username": self.cfg.username,
                "hard_max_order_size": self.cfg.hard_max_order_size,
                "hard_max_open_orders": self.cfg.hard_max_open_orders,
                "hard_max_outstanding_volume": self.cfg.hard_max_outstanding_volume,
                "hard_max_absolute_position": self.cfg.hard_max_absolute_position,
                "soft_b_inventory_cap": self.cfg.soft_b_inventory_cap,
                "soft_option_inventory_cap": self.cfg.soft_option_inventory_cap,
                "soft_outstanding_volume": self.cfg.soft_outstanding_volume,
                "soft_open_orders": self.cfg.soft_open_orders,
                "min_band_leg_qty": self.cfg.min_band_leg_qty,
                "outlier_ticks": self.cfg.outlier_ticks,
                "enter_edge_ticks": self.cfg.enter_edge_ticks,
                "exit_buffer_ticks": self.cfg.exit_buffer_ticks,
                "edge_decay_exit_ticks": self.cfg.edge_decay_exit_ticks,
                "strong_cross_ticks": self.cfg.strong_cross_ticks,
                "conv_rev_min_edge_ticks": self.cfg.conv_rev_min_edge_ticks,
                "conv_rev_size": self.cfg.conv_rev_size,
                "rebound_enabled": self.cfg.rebound_enabled,
                "rebound_lookback_sec": self.cfg.rebound_lookback_sec,
                "rebound_moderate_b_mid": self.cfg.rebound_moderate_b_mid,
                "rebound_strong_b_mid": self.cfg.rebound_strong_b_mid,
                "rebound_moderate_shock_ticks": self.cfg.rebound_moderate_shock_ticks,
                "rebound_strong_shock_ticks": self.cfg.rebound_strong_shock_ticks,
                "rebound_symbol_moderate": self.cfg.rebound_symbol_moderate,
                "rebound_symbol_strong": self.cfg.rebound_symbol_strong,
                "rebound_max_qty": self.cfg.rebound_max_qty,
                "rebound_max_attempts_per_round": self.cfg.rebound_max_attempts_per_round,
            },
        )

    def normalize_value(self, value):
        if isinstance(value, Side):
            return value.name
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self.normalize_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.normalize_value(item) for item in value]
        return value

    def record_event(self, kind: str, **payload) -> None:
        if self.capture_handle is None:
            return
        event = {
            "ts_wall": time.time(),
            "ts_mono": self.now(),
            "kind": kind,
            "payload": self.normalize_value(payload),
        }
        try:
            self.capture_handle.write(json.dumps(event, sort_keys=True) + "\n")
            self.capture_handle.flush()
        except Exception as exc:
            print(f"[CAPTURE-ERROR] kind={kind} error={exc}")

    def log_gate(self, key: str, message: str, throttle_sec: float = 1.5) -> None:
        now = self.now()
        if now - self.last_gate_log_at.get(key, 0.0) < throttle_sec:
            return
        self.last_gate_log_at[key] = now
        print(message)
        self.record_event("gate", key=key, message=message)

    def get_book(self, symbol: str):
        return self.order_books.get(symbol)

    def top(self, symbol: str) -> tuple[Optional[int], int, Optional[int], int]:
        book = self.get_book(symbol)
        bid, bid_qty = live_bid(book)
        ask, ask_qty = live_ask(book)
        return bid, bid_qty, ask, ask_qty

    def mid(self, symbol: str) -> Optional[float]:
        return mid_from_book(self.get_book(symbol), fallback=self.last_trade.get(symbol))

    def exchange_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def position(self, symbol: str) -> int:
        if self.local_positions_seeded:
            return int(self.local_positions.get(symbol, self.exchange_position(symbol)))
        return self.exchange_position(symbol)

    def pending_side_qty(self, symbol: str, side: Side) -> int:
        total = 0
        for order_id, meta in self.order_meta.items():
            if meta.symbol != symbol or meta.side != side:
                continue
            total += self.remaining_qty(order_id)
        return total

    def open_order_count(self) -> int:
        return sum(1 for order_id in self.order_meta if self.remaining_qty(order_id) > 0)

    def outstanding_qty(self) -> int:
        return sum(self.remaining_qty(order_id) for order_id in self.order_meta)

    def remaining_qty(self, order_id: str) -> int:
        order = self.open_orders.get(order_id)
        if not order:
            return 0
        return int(order[1])

    def signed_exposure(self, symbol: str) -> int:
        return (
            self.position(symbol)
            + self.pending_side_qty(symbol, Side.BUY)
            - self.pending_side_qty(symbol, Side.SELL)
        )

    def option_inventory_present(self) -> bool:
        return any(self.position(symbol) != 0 for symbol in self.option_symbols())

    def option_or_stock_orders_present(self) -> bool:
        return self.open_order_count() > 0

    def band_regime(self, band_width: Optional[float]) -> str:
        if band_width is None:
            return "invalid/skip"
        if band_width <= 4:
            return "tight"
        if band_width <= 10:
            return "normal"
        if band_width <= 18:
            return "wide"
        return "invalid/skip"

    def band_order_size(self, regime: str) -> int:
        if regime == "tight":
            return 20
        if regime == "normal":
            return 10
        if regime == "wide":
            return 5
        return 0

    def band_cross_threshold(self, regime: str) -> int:
        if regime == "tight":
            return self.cfg.strong_cross_ticks
        if regime == "normal":
            return self.cfg.strong_cross_ticks + 2
        return 10 ** 9

    def seed_and_reconcile_local_positions(self) -> None:
        tracked = {symbol: self.exchange_position(symbol) for symbol in self.tracked_symbols()}
        if not self.local_positions_seeded:
            if self.now() - self.connection_started_at < 1.0:
                return
            self.local_positions.update(tracked)
            self.local_positions_seeded = True
            print("[SYNC] seeded local positions from exchange")
            self.record_event("seed_local_positions", positions=tracked)
            return

        now = self.now()
        for symbol, exchange_pos in tracked.items():
            local_pos = int(self.local_positions.get(symbol, 0))
            recently_filled = now - self.last_local_fill_at.get(symbol, 0.0) < 2.0
            has_pending = self.pending_side_qty(symbol, Side.BUY) + self.pending_side_qty(symbol, Side.SELL) > 0
            if local_pos != exchange_pos and not recently_filled and not has_pending:
                self.local_positions[symbol] = exchange_pos
                self.record_event(
                    "position_reconcile",
                    symbol=symbol,
                    local=local_pos,
                    exchange=exchange_pos,
                )

    def serialize_book(self, symbol: str) -> dict[str, object]:
        book = self.get_book(symbol)
        if book is None:
            return {"bids": [], "asks": [], "last_trade": self.last_trade.get(symbol)}
        bids = sorted(
            [[int(px), int(qty)] for px, qty in book.bids.items() if qty > 0],
            key=lambda level: level[0],
            reverse=True,
        )
        asks = sorted(
            [[int(px), int(qty)] for px, qty in book.asks.items() if qty > 0],
            key=lambda level: level[0],
        )
        return {"bids": bids, "asks": asks, "last_trade": self.last_trade.get(symbol)}

    def serialize_orders(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for order_id, meta in self.order_meta.items():
            remaining = self.remaining_qty(order_id)
            if remaining <= 0:
                continue
            rows.append(
                {
                    "order_id": order_id,
                    "symbol": meta.symbol,
                    "side": meta.side.name,
                    "price": meta.price,
                    "qty_remaining": remaining,
                    "purpose": meta.purpose,
                    "strategy": meta.strategy,
                    "role": meta.role,
                    "strike": meta.strike,
                    "bundle_id": meta.bundle_id,
                    "age_sec": round(self.now() - meta.created_at, 3),
                    "cancel_pending": meta.cancel_pending,
                }
            )
        return rows

    def serialize_campaign(self) -> Optional[dict[str, object]]:
        if self.active_campaign is None:
            return None
        return {
            "state": self.active_campaign.state,
            "side": self.active_campaign.side.name,
            "regime": self.active_campaign.regime,
            "desired_qty": self.active_campaign.desired_qty,
            "entry_edge_initial": self.active_campaign.entry_edge_initial,
            "entry_reason": self.active_campaign.entry_reason,
            "current_order_id": self.active_campaign.current_order_id,
            "current_order_role": self.active_campaign.current_order_role,
            "filled_qty": self.active_campaign.filled_qty,
            "avg_fill_price": self.active_campaign.avg_fill_price(),
            "ttl_expiries": self.active_campaign.ttl_expiries,
            "cooldown_until": self.active_campaign.cooldown_until,
            "last_reason": self.active_campaign.last_reason,
        }

    def serialize_bundle(self) -> Optional[dict[str, object]]:
        if self.active_bundle is None:
            return None
        return {
            "bundle_id": self.active_bundle.bundle_id,
            "state": self.active_bundle.state,
            "kind": self.active_bundle.kind,
            "strike": self.active_bundle.strike,
            "initial_edge": self.active_bundle.initial_edge,
            "current_edge": self.active_bundle.current_edge,
            "expected_positions": self.active_bundle.expected_positions,
            "fills_by_symbol": self.active_bundle.fills_by_symbol,
            "order_ids": sorted(self.active_bundle.order_ids),
            "first_fill_seen": self.active_bundle.first_fill_seen,
            "post_fill_checked": self.active_bundle.post_fill_checked,
            "last_reason": self.active_bundle.last_reason,
        }

    def serialize_rebound(self) -> dict[str, object]:
        avg_entry = self.rebound.avg_entry_price()
        symbol = self.rebound.symbol
        current_mid = self.mid(symbol) if symbol else None
        unrealized_ticks = None
        if avg_entry is not None and current_mid is not None:
            unrealized_ticks = current_mid - avg_entry
        hold_secs = None
        if self.rebound.active and self.rebound.entry_time > 0:
            hold_secs = self.now() - self.rebound.entry_time
        return {
            "active": self.rebound.active,
            "stage": self.rebound.stage,
            "symbol": self.rebound.symbol,
            "tier": self.rebound.tier,
            "entry_px": avg_entry,
            "filled_qty": self.rebound.filled_qty,
            "entry_time": self.rebound.entry_time,
            "hold_secs": hold_secs,
            "unrealized_ticks": unrealized_ticks,
            "tp_ticks": self.rebound.tp_ticks,
            "sl_ticks": self.rebound.sl_ticks,
            "entry_spread": self.rebound.entry_spread,
            "attempts_used": self.rebound.attempts_used,
            "cooldown_until": self.rebound.cooldown_until,
            "current_order_id": self.rebound.current_order_id,
            "current_order_role": self.rebound.current_order_role,
            "last_reason": self.rebound.last_reason,
            "round_index": self.rebound.round_index,
        }

    def append_mid_history(self, symbol: str, mid_value: Optional[float]) -> None:
        if mid_value is None:
            return
        history = self.mid_history.setdefault(symbol, deque(maxlen=5000))
        now = self.now()
        history.append((now, float(mid_value)))
        cutoff = now - self.history_horizon_sec
        while history and history[0][0] < cutoff:
            history.popleft()

    def history_value_at(self, symbol: str, lookback_sec: float) -> Optional[float]:
        history = self.mid_history.get(symbol)
        if not history:
            return None
        target = self.now() - lookback_sec
        candidate = None
        for timestamp, value in reversed(history):
            if timestamp <= target:
                candidate = value
                break
        return candidate

    def current_round_index(self) -> int:
        elapsed = max(0.0, self.now() - self.connection_started_at)
        return int(elapsed // self.round_duration_sec)

    def current_round_elapsed(self) -> float:
        elapsed = max(0.0, self.now() - self.connection_started_at)
        return elapsed - (self.current_round_index() * self.round_duration_sec)

    def rollover_rebound_round_if_needed(self) -> None:
        round_index = self.current_round_index()
        if round_index == self.rebound.round_index:
            return
        previous = self.rebound.round_index
        self.rebound.round_index = round_index
        self.rebound.attempts_used = 0
        if not self.rebound.active and self.rebound.current_order_id is None:
            self.rebound.cooldown_until = self.now()
        self.record_event(
            "rebound_round_rollover",
            previous_round=previous,
            current_round=round_index,
        )

    def serialize_chain(self, chain: Optional[ChainAnalysis]) -> Optional[dict[str, object]]:
        if chain is None:
            return None
        rows = []
        for analysis in chain.analyses:
            rows.append(
                {
                    "strike": analysis.strike,
                    "call_symbol": analysis.call_symbol,
                    "put_symbol": analysis.put_symbol,
                    "call_bid": analysis.call_bid,
                    "call_bid_qty": analysis.call_bid_qty,
                    "call_ask": analysis.call_ask,
                    "call_ask_qty": analysis.call_ask_qty,
                    "put_bid": analysis.put_bid,
                    "put_bid_qty": analysis.put_bid_qty,
                    "put_ask": analysis.put_ask,
                    "put_ask_qty": analysis.put_ask_qty,
                    "call_mid": analysis.call_mid,
                    "put_mid": analysis.put_mid,
                    "midpoint_fair": analysis.midpoint_fair,
                    "lower": analysis.lower,
                    "upper": analysis.upper,
                    "valid_for_band": analysis.valid_for_band,
                    "rejected_reason": analysis.rejected_reason,
                    "conv_edge": analysis.conv_edge,
                    "conv_qty": analysis.conv_qty,
                    "rev_edge": analysis.rev_edge,
                    "rev_qty": analysis.rev_qty,
                }
            )
        return {
            "spot_bid": chain.spot_bid,
            "spot_bid_qty": chain.spot_bid_qty,
            "spot_ask": chain.spot_ask,
            "spot_ask_qty": chain.spot_ask_qty,
            "spot_mid": chain.spot_mid,
            "filtered_strikes": chain.filtered_strikes,
            "dropped_outlier": chain.dropped_outlier,
            "band_lower": chain.band_lower,
            "band_upper": chain.band_upper,
            "band_width": chain.band_width,
            "regime": chain.regime,
            "consistent": chain.consistent,
            "primary_enabled": chain.primary_enabled,
            "disable_reason": chain.disable_reason,
            "midpoint_fair_median": chain.midpoint_fair_median,
            "best_box_edge": chain.best_box_edge,
            "best_box_desc": chain.best_box_desc,
            "best_struct_violation": chain.best_struct_violation,
            "best_struct_desc": chain.best_struct_desc,
            "strikes": rows,
        }

    def record_snapshot(self) -> None:
        self.record_event(
            "cycle_snapshot",
            cycle=self.cycle_count,
            decision=self.last_decision.action,
            reason=self.last_decision.reason,
            block_code=self.last_decision.block_code,
            decision_details=self.last_decision.details,
            positions={symbol: self.position(symbol) for symbol in self.tracked_symbols()},
            open_orders=self.serialize_orders(),
            books={symbol: self.serialize_book(symbol) for symbol in self.tracked_symbols()},
            chain=self.serialize_chain(self.last_chain),
            campaign=self.serialize_campaign(),
            bundle=self.serialize_bundle(),
            rebound=self.serialize_rebound(),
            outstanding_qty=self.outstanding_qty(),
            open_order_count=self.open_order_count(),
            action_counts=self.action_counts,
        )

    def strongest_conv_rev(self, chain: Optional[ChainAnalysis]) -> tuple[Optional[float], Optional[str]]:
        if chain is None:
            return None, None
        best_edge = None
        best_desc = None
        for analysis in chain.analyses:
            if analysis.conv_edge is not None and (best_edge is None or analysis.conv_edge > best_edge):
                best_edge = analysis.conv_edge
                best_desc = f"conv@{analysis.strike}"
            if analysis.rev_edge is not None and (best_edge is None or analysis.rev_edge > best_edge):
                best_edge = analysis.rev_edge
                best_desc = f"rev@{analysis.strike}"
        return best_edge, best_desc

    def current_band_edges(self, chain: Optional[ChainAnalysis]) -> tuple[Optional[float], Optional[float]]:
        if chain is None or chain.band_lower is None or chain.band_upper is None:
            return None, None
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        long_edge = chain.band_lower - ask if ask is not None else None
        short_edge = bid - chain.band_upper if bid is not None else None
        return long_edge, short_edge

    def print_status(self) -> None:
        now = self.now()
        if now - self.last_status < self.cfg.status_sec:
            return
        self.last_status = now
        chain = self.last_chain
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        mid = self.mid("B")
        long_edge, short_edge = self.current_band_edges(chain)
        best_conv_rev, best_conv_rev_desc = self.strongest_conv_rev(chain)
        box_edge = chain.best_box_edge if chain else None
        box_desc = chain.best_box_desc if chain else None
        struct_edge = chain.best_struct_violation if chain else None
        struct_desc = chain.best_struct_desc if chain else None
        if box_edge is None and struct_edge is None:
            best_diag_edge = None
            best_diag_desc = None
        elif struct_edge is None or (box_edge is not None and box_edge >= struct_edge):
            best_diag_edge = box_edge
            best_diag_desc = box_desc
        else:
            best_diag_edge = struct_edge
            best_diag_desc = struct_desc
        rebound_payload = self.serialize_rebound()
        mode = (
            "rebound"
            if self.rebound.active or self.rebound.current_order_id is not None
            else "bundle"
            if self.active_bundle is not None
            else "campaign"
            if self.active_campaign is not None
            else "idle"
        )
        inventory = {symbol: pos for symbol, pos in self.local_positions.items() if pos}
        print(
            "[STATUS] "
            f"B=({bid},{ask}) mid={mid if mid is not None else 'NA'} "
            f"band=({chain.band_lower if chain else 'NA'},{chain.band_upper if chain else 'NA'}) "
            f"width={chain.band_width if chain else 'NA'} regime={chain.regime if chain else 'NA'} "
            f"edge_long={long_edge if long_edge is not None else 'NA'} "
            f"edge_short={short_edge if short_edge is not None else 'NA'} "
            f"mode={mode} inventory={inventory or {'flat': 0}} "
            f"entries={self.action_counts['entries']} exits={self.action_counts['exits']} "
            f"repairs={self.action_counts['repairs']} rejects={self.action_counts['rejections']} "
            f"best_conv_rev={best_conv_rev if best_conv_rev is not None else 'NA'}:{best_conv_rev_desc or 'NA'} "
            f"best_diag={best_diag_edge if best_diag_edge is not None else 'NA'}:{best_diag_desc or 'NA'} "
            f"rebound={rebound_payload['stage']}:{rebound_payload['symbol'] or 'NA'} "
            f"qty={rebound_payload['filled_qty']} "
            f"uPnL={rebound_payload['unrealized_ticks'] if rebound_payload['unrealized_ticks'] is not None else 'NA'} "
            f"hold={round(rebound_payload['hold_secs'],1) if rebound_payload['hold_secs'] is not None else 'NA'}"
        )
        self.record_event(
            "status",
            mode=mode,
            inventory=inventory,
            best_conv_rev=best_conv_rev,
            best_conv_rev_desc=best_conv_rev_desc,
            best_box_edge=box_edge,
            best_box_desc=box_desc,
            best_struct_violation=struct_edge,
            best_struct_desc=struct_desc,
            best_diag_edge=best_diag_edge,
            best_diag_desc=best_diag_desc,
            rebound=rebound_payload,
        )

    def compute_structural_diagnostics(
        self,
        analyses: list[StrikeAnalysis],
    ) -> tuple[Optional[float], Optional[str]]:
        call_mid_by_strike = {a.strike: a.call_mid for a in analyses if a.call_mid is not None}
        put_mid_by_strike = {a.strike: a.put_mid for a in analyses if a.put_mid is not None}
        best_edge = None
        best_desc = None

        for first, second in zip(DEFAULT_STRIKES, DEFAULT_STRIKES[1:]):
            call_first = call_mid_by_strike.get(first)
            call_second = call_mid_by_strike.get(second)
            if call_first is not None and call_second is not None:
                monotonic = max(0.0, call_second - call_first)
                if best_edge is None or monotonic > best_edge:
                    best_edge = monotonic
                    best_desc = f"call_monotonic_{first}_{second}"
                upper_bound = max(0.0, (call_first - call_second) - (second - first))
                if upper_bound > (best_edge or 0.0):
                    best_edge = upper_bound
                    best_desc = f"call_vertical_upper_{first}_{second}"

            put_first = put_mid_by_strike.get(first)
            put_second = put_mid_by_strike.get(second)
            if put_first is not None and put_second is not None:
                monotonic = max(0.0, put_first - put_second)
                if monotonic > (best_edge or 0.0):
                    best_edge = monotonic
                    best_desc = f"put_monotonic_{first}_{second}"
                upper_bound = max(0.0, (put_second - put_first) - (second - first))
                if upper_bound > (best_edge or 0.0):
                    best_edge = upper_bound
                    best_desc = f"put_vertical_upper_{first}_{second}"

        if len(DEFAULT_STRIKES) == 3:
            k1, k2, k3 = DEFAULT_STRIKES
            call1 = call_mid_by_strike.get(k1)
            call2 = call_mid_by_strike.get(k2)
            call3 = call_mid_by_strike.get(k3)
            if None not in (call1, call2, call3):
                butterfly = call1 + call3 - 2.0 * call2
                if butterfly < 0 and -butterfly > (best_edge or 0.0):
                    best_edge = -butterfly
                    best_desc = "call_butterfly_convexity"

            put1 = put_mid_by_strike.get(k1)
            put2 = put_mid_by_strike.get(k2)
            put3 = put_mid_by_strike.get(k3)
            if None not in (put1, put2, put3):
                butterfly = put1 + put3 - 2.0 * put2
                if butterfly < 0 and -butterfly > (best_edge or 0.0):
                    best_edge = -butterfly
                    best_desc = "put_butterfly_convexity"

        return best_edge, best_desc

    def compute_box_diagnostics(self) -> tuple[Optional[float], Optional[str]]:
        best_edge = None
        best_desc = None
        for first, second in ((950, 1000), (1000, 1050), (950, 1050)):
            c1_bid, _c1_bid_qty, c1_ask, _c1_ask_qty = self.top(f"B_C_{first}")
            c2_bid, _c2_bid_qty, c2_ask, _c2_ask_qty = self.top(f"B_C_{second}")
            p1_bid, _p1_bid_qty, p1_ask, _p1_ask_qty = self.top(f"B_P_{first}")
            p2_bid, _p2_bid_qty, p2_ask, _p2_ask_qty = self.top(f"B_P_{second}")
            if None in (c1_bid, c1_ask, c2_bid, c2_ask, p1_bid, p1_ask, p2_bid, p2_ask):
                continue
            theoretical = second - first
            buy_cost = c1_ask - c2_bid + p2_ask - p1_bid
            sell_net = c1_bid - c2_ask + p2_bid - p1_ask
            buy_edge = theoretical - buy_cost
            sell_edge = sell_net - theoretical
            edge = max(buy_edge, sell_edge)
            desc = f"box_buy_{first}_{second}" if buy_edge >= sell_edge else f"box_sell_{first}_{second}"
            if best_edge is None or edge > best_edge:
                best_edge = edge
                best_desc = desc
        return best_edge, best_desc

    def analyze_chain(self) -> ChainAnalysis:
        spot_bid, spot_bid_qty, spot_ask, spot_ask_qty = self.top("B")
        spot_mid = self.mid("B")
        analyses: list[StrikeAnalysis] = []
        midpoint_fairs: list[float] = []
        valid_strikes: list[StrikeAnalysis] = []

        for pair in self.option_pairs:
            call_bid, call_bid_qty, call_ask, call_ask_qty = self.top(pair.call_symbol)
            put_bid, put_bid_qty, put_ask, put_ask_qty = self.top(pair.put_symbol)
            call_mid = self.mid(pair.call_symbol)
            put_mid = self.mid(pair.put_symbol)
            midpoint_fair = None
            if call_mid is not None and put_mid is not None:
                midpoint_fair = pair.strike + call_mid - put_mid
                midpoint_fairs.append(midpoint_fair)

            valid_for_band = True
            rejected_reason = None
            if None in (call_bid, call_ask, put_bid, put_ask):
                valid_for_band = False
                rejected_reason = "missing_quote"
            elif min(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty) < self.cfg.min_band_leg_qty:
                valid_for_band = False
                rejected_reason = "insufficient_leg_qty"

            lower = None
            upper = None
            if valid_for_band:
                lower = pair.strike + call_bid - put_ask
                upper = pair.strike + call_ask - put_bid

            conv_edge = None
            conv_qty = 0
            if None not in (call_bid, put_ask, spot_ask):
                conv_edge = call_bid - put_ask - spot_ask + pair.strike
                conv_qty = min(call_bid_qty, put_ask_qty, spot_ask_qty)

            rev_edge = None
            rev_qty = 0
            if None not in (call_ask, put_bid, spot_bid):
                rev_edge = -call_ask + put_bid + spot_bid - pair.strike
                rev_qty = min(call_ask_qty, put_bid_qty, spot_bid_qty)

            analysis = StrikeAnalysis(
                strike=pair.strike,
                call_symbol=pair.call_symbol,
                put_symbol=pair.put_symbol,
                call_bid=call_bid,
                call_bid_qty=call_bid_qty,
                call_ask=call_ask,
                call_ask_qty=call_ask_qty,
                put_bid=put_bid,
                put_bid_qty=put_bid_qty,
                put_ask=put_ask,
                put_ask_qty=put_ask_qty,
                call_mid=call_mid,
                put_mid=put_mid,
                midpoint_fair=midpoint_fair,
                lower=lower,
                upper=upper,
                valid_for_band=valid_for_band,
                rejected_reason=rejected_reason,
                conv_edge=conv_edge,
                conv_qty=conv_qty,
                rev_edge=rev_edge,
                rev_qty=rev_qty,
            )
            analyses.append(analysis)
            if analysis.valid_for_band:
                valid_strikes.append(analysis)

        dropped_outlier = None
        filtered = list(valid_strikes)
        if filtered:
            lower_median = statistics.median([analysis.lower for analysis in filtered if analysis.lower is not None])
            upper_median = statistics.median([analysis.upper for analysis in filtered if analysis.upper is not None])
            deviations = []
            for analysis in filtered:
                deviation = max(abs(analysis.lower - lower_median), abs(analysis.upper - upper_median))
                deviations.append((deviation, analysis.strike))
            deviations.sort(reverse=True)
            if deviations and deviations[0][0] > self.cfg.outlier_ticks:
                dropped_outlier = deviations[0][1]
                filtered = [analysis for analysis in filtered if analysis.strike != dropped_outlier]

        filtered_strikes = [analysis.strike for analysis in filtered]
        band_lower = None
        band_upper = None
        band_width = None
        consistent = False
        disable_reason = "insufficient_strikes"
        if len(filtered) >= 2:
            band_lower = max(analysis.lower for analysis in filtered if analysis.lower is not None)
            band_upper = min(analysis.upper for analysis in filtered if analysis.upper is not None)
            band_width = band_upper - band_lower
            consistent = band_lower <= band_upper
            if not consistent:
                disable_reason = "inconsistent_chain"
            else:
                disable_reason = ""

        regime = self.band_regime(band_width if consistent else None)
        primary_enabled = len(filtered) >= 2 and consistent and regime != "invalid/skip"
        if len(filtered) < 2:
            disable_reason = "insufficient_strikes"
        elif consistent and regime == "invalid/skip":
            disable_reason = "band_too_wide"

        midpoint_fair_median = statistics.median(midpoint_fairs) if midpoint_fairs else None
        best_box_edge, best_box_desc = self.compute_box_diagnostics()
        best_struct_violation, best_struct_desc = self.compute_structural_diagnostics(analyses)

        return ChainAnalysis(
            spot_bid=spot_bid,
            spot_bid_qty=spot_bid_qty,
            spot_ask=spot_ask,
            spot_ask_qty=spot_ask_qty,
            spot_mid=spot_mid,
            analyses=analyses,
            filtered_strikes=filtered_strikes,
            dropped_outlier=dropped_outlier,
            band_lower=band_lower,
            band_upper=band_upper,
            band_width=band_width,
            regime=regime,
            consistent=consistent,
            primary_enabled=primary_enabled,
            disable_reason=disable_reason,
            midpoint_fair_median=midpoint_fair_median,
            best_box_edge=best_box_edge,
            best_box_desc=best_box_desc,
            best_struct_violation=best_struct_violation,
            best_struct_desc=best_struct_desc,
        )

    def submission_block(
        self,
        symbol: str,
        side: Side,
        qty: int,
        *,
        allow_option: bool = True,
    ) -> tuple[Optional[str], Optional[str]]:
        if symbol not in self.tracked_symbols():
            return "invalid_symbol", f"symbol={symbol}"
        if qty <= 0:
            return "nonpositive_qty", f"qty={qty}"
        if qty > self.cfg.hard_max_order_size:
            return "hard_max_order_size", f"qty={qty} limit={self.cfg.hard_max_order_size}"
        if not allow_option and symbol != "B":
            return "option_disallowed", f"symbol={symbol}"
        current_open = self.open_order_count()
        if current_open + 1 > self.cfg.hard_max_open_orders:
            return "hard_max_open_orders", f"current={current_open} limit={self.cfg.hard_max_open_orders}"
        if current_open + 1 > self.cfg.soft_open_orders:
            return "soft_open_orders", f"current={current_open} soft_limit={self.cfg.soft_open_orders}"
        current_outstanding = self.outstanding_qty()
        if current_outstanding + qty > self.cfg.hard_max_outstanding_volume:
            return (
                "hard_max_outstanding_volume",
                f"current={current_outstanding} add={qty} limit={self.cfg.hard_max_outstanding_volume}",
            )
        if current_outstanding + qty > self.cfg.soft_outstanding_volume:
            return (
                "soft_outstanding_volume",
                f"current={current_outstanding} add={qty} soft_limit={self.cfg.soft_outstanding_volume}",
            )
        projected = self.signed_exposure(symbol) + (qty if side == Side.BUY else -qty)
        if abs(projected) > self.cfg.hard_max_absolute_position:
            return (
                "hard_max_absolute_position",
                f"symbol={symbol} projected={projected} limit={self.cfg.hard_max_absolute_position}",
            )
        soft_limit = self.cfg.soft_b_inventory_cap if symbol == "B" else self.cfg.soft_option_inventory_cap
        if abs(projected) > soft_limit:
            return "soft_inventory_cap", f"symbol={symbol} projected={projected} soft_limit={soft_limit}"
        return None, None

    async def submit_limit(
        self,
        symbol: str,
        side: Side,
        qty: int,
        price: int,
        purpose: str,
        strategy: str,
        *,
        strike: Optional[int] = None,
        role: Optional[str] = None,
        bundle_id: Optional[str] = None,
        allow_option: bool = True,
    ) -> tuple[Optional[str], Optional[Decision]]:
        block_code, reason = self.submission_block(symbol, side, qty, allow_option=allow_option)
        if block_code is not None:
            self.action_counts["blocks"] += 1
            decision = Decision(
                action="blocked",
                reason=purpose,
                block_code=block_code,
                details={"symbol": symbol, "side": side.name, "qty": qty, "price": price, "message": reason},
            )
            self.record_event(
                "order_blocked",
                purpose=purpose,
                strategy=strategy,
                symbol=symbol,
                side=side.name,
                qty=qty,
                price=price,
                strike=strike,
                role=role,
                bundle_id=bundle_id,
                block_code=block_code,
                reason=reason,
            )
            return None, decision
        try:
            order_id = await self.place_order(symbol, int(qty), side, int(price))
        except Exception as exc:
            self.record_event(
                "order_error",
                purpose=purpose,
                strategy=strategy,
                symbol=symbol,
                side=side.name,
                qty=qty,
                price=price,
                strike=strike,
                role=role,
                bundle_id=bundle_id,
                error=str(exc),
            )
            return None, Decision(
                action="order_error",
                reason=purpose,
                details={"symbol": symbol, "side": side.name, "qty": qty, "price": price, "error": str(exc)},
            )
        self.order_meta[order_id] = OrderMeta(
            symbol=symbol,
            side=side,
            price=int(price),
            qty=int(qty),
            purpose=purpose,
            strategy=strategy,
            created_at=self.now(),
            strike=strike,
            role=role,
            bundle_id=bundle_id,
        )
        self.record_event(
            "order_submitted",
            order_id=order_id,
            purpose=purpose,
            strategy=strategy,
            symbol=symbol,
            side=side.name,
            qty=qty,
            price=price,
            strike=strike,
            role=role,
            bundle_id=bundle_id,
        )
        return order_id, None

    async def request_cancel(self, order_id: str, reason: str) -> bool:
        meta = self.order_meta.get(order_id)
        if meta is None:
            return False
        if meta.cancel_pending or order_id not in self.open_orders:
            return False
        meta.cancel_pending = True
        self.record_event(
            "cancel_requested",
            order_id=order_id,
            purpose=meta.purpose,
            strategy=meta.strategy,
            symbol=meta.symbol,
            role=meta.role,
            bundle_id=meta.bundle_id,
            reason=reason,
        )
        try:
            await self.cancel_order(order_id)
            return True
        except Exception as exc:
            meta.cancel_pending = False
            self.record_event("cancel_error", order_id=order_id, reason=reason, error=str(exc))
            return False

    def band_entry_signal(self, chain: ChainAnalysis) -> tuple[Optional[Side], Optional[float], Optional[str]]:
        if not chain.primary_enabled or chain.band_lower is None or chain.band_upper is None:
            return None, None, None
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        if ask is not None:
            long_edge = chain.band_lower - ask
            if long_edge >= self.cfg.enter_edge_ticks:
                return Side.BUY, float(long_edge), "ask_below_band_lower"
        if bid is not None:
            short_edge = bid - chain.band_upper
            if short_edge >= self.cfg.enter_edge_ticks:
                return Side.SELL, float(short_edge), "bid_above_band_upper"
        return None, None, None

    def entry_passive_price(self, side: Side, chain: ChainAnalysis) -> tuple[Optional[int], bool]:
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        if bid is None or ask is None:
            return None, False
        if side == Side.BUY:
            candidate = min(bid + 1, math.floor(chain.band_lower - self.cfg.enter_edge_ticks))
            return int(candidate), candidate < ask
        candidate = max(ask - 1, math.ceil(chain.band_upper + self.cfg.enter_edge_ticks))
        return int(candidate), candidate > bid

    def exit_passive_price(self, side: Side, chain: ChainAnalysis) -> tuple[Optional[int], bool]:
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        if bid is None or ask is None:
            return None, False
        if side == Side.SELL:
            candidate = max(bid + 1, math.ceil(chain.band_lower - self.cfg.exit_buffer_ticks))
            return int(candidate), candidate < ask
        candidate = min(ask - 1, math.floor(chain.band_upper + self.cfg.exit_buffer_ticks))
        return int(candidate), candidate > bid

    def current_campaign_position(self) -> int:
        return self.position("B")

    def current_campaign_edge(self, chain: ChainAnalysis, side: Side) -> Optional[float]:
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        if chain.band_lower is None or chain.band_upper is None:
            return None
        if side == Side.BUY:
            if ask is None:
                return None
            return max(0.0, chain.band_lower - ask)
        if bid is None:
            return None
        return max(0.0, bid - chain.band_upper)

    def campaign_band_flip(self, chain: ChainAnalysis, side: Side) -> bool:
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        if chain.band_lower is None or chain.band_upper is None or bid is None or ask is None:
            return False
        if side == Side.BUY:
            return bid >= chain.band_upper + self.cfg.exit_buffer_ticks
        return ask <= chain.band_lower - self.cfg.exit_buffer_ticks

    def should_start_exit(self, chain: ChainAnalysis, campaign: BCampaign) -> tuple[bool, str]:
        if not chain.primary_enabled or chain.band_lower is None or chain.band_upper is None:
            return True, "band_invalid_skip"
        position = self.current_campaign_position()
        if position == 0:
            return True, "campaign_flat"
        held_for = self.now() - campaign.started_at
        if held_for >= self.cfg.max_hold_sec:
            return True, "max_hold_exit"
        if self.campaign_band_flip(chain, campaign.side):
            return True, "band_flip_exit"
        current_edge = self.current_campaign_edge(chain, campaign.side)
        if current_edge is None:
            return True, "edge_unavailable_exit"
        if campaign.entry_edge_initial - current_edge >= self.cfg.edge_decay_exit_ticks:
            return True, "edge_decay_exit"
        mid = self.mid("B")
        if mid is None:
            return False, ""
        if campaign.side == Side.BUY and mid >= chain.band_lower - self.cfg.exit_buffer_ticks:
            return True, "reentered_band_long"
        if campaign.side == Side.SELL and mid <= chain.band_upper + self.cfg.exit_buffer_ticks:
            return True, "reentered_band_short"
        return False, ""

    async def move_campaign_to_cooldown(self, reason: str) -> Decision:
        if self.active_campaign is None:
            return Decision(action="noop", reason="campaign_missing")
        self.active_campaign.state = "COOLDOWN"
        self.active_campaign.current_order_id = None
        self.active_campaign.current_order_role = None
        self.active_campaign.cooldown_until = self.now() + self.cfg.entry_cooldown_sec
        self.active_campaign.last_reason = reason
        self.record_event(
            "campaign_cooldown",
            state=self.active_campaign.state,
            side=self.active_campaign.side.name,
            reason=reason,
            cooldown_until=self.active_campaign.cooldown_until,
        )
        return Decision(action="campaign_cooldown", reason=reason)

    async def clear_campaign_if_ready(self) -> None:
        if self.active_campaign is None:
            return
        if self.active_campaign.state != "COOLDOWN":
            return
        if self.current_campaign_position() != 0:
            return
        if self.active_campaign.current_order_id and self.remaining_qty(self.active_campaign.current_order_id) > 0:
            return
        if self.now() < self.active_campaign.cooldown_until:
            return
        self.record_event("campaign_cleared", reason=self.active_campaign.last_reason)
        self.active_campaign = None

    async def submit_campaign_order(
        self,
        side: Side,
        qty: int,
        price: int,
        role: str,
        reason: str,
    ) -> Decision:
        if self.active_campaign is None:
            return Decision(action="noop", reason="campaign_missing")
        order_id, block = await self.submit_limit(
            "B",
            side,
            qty,
            price,
            purpose=f"b_campaign_{role}",
            strategy="b_campaign",
            role=role,
            allow_option=False,
        )
        if block is not None:
            return block
        self.active_campaign.current_order_id = order_id
        self.active_campaign.current_order_role = role
        self.active_campaign.stage_started_at = self.now()
        self.active_campaign.last_reason = reason
        if role == "entry":
            self.action_counts["entries"] += 1
        else:
            self.action_counts["exits"] += 1
        return Decision(
            action=f"campaign_{role}",
            reason=reason,
            details={"order_id": order_id, "qty": qty, "price": price, "side": side.name},
        )

    async def maybe_enter_b_campaign(self, chain: ChainAnalysis) -> Optional[Decision]:
        if self.active_campaign is not None:
            return None
        if self.active_bundle is not None:
            return None
        if self.current_campaign_position() != 0:
            return Decision(action="skip_primary", reason="existing_b_inventory")
        if self.option_inventory_present():
            return Decision(action="skip_primary", reason="option_inventory_present")
        side, edge, reason = self.band_entry_signal(chain)
        if side is None or edge is None:
            return None
        regime = chain.regime
        desired_qty = self.band_order_size(regime)
        if desired_qty <= 0:
            return None
        desired_qty = min(desired_qty, self.cfg.hard_max_order_size)
        self.active_campaign = BCampaign(
            state="ENTRY_RESTING",
            side=side,
            regime=regime,
            desired_qty=desired_qty,
            entry_edge_initial=edge,
            entry_reason=reason,
            started_at=self.now(),
            stage_started_at=self.now(),
            last_reason=reason,
        )
        price, passive = self.entry_passive_price(side, chain)
        if price is None:
            await self.move_campaign_to_cooldown("entry_no_price")
            return Decision(action="skip_primary", reason="entry_no_price")
        if passive:
            return await self.submit_campaign_order(side, desired_qty, price, "entry", reason)

        threshold = self.band_cross_threshold(regime)
        if side == Side.BUY:
            residual_edge = chain.band_lower - chain.spot_ask if chain.spot_ask is not None else None
            cross_price = chain.spot_ask
        else:
            residual_edge = chain.spot_bid - chain.band_upper if chain.spot_bid is not None else None
            cross_price = chain.spot_bid
        if residual_edge is None or cross_price is None or residual_edge < threshold:
            await self.move_campaign_to_cooldown("entry_not_passive_and_not_strong_enough")
            return Decision(action="skip_primary", reason="entry_not_passive_and_not_strong_enough")
        return await self.submit_campaign_order(side, desired_qty, int(cross_price), "entry", f"{reason}_strong_cross")

    async def maybe_maintain_b_campaign(self, chain: ChainAnalysis) -> Optional[Decision]:
        if self.active_campaign is None:
            return None
        await self.clear_campaign_if_ready()
        if self.active_campaign is None:
            return None
        campaign = self.active_campaign

        if campaign.current_order_id is not None and campaign.current_order_id in self.open_orders:
            ttl = self.cfg.entry_ttl_sec if campaign.current_order_role == "entry" else self.cfg.exit_ttl_sec
            meta = self.order_meta.get(campaign.current_order_id)
            age = self.now() - meta.created_at if meta is not None else 0.0
            if age >= ttl and meta is not None and not meta.cancel_pending:
                campaign.ttl_expiries += 1
                await self.request_cancel(campaign.current_order_id, f"{campaign.current_order_role}_ttl")
                return Decision(action="campaign_cancel", reason=f"{campaign.current_order_role}_ttl")
            return Decision(action="campaign_wait", reason=f"{campaign.current_order_role}_working")

        position = self.current_campaign_position()
        if campaign.state == "ENTRY_RESTING":
            if position == 0:
                return await self.move_campaign_to_cooldown("entry_resting_unfilled")
            campaign.state = "ENTRY_FILLED"
            campaign.stage_started_at = self.now()

        if campaign.state == "ENTRY_FILLED":
            should_exit, reason = self.should_start_exit(chain, campaign)
            if not should_exit:
                return Decision(action="campaign_hold", reason="position_still_outside_band")
            campaign.state = "EXIT_RESTING"
            campaign.stage_started_at = self.now()
            campaign.last_reason = reason

        if campaign.state in {"EXIT_RESTING", "REPAIR_EXIT"}:
            position = self.current_campaign_position()
            if position == 0:
                return await self.move_campaign_to_cooldown("campaign_flat")
            exit_side = Side.SELL if position > 0 else Side.BUY
            qty = min(abs(position), self.cfg.hard_max_order_size)
            price, passive = self.exit_passive_price(exit_side, chain)
            if campaign.state == "REPAIR_EXIT":
                passive = False
            aggressive = False
            if price is None or not passive:
                aggressive = True
            else:
                should_exit, reason = self.should_start_exit(chain, campaign)
                aggressive = reason in {"band_invalid_skip", "band_flip_exit", "edge_decay_exit", "max_hold_exit"}
            if aggressive:
                if exit_side == Side.SELL:
                    price = chain.spot_bid
                else:
                    price = chain.spot_ask
                if price is None:
                    return Decision(action="campaign_wait", reason="exit_no_liquidity")
                campaign.state = "REPAIR_EXIT"
                campaign.last_reason = "aggressive_exit"
                return await self.submit_campaign_order(exit_side, qty, int(price), "exit", campaign.last_reason)
            return await self.submit_campaign_order(exit_side, qty, int(price), "exit", campaign.last_reason or "passive_exit")

        if campaign.state == "COOLDOWN":
            return Decision(action="campaign_cooldown", reason=campaign.last_reason or "cooldown")
        return None

    def best_conv_rev_candidate(self, chain: ChainAnalysis) -> Optional[tuple[str, StrikeAnalysis, float]]:
        best: Optional[tuple[str, StrikeAnalysis, float]] = None
        now = self.now()
        for analysis in chain.analyses:
            if now < self.bundle_cooldown_until.get(analysis.strike, 0.0):
                continue
            if analysis.conv_edge is not None and analysis.conv_qty >= self.cfg.conv_rev_size:
                if analysis.conv_edge >= self.cfg.conv_rev_min_edge_ticks:
                    if best is None or analysis.conv_edge > best[2]:
                        best = ("conversion", analysis, analysis.conv_edge)
            if analysis.rev_edge is not None and analysis.rev_qty >= self.cfg.conv_rev_size:
                if analysis.rev_edge >= self.cfg.conv_rev_min_edge_ticks:
                    if best is None or analysis.rev_edge > best[2]:
                        best = ("reversal", analysis, analysis.rev_edge)
        return best

    def bundle_expected_positions(self, kind: str, strike: int, qty: int) -> dict[str, int]:
        call = f"B_C_{strike}"
        put = f"B_P_{strike}"
        if kind == "conversion":
            return {"B": qty, call: -qty, put: qty}
        return {"B": -qty, call: qty, put: -qty}

    def bundle_leg_specs(self, kind: str, strike: int, qty: int) -> Optional[dict[str, BundleLeg]]:
        call_symbol = f"B_C_{strike}"
        put_symbol = f"B_P_{strike}"
        bid, _bid_qty, ask, _ask_qty = self.top("B")
        call_bid, _call_bid_qty, call_ask, _call_ask_qty = self.top(call_symbol)
        put_bid, _put_bid_qty, put_ask, _put_ask_qty = self.top(put_symbol)
        if kind == "conversion":
            if None in (bid, ask, call_bid, put_ask):
                return None
            return {
                "stock": BundleLeg(symbol="B", side=Side.BUY, price=int(ask), qty=qty, role="stock"),
                "call": BundleLeg(symbol=call_symbol, side=Side.SELL, price=int(call_bid), qty=qty, role="call"),
                "put": BundleLeg(symbol=put_symbol, side=Side.BUY, price=int(put_ask), qty=qty, role="put"),
            }
        if None in (bid, ask, call_ask, put_bid):
            return None
        return {
            "stock": BundleLeg(symbol="B", side=Side.SELL, price=int(bid), qty=qty, role="stock"),
            "call": BundleLeg(symbol=call_symbol, side=Side.BUY, price=int(call_ask), qty=qty, role="call"),
            "put": BundleLeg(symbol=put_symbol, side=Side.SELL, price=int(put_bid), qty=qty, role="put"),
        }

    def bundle_current_edge(self, bundle: ConvRevBundle) -> Optional[float]:
        strike = bundle.strike
        call_bid, _cbq, call_ask, _caq = self.top(f"B_C_{strike}")
        put_bid, _pbq, put_ask, _paq = self.top(f"B_P_{strike}")
        stock_bid, _sbq, stock_ask, _saq = self.top("B")
        if bundle.kind == "conversion":
            if None in (call_bid, put_ask, stock_ask):
                return None
            return call_bid - put_ask - stock_ask + strike
        if None in (call_ask, put_bid, stock_bid):
            return None
        return -call_ask + put_bid + stock_bid - strike

    async def start_conv_rev_bundle(self, kind: str, analysis: StrikeAnalysis, edge: float) -> Decision:
        qty = min(self.cfg.conv_rev_size, self.cfg.hard_max_order_size)
        legs = self.bundle_leg_specs(kind, analysis.strike, qty)
        if legs is None:
            return Decision(action="skip_conv_rev", reason="bundle_quotes_missing")
        for role in ("stock", "call", "put"):
            leg = legs[role]
            block_code, reason = self.submission_block(leg.symbol, leg.side, leg.qty, allow_option=True)
            if block_code is not None:
                self.action_counts["blocks"] += 1
                self.record_event(
                    "order_blocked",
                    purpose="conv_rev_preflight",
                    strategy="conv_rev",
                    symbol=leg.symbol,
                    side=leg.side.name,
                    qty=leg.qty,
                    price=leg.price,
                    strike=analysis.strike,
                    role=role,
                    block_code=block_code,
                    reason=reason,
                )
                return Decision(
                    action="blocked",
                    reason="conv_rev_preflight",
                    block_code=block_code,
                    details={
                        "symbol": leg.symbol,
                        "side": leg.side.name,
                        "qty": leg.qty,
                        "price": leg.price,
                        "message": reason,
                    },
                )
        bundle_id = f"{kind}:{analysis.strike}:{int(time.time() * 1000)}"
        bundle = ConvRevBundle(
            bundle_id=bundle_id,
            state="PENDING_SUBMIT",
            kind=kind,
            strike=analysis.strike,
            created_at=self.now(),
            initial_edge=edge,
            expected_positions=self.bundle_expected_positions(kind, analysis.strike, qty),
            legs=legs,
            current_edge=edge,
            last_reason="bundle_created",
        )
        self.active_bundle = bundle

        order_ids: list[str] = []
        for role in ("stock", "call", "put"):
            leg = legs[role]
            order_id, block = await self.submit_limit(
                leg.symbol,
                leg.side,
                leg.qty,
                leg.price,
                purpose=f"conv_rev_{kind}",
                strategy="conv_rev",
                strike=analysis.strike,
                role=role,
                bundle_id=bundle_id,
                allow_option=True,
            )
            if block is not None:
                if order_ids:
                    bundle.state = "REPAIR"
                    bundle.order_ids = set(order_ids)
                    bundle.last_reason = "bundle_submit_partial_failure"
                    self.record_event(
                        "bundle_submit_partial_failure",
                        bundle_id=bundle.bundle_id,
                        block_code=block.block_code,
                        reason=block.reason,
                        partial_order_ids=order_ids,
                    )
                    return Decision(
                        action="bundle_repair_pending",
                        reason="bundle_submit_partial_failure",
                        details={"bundle_id": bundle.bundle_id, "partial_order_ids": order_ids},
                    )
                bundle.state = "FAILED_CLEANED"
                bundle.last_reason = block.reason
                self.bundle_cooldown_until[analysis.strike] = self.now() + self.cfg.conv_rev_cooldown_sec
                self.active_bundle = None
                return block
            order_ids.append(order_id)
        bundle.state = "WORKING"
        bundle.order_ids = set(order_ids)
        bundle.last_reason = "bundle_submitted"
        self.record_event(
            "bundle_created",
            bundle_id=bundle.bundle_id,
            kind=bundle.kind,
            strike=bundle.strike,
            initial_edge=bundle.initial_edge,
            expected_positions=bundle.expected_positions,
            order_ids=order_ids,
        )
        return Decision(
            action="conv_rev_bundle",
            reason=f"{kind}_submitted",
            details={"bundle_id": bundle_id, "strike": analysis.strike, "edge": edge, "qty": qty},
        )

    async def maybe_start_conv_rev_bundle(self, chain: ChainAnalysis) -> Optional[Decision]:
        if self.active_bundle is not None or self.active_campaign is not None:
            return None
        if not self.startup_flatten_complete:
            return None
        if self.current_campaign_position() != 0:
            return Decision(action="skip_conv_rev", reason="b_inventory_not_flat")
        if self.option_inventory_present():
            return Decision(action="skip_conv_rev", reason="option_inventory_present")
        candidate = self.best_conv_rev_candidate(chain)
        if candidate is None:
            return None
        kind, analysis, edge = candidate
        return await self.start_conv_rev_bundle(kind, analysis, edge)

    def bundle_live_order_ids(self, bundle: ConvRevBundle) -> list[str]:
        live = []
        for order_id in bundle.order_ids:
            if self.remaining_qty(order_id) > 0:
                live.append(order_id)
        return live

    def bundle_positions(self, bundle: ConvRevBundle) -> dict[str, int]:
        return {symbol: self.position(symbol) for symbol in bundle.expected_positions}

    def bundle_fully_filled(self, bundle: ConvRevBundle) -> bool:
        if self.bundle_live_order_ids(bundle):
            return False
        current = self.bundle_positions(bundle)
        return current == bundle.expected_positions

    async def cancel_bundle_live_orders(self, bundle: ConvRevBundle, reason: str) -> bool:
        live_ids = self.bundle_live_order_ids(bundle)
        if not live_ids:
            return False
        cancelled_any = False
        for order_id in live_ids:
            cancelled_any = await self.request_cancel(order_id, reason) or cancelled_any
        if cancelled_any:
            self.record_event(
                "bundle_cancel_requested",
                bundle_id=bundle.bundle_id,
                reason=reason,
                live_ids=live_ids,
            )
        return cancelled_any

    def repair_priority(self, bundle: ConvRevBundle) -> list[tuple[str, int]]:
        positions = self.bundle_positions(bundle)
        option_symbols = [symbol for symbol in positions if symbol != "B" and positions[symbol] != 0]
        option_symbols.sort(
            key=lambda symbol: (
                self.option_spread(symbol),
                -self.top_depth(symbol),
                0 if "_1000" in symbol else 1,
                symbol,
            )
        )
        ordered = []
        if positions.get("B", 0) != 0:
            ordered.append(("B", positions["B"]))
        for symbol in option_symbols:
            ordered.append((symbol, positions[symbol]))
        return ordered

    def option_spread(self, symbol: str) -> int:
        bid, _bid_qty, ask, _ask_qty = self.top(symbol)
        if bid is None or ask is None:
            return 10 ** 6
        return max(0, ask - bid)

    def top_depth(self, symbol: str) -> int:
        bid, bid_qty, ask, ask_qty = self.top(symbol)
        if bid is None and ask is None:
            return 0
        return max(bid_qty, ask_qty)

    async def bundle_repair_step(self, bundle: ConvRevBundle) -> Optional[Decision]:
        live_ids = self.bundle_live_order_ids(bundle)
        if live_ids:
            await self.cancel_bundle_live_orders(bundle, bundle.last_reason or "repair")
            return Decision(action="bundle_cancel", reason=bundle.last_reason or "repair")

        outstanding = [(symbol, pos) for symbol, pos in self.bundle_positions(bundle).items() if pos != 0]
        if not outstanding:
            self.record_event(
                "bundle_failed_cleaned",
                bundle_id=bundle.bundle_id,
                strike=bundle.strike,
                reason=bundle.last_reason,
            )
            self.bundle_cooldown_until[bundle.strike] = self.now() + self.cfg.conv_rev_cooldown_sec
            self.active_bundle = None
            return Decision(action="bundle_cleaned", reason=bundle.last_reason or "repair_completed")

        symbol, pos = self.repair_priority(bundle)[0]
        bid, _bid_qty, ask, _ask_qty = self.top(symbol)
        if pos > 0:
            side = Side.SELL
            price = bid
        else:
            side = Side.BUY
            price = ask
        if price is None:
            return Decision(action="bundle_wait", reason=f"repair_no_liquidity_{symbol}")
        qty = min(abs(pos), self.cfg.hard_max_order_size)
        order_id, block = await self.submit_limit(
            symbol,
            side,
            qty,
            int(price),
            purpose="bundle_repair",
            strategy="bundle_repair",
            strike=bundle.strike,
            role="repair",
            bundle_id=bundle.bundle_id,
            allow_option=True,
        )
        if block is not None:
            return block
        bundle.last_reason = f"repair_{symbol}"
        self.action_counts["repairs"] += 1
        return Decision(
            action="bundle_repair",
            reason=f"repair_{symbol}",
            details={"order_id": order_id, "qty": qty, "price": price},
        )

    async def maybe_maintain_bundle(self) -> Optional[Decision]:
        bundle = self.active_bundle
        if bundle is None:
            return None
        if bundle.state == "DONE":
            self.bundle_cooldown_until[bundle.strike] = self.now() + self.cfg.conv_rev_cooldown_sec
            self.record_event(
                "bundle_done",
                bundle_id=bundle.bundle_id,
                strike=bundle.strike,
                positions=self.bundle_positions(bundle),
            )
            self.active_bundle = None
            return Decision(action="bundle_done", reason="bundle_complete")
        if bundle.state in {"REPAIR", "PARTIAL"}:
            return await self.bundle_repair_step(bundle)

        if bundle.first_fill_seen and not bundle.post_fill_checked:
            bundle.current_edge = self.bundle_current_edge(bundle)
            bundle.post_fill_checked = True
            if bundle.current_edge is None or bundle.current_edge < self.cfg.post_fill_keep_edge:
                bundle.state = "REPAIR"
                bundle.last_reason = "conv_rev_post_fill_edge_collapsed"
                self.record_event(
                    "bundle_edge_collapse",
                    bundle_id=bundle.bundle_id,
                    current_edge=bundle.current_edge,
                    threshold=self.cfg.post_fill_keep_edge,
                )
                return await self.bundle_repair_step(bundle)

        if self.bundle_fully_filled(bundle):
            bundle.state = "DONE"
            bundle.last_reason = "bundle_filled"
            return Decision(action="bundle_done", reason="bundle_filled")

        live_ids = self.bundle_live_order_ids(bundle)
        if not live_ids:
            bundle.state = "PARTIAL"
            bundle.last_reason = "bundle_partial_missing_legs"
            return await self.bundle_repair_step(bundle)

        return Decision(action="bundle_wait", reason="bundle_working")

    async def maybe_startup_flatten(self) -> Optional[Decision]:
        if self.startup_flatten_complete:
            return None
        if not self.local_positions_seeded:
            return Decision(action="startup_wait", reason="waiting_for_positions")

        live_orders = [order_id for order_id in self.order_meta if self.remaining_qty(order_id) > 0]
        if live_orders:
            cancelled = False
            for order_id in live_orders:
                cancelled = await self.request_cancel(order_id, "startup_flatten") or cancelled
            if cancelled:
                return Decision(action="startup_cancel", reason="cancel_open_orders")

        for symbol in self.tracked_symbols():
            pos = self.position(symbol)
            if pos == 0:
                continue
            bid, _bid_qty, ask, _ask_qty = self.top(symbol)
            if pos > 0:
                side = Side.SELL
                price = bid
            else:
                side = Side.BUY
                price = ask
            if price is None:
                return Decision(action="startup_wait", reason=f"no_liquidity_{symbol}")
            qty = min(abs(pos), self.cfg.hard_max_order_size)
            order_id, block = await self.submit_limit(
                symbol,
                side,
                qty,
                int(price),
                purpose="startup_flatten",
                strategy="startup_flatten",
                role="flatten",
                allow_option=True,
            )
            if block is not None:
                return block
            return Decision(
                action="startup_flatten",
                reason=f"flatten_{symbol}",
                details={"order_id": order_id, "qty": qty, "price": price},
            )

        self.startup_flatten_complete = True
        self.record_event("startup_flatten_complete")
        return Decision(action="startup_complete", reason="all_positions_flat")

    async def maybe_cancel_unmanaged_stale(self) -> None:
        for order_id, meta in list(self.order_meta.items()):
            if meta.cancel_pending:
                continue
            if meta.strategy in {"b_campaign", "conv_rev", "bundle_repair", "startup_flatten"}:
                continue
            if self.remaining_qty(order_id) <= 0:
                self.order_meta.pop(order_id, None)
                continue
            if self.now() - meta.created_at >= max(self.cfg.entry_ttl_sec, self.cfg.exit_ttl_sec):
                await self.request_cancel(order_id, "generic_stale")

    async def run_strategy(self) -> None:
        async with self._strategy_lock:
            self.cycle_count += 1
            self.seed_and_reconcile_local_positions()
            self.last_chain = self.analyze_chain()
            await self.clear_campaign_if_ready()
            await self.maybe_cancel_unmanaged_stale()

            decision = await self.maybe_startup_flatten()
            if decision is None:
                decision = await self.maybe_maintain_bundle()
            if decision is None:
                decision = await self.maybe_maintain_b_campaign(self.last_chain)
            if decision is None:
                decision = await self.maybe_enter_b_campaign(self.last_chain)
            if decision is None:
                decision = await self.maybe_start_conv_rev_bundle(self.last_chain)
            if decision is None:
                decision = Decision(action="diagnostics_only", reason="no_live_action")

            self.last_decision = decision
            self.record_snapshot()
            self.print_status()

    async def process_message(self, msg) -> None:
        await super().process_message(msg)

    async def strategy_loop(self) -> None:
        await asyncio.sleep(1.0)
        print(
            f"[INIT] host={self.cfg.host} user={self.cfg.username} "
            f"enter_edge={self.cfg.enter_edge_ticks} strong_cross={self.cfg.strong_cross_ticks} "
            f"conv_rev_min_edge={self.cfg.conv_rev_min_edge_ticks}"
        )
        while True:
            try:
                try:
                    await asyncio.wait_for(self._book_event.wait(), timeout=self.cfg.poll_sec)
                except asyncio.TimeoutError:
                    pass
                self._book_event.clear()
                await self.run_strategy()
            except Exception as exc:
                print(f"[STRATEGY-ERROR] {exc}")
                self.record_event("strategy_error", error=str(exc))
                await asyncio.sleep(0.25)

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol == "B" or OPTION_RE.fullmatch(symbol):
            self._book_event.set()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        self.last_trade[symbol] = int(price)
        self.record_event("trade_msg", symbol=symbol, price=price, qty=qty)
        if symbol == "B" or OPTION_RE.fullmatch(symbol):
            self._book_event.set()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        meta = self.order_meta.get(order_id)
        if meta is not None:
            current_local = self.local_positions.get(meta.symbol, self.exchange_position(meta.symbol))
            delta = qty if meta.side == Side.BUY else -qty
            self.local_positions[meta.symbol] = current_local + delta
            self.last_local_fill_at[meta.symbol] = self.now()
            self.record_event(
                "order_fill",
                order_id=order_id,
                purpose=meta.purpose,
                strategy=meta.strategy,
                symbol=meta.symbol,
                side=meta.side.name,
                qty=qty,
                price=price,
                local_position=self.local_positions[meta.symbol],
                exchange_position=self.exchange_position(meta.symbol),
            )
            if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
                self.active_campaign.filled_qty += qty
                self.active_campaign.fill_notional += qty * price
                if self.active_campaign.current_order_role == "entry":
                    self.active_campaign.state = "ENTRY_FILLED"
                    self.active_campaign.last_reason = "entry_fill_observed"
                    if order_id in self.open_orders:
                        await self.request_cancel(order_id, "entry_partial_fill")
                    else:
                        self.active_campaign.current_order_id = None
                        self.active_campaign.current_order_role = None
                elif self.active_campaign.current_order_role == "exit":
                    if order_id in self.open_orders:
                        await self.request_cancel(order_id, "exit_partial_fill")
                    else:
                        self.active_campaign.current_order_id = None
                        self.active_campaign.current_order_role = None
            if self.active_bundle is not None and meta.bundle_id == self.active_bundle.bundle_id:
                self.active_bundle.first_fill_seen = True
                self.active_bundle.fills_by_symbol[meta.symbol] = (
                    self.active_bundle.fills_by_symbol.get(meta.symbol, 0) + (qty if meta.side == Side.BUY else -qty)
                )
                self.active_bundle.fill_notional_by_symbol[meta.symbol] = (
                    self.active_bundle.fill_notional_by_symbol.get(meta.symbol, 0.0) + qty * price
                )
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)
        self._book_event.set()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        meta = self.order_meta.get(order_id)
        self.action_counts["rejections"] += 1
        if meta is not None:
            self.record_event(
                "order_rejected",
                order_id=order_id,
                purpose=meta.purpose,
                strategy=meta.strategy,
                symbol=meta.symbol,
                side=meta.side.name,
                reason=reason,
                role=meta.role,
                bundle_id=meta.bundle_id,
            )
            if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
                self.active_campaign.current_order_id = None
                self.active_campaign.current_order_role = None
                if self.current_campaign_position() == 0:
                    await self.move_campaign_to_cooldown(f"order_rejected:{reason}")
                else:
                    self.active_campaign.state = "REPAIR_EXIT"
                    self.active_campaign.last_reason = f"order_rejected:{reason}"
            if self.active_bundle is not None and meta.bundle_id == self.active_bundle.bundle_id:
                self.active_bundle.state = "REPAIR"
                self.active_bundle.last_reason = f"order_rejected:{reason}"
        self.order_meta.pop(order_id, None)
        self._book_event.set()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        meta = self.order_meta.get(order_id)
        if meta is not None:
            meta.cancel_pending = False
        if success:
            if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
                self.active_campaign.current_order_id = None
                self.active_campaign.current_order_role = None
                if self.current_campaign_position() == 0 and self.active_campaign.state == "ENTRY_RESTING":
                    await self.move_campaign_to_cooldown("entry_cancelled")
                elif self.current_campaign_position() == 0 and self.active_campaign.state in {"EXIT_RESTING", "REPAIR_EXIT"}:
                    await self.move_campaign_to_cooldown("exit_cancelled_flat")
                elif self.current_campaign_position() != 0 and self.active_campaign.state in {"EXIT_RESTING", "REPAIR_EXIT"}:
                    self.active_campaign.state = "REPAIR_EXIT"
                    self.active_campaign.last_reason = "exit_cancelled_with_position"
            self.order_meta.pop(order_id, None)
        else:
            self.record_event("cancel_failed", order_id=order_id, error=error)
        if self.active_bundle is not None and meta is not None and meta.bundle_id == self.active_bundle.bundle_id:
            if not success:
                self.active_bundle.state = "REPAIR"
                self.active_bundle.last_reason = "bundle_cancel_failed"
        self.record_event("cancel_response", order_id=order_id, success=success, error=error)
        self._book_event.set()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        return

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        return

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        return

    async def start(self) -> None:
        asyncio.create_task(self.strategy_loop())
        await self.connect()


async def main() -> None:
    bot = MarketBExecutableBandBot(load_config())
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
