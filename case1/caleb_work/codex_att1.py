from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import sys
import time
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


@dataclass(frozen=True)
class BotConfig:
    host: str
    username: str
    password: str
    risk_free_rate: float
    time_to_expiry_years: float
    max_position_per_option: int
    max_total_outstanding: int
    max_trade_qty: int
    parity_entry_edge: float
    parity_exit_band: float
    per_strike_cooldown_sec: float
    order_ttl_sec: float
    poll_sec: float
    status_sec: float
    startup_probe_enabled: bool
    startup_probe_qty: int
    startup_probe_timeout_sec: float
    startup_probe_max_attempts: int
    passive_entry_ttl_sec: float
    passive_hedge_ttl_sec: float
    passive_unwind_ttl_sec: float
    passive_unwind_reprice_sec: float
    capture_enabled: bool
    capture_snapshot_interval_sec: float
    capture_dir: str


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
    strike: int
    created_at: float
    cancel_pending: bool = False


@dataclass(frozen=True)
class ParitySnapshot:
    strike: int
    spot: float
    discounted_strike: float
    theoretical_cp_diff: float
    call_bid: Optional[int]
    call_bid_qty: int
    call_ask: Optional[int]
    call_ask_qty: int
    put_bid: Optional[int]
    put_bid_qty: int
    put_ask: Optional[int]
    put_ask_qty: int
    mid_gap: Optional[float]
    long_synth_edge: Optional[float]
    short_synth_edge: Optional[float]


@dataclass(frozen=True)
class PassiveQuotePlan:
    symbol: str
    side: Side
    price: int
    qty: int
    preserved_edge: float
    option_spread: int
    top_depth: int
    note: str


@dataclass(frozen=True)
class CampaignCandidate:
    kind: str
    pair: OptionPair
    first_leg: PassiveQuotePlan
    hedge_symbol: str
    hedge_side: Side
    score: float
    note: str


@dataclass
class PassiveCampaign:
    state: str
    kind: str
    pair: OptionPair
    first_leg: PassiveQuotePlan
    hedge_symbol: str
    hedge_side: Side
    started_at: float
    stage_started_at: float
    current_order_id: Optional[str] = None
    current_role: Optional[str] = None
    first_filled_qty: int = 0
    first_fill_notional: float = 0.0
    hedge_filled_qty: int = 0
    hedge_fill_notional: float = 0.0
    unwind_filled_qty: int = 0
    unwind_fill_notional: float = 0.0
    unwind_reprices: int = 0
    last_reprice_at: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)

    def first_fill_price(self) -> Optional[float]:
        if self.first_filled_qty <= 0:
            return None
        return self.first_fill_notional / self.first_filled_qty

    def residual_qty(self) -> int:
        return max(0, self.first_filled_qty - self.hedge_filled_qty - self.unwind_filled_qty)


def load_config() -> BotConfig:
    return BotConfig(
        host=env_str("UTC_HOST", "34.197.188.76:3333"),
        username=env_str("UTC_USERNAME", "uiuc"),
        password=env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
        risk_free_rate=env_float("B_RISK_FREE_RATE", 0.0),
        time_to_expiry_years=env_float("B_TIME_TO_EXPIRY_YEARS", 1.0),
        max_position_per_option=env_int("B_MAX_OPTION_POSITION", 6),
        max_total_outstanding=env_int("B_MAX_TOTAL_OUTSTANDING", 18),
        max_trade_qty=env_int("B_MAX_TRADE_QTY", 1),
        parity_entry_edge=env_float("B_PARITY_ENTRY_EDGE", 1.5),
        parity_exit_band=env_float("B_PARITY_EXIT_BAND", 1.0),
        per_strike_cooldown_sec=env_float("B_STRIKE_COOLDOWN_SEC", 0.75),
        order_ttl_sec=env_float("B_ORDER_TTL_SEC", 0.75),
        poll_sec=env_float("B_POLL_SEC", 0.25),
        status_sec=env_float("B_STATUS_SEC", 5.0),
        startup_probe_enabled=env_bool("B_STARTUP_PROBE", False),
        startup_probe_qty=max(1, env_int("B_STARTUP_PROBE_QTY", 1)),
        startup_probe_timeout_sec=env_float("B_STARTUP_PROBE_TIMEOUT_SEC", 6.0),
        startup_probe_max_attempts=max(1, env_int("B_STARTUP_PROBE_MAX_ATTEMPTS", 2)),
        passive_entry_ttl_sec=env_float("B_PASSIVE_ENTRY_TTL_SEC", 5.0),
        passive_hedge_ttl_sec=env_float("B_PASSIVE_HEDGE_TTL_SEC", 3.0),
        passive_unwind_ttl_sec=env_float("B_PASSIVE_UNWIND_TTL_SEC", 5.0),
        passive_unwind_reprice_sec=env_float("B_PASSIVE_UNWIND_REPRICE_SEC", 1.0),
        capture_enabled=env_bool("B_CAPTURE_ENABLED", True),
        capture_snapshot_interval_sec=env_float("B_CAPTURE_SNAPSHOT_INTERVAL_SEC", 1.0),
        capture_dir=env_str("B_CAPTURE_DIR", str(Path(__file__).resolve().parent / "run_logs")),
    )


def live_bid(book) -> tuple[Optional[int], int]:
    levels = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
    if not levels:
        return None, 0
    return max(levels, key=lambda level: level[0])


def live_ask(book) -> tuple[Optional[int], int]:
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


class MarketBBot(XChangeClient):
    def __init__(self, config: BotConfig):
        super().__init__(config.host, config.username, config.password, silent=False)
        self.cfg = config
        self.option_pairs: list[OptionPair] = []
        self.last_trade: dict[str, int] = {}
        self.order_meta: dict[str, OrderMeta] = {}
        self.local_positions: dict[str, int] = {}
        self.exchange_positions_view: dict[str, int] = {}
        self.last_local_fill_at: dict[str, float] = {}
        self.last_action_at: dict[int, float] = {}
        self.last_gate_log_at: dict[str, float] = {}
        self.last_status = 0.0
        self._book_event = asyncio.Event()
        self._strategy_lock = asyncio.Lock()
        self.local_positions_seeded = False
        self.startup_probe_state = "disabled" if not config.startup_probe_enabled else "entry_pending"
        self.startup_probe_symbol: Optional[str] = None
        self.startup_probe_strike: Optional[int] = None
        self.startup_probe_started_at = self.now()
        self.startup_probe_entry_attempts = 0
        self.startup_probe_exit_attempts = 0
        self.active_campaign: Optional[PassiveCampaign] = None
        self.capture_file_path: Optional[Path] = None
        self.capture_handle = None
        self.discover_option_pairs()
        self.setup_capture()

    def now(self) -> float:
        return time.monotonic()

    def setup_capture(self) -> None:
        if not self.cfg.capture_enabled:
            return
        capture_dir = Path(self.cfg.capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        session_name = time.strftime("market_b_%Y%m%d_%H%M%S")
        self.capture_file_path = capture_dir / f"{session_name}.jsonl"
        self.capture_handle = self.capture_file_path.open("a", encoding="utf-8", buffering=1)
        print(f"[CAPTURE] file={self.capture_file_path}")
        self.record_event(
            "session_start",
            config={
                "host": self.cfg.host,
                "username": self.cfg.username,
                "risk_free_rate": self.cfg.risk_free_rate,
                "time_to_expiry_years": self.cfg.time_to_expiry_years,
                "parity_entry_edge": self.cfg.parity_entry_edge,
                "parity_exit_band": self.cfg.parity_exit_band,
                "startup_probe_enabled": self.cfg.startup_probe_enabled,
                "startup_probe_timeout_sec": self.cfg.startup_probe_timeout_sec,
                "startup_probe_max_attempts": self.cfg.startup_probe_max_attempts,
                "passive_entry_ttl_sec": self.cfg.passive_entry_ttl_sec,
                "passive_hedge_ttl_sec": self.cfg.passive_hedge_ttl_sec,
                "passive_unwind_ttl_sec": self.cfg.passive_unwind_ttl_sec,
                "passive_unwind_reprice_sec": self.cfg.passive_unwind_reprice_sec,
                "capture_snapshot_interval_sec": self.cfg.capture_snapshot_interval_sec,
            },
        )

    def discover_option_pairs(self) -> None:
        discovered: dict[int, dict[str, str]] = {}
        for symbol in set(self.symbols) | set(self.order_books.keys()):
            match = OPTION_RE.fullmatch(symbol)
            if not match:
                continue
            side, strike_raw = match.groups()
            discovered.setdefault(int(strike_raw), {})[side] = symbol

        pairs: list[OptionPair] = []
        for strike in DEFAULT_STRIKES:
            symbols = discovered.get(strike, {})
            call_symbol = symbols.get("C", f"B_C_{strike}")
            put_symbol = symbols.get("P", f"B_P_{strike}")
            pairs.append(OptionPair(strike=strike, call_symbol=call_symbol, put_symbol=put_symbol))
        self.option_pairs = pairs

    def get_book(self, symbol: str):
        return self.order_books.get(symbol)

    def option_symbols(self) -> list[str]:
        symbols: list[str] = []
        for pair in self.option_pairs:
            symbols.extend((pair.call_symbol, pair.put_symbol))
        return symbols

    def tracked_symbols(self) -> list[str]:
        return ["B", *self.option_symbols()]

    def normalize_value(self, value):
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

    def serialize_book(self, symbol: str) -> dict:
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
        return {
            "bids": bids,
            "asks": asks,
            "last_trade": self.last_trade.get(symbol),
        }

    def serialize_open_orders(self) -> list[dict]:
        rows: list[dict] = []
        for order_id, meta in self.order_meta.items():
            order = self.open_orders.get(order_id)
            remaining_qty = int(order[1]) if order else 0
            rows.append(
                {
                    "order_id": order_id,
                    "symbol": meta.symbol,
                    "side": meta.side.name,
                    "price": meta.price,
                    "remaining_qty": remaining_qty,
                    "purpose": meta.purpose,
                    "strike": meta.strike,
                    "cancel_pending": meta.cancel_pending,
                    "age_sec": round(self.now() - meta.created_at, 3),
                }
            )
        return rows

    def serialize_parity(self, spot: Optional[float]) -> dict:
        rows: dict[str, dict] = {}
        if spot is None:
            return rows
        for pair in self.option_pairs:
            snapshot = self.parity_snapshot(pair, spot)
            if snapshot is None:
                continue
            rows[str(pair.strike)] = {
                "spot": snapshot.spot,
                "discounted_strike": snapshot.discounted_strike,
                "theoretical_cp_diff": snapshot.theoretical_cp_diff,
                "call_bid": snapshot.call_bid,
                "call_ask": snapshot.call_ask,
                "put_bid": snapshot.put_bid,
                "put_ask": snapshot.put_ask,
                "mid_gap": snapshot.mid_gap,
                "long_synth_edge": snapshot.long_synth_edge,
                "short_synth_edge": snapshot.short_synth_edge,
                "pending_order_count": self.pending_order_count_for_strike(pair.strike),
                "local_call_pos": self.position(pair.call_symbol),
                "local_put_pos": self.position(pair.put_symbol),
                "exchange_call_pos": self.exchange_position(pair.call_symbol),
                "exchange_put_pos": self.exchange_position(pair.put_symbol),
            }
        return rows

    def serialize_campaign(self) -> Optional[dict]:
        if self.active_campaign is None:
            return None
        campaign = self.active_campaign
        return {
            "state": campaign.state,
            "kind": campaign.kind,
            "strike": campaign.pair.strike,
            "current_order_id": campaign.current_order_id,
            "current_role": campaign.current_role,
            "first_leg": {
                "symbol": campaign.first_leg.symbol,
                "side": campaign.first_leg.side.name,
                "price": campaign.first_leg.price,
                "qty": campaign.first_leg.qty,
                "preserved_edge": campaign.first_leg.preserved_edge,
                "note": campaign.first_leg.note,
            },
            "hedge_symbol": campaign.hedge_symbol,
            "hedge_side": campaign.hedge_side.name,
            "first_filled_qty": campaign.first_filled_qty,
            "first_fill_price": campaign.first_fill_price(),
            "hedge_filled_qty": campaign.hedge_filled_qty,
            "unwind_filled_qty": campaign.unwind_filled_qty,
            "residual_qty": campaign.residual_qty(),
            "unwind_reprices": campaign.unwind_reprices,
            "notes": campaign.notes,
        }

    def record_snapshot(self, reason: str) -> None:
        spot = self.extract_spot()
        self.record_event(
            "snapshot",
            reason=reason,
            probe_state=self.startup_probe_state,
            campaign=self.serialize_campaign(),
            spot=spot,
            positions={
                "local": {symbol: self.position(symbol) for symbol in self.option_symbols()},
                "exchange": {symbol: self.exchange_position(symbol) for symbol in self.option_symbols()},
                "cash": self.positions.get("cash", 0),
            },
            open_orders=self.serialize_open_orders(),
            books={symbol: self.serialize_book(symbol) for symbol in self.tracked_symbols()},
            parity=self.serialize_parity(spot),
        )

    def top(self, symbol: str) -> tuple[Optional[int], int, Optional[int], int]:
        book = self.get_book(symbol)
        if book is None:
            return None, 0, None, 0
        bid, bid_qty = live_bid(book)
        ask, ask_qty = live_ask(book)
        return bid, bid_qty, ask, ask_qty

    def mid(self, symbol: str) -> Optional[float]:
        book = self.get_book(symbol)
        if book is None:
            return self.last_trade.get(symbol)
        return mid_from_book(book, fallback=self.last_trade.get(symbol))

    def extract_spot(self) -> Optional[float]:
        return self.mid("B")

    def discounted_strike(self, strike: int) -> float:
        return float(strike) * math.exp(-self.cfg.risk_free_rate * self.cfg.time_to_expiry_years)

    def exchange_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def position(self, symbol: str) -> int:
        if OPTION_RE.fullmatch(symbol) and self.local_positions_seeded:
            return int(self.local_positions.get(symbol, self.exchange_position(symbol)))
        return self.exchange_position(symbol)

    def seed_and_reconcile_local_positions(self) -> None:
        tracked = {symbol: self.exchange_position(symbol) for symbol in self.option_symbols()}
        self.exchange_positions_view = tracked
        if not tracked:
            return
        if not self.local_positions_seeded:
            self.local_positions = dict(tracked)
            self.local_positions_seeded = True
            print("[SYNC] seeded local option positions from exchange")
            return

        now = self.now()
        for symbol, exchange_pos in tracked.items():
            local_pos = int(self.local_positions.get(symbol, 0))
            has_pending = self.pending_total_for_symbol(symbol) > 0
            recently_filled = now - self.last_local_fill_at.get(symbol, 0.0) < max(2.0, self.cfg.order_ttl_sec * 2.0)
            if local_pos != exchange_pos and not has_pending and not recently_filled:
                print(f"[SYNC] reconcile symbol={symbol} local={local_pos} exchange={exchange_pos}")
                self.local_positions[symbol] = exchange_pos

    def outstanding_qty(self) -> int:
        total = 0
        for order_id, meta in self.order_meta.items():
            if meta.cancel_pending:
                continue
            order = self.open_orders.get(order_id)
            if not order:
                continue
            total += int(order[1])
        return total

    def pending_side_qty(self, symbol: str, side: Side) -> int:
        total = 0
        for order_id, meta in self.order_meta.items():
            if meta.symbol != symbol or meta.side != side or meta.cancel_pending:
                continue
            order = self.open_orders.get(order_id)
            if not order:
                continue
            total += int(order[1])
        return total

    def pending_total_for_symbol(self, symbol: str) -> int:
        total = 0
        for order_id, meta in self.order_meta.items():
            if meta.symbol != symbol or meta.cancel_pending:
                continue
            order = self.open_orders.get(order_id)
            if not order:
                continue
            total += int(order[1])
        return total

    def pending_order_count_for_strike(self, strike: int) -> int:
        total = 0
        for order_id, meta in self.order_meta.items():
            if meta.strike != strike or meta.cancel_pending:
                continue
            if order_id in self.open_orders:
                total += 1
        return total

    def has_any_option_inventory_or_orders(self) -> bool:
        if self.outstanding_qty() > 0:
            return True
        for symbol in self.option_symbols():
            if self.position(symbol) != 0:
                return True
        return False

    def signed_exposure(self, symbol: str) -> int:
        return self.position(symbol) + self.pending_side_qty(symbol, Side.BUY) - self.pending_side_qty(symbol, Side.SELL)

    def submission_block_reason(self, symbol: str, side: Side, qty: int) -> Optional[str]:
        if qty <= 0:
            return f"nonpositive_qty={qty}"
        if not OPTION_RE.fullmatch(symbol):
            return f"invalid_symbol={symbol}"
        if self.outstanding_qty() + qty > self.cfg.max_total_outstanding:
            return (
                f"outstanding_limit current={self.outstanding_qty()} add={qty} "
                f"limit={self.cfg.max_total_outstanding}"
            )
        projected = self.signed_exposure(symbol) + (qty if side == Side.BUY else -qty)
        if abs(projected) > self.cfg.max_position_per_option:
            return (
                f"position_limit symbol={symbol} projected={projected} "
                f"limit={self.cfg.max_position_per_option}"
            )
        return None

    def log_gate(self, key: str, message: str, throttle_sec: float = 1.5) -> None:
        now = self.now()
        if now - self.last_gate_log_at.get(key, 0.0) < throttle_sec:
            return
        self.last_gate_log_at[key] = now
        print(message)
        self.record_event("gate", key=key, message=message)

    def probe_active(self) -> bool:
        return self.startup_probe_state not in {"disabled", "done", "failed"}

    def set_probe_state(self, state: str, reason: Optional[str] = None) -> None:
        if state == self.startup_probe_state and reason is None:
            return
        self.startup_probe_state = state
        self.record_event("probe_state", state=state, reason=reason)
        if reason:
            print(f"[PROBE] state={state} reason={reason}")
        else:
            print(f"[PROBE] state={state}")

    def mark_probe_failed(self, reason: str) -> None:
        self.startup_probe_symbol = None
        self.startup_probe_strike = None
        self.set_probe_state("failed", reason)

    def mark_probe_done(self, reason: Optional[str] = None) -> None:
        self.startup_probe_symbol = None
        self.startup_probe_strike = None
        self.set_probe_state("done", reason)

    async def submit_cross(self, symbol: str, side: Side, qty: int, price: int, purpose: str, strike: int) -> Optional[str]:
        reason = self.submission_block_reason(symbol, side, qty)
        if reason is not None:
            print(f"[BLOCK] {purpose} {symbol} {side.name} qty={qty} px={price}: {reason}")
            self.record_event(
                "order_blocked",
                purpose=purpose,
                symbol=symbol,
                side=side.name,
                qty=qty,
                price=price,
                strike=strike,
                reason=reason,
            )
            return None
        try:
            order_id = await self.place_order(symbol, int(qty), side, int(price))
        except Exception as exc:
            print(f"[ORDER-ERROR] {purpose} {symbol} {side.name} {qty}@{price}: {exc}")
            self.record_event(
                "order_error",
                purpose=purpose,
                symbol=symbol,
                side=side.name,
                qty=qty,
                price=price,
                strike=strike,
                error=str(exc),
            )
            return None

        self.order_meta[order_id] = OrderMeta(
            symbol=symbol,
            side=side,
            price=int(price),
            qty=int(qty),
            purpose=purpose,
            strike=int(strike),
            created_at=self.now(),
        )
        self.record_event(
            "order_submitted",
            order_id=order_id,
            purpose=purpose,
            symbol=symbol,
            side=side.name,
            qty=qty,
            price=price,
            strike=strike,
        )
        return order_id

    async def cancel_stale_orders(self) -> None:
        now = self.now()
        for order_id, meta in list(self.order_meta.items()):
            if meta.cancel_pending:
                continue
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)
                continue
            if meta.purpose.startswith("campaign:"):
                continue
            if now - meta.created_at < self.cfg.order_ttl_sec:
                continue
            meta.cancel_pending = True
            print(
                f"[STALE] cancel order_id={order_id} purpose={meta.purpose} "
                f"symbol={meta.symbol} age={now - meta.created_at:.2f}s"
            )
            self.record_event(
                "order_cancel_requested",
                order_id=order_id,
                purpose=meta.purpose,
                symbol=meta.symbol,
                strike=meta.strike,
                age_sec=round(now - meta.created_at, 3),
                reason="stale",
            )
            try:
                await self.cancel_order(order_id)
            except Exception as exc:
                meta.cancel_pending = False
                print(f"[CANCEL-ERROR] {order_id}: {exc}")

    def paired_inventory(self, pair: OptionPair) -> tuple[int, int, int]:
        call_pos = self.position(pair.call_symbol)
        put_pos = self.position(pair.put_symbol)
        synthetic_long = min(max(call_pos, 0), max(-put_pos, 0))
        synthetic_short = min(max(-call_pos, 0), max(put_pos, 0))
        imbalance = abs(call_pos + put_pos)
        return synthetic_long, synthetic_short, imbalance

    def parity_snapshot(self, pair: OptionPair, spot: float) -> Optional[ParitySnapshot]:
        call_bid, call_bid_qty, call_ask, call_ask_qty = self.top(pair.call_symbol)
        put_bid, put_bid_qty, put_ask, put_ask_qty = self.top(pair.put_symbol)

        call_mid = self.mid(pair.call_symbol)
        put_mid = self.mid(pair.put_symbol)
        discounted = self.discounted_strike(pair.strike)
        theoretical = spot - discounted

        mid_gap = None
        if call_mid is not None and put_mid is not None:
            mid_gap = (call_mid - put_mid) - theoretical

        long_edge = None
        if call_ask is not None and put_bid is not None:
            long_edge = theoretical - (call_ask - put_bid)

        short_edge = None
        if call_bid is not None and put_ask is not None:
            short_edge = (call_bid - put_ask) - theoretical

        return ParitySnapshot(
            strike=pair.strike,
            spot=float(spot),
            discounted_strike=float(discounted),
            theoretical_cp_diff=float(theoretical),
            call_bid=call_bid,
            call_bid_qty=call_bid_qty,
            call_ask=call_ask,
            call_ask_qty=call_ask_qty,
            put_bid=put_bid,
            put_bid_qty=put_bid_qty,
            put_ask=put_ask,
            put_ask_qty=put_ask_qty,
            mid_gap=mid_gap,
            long_synth_edge=long_edge,
            short_synth_edge=short_edge,
        )

    def cooldown_ready(self, strike: int) -> bool:
        return self.now() - self.last_action_at.get(strike, 0.0) >= self.cfg.per_strike_cooldown_sec

    def strike_has_pending_orders(self, strike: int) -> bool:
        for order_id, meta in self.order_meta.items():
            if meta.strike != strike or meta.cancel_pending:
                continue
            if order_id in self.open_orders:
                return True
        return False

    def has_live_probe_orders(self) -> bool:
        for order_id, meta in self.order_meta.items():
            if not meta.purpose.startswith("startup-probe") or meta.cancel_pending:
                continue
            if order_id in self.open_orders:
                return True
        return False

    def best_probe_candidate(self) -> Optional[tuple[str, int, int, int, int]]:
        candidates: list[tuple[int, int, str, int, int, int]] = []
        for pair in self.option_pairs:
            for symbol in (pair.call_symbol, pair.put_symbol):
                bid, bid_qty, ask, ask_qty = self.top(symbol)
                if bid is None or ask is None or bid_qty <= 0 or ask_qty <= 0:
                    continue
                spread = ask - bid
                depth = min(bid_qty, ask_qty)
                candidates.append((spread, -depth, symbol, pair.strike, bid, ask))
        if not candidates:
            return None
        spread, _neg_depth, symbol, strike, bid, ask = min(candidates)
        return symbol, strike, bid, ask, spread

    async def maybe_run_startup_probe(self) -> bool:
        if not self.cfg.startup_probe_enabled:
            self.set_probe_state("disabled")
            return False
        if self.startup_probe_state in {"done", "failed"}:
            return False

        elapsed = self.now() - self.startup_probe_started_at
        if elapsed >= self.cfg.startup_probe_timeout_sec and not self.has_live_probe_orders():
            self.mark_probe_failed(f"timeout elapsed={elapsed:.2f}s")
            return False

        symbol = self.startup_probe_symbol
        strike = self.startup_probe_strike

        if self.startup_probe_state == "entry_pending":
            if symbol is not None and self.has_live_probe_orders():
                return False

            if symbol is not None and self.position(symbol) > 0:
                self.set_probe_state("entry_filled_waiting_exit", f"symbol={symbol} pos={self.position(symbol)}")
                return True

            if symbol is not None and not self.has_live_probe_orders():
                if self.startup_probe_entry_attempts >= self.cfg.startup_probe_max_attempts:
                    self.mark_probe_failed(
                        f"entry_unfilled symbol={symbol} attempts={self.startup_probe_entry_attempts}"
                    )
                    return False
                print(f"[PROBE] entry unfilled symbol={symbol}; retrying")
                self.startup_probe_symbol = None
                self.startup_probe_strike = None
                symbol = None
                strike = None

            if symbol is None:
                candidate = self.best_probe_candidate()
                if candidate is None:
                    self.log_gate("probe:no-candidate", "[PROBE] waiting reason=no_liquid_option")
                    return False
                symbol, strike, bid, ask, spread = candidate
                qty = min(self.cfg.startup_probe_qty, self.cfg.max_trade_qty)
                self.startup_probe_entry_attempts += 1
                self.startup_probe_symbol = symbol
                self.startup_probe_strike = strike
                print(
                    f"[PROBE] candidate symbol={symbol} strike={strike} bid={bid} ask={ask} "
                    f"spread={spread} attempt={self.startup_probe_entry_attempts}"
                )
                order_id = await self.submit_cross(symbol, Side.BUY, qty, ask, "startup-probe-entry", strike)
                if order_id:
                    self.set_probe_state("entry_pending", f"entry_order={order_id}")
                    return True
                self.startup_probe_symbol = None
                self.startup_probe_strike = None
                if self.startup_probe_entry_attempts >= self.cfg.startup_probe_max_attempts:
                    self.mark_probe_failed(
                        f"entry_blocked attempts={self.startup_probe_entry_attempts}"
                    )
                return False

        if self.startup_probe_state == "entry_filled_waiting_exit":
            if symbol is None or strike is None:
                self.mark_probe_failed("missing_symbol_after_entry_fill")
                return False
            if self.has_live_probe_orders():
                return False
            pos = self.position(symbol)
            if pos <= 0:
                self.mark_probe_done("flat_after_entry")
                return False
            bid, bid_qty, _ask, _ask_qty = self.top(symbol)
            if bid is None or bid_qty <= 0:
                self.log_gate(f"probe:no-bid:{symbol}", f"[PROBE] waiting_exit symbol={symbol} reason=no_bid")
                return False
            if self.startup_probe_exit_attempts >= self.cfg.startup_probe_max_attempts:
                self.mark_probe_failed(f"exit_attempts_exhausted symbol={symbol} pos={pos}")
                return False
            qty = min(pos, self.cfg.startup_probe_qty, self.cfg.max_trade_qty)
            self.startup_probe_exit_attempts += 1
            order_id = await self.submit_cross(symbol, Side.SELL, qty, bid, "startup-probe-exit", strike)
            if order_id:
                print(
                    f"[PROBE] exit symbol={symbol} qty={qty} sell_at={bid} "
                    f"attempt={self.startup_probe_exit_attempts}"
                )
                self.set_probe_state("exit_pending", f"exit_order={order_id}")
                return True
            if self.startup_probe_exit_attempts >= self.cfg.startup_probe_max_attempts:
                self.mark_probe_failed(f"exit_blocked symbol={symbol} pos={pos}")
            return False

        if self.startup_probe_state == "exit_pending":
            if self.has_live_probe_orders():
                return False
            if symbol is None:
                self.mark_probe_failed("missing_symbol_during_exit")
                return False
            pos = self.position(symbol)
            if pos <= 0:
                self.mark_probe_done("exit_fill_observed")
                return True
            if elapsed >= self.cfg.startup_probe_timeout_sec:
                self.mark_probe_failed(f"exit_timeout symbol={symbol} residual_pos={pos}")
                return False
            self.set_probe_state("entry_filled_waiting_exit", f"retry_exit symbol={symbol} residual_pos={pos}")
            return False

        return False

    def option_spread(self, symbol: str) -> Optional[int]:
        bid, _bid_qty, ask, _ask_qty = self.top(symbol)
        if bid is None or ask is None or ask <= bid:
            return None
        return ask - bid

    def quote_depth(self, bid_qty: int, ask_qty: int) -> int:
        return max(int(bid_qty), int(ask_qty))

    def trade_size_for_pair(self, pair: OptionPair, cap_qty: Optional[int] = None) -> int:
        call_capacity = max(0, self.cfg.max_position_per_option - abs(self.position(pair.call_symbol)))
        put_capacity = max(0, self.cfg.max_position_per_option - abs(self.position(pair.put_symbol)))
        qty = min(self.cfg.max_trade_qty, call_capacity, put_capacity)
        if cap_qty is not None:
            qty = min(qty, cap_qty)
        return int(max(0, qty))

    def passive_buy_price(self, bid: Optional[int], ask: Optional[int], cap_price: float) -> Optional[int]:
        if bid is None or ask is None or ask <= bid:
            return None
        price = min(ask - 1, math.floor(cap_price))
        if price < bid:
            return None
        return int(price)

    def passive_sell_price(self, bid: Optional[int], ask: Optional[int], floor_price: float) -> Optional[int]:
        if bid is None or ask is None or ask <= bid:
            return None
        price = max(bid + 1, math.ceil(floor_price))
        if price > ask:
            return None
        return int(price)

    def current_campaign_state(self) -> str:
        if self.active_campaign is not None:
            return self.active_campaign.state
        for pair in self.option_pairs:
            synthetic_long, synthetic_short, _ = self.paired_inventory(pair)
            if synthetic_long > 0 or synthetic_short > 0:
                return "PAIRED_POSITION"
        return "IDLE"

    def transition_campaign(self, state: str, reason: str) -> None:
        if self.active_campaign is None:
            return
        self.active_campaign.state = state
        self.active_campaign.stage_started_at = self.now()
        self.record_event(
            "campaign_state",
            state=state,
            campaign_kind=self.active_campaign.kind,
            strike=self.active_campaign.pair.strike,
            reason=reason,
        )
        print(
            f"[CAMPAIGN] state={state} kind={self.active_campaign.kind} "
            f"strike={self.active_campaign.pair.strike} reason={reason}"
        )

    def clear_campaign(self, reason: str) -> None:
        if self.active_campaign is None:
            return
        strike = self.active_campaign.pair.strike
        kind = self.active_campaign.kind
        self.record_event(
            "campaign_cleared",
            campaign_kind=kind,
            strike=strike,
            state=self.active_campaign.state,
            reason=reason,
            residual_qty=self.active_campaign.residual_qty(),
        )
        print(f"[CAMPAIGN] clear kind={kind} strike={strike} reason={reason}")
        self.last_action_at[strike] = self.now()
        self.active_campaign = None

    def campaign_has_live_order(self) -> bool:
        if self.active_campaign is None or self.active_campaign.current_order_id is None:
            return False
        order_id = self.active_campaign.current_order_id
        meta = self.order_meta.get(order_id)
        if meta is not None and meta.cancel_pending:
            return False
        return order_id in self.open_orders

    async def cancel_campaign_order(self, reason: str) -> None:
        if self.active_campaign is None or self.active_campaign.current_order_id is None:
            return
        order_id = self.active_campaign.current_order_id
        if order_id not in self.open_orders:
            return
        meta = self.order_meta.get(order_id)
        if meta is not None:
            if meta.cancel_pending:
                return
            meta.cancel_pending = True
        self.record_event(
            "campaign_cancel_requested",
            campaign_kind=self.active_campaign.kind,
            strike=self.active_campaign.pair.strike,
            state=self.active_campaign.state,
            role=self.active_campaign.current_role,
            order_id=order_id,
            reason=reason,
        )
        print(
            f"[CAMPAIGN] cancel kind={self.active_campaign.kind} strike={self.active_campaign.pair.strike} "
            f"role={self.active_campaign.current_role} order_id={order_id} reason={reason}"
        )
        try:
            await self.cancel_order(order_id)
        except Exception as exc:
            if meta is not None:
                meta.cancel_pending = False
            print(f"[CANCEL-ERROR] {order_id}: {exc}")

    async def submit_campaign_plan(self, plan: PassiveQuotePlan, role: str) -> bool:
        if self.active_campaign is None:
            return False
        purpose = f"campaign:{self.active_campaign.kind}:{role}"
        order_id = await self.submit_cross(
            plan.symbol,
            plan.side,
            plan.qty,
            plan.price,
            purpose,
            self.active_campaign.pair.strike,
        )
        if order_id is None:
            return False
        self.active_campaign.current_order_id = order_id
        self.active_campaign.current_role = role
        self.active_campaign.stage_started_at = self.now()
        if role == "unwind":
            self.active_campaign.unwind_reprices += 1
            self.active_campaign.last_reprice_at = self.now()
        self.record_event(
            "campaign_quote",
            campaign_kind=self.active_campaign.kind,
            strike=self.active_campaign.pair.strike,
            state=self.active_campaign.state,
            role=role,
            symbol=plan.symbol,
            side=plan.side.name,
            qty=plan.qty,
            price=plan.price,
            preserved_edge=plan.preserved_edge,
            note=plan.note,
        )
        print(
            f"[CAMPAIGN] submit kind={self.active_campaign.kind} strike={self.active_campaign.pair.strike} "
            f"state={self.active_campaign.state} role={role} symbol={plan.symbol} "
            f"side={plan.side.name} qty={plan.qty} px={plan.price} edge={plan.preserved_edge:.2f}"
        )
        return True

    async def start_campaign(self, candidate: CampaignCandidate) -> bool:
        if self.active_campaign is not None:
            return False
        state = "ENTRY_RESTING" if candidate.kind.startswith("entry_") else "EXIT_RESTING"
        self.active_campaign = PassiveCampaign(
            state=state,
            kind=candidate.kind,
            pair=candidate.pair,
            first_leg=candidate.first_leg,
            hedge_symbol=candidate.hedge_symbol,
            hedge_side=candidate.hedge_side,
            started_at=self.now(),
            stage_started_at=self.now(),
            notes={"origin": candidate.note},
        )
        self.record_event(
            "campaign_created",
            campaign_kind=candidate.kind,
            strike=candidate.pair.strike,
            state=state,
            score=candidate.score,
            note=candidate.note,
            first_leg={
                "symbol": candidate.first_leg.symbol,
                "side": candidate.first_leg.side.name,
                "price": candidate.first_leg.price,
                "qty": candidate.first_leg.qty,
                "preserved_edge": candidate.first_leg.preserved_edge,
                "option_spread": candidate.first_leg.option_spread,
                "top_depth": candidate.first_leg.top_depth,
                "note": candidate.first_leg.note,
            },
            hedge_symbol=candidate.hedge_symbol,
            hedge_side=candidate.hedge_side.name,
        )
        if await self.submit_campaign_plan(candidate.first_leg, "first"):
            return True
        self.clear_campaign("first_leg_submit_failed")
        return False

    def entry_candidates_for_pair(self, pair: OptionPair, snapshot: ParitySnapshot) -> list[CampaignCandidate]:
        candidates: list[CampaignCandidate] = []
        qty = self.trade_size_for_pair(pair)
        if qty <= 0:
            return candidates

        call_spread = self.option_spread(pair.call_symbol)
        put_spread = self.option_spread(pair.put_symbol)
        call_depth = self.quote_depth(snapshot.call_bid_qty, snapshot.call_ask_qty)
        put_depth = self.quote_depth(snapshot.put_bid_qty, snapshot.put_ask_qty)

        if snapshot.put_bid is not None and call_spread is not None:
            price = self.passive_buy_price(
                snapshot.call_bid,
                snapshot.call_ask,
                snapshot.theoretical_cp_diff + snapshot.put_bid - self.cfg.parity_entry_edge,
            )
            if price is not None:
                edge = snapshot.theoretical_cp_diff - (price - snapshot.put_bid)
                candidates.append(
                    CampaignCandidate(
                        kind="entry_long",
                        pair=pair,
                        first_leg=PassiveQuotePlan(
                            symbol=pair.call_symbol,
                            side=Side.BUY,
                            price=price,
                            qty=qty,
                            preserved_edge=edge,
                            option_spread=call_spread,
                            top_depth=call_depth,
                            note="long_call_first",
                        ),
                        hedge_symbol=pair.put_symbol,
                        hedge_side=Side.SELL,
                        score=edge - self.cfg.parity_entry_edge,
                        note="passive_entry",
                    )
                )

        if snapshot.call_ask is not None and put_spread is not None:
            price = self.passive_sell_price(
                snapshot.put_bid,
                snapshot.put_ask,
                snapshot.call_ask - snapshot.theoretical_cp_diff + self.cfg.parity_entry_edge,
            )
            if price is not None:
                edge = snapshot.theoretical_cp_diff - (snapshot.call_ask - price)
                candidates.append(
                    CampaignCandidate(
                        kind="entry_long",
                        pair=pair,
                        first_leg=PassiveQuotePlan(
                            symbol=pair.put_symbol,
                            side=Side.SELL,
                            price=price,
                            qty=qty,
                            preserved_edge=edge,
                            option_spread=put_spread,
                            top_depth=put_depth,
                            note="long_put_first",
                        ),
                        hedge_symbol=pair.call_symbol,
                        hedge_side=Side.BUY,
                        score=edge - self.cfg.parity_entry_edge,
                        note="passive_entry",
                    )
                )

        if snapshot.put_ask is not None and call_spread is not None:
            price = self.passive_sell_price(
                snapshot.call_bid,
                snapshot.call_ask,
                snapshot.theoretical_cp_diff + snapshot.put_ask + self.cfg.parity_entry_edge,
            )
            if price is not None:
                edge = (price - snapshot.put_ask) - snapshot.theoretical_cp_diff
                candidates.append(
                    CampaignCandidate(
                        kind="entry_short",
                        pair=pair,
                        first_leg=PassiveQuotePlan(
                            symbol=pair.call_symbol,
                            side=Side.SELL,
                            price=price,
                            qty=qty,
                            preserved_edge=edge,
                            option_spread=call_spread,
                            top_depth=call_depth,
                            note="short_call_first",
                        ),
                        hedge_symbol=pair.put_symbol,
                        hedge_side=Side.BUY,
                        score=edge - self.cfg.parity_entry_edge,
                        note="passive_entry",
                    )
                )

        if snapshot.call_bid is not None and put_spread is not None:
            price = self.passive_buy_price(
                snapshot.put_bid,
                snapshot.put_ask,
                snapshot.call_bid - snapshot.theoretical_cp_diff - self.cfg.parity_entry_edge,
            )
            if price is not None:
                edge = (snapshot.call_bid - price) - snapshot.theoretical_cp_diff
                candidates.append(
                    CampaignCandidate(
                        kind="entry_short",
                        pair=pair,
                        first_leg=PassiveQuotePlan(
                            symbol=pair.put_symbol,
                            side=Side.BUY,
                            price=price,
                            qty=qty,
                            preserved_edge=edge,
                            option_spread=put_spread,
                            top_depth=put_depth,
                            note="short_put_first",
                        ),
                        hedge_symbol=pair.call_symbol,
                        hedge_side=Side.SELL,
                        score=edge - self.cfg.parity_entry_edge,
                        note="passive_entry",
                    )
                )

        return [candidate for candidate in candidates if candidate.score >= 0]

    def best_entry_candidate(self, spot: float) -> Optional[CampaignCandidate]:
        best_per_pair: list[CampaignCandidate] = []
        for pair in self.option_pairs:
            if not self.cooldown_ready(pair.strike):
                self.log_gate(
                    f"entry:{pair.strike}:cooldown",
                    f"[SKIP] strike={pair.strike} reason=entry_cooldown",
                )
                continue
            if self.strike_has_pending_orders(pair.strike):
                self.log_gate(
                    f"entry:{pair.strike}:pending",
                    f"[SKIP] strike={pair.strike} reason=pending_orders count={self.pending_order_count_for_strike(pair.strike)}",
                )
                continue
            synthetic_long, synthetic_short, imbalance = self.paired_inventory(pair)
            if synthetic_long > 0 or synthetic_short > 0 or imbalance > 0:
                self.log_gate(
                    f"entry:{pair.strike}:inventory",
                    f"[SKIP] strike={pair.strike} reason=existing_inventory "
                    f"call={self.position(pair.call_symbol)} put={self.position(pair.put_symbol)}",
                )
                continue
            snapshot = self.parity_snapshot(pair, spot)
            if snapshot is None:
                continue
            candidates = self.entry_candidates_for_pair(pair, snapshot)
            if not candidates:
                self.log_gate(
                    f"entry:{pair.strike}:no-feasible",
                    f"[SKIP] strike={pair.strike} reason=no_passive_entry "
                    f"long={snapshot.long_synth_edge} short={snapshot.short_synth_edge}",
                )
                continue
            best_per_pair.append(
                max(
                    candidates,
                    key=lambda candidate: (
                        candidate.first_leg.option_spread,
                        candidate.first_leg.preserved_edge,
                        candidate.first_leg.top_depth,
                    ),
                )
            )
        if not best_per_pair:
            return None
        return max(
            best_per_pair,
            key=lambda candidate: (
                candidate.first_leg.preserved_edge,
                candidate.first_leg.option_spread,
                candidate.first_leg.top_depth,
            ),
        )

    async def maybe_start_entry_campaign(self, spot: float) -> bool:
        candidate = self.best_entry_candidate(spot)
        if candidate is None:
            return False
        return await self.start_campaign(candidate)

    def exit_candidates_for_pair(self, pair: OptionPair, snapshot: ParitySnapshot) -> list[CampaignCandidate]:
        candidates: list[CampaignCandidate] = []
        synthetic_long, synthetic_short, _ = self.paired_inventory(pair)
        call_spread = self.option_spread(pair.call_symbol)
        put_spread = self.option_spread(pair.put_symbol)
        call_depth = self.quote_depth(snapshot.call_bid_qty, snapshot.call_ask_qty)
        put_depth = self.quote_depth(snapshot.put_bid_qty, snapshot.put_ask_qty)

        if synthetic_long > 0:
            qty = self.trade_size_for_pair(pair, synthetic_long)
            if qty > 0 and snapshot.mid_gap is not None and abs(snapshot.mid_gap) <= self.cfg.parity_exit_band:
                if snapshot.put_ask is not None and call_spread is not None:
                    price = self.passive_sell_price(
                        snapshot.call_bid,
                        snapshot.call_ask,
                        snapshot.theoretical_cp_diff + snapshot.put_ask - self.cfg.parity_exit_band,
                    )
                    if price is not None:
                        slack = price - snapshot.put_ask - (snapshot.theoretical_cp_diff - self.cfg.parity_exit_band)
                        candidates.append(
                            CampaignCandidate(
                                kind="exit_long",
                                pair=pair,
                                first_leg=PassiveQuotePlan(
                                    symbol=pair.call_symbol,
                                    side=Side.SELL,
                                    price=price,
                                    qty=qty,
                                    preserved_edge=slack,
                                    option_spread=call_spread,
                                    top_depth=call_depth,
                                    note="exit_long_call_first",
                                ),
                                hedge_symbol=pair.put_symbol,
                                hedge_side=Side.BUY,
                                score=slack,
                                note="passive_exit",
                            )
                        )
                if snapshot.call_bid is not None and put_spread is not None:
                    price = self.passive_buy_price(
                        snapshot.put_bid,
                        snapshot.put_ask,
                        snapshot.call_bid - snapshot.theoretical_cp_diff + self.cfg.parity_exit_band,
                    )
                    if price is not None:
                        slack = snapshot.call_bid - price - (snapshot.theoretical_cp_diff - self.cfg.parity_exit_band)
                        candidates.append(
                            CampaignCandidate(
                                kind="exit_long",
                                pair=pair,
                                first_leg=PassiveQuotePlan(
                                    symbol=pair.put_symbol,
                                    side=Side.BUY,
                                    price=price,
                                    qty=qty,
                                    preserved_edge=slack,
                                    option_spread=put_spread,
                                    top_depth=put_depth,
                                    note="exit_long_put_first",
                                ),
                                hedge_symbol=pair.call_symbol,
                                hedge_side=Side.SELL,
                                score=slack,
                                note="passive_exit",
                            )
                        )

        if synthetic_short > 0:
            qty = self.trade_size_for_pair(pair, synthetic_short)
            if qty > 0 and snapshot.mid_gap is not None and abs(snapshot.mid_gap) <= self.cfg.parity_exit_band:
                if snapshot.put_bid is not None and call_spread is not None:
                    price = self.passive_buy_price(
                        snapshot.call_bid,
                        snapshot.call_ask,
                        snapshot.theoretical_cp_diff + snapshot.put_bid + self.cfg.parity_exit_band,
                    )
                    if price is not None:
                        slack = snapshot.theoretical_cp_diff + self.cfg.parity_exit_band - (price - snapshot.put_bid)
                        candidates.append(
                            CampaignCandidate(
                                kind="exit_short",
                                pair=pair,
                                first_leg=PassiveQuotePlan(
                                    symbol=pair.call_symbol,
                                    side=Side.BUY,
                                    price=price,
                                    qty=qty,
                                    preserved_edge=slack,
                                    option_spread=call_spread,
                                    top_depth=call_depth,
                                    note="exit_short_call_first",
                                ),
                                hedge_symbol=pair.put_symbol,
                                hedge_side=Side.SELL,
                                score=slack,
                                note="passive_exit",
                            )
                        )
                if snapshot.call_ask is not None and put_spread is not None:
                    price = self.passive_sell_price(
                        snapshot.put_bid,
                        snapshot.put_ask,
                        snapshot.call_ask - snapshot.theoretical_cp_diff - self.cfg.parity_exit_band,
                    )
                    if price is not None:
                        slack = snapshot.theoretical_cp_diff + self.cfg.parity_exit_band - (snapshot.call_ask - price)
                        candidates.append(
                            CampaignCandidate(
                                kind="exit_short",
                                pair=pair,
                                first_leg=PassiveQuotePlan(
                                    symbol=pair.put_symbol,
                                    side=Side.SELL,
                                    price=price,
                                    qty=qty,
                                    preserved_edge=slack,
                                    option_spread=put_spread,
                                    top_depth=put_depth,
                                    note="exit_short_put_first",
                                ),
                                hedge_symbol=pair.call_symbol,
                                hedge_side=Side.BUY,
                                score=slack,
                                note="passive_exit",
                            )
                        )

        return [candidate for candidate in candidates if candidate.score >= 0]

    def best_exit_candidate(self, spot: float) -> Optional[CampaignCandidate]:
        best_per_pair: list[CampaignCandidate] = []
        for pair in self.option_pairs:
            if not self.cooldown_ready(pair.strike):
                continue
            if self.strike_has_pending_orders(pair.strike):
                continue
            snapshot = self.parity_snapshot(pair, spot)
            if snapshot is None:
                continue
            candidates = self.exit_candidates_for_pair(pair, snapshot)
            if not candidates:
                continue
            best_per_pair.append(
                max(
                    candidates,
                    key=lambda candidate: (
                        candidate.first_leg.option_spread,
                        candidate.first_leg.preserved_edge,
                        candidate.first_leg.top_depth,
                    ),
                )
            )
        if not best_per_pair:
            return None
        return max(
            best_per_pair,
            key=lambda candidate: (
                candidate.first_leg.preserved_edge,
                candidate.first_leg.option_spread,
                candidate.first_leg.top_depth,
            ),
        )

    async def maybe_start_exit_campaign(self, spot: float) -> bool:
        candidate = self.best_exit_candidate(spot)
        if candidate is None:
            return False
        return await self.start_campaign(candidate)

    def build_hedge_plan(self, campaign: PassiveCampaign, snapshot: ParitySnapshot) -> Optional[PassiveQuotePlan]:
        fill_price = campaign.first_fill_price()
        if fill_price is None:
            return None
        qty = max(0, campaign.first_filled_qty - campaign.hedge_filled_qty)
        if qty <= 0:
            return None
        symbol = campaign.hedge_symbol
        bid, bid_qty, ask, ask_qty = self.top(symbol)
        spread = self.option_spread(symbol)
        if spread is None:
            return None
        depth = self.quote_depth(bid_qty, ask_qty)
        theoretical = snapshot.theoretical_cp_diff
        note = f"hedge_after_{campaign.first_leg.note}"

        if campaign.kind == "entry_long" and campaign.first_leg.symbol == campaign.pair.call_symbol:
            price = self.passive_sell_price(bid, ask, fill_price - theoretical + self.cfg.parity_entry_edge)
            if price is None:
                return None
            edge = theoretical - (fill_price - price)
            return PassiveQuotePlan(symbol=symbol, side=Side.SELL, price=price, qty=qty, preserved_edge=edge, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "entry_long":
            price = self.passive_buy_price(bid, ask, theoretical + fill_price - self.cfg.parity_entry_edge)
            if price is None:
                return None
            edge = theoretical - (price - fill_price)
            return PassiveQuotePlan(symbol=symbol, side=Side.BUY, price=price, qty=qty, preserved_edge=edge, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "entry_short" and campaign.first_leg.symbol == campaign.pair.call_symbol:
            price = self.passive_buy_price(bid, ask, fill_price - theoretical - self.cfg.parity_entry_edge)
            if price is None:
                return None
            edge = (fill_price - price) - theoretical
            return PassiveQuotePlan(symbol=symbol, side=Side.BUY, price=price, qty=qty, preserved_edge=edge, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "entry_short":
            price = self.passive_sell_price(bid, ask, theoretical + fill_price + self.cfg.parity_entry_edge)
            if price is None:
                return None
            edge = (price - fill_price) - theoretical
            return PassiveQuotePlan(symbol=symbol, side=Side.SELL, price=price, qty=qty, preserved_edge=edge, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "exit_long" and campaign.first_leg.symbol == campaign.pair.call_symbol:
            price = self.passive_buy_price(bid, ask, fill_price - theoretical + self.cfg.parity_exit_band)
            if price is None:
                return None
            slack = fill_price - price - (theoretical - self.cfg.parity_exit_band)
            return PassiveQuotePlan(symbol=symbol, side=Side.BUY, price=price, qty=qty, preserved_edge=slack, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "exit_long":
            price = self.passive_sell_price(bid, ask, theoretical + fill_price - self.cfg.parity_exit_band)
            if price is None:
                return None
            slack = price - fill_price - (theoretical - self.cfg.parity_exit_band)
            return PassiveQuotePlan(symbol=symbol, side=Side.SELL, price=price, qty=qty, preserved_edge=slack, option_spread=spread, top_depth=depth, note=note)

        if campaign.kind == "exit_short" and campaign.first_leg.symbol == campaign.pair.call_symbol:
            price = self.passive_sell_price(bid, ask, fill_price - theoretical - self.cfg.parity_exit_band)
            if price is None:
                return None
            slack = price - (fill_price - theoretical - self.cfg.parity_exit_band)
            return PassiveQuotePlan(symbol=symbol, side=Side.SELL, price=price, qty=qty, preserved_edge=slack, option_spread=spread, top_depth=depth, note=note)

        price = self.passive_buy_price(bid, ask, theoretical + fill_price + self.cfg.parity_exit_band)
        if price is None:
            return None
        slack = theoretical + self.cfg.parity_exit_band - (price - fill_price)
        return PassiveQuotePlan(symbol=symbol, side=Side.BUY, price=price, qty=qty, preserved_edge=slack, option_spread=spread, top_depth=depth, note=note)

    def build_unwind_plan(self, campaign: PassiveCampaign) -> Optional[PassiveQuotePlan]:
        qty = campaign.residual_qty()
        if qty <= 0:
            return None
        symbol = campaign.first_leg.symbol
        bid, bid_qty, ask, ask_qty = self.top(symbol)
        if bid is None or ask is None or ask <= bid:
            return None
        side = Side.SELL if campaign.first_leg.side == Side.BUY else Side.BUY
        spread = ask - bid
        step = campaign.unwind_reprices
        if side == Side.SELL:
            base = max(bid + 1, math.floor((bid + ask) / 2)) if spread >= 2 else ask
            price = min(ask, base + step)
        else:
            base = min(ask - 1, math.ceil((bid + ask) / 2)) if spread >= 2 else bid
            price = max(bid, base - step)
        return PassiveQuotePlan(
            symbol=symbol,
            side=side,
            price=int(price),
            qty=qty,
            preserved_edge=0.0,
            option_spread=spread,
            top_depth=self.quote_depth(bid_qty, ask_qty),
            note="passive_unwind",
        )

    async def advance_campaign(self, spot: float) -> bool:
        campaign = self.active_campaign
        if campaign is None:
            return False
        snapshot = self.parity_snapshot(campaign.pair, spot)
        if snapshot is None:
            self.log_gate("campaign:no-snapshot", "[SKIP] campaign reason=no_snapshot")
            return True
        now = self.now()

        if campaign.state in {"ENTRY_RESTING", "EXIT_RESTING"}:
            if self.campaign_has_live_order():
                if now - campaign.stage_started_at >= self.cfg.passive_entry_ttl_sec:
                    await self.cancel_campaign_order("first_leg_ttl")
                return True
            if campaign.first_filled_qty > 0:
                self.transition_campaign("FIRST_LEG_FILLED_WAITING_HEDGE", "first_leg_filled")
            else:
                self.clear_campaign("first_leg_unfilled")
            return True

        if campaign.state == "FIRST_LEG_FILLED_WAITING_HEDGE":
            if self.campaign_has_live_order():
                return True
            hedge_plan = self.build_hedge_plan(campaign, snapshot)
            if hedge_plan is None:
                self.transition_campaign("UNWIND_RESTING", "hedge_not_passively_feasible")
                return True
            self.transition_campaign("HEDGE_RESTING", "submit_passive_hedge")
            if not await self.submit_campaign_plan(hedge_plan, "hedge"):
                self.transition_campaign("UNWIND_RESTING", "hedge_submit_failed")
            return True

        if campaign.state == "HEDGE_RESTING":
            if self.campaign_has_live_order():
                if now - campaign.stage_started_at >= self.cfg.passive_hedge_ttl_sec:
                    await self.cancel_campaign_order("hedge_ttl")
                return True
            if campaign.first_filled_qty <= campaign.hedge_filled_qty:
                if campaign.kind.startswith("entry_"):
                    self.record_event(
                        "parity_trade_open",
                        strike=campaign.pair.strike,
                        action=campaign.kind,
                        qty=campaign.hedge_filled_qty,
                        first_fill_price=campaign.first_fill_price(),
                    )
                else:
                    self.record_event(
                        "parity_trade_exit",
                        strike=campaign.pair.strike,
                        action=campaign.kind,
                        qty=campaign.hedge_filled_qty,
                        first_fill_price=campaign.first_fill_price(),
                    )
                self.clear_campaign("hedge_completed")
                return True
            self.transition_campaign("UNWIND_RESTING", f"hedge_incomplete residual={campaign.residual_qty()}")
            return True

        if campaign.state == "UNWIND_RESTING":
            if campaign.residual_qty() <= 0:
                if campaign.hedge_filled_qty > 0:
                    self.record_event(
                        "parity_trade_open" if campaign.kind.startswith("entry_") else "parity_trade_exit",
                        strike=campaign.pair.strike,
                        action=campaign.kind,
                        qty=campaign.hedge_filled_qty,
                        partial=True,
                        first_fill_price=campaign.first_fill_price(),
                        unwind_filled_qty=campaign.unwind_filled_qty,
                    )
                else:
                    self.record_event(
                        "campaign_aborted",
                        strike=campaign.pair.strike,
                        action=campaign.kind,
                        first_fill_price=campaign.first_fill_price(),
                        unwind_filled_qty=campaign.unwind_filled_qty,
                    )
                self.clear_campaign("unwind_completed")
                return True
            if self.campaign_has_live_order():
                if now - campaign.stage_started_at < self.cfg.passive_unwind_ttl_sec:
                    if now - campaign.last_reprice_at >= self.cfg.passive_unwind_reprice_sec:
                        await self.cancel_campaign_order("unwind_reprice")
                return True
            plan = self.build_unwind_plan(campaign)
            if plan is None:
                self.log_gate(
                    f"campaign:{campaign.pair.strike}:unwind_no_quote",
                    f"[SKIP] strike={campaign.pair.strike} reason=no_unwind_quote",
                )
                return True
            if not await self.submit_campaign_plan(plan, "unwind"):
                self.log_gate(
                    f"campaign:{campaign.pair.strike}:unwind_blocked",
                    f"[SKIP] strike={campaign.pair.strike} reason=unwind_submit_blocked",
                )
            return True

        return True

    def prune_order_meta(self) -> None:
        for order_id in list(self.order_meta):
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)

    def print_status(self, spot: Optional[float]) -> None:
        now = self.now()
        if now - self.last_status < self.cfg.status_sec:
            return
        self.last_status = now

        local_parts: list[str] = []
        diff_parts: list[str] = []
        edge_parts: list[str] = []
        for pair in self.option_pairs:
            call_pos = self.position(pair.call_symbol)
            put_pos = self.position(pair.put_symbol)
            call_exchange = self.exchange_position(pair.call_symbol)
            put_exchange = self.exchange_position(pair.put_symbol)
            if call_pos or put_pos:
                local_parts.append(f"{pair.strike}:C={call_pos},P={put_pos}")
            if call_pos != call_exchange or put_pos != put_exchange:
                diff_parts.append(f"{pair.strike}:C={call_pos}/{call_exchange},P={put_pos}/{put_exchange}")
            snapshot = self.parity_snapshot(pair, spot) if spot is not None else None
            if snapshot is None:
                continue
            edge_parts.append(
                f"{pair.strike}:L={snapshot.long_synth_edge if snapshot.long_synth_edge is not None else 'NA'}"
                f"/S={snapshot.short_synth_edge if snapshot.short_synth_edge is not None else 'NA'}"
                f"/P={self.pending_order_count_for_strike(pair.strike)}"
            )

        campaign = self.serialize_campaign()
        print(
            "[STATUS]",
            f"spot={spot:.2f}" if spot is not None else "spot=NA",
            f"pairs={len(self.option_pairs)}",
            f"local={' | '.join(local_parts) if local_parts else 'flat'}",
            f"exchange_diff={' | '.join(diff_parts) if diff_parts else 'none'}",
            f"outstanding={self.outstanding_qty()}",
            f"edges={' | '.join(edge_parts) if edge_parts else 'NA'}",
            f"probe={self.startup_probe_state}",
            f"campaign={self.current_campaign_state()}",
            f"campaign_strike={campaign['strike'] if campaign else 'NA'}",
        )
        self.record_event(
            "status",
            spot=spot,
            local_positions=local_parts,
            exchange_diff=diff_parts,
            edges=edge_parts,
            outstanding=self.outstanding_qty(),
            probe=self.startup_probe_state,
            campaign=campaign,
        )

    async def run_strategy(self) -> None:
        async with self._strategy_lock:
            self.prune_order_meta()
            await self.cancel_stale_orders()
            self.discover_option_pairs()
            self.seed_and_reconcile_local_positions()

            spot = self.extract_spot()
            if spot is None:
                self.log_gate("spot:none", "[SKIP] strategy reason=no_spot")
                return

            await self.maybe_run_startup_probe()
            if self.probe_active():
                self.print_status(spot)
                return

            if await self.advance_campaign(spot):
                self.print_status(spot)
                return

            if await self.maybe_start_exit_campaign(spot):
                self.print_status(spot)
                return

            if self.has_any_option_inventory_or_orders():
                self.log_gate("strategy:inventory", "[SKIP] strategy reason=inventory_or_orders_present")
                self.print_status(spot)
                return

            await self.maybe_start_entry_campaign(spot)
            self.print_status(spot)

    async def capture_loop(self) -> None:
        if not self.cfg.capture_enabled:
            return
        while True:
            try:
                self.record_snapshot("periodic")
            except Exception as exc:
                print(f"[CAPTURE-ERROR] snapshot error={exc}")
            await asyncio.sleep(self.cfg.capture_snapshot_interval_sec)

    async def process_message(self, msg) -> None:
        if hasattr(msg, "WhichOneof"):
            msg_type = msg.WhichOneof("body")
            index = getattr(msg, "index", None)
            if msg_type == "trade":
                self.record_event(
                    "exchange_trade",
                    index=index,
                    symbol=msg.trade.symbol,
                    price=msg.trade.px,
                    qty=msg.trade.qty,
                )
            elif msg_type == "book_update":
                side = "BUY" if int(msg.book_update.side) == 1 else "SELL"
                self.record_event(
                    "exchange_book_update",
                    index=index,
                    symbol=msg.book_update.symbol,
                    side=side,
                    price=msg.book_update.px,
                    delta_qty=msg.book_update.dq,
                )
            elif msg_type == "book_snapshot":
                self.record_event(
                    "exchange_book_snapshot",
                    index=index,
                    symbol=msg.book_snapshot.symbol,
                    bids=[[int(level.px), int(level.qty)] for level in msg.book_snapshot.bids],
                    asks=[[int(level.px), int(level.qty)] for level in msg.book_snapshot.asks],
                )
            elif msg_type == "order_fill":
                self.record_event(
                    "exchange_order_fill",
                    index=index,
                    order_id=msg.order_fill.id,
                    qty=msg.order_fill.qty,
                    price=msg.order_fill.px,
                    known_order=msg.order_fill.id in self.open_orders,
                )
            elif msg_type == "order_rejected":
                self.record_event(
                    "exchange_order_rejected",
                    index=index,
                    order_id=msg.order_rejected.id,
                    reason=msg.order_rejected.reason,
                )
            elif msg_type == "cancel_response":
                result = "ok" if msg.cancel_response.WhichOneof("result") == "ok" else "error"
                self.record_event(
                    "exchange_cancel_response",
                    index=index,
                    order_id=msg.cancel_response.id,
                    result=result,
                )
            elif msg_type == "position_update":
                symbol = msg.position_update.symbol
                self.record_event(
                    "exchange_position_update",
                    index=index,
                    symbol=symbol,
                    old_value=self.exchange_position(symbol),
                    new_value=msg.position_update.value,
                )
            elif msg_type == "cash_update":
                self.record_event(
                    "exchange_cash_update",
                    index=index,
                    old_value=self.positions.get("cash", 0),
                    new_value=msg.cash_update.value,
                )
            elif msg_type == "position_snapshot":
                snapshot_positions = {
                    position.symbol: position.position for position in msg.position_snapshot.positions
                }
                self.record_event(
                    "exchange_position_snapshot",
                    index=index,
                    cash=msg.position_snapshot.cash,
                    positions=snapshot_positions,
                )
        await super().process_message(msg)

    async def strategy_loop(self) -> None:
        await asyncio.sleep(1.0)
        print(
            f"[INIT] host={self.cfg.host} user={self.cfg.username} "
            f"r={self.cfg.risk_free_rate} T={self.cfg.time_to_expiry_years} "
            f"entry_edge={self.cfg.parity_entry_edge} "
            f"startup_probe={self.cfg.startup_probe_enabled} "
            f"probe_timeout={self.cfg.startup_probe_timeout_sec} "
            f"probe_attempts={self.cfg.startup_probe_max_attempts} "
            f"passive_entry_ttl={self.cfg.passive_entry_ttl_sec} "
            f"passive_hedge_ttl={self.cfg.passive_hedge_ttl_sec} "
            f"passive_unwind_ttl={self.cfg.passive_unwind_ttl_sec}"
        )
        self.record_event(
            "init",
            host=self.cfg.host,
            username=self.cfg.username,
            parity_entry_edge=self.cfg.parity_entry_edge,
            startup_probe=self.cfg.startup_probe_enabled,
            passive_entry_ttl_sec=self.cfg.passive_entry_ttl_sec,
            passive_hedge_ttl_sec=self.cfg.passive_hedge_ttl_sec,
            passive_unwind_ttl_sec=self.cfg.passive_unwind_ttl_sec,
        )
        while True:
            try:
                try:
                    await asyncio.wait_for(self._book_event.wait(), timeout=self.cfg.poll_sec)
                except TimeoutError:
                    pass
                self._book_event.clear()
                await self.run_strategy()
            except Exception as exc:
                print(f"[STRATEGY-ERROR] {exc}")
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
        if meta:
            current_local = int(self.local_positions.get(meta.symbol, self.exchange_position(meta.symbol)))
            delta = qty if meta.side == Side.BUY else -qty
            self.local_positions[meta.symbol] = current_local + delta
            self.last_local_fill_at[meta.symbol] = self.now()
            print(f"[FILL] {meta.purpose} {meta.symbol} {meta.side.name} {qty}@{price}")
            print(
                f"[FILL-STATE] symbol={meta.symbol} local={self.local_positions[meta.symbol]} "
                f"exchange={self.exchange_position(meta.symbol)}"
            )
            self.record_event(
                "order_fill",
                order_id=order_id,
                purpose=meta.purpose,
                symbol=meta.symbol,
                side=meta.side.name,
                qty=qty,
                price=price,
                local_position=self.local_positions[meta.symbol],
                exchange_position=self.exchange_position(meta.symbol),
            )
            if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
                if self.active_campaign.current_role == "first":
                    self.active_campaign.first_filled_qty += qty
                    self.active_campaign.first_fill_notional += qty * price
                    if order_id in self.open_orders:
                        await self.cancel_campaign_order("first_leg_partial_fill")
                    else:
                        self.active_campaign.current_order_id = None
                        self.active_campaign.current_role = None
                        self.transition_campaign("FIRST_LEG_FILLED_WAITING_HEDGE", "first_leg_filled")
                elif self.active_campaign.current_role == "hedge":
                    self.active_campaign.hedge_filled_qty += qty
                    self.active_campaign.hedge_fill_notional += qty * price
                    if order_id not in self.open_orders:
                        self.active_campaign.current_order_id = None
                        self.active_campaign.current_role = None
                elif self.active_campaign.current_role == "unwind":
                    self.active_campaign.unwind_filled_qty += qty
                    self.active_campaign.unwind_fill_notional += qty * price
                    if order_id not in self.open_orders:
                        self.active_campaign.current_order_id = None
                        self.active_campaign.current_role = None
            if meta.purpose == "startup-probe-entry":
                self.set_probe_state("entry_filled_waiting_exit", f"symbol={meta.symbol} pos={self.position(meta.symbol)}")
            elif meta.purpose == "startup-probe-exit" and self.position(meta.symbol) == 0:
                self.mark_probe_done("probe_round_trip_complete")
                print("[PROBE] complete")
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)
        self._book_event.set()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
            self.record_event(
                "campaign_order_rejected",
                campaign_kind=self.active_campaign.kind,
                strike=self.active_campaign.pair.strike,
                state=self.active_campaign.state,
                role=self.active_campaign.current_role,
                order_id=order_id,
                reason=reason,
            )
            if self.active_campaign.current_role == "first":
                self.clear_campaign(f"first_leg_rejected:{reason}")
            elif self.active_campaign.current_role == "hedge":
                self.active_campaign.current_order_id = None
                self.active_campaign.current_role = None
                self.transition_campaign("UNWIND_RESTING", f"hedge_rejected:{reason}")
            else:
                self.active_campaign.current_order_id = None
                self.active_campaign.current_role = None
        self.order_meta.pop(order_id, None)
        print(f"[REJECT] {order_id}: {reason}")
        self.record_event("order_rejected", order_id=order_id, reason=reason)
        self._book_event.set()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        meta = self.order_meta.get(order_id)
        if meta:
            meta.cancel_pending = False
        if success:
            if self.active_campaign is not None and order_id == self.active_campaign.current_order_id:
                self.active_campaign.current_order_id = None
                self.active_campaign.current_role = None
            if meta and meta.purpose.startswith("startup-probe"):
                print(f"[PROBE] cancel order_id={order_id} purpose={meta.purpose} symbol={meta.symbol}")
            self.order_meta.pop(order_id, None)
        else:
            print(f"[CANCEL-FAIL] {order_id}: {error}")
        self.record_event("cancel_response", order_id=order_id, success=success, error=error)
        self._book_event.set()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        return

    async def start(self):
        asyncio.create_task(self.strategy_loop())
        if self.cfg.capture_enabled:
            asyncio.create_task(self.capture_loop())
        await self.connect()


async def main() -> None:
    config = load_config()
    bot = MarketBBot(config)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
