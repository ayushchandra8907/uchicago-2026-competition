from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    force_entry_after_sec: float
    force_entry_edge_floor: float


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
        startup_probe_enabled=env_bool("B_STARTUP_PROBE", True),
        startup_probe_qty=max(1, env_int("B_STARTUP_PROBE_QTY", 1)),
        startup_probe_timeout_sec=env_float("B_STARTUP_PROBE_TIMEOUT_SEC", 6.0),
        startup_probe_max_attempts=max(1, env_int("B_STARTUP_PROBE_MAX_ATTEMPTS", 2)),
        force_entry_after_sec=env_float("B_FORCE_ENTRY_AFTER_SEC", 0.0),
        force_entry_edge_floor=env_float("B_FORCE_ENTRY_EDGE_FLOOR", -6.0),
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
        self.last_open_trade_at = 0.0
        self.discover_option_pairs()

    def now(self) -> float:
        return time.monotonic()

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

    def probe_active(self) -> bool:
        return self.startup_probe_state not in {"disabled", "done", "failed"}

    def set_probe_state(self, state: str, reason: Optional[str] = None) -> None:
        if state == self.startup_probe_state and reason is None:
            return
        self.startup_probe_state = state
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
            return None
        try:
            order_id = await self.place_order(symbol, int(qty), side, int(price))
        except Exception as exc:
            print(f"[ORDER-ERROR] {purpose} {symbol} {side.name} {qty}@{price}: {exc}")
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
        return order_id

    async def cancel_stale_orders(self) -> None:
        now = self.now()
        for order_id, meta in list(self.order_meta.items()):
            if meta.cancel_pending:
                continue
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)
                continue
            if now - meta.created_at < self.cfg.order_ttl_sec:
                continue
            meta.cancel_pending = True
            print(
                f"[STALE] cancel order_id={order_id} purpose={meta.purpose} "
                f"symbol={meta.symbol} age={now - meta.created_at:.2f}s"
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

    def trade_size_for_open(self, pair: OptionPair, snapshot: ParitySnapshot) -> int:
        if snapshot.call_ask is None or snapshot.call_bid is None or snapshot.put_ask is None or snapshot.put_bid is None:
            return 0
        call_pos = abs(self.position(pair.call_symbol))
        put_pos = abs(self.position(pair.put_symbol))
        call_capacity = max(0, self.cfg.max_position_per_option - call_pos)
        put_capacity = max(0, self.cfg.max_position_per_option - put_pos)
        size = min(
            self.cfg.max_trade_qty,
            call_capacity,
            put_capacity,
        )
        return int(size)

    async def open_synthetic_long(self, pair: OptionPair, snapshot: ParitySnapshot) -> bool:
        if snapshot.call_ask is None or snapshot.put_bid is None:
            return False
        size = min(
            self.trade_size_for_open(pair, snapshot),
            snapshot.call_ask_qty or self.cfg.max_trade_qty,
            snapshot.put_bid_qty or self.cfg.max_trade_qty,
        )
        if size <= 0:
            return False

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.BUY, size, snapshot.call_ask, "pcp-open-long", pair.strike),
            self.submit_cross(pair.put_symbol, Side.SELL, size, snapshot.put_bid, "pcp-open-long", pair.strike),
        )
        if any(results):
            now = self.now()
            self.last_action_at[pair.strike] = now
            self.last_open_trade_at = now
            print(
                f"[TRADE] strike={pair.strike} action=BUY_CALL_SELL_PUT "
                f"edge={snapshot.long_synth_edge:.2f} spot={snapshot.spot:.2f}"
            )
            return True
        return False

    async def open_synthetic_short(self, pair: OptionPair, snapshot: ParitySnapshot) -> bool:
        if snapshot.call_bid is None or snapshot.put_ask is None:
            return False
        size = min(
            self.trade_size_for_open(pair, snapshot),
            snapshot.call_bid_qty or self.cfg.max_trade_qty,
            snapshot.put_ask_qty or self.cfg.max_trade_qty,
        )
        if size <= 0:
            return False

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.SELL, size, snapshot.call_bid, "pcp-open-short", pair.strike),
            self.submit_cross(pair.put_symbol, Side.BUY, size, snapshot.put_ask, "pcp-open-short", pair.strike),
        )
        if any(results):
            now = self.now()
            self.last_action_at[pair.strike] = now
            self.last_open_trade_at = now
            print(
                f"[TRADE] strike={pair.strike} action=SELL_CALL_BUY_PUT "
                f"edge={snapshot.short_synth_edge:.2f} spot={snapshot.spot:.2f}"
            )
            return True
        return False

    async def close_synthetic_long(self, pair: OptionPair, snapshot: ParitySnapshot, qty: int) -> None:
        if qty <= 0 or snapshot.call_bid is None or snapshot.put_ask is None:
            return
        qty = min(qty, snapshot.call_bid_qty or qty, snapshot.put_ask_qty or qty, self.cfg.max_trade_qty)
        if qty <= 0:
            return

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.SELL, qty, snapshot.call_bid, "pcp-close-long", pair.strike),
            self.submit_cross(pair.put_symbol, Side.BUY, qty, snapshot.put_ask, "pcp-close-long", pair.strike),
        )
        if any(results):
            self.last_action_at[pair.strike] = self.now()
            print(f"[REDUCE] strike={pair.strike} action=CLOSE_SYNTH_LONG gap={snapshot.mid_gap}")

    async def close_synthetic_short(self, pair: OptionPair, snapshot: ParitySnapshot, qty: int) -> None:
        if qty <= 0 or snapshot.call_ask is None or snapshot.put_bid is None:
            return
        qty = min(qty, snapshot.call_ask_qty or qty, snapshot.put_bid_qty or qty, self.cfg.max_trade_qty)
        if qty <= 0:
            return

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.BUY, qty, snapshot.call_ask, "pcp-close-short", pair.strike),
            self.submit_cross(pair.put_symbol, Side.SELL, qty, snapshot.put_bid, "pcp-close-short", pair.strike),
        )
        if any(results):
            self.last_action_at[pair.strike] = self.now()
            print(f"[REDUCE] strike={pair.strike} action=CLOSE_SYNTH_SHORT gap={snapshot.mid_gap}")

    async def maybe_reduce_when_normalized(self, pair: OptionPair, snapshot: ParitySnapshot) -> bool:
        synthetic_long, synthetic_short, _ = self.paired_inventory(pair)
        if snapshot.mid_gap is None:
            return False
        if abs(snapshot.mid_gap) > self.cfg.parity_exit_band:
            return False
        if not self.cooldown_ready(pair.strike):
            return False

        if synthetic_long > 0:
            executable_close = None
            if snapshot.call_bid is not None and snapshot.put_ask is not None:
                executable_close = snapshot.call_bid - snapshot.put_ask
            if executable_close is not None and executable_close >= snapshot.theoretical_cp_diff - self.cfg.parity_exit_band:
                await self.close_synthetic_long(pair, snapshot, synthetic_long)
                return True

        if synthetic_short > 0:
            executable_close = None
            if snapshot.call_ask is not None and snapshot.put_bid is not None:
                executable_close = snapshot.call_ask - snapshot.put_bid
            if executable_close is not None and executable_close <= snapshot.theoretical_cp_diff + self.cfg.parity_exit_band:
                await self.close_synthetic_short(pair, snapshot, synthetic_short)
                return True

        return False

    async def maybe_reduce_unpaired_inventory(self, pair: OptionPair, snapshot: ParitySnapshot) -> bool:
        call_pos = self.position(pair.call_symbol)
        put_pos = self.position(pair.put_symbol)
        cleanup_band = max(self.cfg.parity_exit_band, self.cfg.parity_entry_edge)
        if snapshot.mid_gap is not None and abs(snapshot.mid_gap) > cleanup_band:
            return False
        if not self.cooldown_ready(pair.strike):
            return False
        if self.strike_has_pending_orders(pair.strike):
            return False

        if call_pos > 0 and snapshot.call_bid is not None:
            qty = min(call_pos, snapshot.call_bid_qty or call_pos, self.cfg.max_trade_qty)
            if qty > 0:
                order_id = await self.submit_cross(pair.call_symbol, Side.SELL, qty, snapshot.call_bid, "pcp-cleanup", pair.strike)
                if order_id:
                    self.last_action_at[pair.strike] = self.now()
                    return True
        if call_pos < 0 and snapshot.call_ask is not None:
            qty = min(abs(call_pos), snapshot.call_ask_qty or abs(call_pos), self.cfg.max_trade_qty)
            if qty > 0:
                order_id = await self.submit_cross(pair.call_symbol, Side.BUY, qty, snapshot.call_ask, "pcp-cleanup", pair.strike)
                if order_id:
                    self.last_action_at[pair.strike] = self.now()
                    return True
        if put_pos > 0 and snapshot.put_bid is not None:
            qty = min(put_pos, snapshot.put_bid_qty or put_pos, self.cfg.max_trade_qty)
            if qty > 0:
                order_id = await self.submit_cross(pair.put_symbol, Side.SELL, qty, snapshot.put_bid, "pcp-cleanup", pair.strike)
                if order_id:
                    self.last_action_at[pair.strike] = self.now()
                    return True
        if put_pos < 0 and snapshot.put_ask is not None:
            qty = min(abs(put_pos), snapshot.put_ask_qty or abs(put_pos), self.cfg.max_trade_qty)
            if qty > 0:
                order_id = await self.submit_cross(pair.put_symbol, Side.BUY, qty, snapshot.put_ask, "pcp-cleanup", pair.strike)
                if order_id:
                    self.last_action_at[pair.strike] = self.now()
                    return True

        return False

    async def maybe_open_parity_trade(self, pair: OptionPair, snapshot: ParitySnapshot) -> bool:
        synthetic_long, synthetic_short, imbalance = self.paired_inventory(pair)
        if imbalance > 0:
            self.log_gate(
                f"open:{pair.strike}:imbalance",
                f"[SKIP] strike={pair.strike} reason=unpaired_inventory "
                f"call={self.position(pair.call_symbol)} put={self.position(pair.put_symbol)}",
            )
            return False
        if not self.cooldown_ready(pair.strike):
            self.log_gate(
                f"open:{pair.strike}:cooldown",
                f"[SKIP] strike={pair.strike} reason=cooldown "
                f"wait={self.cfg.per_strike_cooldown_sec - (self.now() - self.last_action_at.get(pair.strike, 0.0)):.2f}s",
            )
            return False
        if self.strike_has_pending_orders(pair.strike):
            self.log_gate(
                f"open:{pair.strike}:pending",
                f"[SKIP] strike={pair.strike} reason=pending_orders count={self.pending_order_count_for_strike(pair.strike)}",
            )
            return False
        if synthetic_long > 0 or synthetic_short > 0:
            self.log_gate(
                f"open:{pair.strike}:paired",
                f"[SKIP] strike={pair.strike} reason=existing_pair long={synthetic_long} short={synthetic_short}",
            )
            return False

        if snapshot.long_synth_edge is None and snapshot.short_synth_edge is None:
            self.log_gate(f"open:{pair.strike}:quotes", f"[SKIP] strike={pair.strike} reason=no_executable_quotes")
            return False

        if snapshot.long_synth_edge is not None and snapshot.long_synth_edge >= self.cfg.parity_entry_edge:
            return await self.open_synthetic_long(pair, snapshot)

        if snapshot.short_synth_edge is not None and snapshot.short_synth_edge >= self.cfg.parity_entry_edge:
            return await self.open_synthetic_short(pair, snapshot)

        self.log_gate(
            f"open:{pair.strike}:edge",
            f"[SKIP] strike={pair.strike} reason=edge_below_threshold "
            f"long={snapshot.long_synth_edge} short={snapshot.short_synth_edge} "
            f"threshold={self.cfg.parity_entry_edge}",
        )

        return False

    async def maybe_force_parity_trade(self, snapshots: list[tuple[OptionPair, ParitySnapshot]]) -> bool:
        if self.probe_active():
            return False
        if self.has_any_option_inventory_or_orders():
            return False
        if self.now() - self.last_open_trade_at < self.cfg.force_entry_after_sec:
            return False

        best_pair: Optional[OptionPair] = None
        best_snapshot: Optional[ParitySnapshot] = None
        best_side: Optional[str] = None
        best_edge = float("-inf")

        for pair, snapshot in snapshots:
            if not self.cooldown_ready(pair.strike):
                continue
            if self.strike_has_pending_orders(pair.strike):
                continue
            if snapshot.long_synth_edge is not None and snapshot.long_synth_edge > best_edge:
                best_pair = pair
                best_snapshot = snapshot
                best_side = "long"
                best_edge = snapshot.long_synth_edge
            if snapshot.short_synth_edge is not None and snapshot.short_synth_edge > best_edge:
                best_pair = pair
                best_snapshot = snapshot
                best_side = "short"
                best_edge = snapshot.short_synth_edge

        if best_pair is None or best_snapshot is None or best_side is None:
            self.log_gate("force:none", "[FORCE] skip reason=no_force_candidate")
            return False

        if best_edge < self.cfg.force_entry_edge_floor:
            self.log_gate(
                "force:floor",
                f"[FORCE] skip reason=edge_below_floor best_edge={best_edge} "
                f"floor={self.cfg.force_entry_edge_floor}",
            )
            return False

        print(
            f"[FORCE] strike={best_pair.strike} side={best_side} "
            f"edge={best_edge:.2f} floor={self.cfg.force_entry_edge_floor}"
        )
        if best_side == "long":
            return await self.open_synthetic_long(best_pair, best_snapshot)
        return await self.open_synthetic_short(best_pair, best_snapshot)

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
                diff_parts.append(
                    f"{pair.strike}:C={call_pos}/{call_exchange},P={put_pos}/{put_exchange}"
                )
            snapshot = self.parity_snapshot(pair, spot) if spot is not None else None
            if snapshot is None:
                continue
            edge_parts.append(
                f"{pair.strike}:L={snapshot.long_synth_edge if snapshot.long_synth_edge is not None else 'NA'}"
                f"/S={snapshot.short_synth_edge if snapshot.short_synth_edge is not None else 'NA'}"
                f"/P={self.pending_order_count_for_strike(pair.strike)}"
            )

        print(
            "[STATUS]",
            f"spot={spot:.2f}" if spot is not None else "spot=NA",
            f"pairs={len(self.option_pairs)}",
            f"local={' | '.join(local_parts) if local_parts else 'flat'}",
            f"exchange_diff={' | '.join(diff_parts) if diff_parts else 'none'}",
            f"outstanding={self.outstanding_qty()}",
            f"edges={' | '.join(edge_parts) if edge_parts else 'NA'}",
            f"probe={self.startup_probe_state}",
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

            snapshots: list[tuple[OptionPair, ParitySnapshot]] = []
            opened = False
            for pair in self.option_pairs:
                if self.probe_active() and self.startup_probe_strike == pair.strike:
                    self.log_gate(
                        f"probe:strike:{pair.strike}",
                        f"[SKIP] strike={pair.strike} reason=probe_active state={self.startup_probe_state}",
                    )
                    continue
                snapshot = self.parity_snapshot(pair, spot)
                if snapshot is None:
                    continue
                snapshots.append((pair, snapshot))
                reduced = await self.maybe_reduce_when_normalized(pair, snapshot)
                if reduced:
                    continue
                cleaned = await self.maybe_reduce_unpaired_inventory(pair, snapshot)
                if cleaned:
                    continue
                opened = await self.maybe_open_parity_trade(pair, snapshot)
                if opened:
                    break

            if not opened:
                await self.maybe_force_parity_trade(snapshots)

            self.print_status(spot)

    async def strategy_loop(self) -> None:
        await asyncio.sleep(1.0)
        print(
            f"[INIT] host={self.cfg.host} user={self.cfg.username} "
            f"r={self.cfg.risk_free_rate} T={self.cfg.time_to_expiry_years} "
            f"entry_edge={self.cfg.parity_entry_edge} "
            f"startup_probe={self.cfg.startup_probe_enabled} "
            f"probe_timeout={self.cfg.startup_probe_timeout_sec} "
            f"probe_attempts={self.cfg.startup_probe_max_attempts} "
            f"force_after={self.cfg.force_entry_after_sec} "
            f"force_floor={self.cfg.force_entry_edge_floor}"
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
            if meta.purpose == "startup-probe-entry":
                self.set_probe_state("entry_filled_waiting_exit", f"symbol={meta.symbol} pos={self.position(meta.symbol)}")
            elif meta.purpose == "startup-probe-exit" and self.position(meta.symbol) == 0:
                self.mark_probe_done("probe_round_trip_complete")
                print("[PROBE] complete")
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)
        self._book_event.set()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self.order_meta.pop(order_id, None)
        print(f"[REJECT] {order_id}: {reason}")
        self._book_event.set()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        meta = self.order_meta.get(order_id)
        if meta:
            meta.cancel_pending = False
        if success:
            if meta and meta.purpose.startswith("startup-probe"):
                print(f"[PROBE] cancel order_id={order_id} purpose={meta.purpose} symbol={meta.symbol}")
            self.order_meta.pop(order_id, None)
        else:
            print(f"[CANCEL-FAIL] {order_id}: {error}")
        self._book_event.set()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        return

    async def start(self):
        asyncio.create_task(self.strategy_loop())
        await self.connect()


async def main() -> None:
    config = load_config()
    bot = MarketBBot(config)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
