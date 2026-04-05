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
    sys.path.append(str(LOCAL_LIB))

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
        parity_entry_edge=env_float("B_PARITY_ENTRY_EDGE", 3.0),
        parity_exit_band=env_float("B_PARITY_EXIT_BAND", 1.0),
        per_strike_cooldown_sec=env_float("B_STRIKE_COOLDOWN_SEC", 0.75),
        order_ttl_sec=env_float("B_ORDER_TTL_SEC", 0.75),
        poll_sec=env_float("B_POLL_SEC", 0.25),
        status_sec=env_float("B_STATUS_SEC", 5.0),
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
        self.last_action_at: dict[int, float] = {}
        self.last_status = 0.0
        self._book_event = asyncio.Event()
        self._strategy_lock = asyncio.Lock()
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

    def position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

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

    def signed_exposure(self, symbol: str) -> int:
        return self.position(symbol) + self.pending_side_qty(symbol, Side.BUY) - self.pending_side_qty(symbol, Side.SELL)

    def can_submit(self, symbol: str, side: Side, qty: int) -> bool:
        if qty <= 0:
            return False
        if not OPTION_RE.fullmatch(symbol):
            return False
        if self.outstanding_qty() + qty > self.cfg.max_total_outstanding:
            return False
        projected = self.signed_exposure(symbol) + (qty if side == Side.BUY else -qty)
        return abs(projected) <= self.cfg.max_position_per_option

    async def submit_cross(self, symbol: str, side: Side, qty: int, price: int, purpose: str, strike: int) -> Optional[str]:
        if not self.can_submit(symbol, side, qty):
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

    async def open_synthetic_long(self, pair: OptionPair, snapshot: ParitySnapshot) -> None:
        if snapshot.call_ask is None or snapshot.put_bid is None:
            return
        size = min(
            self.trade_size_for_open(pair, snapshot),
            snapshot.call_ask_qty or self.cfg.max_trade_qty,
            snapshot.put_bid_qty or self.cfg.max_trade_qty,
        )
        if size <= 0:
            return

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.BUY, size, snapshot.call_ask, "pcp-open-long", pair.strike),
            self.submit_cross(pair.put_symbol, Side.SELL, size, snapshot.put_bid, "pcp-open-long", pair.strike),
        )
        if any(results):
            self.last_action_at[pair.strike] = self.now()
            print(
                f"[TRADE] strike={pair.strike} action=BUY_CALL_SELL_PUT "
                f"edge={snapshot.long_synth_edge:.2f} spot={snapshot.spot:.2f}"
            )

    async def open_synthetic_short(self, pair: OptionPair, snapshot: ParitySnapshot) -> None:
        if snapshot.call_bid is None or snapshot.put_ask is None:
            return
        size = min(
            self.trade_size_for_open(pair, snapshot),
            snapshot.call_bid_qty or self.cfg.max_trade_qty,
            snapshot.put_ask_qty or self.cfg.max_trade_qty,
        )
        if size <= 0:
            return

        results = await asyncio.gather(
            self.submit_cross(pair.call_symbol, Side.SELL, size, snapshot.call_bid, "pcp-open-short", pair.strike),
            self.submit_cross(pair.put_symbol, Side.BUY, size, snapshot.put_ask, "pcp-open-short", pair.strike),
        )
        if any(results):
            self.last_action_at[pair.strike] = self.now()
            print(
                f"[TRADE] strike={pair.strike} action=SELL_CALL_BUY_PUT "
                f"edge={snapshot.short_synth_edge:.2f} spot={snapshot.spot:.2f}"
            )

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
        if snapshot.mid_gap is None:
            return False
        if abs(snapshot.mid_gap) > self.cfg.parity_exit_band:
            return False
        if not self.cooldown_ready(pair.strike):
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
            return False
        if not self.cooldown_ready(pair.strike):
            return False
        if synthetic_long > 0 or synthetic_short > 0:
            return False

        if snapshot.long_synth_edge is not None and snapshot.long_synth_edge >= self.cfg.parity_entry_edge:
            await self.open_synthetic_long(pair, snapshot)
            return True

        if snapshot.short_synth_edge is not None and snapshot.short_synth_edge >= self.cfg.parity_entry_edge:
            await self.open_synthetic_short(pair, snapshot)
            return True

        return False

    def prune_order_meta(self) -> None:
        for order_id in list(self.order_meta):
            if order_id not in self.open_orders:
                self.order_meta.pop(order_id, None)

    def print_status(self, spot: Optional[float]) -> None:
        now = self.now()
        if now - self.last_status < self.cfg.status_sec:
            return
        self.last_status = now

        parts: list[str] = []
        for pair in self.option_pairs:
            call_pos = self.position(pair.call_symbol)
            put_pos = self.position(pair.put_symbol)
            if call_pos or put_pos:
                parts.append(f"{pair.strike}:C={call_pos},P={put_pos}")

        print(
            "[STATUS]",
            f"spot={spot:.2f}" if spot is not None else "spot=NA",
            f"pairs={len(self.option_pairs)}",
            f"positions={' | '.join(parts) if parts else 'flat'}",
            f"outstanding={self.outstanding_qty()}",
        )

    async def run_strategy(self) -> None:
        async with self._strategy_lock:
            self.prune_order_meta()
            await self.cancel_stale_orders()
            self.discover_option_pairs()

            spot = self.extract_spot()
            if spot is None:
                return

            for pair in self.option_pairs:
                snapshot = self.parity_snapshot(pair, spot)
                if snapshot is None:
                    continue
                reduced = await self.maybe_reduce_when_normalized(pair, snapshot)
                if reduced:
                    continue
                cleaned = await self.maybe_reduce_unpaired_inventory(pair, snapshot)
                if cleaned:
                    continue
                await self.maybe_open_parity_trade(pair, snapshot)

            self.print_status(spot)

    async def strategy_loop(self) -> None:
        await asyncio.sleep(1.0)
        print(
            f"[INIT] host={self.cfg.host} user={self.cfg.username} "
            f"r={self.cfg.risk_free_rate} T={self.cfg.time_to_expiry_years} "
            f"entry_edge={self.cfg.parity_entry_edge}"
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
            print(f"[FILL] {meta.purpose} {meta.symbol} {meta.side.name} {qty}@{price}")
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
