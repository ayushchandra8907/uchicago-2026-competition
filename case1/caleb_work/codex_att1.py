from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from statistics import median
import sys
import time
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.append(str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient


OPTION_RE = re.compile(r"^B_(C|P)_(\d+)$")


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
    max_b_position: int
    max_hedge_position: int
    max_option_position: int
    max_outstanding_qty: int
    base_quote_size: int
    take_size: int
    option_arb_size: int
    basket_arb_size: int
    quote_half_spread: int
    take_edge: int
    parity_edge: int
    box_edge: int
    etf_edge: int
    inventory_skew_ticks: float
    swap_fee: int
    strategy_poll_sec: float
    reprice_sec: float
    stale_quote_sec: float
    hedge_cooldown_sec: float
    status_sec: float


@dataclass
class PriceBand:
    lower: Optional[float] = None
    upper: Optional[float] = None
    mid: Optional[float] = None


@dataclass
class OrderMeta:
    symbol: str
    side: Side
    price: int
    qty: int
    purpose: str
    created_at: float
    cancel_pending: bool = False


def load_config() -> BotConfig:
    return BotConfig(
        host=env_str("UTC_HOST", "34.197.188.76:3333"),
        username=env_str("UTC_USERNAME", "uiuc"),
        password=env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
        max_b_position=env_int("B_MAX_POSITION", 50),
        max_hedge_position=env_int("B_MAX_HEDGE_POSITION", 20),
        max_option_position=env_int("B_MAX_OPTION_POSITION", 10),
        max_outstanding_qty=env_int("B_MAX_OUTSTANDING_QTY", 120),
        base_quote_size=env_int("B_QUOTE_SIZE", 3),
        take_size=env_int("B_TAKE_SIZE", 4),
        option_arb_size=env_int("B_OPTION_ARB_SIZE", 1),
        basket_arb_size=env_int("B_BASKET_ARB_SIZE", 1),
        quote_half_spread=env_int("B_QUOTE_HALF_SPREAD", 4),
        take_edge=env_int("B_TAKE_EDGE", 5),
        parity_edge=env_int("B_PARITY_EDGE", 2),
        box_edge=env_int("B_BOX_EDGE", 2),
        etf_edge=env_int("B_ETF_EDGE", 2),
        inventory_skew_ticks=env_float("B_INVENTORY_SKEW_TICKS", 0.20),
        swap_fee=env_int("B_SWAP_FEE", 5),
        strategy_poll_sec=env_float("B_STRATEGY_POLL_SEC", 0.20),
        reprice_sec=env_float("B_REPRICE_SEC", 0.45),
        stale_quote_sec=env_float("B_STALE_QUOTE_SEC", 1.50),
        hedge_cooldown_sec=env_float("B_HEDGE_COOLDOWN_SEC", 0.60),
        status_sec=env_float("B_STATUS_SEC", 5.00),
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


def mid_price(book, fallback: Optional[float] = None) -> Optional[float]:
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

        self.option_chain: list[tuple[int, str, str]] = []
        self.last_trade: dict[str, int] = {}
        self.order_meta: dict[str, OrderMeta] = {}
        self.quote_order_ids: dict[str, Optional[str]] = {"BUY": None, "SELL": None}
        self.last_quote_refresh = 0.0
        self.last_status = 0.0
        self.last_hedge_action: dict[str, float] = {}
        self.last_swap_request = 0.0
        self.swap_in_flight: Optional[str] = None
        self._book_event = asyncio.Event()
        self._strategy_lock = asyncio.Lock()
        self._running = False
        self.discover_option_chain()

    def now(self) -> float:
        return time.monotonic()

    def symbol_limit(self, symbol: str) -> int:
        if symbol == "B":
            return self.cfg.max_b_position
        if symbol in {"A", "C", "ETF"}:
            return self.cfg.max_hedge_position
        if OPTION_RE.match(symbol):
            return self.cfg.max_option_position
        return self.cfg.max_hedge_position

    def discover_option_chain(self) -> None:
        strikes: dict[int, dict[str, str]] = {}
        for symbol in set(self.symbols) | set(self.order_books.keys()):
            match = OPTION_RE.fullmatch(symbol)
            if not match:
                continue
            side, strike_raw = match.groups()
            strike = int(strike_raw)
            strikes.setdefault(strike, {})[side] = symbol

        chain: list[tuple[int, str, str]] = []
        for strike in sorted(strikes):
            if "C" in strikes[strike] and "P" in strikes[strike]:
                chain.append((strike, strikes[strike]["C"], strikes[strike]["P"]))
        self.option_chain = chain

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
        return mid_price(book, fallback=self.last_trade.get(symbol))

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
        if self.outstanding_qty() + qty > self.cfg.max_outstanding_qty:
            return False

        projected = self.signed_exposure(symbol) + (qty if side == Side.BUY else -qty)
        return abs(projected) <= self.symbol_limit(symbol)

    async def submit_limit(self, symbol: str, qty: int, side: Side, price: int, purpose: str) -> Optional[str]:
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
            created_at=self.now(),
        )
        if purpose == "quote" and symbol == "B":
            self.quote_order_ids["BUY" if side == Side.BUY else "SELL"] = order_id
        return order_id

    async def cancel_if_needed(self, order_id: Optional[str]) -> None:
        if not order_id:
            return
        meta = self.order_meta.get(order_id)
        if meta is None or meta.cancel_pending:
            return
        if order_id not in self.open_orders:
            return
        meta.cancel_pending = True
        try:
            await self.cancel_order(order_id)
        except Exception as exc:
            meta.cancel_pending = False
            print(f"[CANCEL-ERROR] {order_id}: {exc}")

    def prune_order_meta(self) -> None:
        live_quotes = {order_id for order_id in self.quote_order_ids.values() if order_id}
        stale_ids = [order_id for order_id in self.order_meta if order_id not in self.open_orders and order_id not in live_quotes]
        for order_id in stale_ids:
            self.order_meta.pop(order_id, None)

    def option_parity_band(self) -> PriceBand:
        lowers: list[float] = []
        uppers: list[float] = []
        mids: list[float] = []

        for strike, call_sym, put_sym in self.option_chain:
            call_book = self.get_book(call_sym)
            put_book = self.get_book(put_sym)
            if call_book is None or put_book is None:
                continue

            call_bid, _, call_ask, _ = self.top(call_sym)
            put_bid, _, put_ask, _ = self.top(put_sym)
            call_mid = self.mid(call_sym)
            put_mid = self.mid(put_sym)

            if call_bid is not None and put_ask is not None:
                lowers.append(call_bid - put_ask + strike)
            if call_ask is not None and put_bid is not None:
                uppers.append(call_ask - put_bid + strike)
            if call_mid is not None and put_mid is not None:
                mids.append(call_mid - put_mid + strike)

        band = PriceBand()
        if lowers:
            band.lower = max(lowers)
        if uppers:
            band.upper = min(uppers)
        if mids:
            band.mid = float(median(mids))
        elif band.lower is not None and band.upper is not None:
            band.mid = (band.lower + band.upper) / 2.0
        return band

    def etf_b_band(self) -> PriceBand:
        a_bid, _, a_ask, _ = self.top("A")
        c_bid, _, c_ask, _ = self.top("C")
        etf_bid, _, etf_ask, _ = self.top("ETF")

        band = PriceBand()
        if etf_bid is not None and a_ask is not None and c_ask is not None:
            band.lower = etf_bid - self.cfg.swap_fee - a_ask - c_ask
        if etf_ask is not None and a_bid is not None and c_bid is not None:
            band.upper = etf_ask + self.cfg.swap_fee - a_bid - c_bid
        if band.lower is not None and band.upper is not None:
            band.mid = (band.lower + band.upper) / 2.0
        return band

    def combined_b_band(self) -> PriceBand:
        option_band = self.option_parity_band()
        etf_band = self.etf_b_band()
        direct_mid = self.mid("B")

        lowers = [value for value in (option_band.lower, etf_band.lower) if value is not None]
        uppers = [value for value in (option_band.upper, etf_band.upper) if value is not None]
        mids = [value for value in (option_band.mid, etf_band.mid, direct_mid) if value is not None]

        band = PriceBand()
        if lowers:
            band.lower = max(lowers)
        if uppers:
            band.upper = min(uppers)
        if mids:
            fair = float(median(mids))
            if band.lower is not None:
                fair = max(fair, band.lower)
            if band.upper is not None:
                fair = min(fair, band.upper)
            band.mid = fair
        elif band.lower is not None and band.upper is not None:
            band.mid = (band.lower + band.upper) / 2.0
        return band

    async def direct_b_take(self, band: PriceBand) -> None:
        bid, bid_qty, ask, ask_qty = self.top("B")
        if bid is None and ask is None:
            return

        if ask is not None:
            buy_target = band.lower if band.lower is not None else band.mid
            if buy_target is not None and ask <= math.floor(buy_target - self.cfg.take_edge):
                qty = min(self.cfg.take_size, ask_qty or self.cfg.take_size)
                await self.submit_limit("B", qty, Side.BUY, ask, "take")

        if bid is not None:
            sell_target = band.upper if band.upper is not None else band.mid
            if sell_target is not None and bid >= math.ceil(sell_target + self.cfg.take_edge):
                qty = min(self.cfg.take_size, bid_qty or self.cfg.take_size)
                await self.submit_limit("B", qty, Side.SELL, bid, "take")

    def quote_targets(self, fair: float) -> tuple[Optional[int], int, Optional[int], int]:
        bid, _, ask, _ = self.top("B")
        inv = self.position("B")
        reservation = fair - (inv * self.cfg.inventory_skew_ticks)
        target_bid = math.floor(reservation - self.cfg.quote_half_spread)
        target_ask = math.ceil(reservation + self.cfg.quote_half_spread)

        if bid is not None:
            target_bid = max(target_bid, bid)
        if ask is not None:
            target_ask = min(target_ask, ask)

        if bid is not None and ask is not None:
            if ask - bid > 1:
                target_bid = min(max(target_bid, bid + 1), ask - 1)
                target_ask = max(min(target_ask, ask - 1), target_bid + 1)
            else:
                target_bid = bid
                target_ask = ask

        if target_bid >= target_ask:
            target_bid = math.floor(fair) - 1
            target_ask = math.ceil(fair) + 1

        bid_qty = self.cfg.base_quote_size
        ask_qty = self.cfg.base_quote_size
        if inv > 0:
            bid_qty = max(0, self.cfg.base_quote_size - inv // 4)
            ask_qty = min(self.cfg.take_size, self.cfg.base_quote_size + max(1, inv // 4))
        elif inv < 0:
            ask_qty = max(0, self.cfg.base_quote_size - abs(inv) // 4)
            bid_qty = min(self.cfg.take_size, self.cfg.base_quote_size + max(1, abs(inv) // 4))

        soft_limit = int(self.cfg.max_b_position * 0.8)
        if inv >= soft_limit:
            bid_qty = 0
        if inv <= -soft_limit:
            ask_qty = 0

        return int(target_bid), int(bid_qty), int(target_ask), int(ask_qty)

    async def refresh_b_quotes(self, band: PriceBand) -> None:
        if band.mid is None:
            await self.cancel_if_needed(self.quote_order_ids["BUY"])
            await self.cancel_if_needed(self.quote_order_ids["SELL"])
            return

        now = self.now()
        if now - self.last_quote_refresh < self.cfg.reprice_sec:
            return

        desired_bid, bid_qty, desired_ask, ask_qty = self.quote_targets(band.mid)
        self.last_quote_refresh = now

        await self.sync_quote_side("BUY", desired_bid, bid_qty)
        await self.sync_quote_side("SELL", desired_ask, ask_qty)

    async def sync_quote_side(self, side_name: str, desired_price: int, desired_qty: int) -> None:
        current_id = self.quote_order_ids[side_name]
        current_meta = self.order_meta.get(current_id) if current_id else None
        side = Side.BUY if side_name == "BUY" else Side.SELL

        if desired_qty <= 0:
            await self.cancel_if_needed(current_id)
            return

        if current_id and current_id in self.open_orders and current_meta is not None:
            age = self.now() - current_meta.created_at
            if (
                current_meta.price == desired_price
                and current_meta.qty == desired_qty
                and age <= self.cfg.stale_quote_sec
                and not current_meta.cancel_pending
            ):
                return
            await self.cancel_if_needed(current_id)
            return

        order_id = await self.submit_limit("B", desired_qty, side, desired_price, "quote")
        if order_id:
            self.quote_order_ids[side_name] = order_id

    async def execute_parity_arbs(self) -> None:
        stock_bid, stock_bid_qty, stock_ask, stock_ask_qty = self.top("B")
        if stock_bid is None or stock_ask is None:
            return

        for strike, call_sym, put_sym in self.option_chain:
            call_bid, call_bid_qty, call_ask, call_ask_qty = self.top(call_sym)
            put_bid, put_bid_qty, put_ask, put_ask_qty = self.top(put_sym)

            if call_bid is not None and put_ask is not None:
                conversion_edge = call_bid - put_ask - stock_ask + strike
                if conversion_edge >= self.cfg.parity_edge:
                    qty = min(self.cfg.option_arb_size, stock_ask_qty, call_bid_qty, put_ask_qty)
                    if qty > 0:
                        await asyncio.gather(
                            self.submit_limit("B", qty, Side.BUY, stock_ask, "parity"),
                            self.submit_limit(call_sym, qty, Side.SELL, call_bid, "parity"),
                            self.submit_limit(put_sym, qty, Side.BUY, put_ask, "parity"),
                        )

            if call_ask is not None and put_bid is not None:
                reversal_edge = stock_bid - strike - call_ask + put_bid
                if reversal_edge >= self.cfg.parity_edge:
                    qty = min(self.cfg.option_arb_size, stock_bid_qty, call_ask_qty, put_bid_qty)
                    if qty > 0:
                        await asyncio.gather(
                            self.submit_limit("B", qty, Side.SELL, stock_bid, "parity"),
                            self.submit_limit(call_sym, qty, Side.BUY, call_ask, "parity"),
                            self.submit_limit(put_sym, qty, Side.SELL, put_bid, "parity"),
                        )

    async def execute_box_arbs(self) -> None:
        for index, (k1, c1_sym, p1_sym) in enumerate(self.option_chain):
            for k2, c2_sym, p2_sym in self.option_chain[index + 1 :]:
                c1_bid, c1_bid_qty, c1_ask, c1_ask_qty = self.top(c1_sym)
                c2_bid, c2_bid_qty, c2_ask, c2_ask_qty = self.top(c2_sym)
                p1_bid, p1_bid_qty, p1_ask, p1_ask_qty = self.top(p1_sym)
                p2_bid, p2_bid_qty, p2_ask, p2_ask_qty = self.top(p2_sym)

                theoretical = k2 - k1

                if None not in (c1_ask, c2_bid, p2_ask, p1_bid):
                    buy_cost = c1_ask - c2_bid + p2_ask - p1_bid
                    buy_edge = theoretical - buy_cost
                    if buy_edge >= self.cfg.box_edge:
                        qty = min(
                            self.cfg.option_arb_size,
                            c1_ask_qty,
                            c2_bid_qty,
                            p2_ask_qty,
                            p1_bid_qty,
                        )
                        if qty > 0:
                            await asyncio.gather(
                                self.submit_limit(c1_sym, qty, Side.BUY, c1_ask, "box"),
                                self.submit_limit(c2_sym, qty, Side.SELL, c2_bid, "box"),
                                self.submit_limit(p2_sym, qty, Side.BUY, p2_ask, "box"),
                                self.submit_limit(p1_sym, qty, Side.SELL, p1_bid, "box"),
                            )

                if None not in (c1_bid, c2_ask, p2_bid, p1_ask):
                    sell_net = c1_bid - c2_ask + p2_bid - p1_ask
                    sell_edge = sell_net - theoretical
                    if sell_edge >= self.cfg.box_edge:
                        qty = min(
                            self.cfg.option_arb_size,
                            c1_bid_qty,
                            c2_ask_qty,
                            p2_bid_qty,
                            p1_ask_qty,
                        )
                        if qty > 0:
                            await asyncio.gather(
                                self.submit_limit(c1_sym, qty, Side.SELL, c1_bid, "box"),
                                self.submit_limit(c2_sym, qty, Side.BUY, c2_ask, "box"),
                                self.submit_limit(p2_sym, qty, Side.SELL, p2_bid, "box"),
                                self.submit_limit(p1_sym, qty, Side.BUY, p1_ask, "box"),
                            )

    async def execute_etf_b_arbs(self) -> None:
        a_bid, a_bid_qty, a_ask, a_ask_qty = self.top("A")
        b_bid, b_bid_qty, b_ask, b_ask_qty = self.top("B")
        c_bid, c_bid_qty, c_ask, c_ask_qty = self.top("C")
        etf_bid, etf_bid_qty, etf_ask, etf_ask_qty = self.top("ETF")
        etf_band = self.etf_b_band()

        if etf_band.upper is not None and b_bid is not None and b_bid >= etf_band.upper + self.cfg.etf_edge:
            if None not in (a_bid, c_bid, etf_ask):
                qty = min(self.cfg.basket_arb_size, a_bid_qty, b_bid_qty, c_bid_qty, etf_ask_qty)
                if qty > 0:
                    await asyncio.gather(
                        self.submit_limit("ETF", qty, Side.BUY, etf_ask, "basket"),
                        self.submit_limit("A", qty, Side.SELL, a_bid, "basket"),
                        self.submit_limit("B", qty, Side.SELL, b_bid, "basket"),
                        self.submit_limit("C", qty, Side.SELL, c_bid, "basket"),
                    )

        if etf_band.lower is not None and b_ask is not None and b_ask <= etf_band.lower - self.cfg.etf_edge:
            if None not in (a_ask, c_ask, etf_bid):
                qty = min(self.cfg.basket_arb_size, a_ask_qty, b_ask_qty, c_ask_qty, etf_bid_qty)
                if qty > 0:
                    await asyncio.gather(
                        self.submit_limit("ETF", qty, Side.SELL, etf_bid, "basket"),
                        self.submit_limit("A", qty, Side.BUY, a_ask, "basket"),
                        self.submit_limit("B", qty, Side.BUY, b_ask, "basket"),
                        self.submit_limit("C", qty, Side.BUY, c_ask, "basket"),
                    )

    async def maybe_swap_spreads(self) -> None:
        now = self.now()
        if self.swap_in_flight is not None or now - self.last_swap_request < self.cfg.hedge_cooldown_sec:
            return

        to_etf_qty = min(
            max(0, self.position("A")),
            max(0, self.position("B")),
            max(0, self.position("C")),
            max(0, -self.position("ETF")),
        )
        from_etf_qty = min(
            max(0, self.position("ETF")),
            max(0, -self.position("A")),
            max(0, -self.position("B")),
            max(0, -self.position("C")),
        )

        if to_etf_qty > 0:
            self.swap_in_flight = "toETF"
            self.last_swap_request = now
            await self.place_swap_order("toETF", int(to_etf_qty))
            return

        if from_etf_qty > 0:
            self.swap_in_flight = "fromETF"
            self.last_swap_request = now
            await self.place_swap_order("fromETF", int(from_etf_qty))

    async def reduce_unwanted_hedges(self) -> None:
        for symbol in ("A", "C", "ETF"):
            pos = self.position(symbol)
            if pos == 0:
                continue
            last_action = self.last_hedge_action.get(symbol, 0.0)
            if self.now() - last_action < self.cfg.hedge_cooldown_sec:
                continue

            bid, bid_qty, ask, ask_qty = self.top(symbol)
            if pos > 0 and bid is not None:
                qty = min(abs(pos), self.cfg.take_size, bid_qty or self.cfg.take_size)
                if qty > 0:
                    self.last_hedge_action[symbol] = self.now()
                    await self.submit_limit(symbol, qty, Side.SELL, bid, "hedge-flatten")
            elif pos < 0 and ask is not None:
                qty = min(abs(pos), self.cfg.take_size, ask_qty or self.cfg.take_size)
                if qty > 0:
                    self.last_hedge_action[symbol] = self.now()
                    await self.submit_limit(symbol, qty, Side.BUY, ask, "hedge-flatten")

    async def reduce_extreme_option_inventory(self) -> None:
        threshold = max(2, self.cfg.max_option_position // 2)
        for _, call_sym, put_sym in self.option_chain:
            for symbol in (call_sym, put_sym):
                pos = self.position(symbol)
                if abs(pos) < threshold:
                    continue
                last_action = self.last_hedge_action.get(symbol, 0.0)
                if self.now() - last_action < self.cfg.hedge_cooldown_sec:
                    continue

                bid, bid_qty, ask, ask_qty = self.top(symbol)
                if pos > 0 and bid is not None:
                    qty = min(abs(pos), self.cfg.option_arb_size, bid_qty or self.cfg.option_arb_size)
                    if qty > 0:
                        self.last_hedge_action[symbol] = self.now()
                        await self.submit_limit(symbol, qty, Side.SELL, bid, "option-flatten")
                elif pos < 0 and ask is not None:
                    qty = min(abs(pos), self.cfg.option_arb_size, ask_qty or self.cfg.option_arb_size)
                    if qty > 0:
                        self.last_hedge_action[symbol] = self.now()
                        await self.submit_limit(symbol, qty, Side.BUY, ask, "option-flatten")

    async def emergency_b_flatten(self) -> None:
        pos = self.position("B")
        soft_limit = int(self.cfg.max_b_position * 0.9)
        if abs(pos) < soft_limit:
            return
        bid, bid_qty, ask, ask_qty = self.top("B")
        if pos > 0 and bid is not None:
            qty = min(pos - soft_limit + 1, self.cfg.take_size, bid_qty or self.cfg.take_size)
            if qty > 0:
                await self.submit_limit("B", qty, Side.SELL, bid, "emergency")
        elif pos < 0 and ask is not None:
            qty = min(abs(pos) - soft_limit + 1, self.cfg.take_size, ask_qty or self.cfg.take_size)
            if qty > 0:
                await self.submit_limit("B", qty, Side.BUY, ask, "emergency")

    def print_status(self, band: PriceBand) -> None:
        now = self.now()
        if now - self.last_status < self.cfg.status_sec:
            return
        self.last_status = now

        b_bid, _, b_ask, _ = self.top("B")
        fair = f"{band.mid:.2f}" if band.mid is not None else "NA"
        lower = f"{band.lower:.2f}" if band.lower is not None else "NA"
        upper = f"{band.upper:.2f}" if band.upper is not None else "NA"
        print(
            "[STATUS]",
            f"Bpos={self.position('B')}",
            f"A={self.position('A')}",
            f"C={self.position('C')}",
            f"ETF={self.position('ETF')}",
            f"Bbook={b_bid}/{b_ask}",
            f"fair={fair}",
            f"band={lower}..{upper}",
            f"options={len(self.option_chain)}",
            f"outstanding={self.outstanding_qty()}",
        )

    async def run_strategy(self) -> None:
        async with self._strategy_lock:
            self.prune_order_meta()
            self.discover_option_chain()
            band = self.combined_b_band()

            await self.maybe_swap_spreads()
            await self.execute_parity_arbs()
            await self.execute_box_arbs()
            await self.execute_etf_b_arbs()
            await self.direct_b_take(band)
            await self.refresh_b_quotes(band)
            await self.emergency_b_flatten()
            await self.reduce_unwanted_hedges()
            await self.reduce_extreme_option_inventory()
            self.print_status(band)

    async def strategy_loop(self) -> None:
        await asyncio.sleep(1.0)
        print(f"[INIT] host={self.cfg.host} user={self.cfg.username} option_chain={self.option_chain}")
        self._running = True
        while True:
            try:
                try:
                    await asyncio.wait_for(self._book_event.wait(), timeout=self.cfg.strategy_poll_sec)
                except TimeoutError:
                    pass
                self._book_event.clear()
                await self.run_strategy()
            except Exception as exc:
                print(f"[STRATEGY-ERROR] {exc}")
                await asyncio.sleep(0.25)

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol == "B" or symbol in {"A", "C", "ETF"} or OPTION_RE.match(symbol):
            self._book_event.set()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        self.last_trade[symbol] = int(price)
        if symbol == "B" or symbol in {"A", "C", "ETF"} or OPTION_RE.match(symbol):
            self._book_event.set()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        meta = self.order_meta.get(order_id)
        if meta:
            remaining = int(self.open_orders.get(order_id, [None, 0])[1]) if order_id in self.open_orders else 0
            if meta.purpose == "quote" and meta.symbol == "B":
                if remaining <= 0:
                    key = "BUY" if meta.side == Side.BUY else "SELL"
                    self.quote_order_ids[key] = None
            if remaining <= 0:
                self.order_meta.pop(order_id, None)
            print(f"[FILL] {meta.purpose} {meta.symbol} {meta.side.name} {qty}@{price}")
        self._book_event.set()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        meta = self.order_meta.pop(order_id, None)
        if meta and meta.purpose == "quote" and meta.symbol == "B":
            key = "BUY" if meta.side == Side.BUY else "SELL"
            if self.quote_order_ids[key] == order_id:
                self.quote_order_ids[key] = None
        print(f"[REJECT] {order_id}: {reason}")
        self._book_event.set()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        meta = self.order_meta.get(order_id)
        if meta:
            meta.cancel_pending = False
        if success:
            if meta and meta.purpose == "quote" and meta.symbol == "B":
                key = "BUY" if meta.side == Side.BUY else "SELL"
                if self.quote_order_ids[key] == order_id:
                    self.quote_order_ids[key] = None
            self.order_meta.pop(order_id, None)
        else:
            print(f"[CANCEL-FAIL] {order_id}: {error}")
        self._book_event.set()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        print(f"[SWAP] {swap} qty={qty} success={success}")
        self.swap_in_flight = None
        self._book_event.set()

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
