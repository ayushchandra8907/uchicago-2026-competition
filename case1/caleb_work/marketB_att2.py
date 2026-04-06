from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from utcxchangelib import Side, XChangeClient


@dataclass
class Bundle:
    strategy: str
    strike: int | tuple[int, int]
    direction: str
    order_ids: set[str] = field(default_factory=set)
    started_at: float = 0.0
    warned_stalled: bool = False
    cancel_requested: bool = False


class PCPBoxBot(XChangeClient):
    STRIKES = (950, 1000, 1050)
    BOX_PAIRS = ((950, 1000), (1000, 1050), (950, 1050))
    EDGE_THRESHOLD = 2
    TRADE_SIZE = 5
    STALL_WARN_SEC = 5.0
    CANCEL_AFTER_SEC = 5.0

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.relevant = {"B"} | {f"B_{kind}_{strike}" for strike in self.STRIKES for kind in ("C", "P")}
        self.positions_ready = False
        self.local_positions = {symbol: 0 for symbol in self.relevant}
        self.active: Optional[Bundle] = None
        self.check_lock = asyncio.Lock()
        self.last_no_trade_at = 0.0

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        if not self.positions_ready:
            self.local_positions = {symbol: int(self.positions.get(symbol, 0)) for symbol in self.relevant}
            self.positions_ready = True

    def best_bid(self, symbol: str) -> Optional[int]:
        book = self.order_books.get(symbol)
        levels = book.bids.items() if book else ()
        live = [int(px) for px, qty in levels if qty > 0]
        return max(live) if live else None

    def best_ask(self, symbol: str) -> Optional[int]:
        book = self.order_books.get(symbol)
        levels = book.asks.items() if book else ()
        live = [int(px) for px, qty in levels if qty > 0]
        return min(live) if live else None

    def mid(self, bid: Optional[int], ask: Optional[int]) -> Optional[float]:
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def fmt(self, value: Optional[float]) -> str:
        return "NA" if value is None else f"{value:.1f}"

    def clear_bundle_leg(self, order_id: str) -> None:
        if self.active is None:
            return
        self.active.order_ids.discard(order_id)
        if not self.active.order_ids:
            self.active = None

    def bundle_active(self) -> bool:
        if self.active is None:
            return False
        if not self.active.order_ids:
            self.active = None
            return False
        if not self.active.warned_stalled and time.monotonic() - self.active.started_at > self.STALL_WARN_SEC:
            self.active.warned_stalled = True
            print(
                f"[STALL] strategy={self.active.strategy} strike={self.active.strike} "
                f"direction={self.active.direction} order_ids={sorted(self.active.order_ids)}"
            )
        return True

    async def cancel_stale_bundle(self) -> None:
        if self.active is None:
            return
        for order_id in [oid for oid in self.active.order_ids if oid not in self.open_orders]:
            self.clear_bundle_leg(order_id)
        if self.active is None:
            return
        if self.active.cancel_requested or time.monotonic() - self.active.started_at < self.CANCEL_AFTER_SEC:
            return
        open_ids = [oid for oid in self.active.order_ids if oid in self.open_orders]
        if not open_ids:
            return
        self.active.cancel_requested = True
        print(
            f"[CANCEL] strategy={self.active.strategy} strike={self.active.strike} "
            f"direction={self.active.direction} order_ids={sorted(open_ids)}"
        )
        for order_id in open_ids:
            await self.cancel_order(order_id)

    async def submit_bundle(
        self,
        strategy: str,
        strike: int | tuple[int, int],
        direction: str,
        legs: list[tuple[str, Side, int]],
    ) -> None:
        self.active = Bundle(strategy=strategy, strike=strike, direction=direction, started_at=time.monotonic())
        for symbol, side, px in legs:
            try:
                order_id = await self.place_order(symbol, self.TRADE_SIZE, side, px)
            except Exception as exc:
                print(f"[ERROR] submit {symbol} {side.name} @{px}: {exc}")
                break
            self.active.order_ids.add(order_id)
        print(
            f"[{strategy.upper()}] strike={strike} direction={direction} "
            f"order_ids={sorted(self.active.order_ids)}"
        )
        if not self.active.order_ids:
            self.active = None

    def evaluate_pcp(self):
        b_bid, b_ask = self.best_bid("B"), self.best_ask("B")
        if b_bid is None or b_ask is None:
            return None, [], []
        s0_mid = self.mid(b_bid, b_ask)
        best = None
        available = []
        rows = []
        for strike in self.STRIKES:
            c_sym, p_sym = f"B_C_{strike}", f"B_P_{strike}"
            c_bid, c_ask = self.best_bid(c_sym), self.best_ask(c_sym)
            p_bid, p_ask = self.best_bid(p_sym), self.best_ask(p_sym)
            c_mid, p_mid = self.mid(c_bid, c_ask), self.mid(p_bid, p_ask)
            cp_mid = None if c_mid is None or p_mid is None else c_mid - p_mid
            sk_mid = None if s0_mid is None else s0_mid - strike
            gap = None if cp_mid is None or sk_mid is None else cp_mid - sk_mid
            edge1 = None if None in (c_bid, p_ask) else (c_bid - p_ask) - (b_ask - strike)
            edge2 = None if None in (c_ask, p_bid) else (b_bid - strike) - (c_ask - p_bid)
            if edge1 is not None:
                available.append((edge1, "pcp", strike, "case1"))
            if edge2 is not None:
                available.append((edge2, "pcp", strike, "case2"))
            if edge1 is not None and edge1 > self.EDGE_THRESHOLD and (best is None or edge1 > best[0]):
                best = (
                    edge1,
                    "pcp",
                    strike,
                    "case1",
                    [(c_sym, Side.SELL, c_bid), (p_sym, Side.BUY, p_ask), ("B", Side.BUY, b_ask)],
                )
            if edge2 is not None and edge2 > self.EDGE_THRESHOLD and (best is None or edge2 > best[0]):
                best = (
                    edge2,
                    "pcp",
                    strike,
                    "case2",
                    [(c_sym, Side.BUY, c_ask), (p_sym, Side.SELL, p_bid), ("B", Side.SELL, b_bid)],
                )
            rows.append(
                f"K={strike} "
                f"C=({c_bid if c_bid is not None else 'NA'},{c_ask if c_ask is not None else 'NA'}) "
                f"P=({p_bid if p_bid is not None else 'NA'},{p_ask if p_ask is not None else 'NA'}) "
                f"S0=({b_bid},{b_ask}) "
                f"mid:(C-P)={self.fmt(cp_mid)} (S-K)={self.fmt(sk_mid)} gap={self.fmt(gap)} "
                f"case1:[({c_bid if c_bid is not None else 'NA'}-{p_ask if p_ask is not None else 'NA'})-({b_ask}-{strike})]={self.fmt(edge1)} "
                f"case2:[({b_bid}-{strike})-({c_ask if c_ask is not None else 'NA'}-{p_bid if p_bid is not None else 'NA'})]={self.fmt(edge2)}"
            )
        return best, available, rows

    def evaluate_box(self):
        best = None
        available = []
        rows = []
        for k1, k2 in self.BOX_PAIRS:
            c1_sym, c2_sym = f"B_C_{k1}", f"B_C_{k2}"
            p1_sym, p2_sym = f"B_P_{k1}", f"B_P_{k2}"
            c1_bid, c1_ask = self.best_bid(c1_sym), self.best_ask(c1_sym)
            c2_bid, c2_ask = self.best_bid(c2_sym), self.best_ask(c2_sym)
            p1_bid, p1_ask = self.best_bid(p1_sym), self.best_ask(p1_sym)
            p2_bid, p2_ask = self.best_bid(p2_sym), self.best_ask(p2_sym)
            fair = k2 - k1
            buy_cost = None if None in (c1_ask, c2_bid, p2_ask, p1_bid) else c1_ask - c2_bid + p2_ask - p1_bid
            sell_rev = None if None in (c1_bid, c2_ask, p2_bid, p1_ask) else c1_bid - c2_ask + p2_bid - p1_ask
            buy_edge = None if buy_cost is None else fair - buy_cost
            sell_edge = None if sell_rev is None else sell_rev - fair
            if buy_edge is not None:
                available.append((buy_edge, "box", (k1, k2), "buy"))
            if sell_edge is not None:
                available.append((sell_edge, "box", (k1, k2), "sell"))
            if buy_edge is not None and buy_edge > self.EDGE_THRESHOLD and (best is None or buy_edge > best[0]):
                best = (
                    buy_edge,
                    "box",
                    (k1, k2),
                    "buy",
                    [(c1_sym, Side.BUY, c1_ask), (c2_sym, Side.SELL, c2_bid), (p2_sym, Side.BUY, p2_ask), (p1_sym, Side.SELL, p1_bid)],
                )
            if sell_edge is not None and sell_edge > self.EDGE_THRESHOLD and (best is None or sell_edge > best[0]):
                best = (
                    sell_edge,
                    "box",
                    (k1, k2),
                    "sell",
                    [(c1_sym, Side.SELL, c1_bid), (c2_sym, Side.BUY, c2_ask), (p2_sym, Side.SELL, p2_bid), (p1_sym, Side.BUY, p1_ask)],
                )
            rows.append(
                f"K=({k1},{k2}) fair={fair} "
                f"C1=({c1_bid if c1_bid is not None else 'NA'},{c1_ask if c1_ask is not None else 'NA'}) "
                f"C2=({c2_bid if c2_bid is not None else 'NA'},{c2_ask if c2_ask is not None else 'NA'}) "
                f"P1=({p1_bid if p1_bid is not None else 'NA'},{p1_ask if p1_ask is not None else 'NA'}) "
                f"P2=({p2_bid if p2_bid is not None else 'NA'},{p2_ask if p2_ask is not None else 'NA'}) "
                f"buy:[{c1_ask if c1_ask is not None else 'NA'}-{c2_bid if c2_bid is not None else 'NA'}+{p2_ask if p2_ask is not None else 'NA'}-{p1_bid if p1_bid is not None else 'NA'}]={self.fmt(buy_cost)} edge={self.fmt(buy_edge)} "
                f"sell:[{c1_bid if c1_bid is not None else 'NA'}-{c2_ask if c2_ask is not None else 'NA'}+{p2_bid if p2_bid is not None else 'NA'}-{p1_ask if p1_ask is not None else 'NA'}]={self.fmt(sell_rev)} edge={self.fmt(sell_edge)}"
            )
        return best, available, rows

    async def check_all(self) -> None:
        if not self.positions_ready:
            return
        async with self.check_lock:
            if not self.positions_ready:
                return
            await self.cancel_stale_bundle()
            if self.bundle_active():
                return
            pcp_best, pcp_live, pcp_rows = self.evaluate_pcp()
            box_best, box_live, box_rows = self.evaluate_box()
            winner = max([item for item in (pcp_best, box_best) if item is not None], default=None, key=lambda item: item[0])
            if winner is not None:
                edge, strategy, strike, direction, legs = winner
                print(f"[EDGE] strategy={strategy} strike={strike} direction={direction} edge={edge:.1f}")
                await self.submit_bundle(strategy, strike, direction, legs)
                return
            if time.monotonic() - self.last_no_trade_at < 1.0:
                return
            self.last_no_trade_at = time.monotonic()
            best_live = max(pcp_live + box_live, default=None, key=lambda item: item[0])
            summary = (
                f" best_live=strategy={best_live[1]} strike={best_live[2]} direction={best_live[3]} edge={best_live[0]:.1f}"
                if best_live is not None else
                " best_live=NA"
            )
            print(f"[NO-TRADE] threshold={self.EDGE_THRESHOLD}{summary}")
            for row in pcp_rows:
                print(f"[PCP-DEBUG] {row}")
            for row in box_rows:
                print(f"[BOX-DEBUG] {row}")

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.relevant:
            await self.check_all()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.relevant:
            await self.check_all()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        order = self.open_orders.get(order_id)
        if order:
            req, remaining, _is_market = order
            delta = qty if int(req.side) == 1 else -qty
            self.local_positions[req.symbol] = self.local_positions.get(req.symbol, 0) + delta
            live = {symbol: pos for symbol, pos in self.local_positions.items() if pos}
            print(f"[FILL] order={order_id} symbol={req.symbol} qty={qty} price={price} local={live or {'flat': 0}}")
            if remaining == 0:
                self.clear_bundle_leg(order_id)
            return
        self.clear_bundle_leg(order_id)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print(f"[REJECTED] order={order_id}: {reason}")
        self.clear_bundle_leg(order_id)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        if not success:
            print(f"[CANCEL-FAIL] order={order_id}: {error}")
            if self.active is not None and order_id in self.active.order_ids:
                self.active.cancel_requested = False
            return
        self.clear_bundle_leg(order_id)

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        return

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        return

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        return

    async def start(self):
        await self.connect()


async def main():
    client = PCPBoxBot(
        os.getenv("UTC_HOST", "34.197.188.76:3333"),
        os.getenv("UTC_USERNAME", "uiuc"),
        os.getenv("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
