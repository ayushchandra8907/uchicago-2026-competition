from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

from utcxchangelib import Side, XChangeClient


@dataclass
class Bundle:
    strike: int
    direction: str
    order_ids: set[str]
    started_at: float
    warned_stalled: bool = False


class PCPBot(XChangeClient):
    STRIKES = (950, 1000, 1050)
    EDGE_THRESHOLD = 2
    TRADE_SIZE = 5
    STALL_WARN_SEC = 5.0

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
        bids = book.bids if book else {}
        live = [int(px) for px, qty in bids.items() if qty > 0]
        return max(live) if live else None

    def best_ask(self, symbol: str) -> Optional[int]:
        book = self.order_books.get(symbol)
        asks = book.asks if book else {}
        live = [int(px) for px, qty in asks.items() if qty > 0]
        return min(live) if live else None

    def mid(self, bid: Optional[int], ask: Optional[int]) -> Optional[float]:
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def fmt(self, value: Optional[float]) -> str:
        return "NA" if value is None else f"{value:.1f}"

    def relevant_symbols(self) -> set[str]:
        return self.relevant

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
            print(f"[STALL] strike={self.active.strike} direction={self.active.direction} order_ids={sorted(self.active.order_ids)}")
        return True

    async def submit_bundle(self, strike: int, direction: str, legs: list[tuple[str, Side, int]]) -> None:
        self.active = Bundle(strike=strike, direction=direction, order_ids=set(), started_at=time.monotonic())
        for symbol, side, px in legs:
            try:
                order_id = await self.place_order(symbol, self.TRADE_SIZE, side, px)
            except Exception as exc:
                print(f"[ERROR] submit {symbol} {side.name} @{px}: {exc}")
                break
            self.active.order_ids.add(order_id)
        print(f"[PCP] strike={strike} direction={direction} order_ids={sorted(self.active.order_ids)}")
        if not self.active.order_ids:
            self.active = None

    async def check_pcp(self) -> None:
        if not self.positions_ready or self.bundle_active():
            return
        async with self.check_lock:
            if not self.positions_ready or self.bundle_active():
                return
            b_bid, b_ask = self.best_bid("B"), self.best_ask("B")
            if b_bid is None or b_ask is None:
                return
            s0_mid = self.mid(b_bid, b_ask)
            best = None
            available = []
            debug_rows = []
            for strike in self.STRIKES:
                c, p = f"B_C_{strike}", f"B_P_{strike}"
                call_bid, call_ask = self.best_bid(c), self.best_ask(c)
                put_bid, put_ask = self.best_bid(p), self.best_ask(p)
                call_mid = self.mid(call_bid, call_ask)
                put_mid = self.mid(put_bid, put_ask)
                cp_mid = None if call_mid is None or put_mid is None else call_mid - put_mid
                sk_mid = None if s0_mid is None else s0_mid - strike
                parity_mid = None if cp_mid is None or sk_mid is None else cp_mid - sk_mid
                edge1 = None if None in (call_bid, put_ask) else (call_bid - put_ask) - (b_ask - strike)
                if edge1 is not None:
                    available.append((edge1, strike, "case1"))
                if edge1 is not None and edge1 > self.EDGE_THRESHOLD and (best is None or edge1 > best[0]):
                    best = (edge1, strike, "case1", [(c, Side.SELL, call_bid), (p, Side.BUY, put_ask), ("B", Side.BUY, b_ask)])
                edge2 = None if None in (call_ask, put_bid) else (b_bid - strike) - (call_ask - put_bid)
                if edge2 is not None:
                    available.append((edge2, strike, "case2"))
                if edge2 is not None and edge2 > self.EDGE_THRESHOLD and (best is None or edge2 > best[0]):
                    best = (edge2, strike, "case2", [(c, Side.BUY, call_ask), (p, Side.SELL, put_bid), ("B", Side.SELL, b_bid)])
                debug_rows.append(
                    f"K={strike} "
                    f"C=({call_bid if call_bid is not None else 'NA'},{call_ask if call_ask is not None else 'NA'}) "
                    f"P=({put_bid if put_bid is not None else 'NA'},{put_ask if put_ask is not None else 'NA'}) "
                    f"S0=({b_bid},{b_ask}) "
                    f"mid:(C-P)={self.fmt(cp_mid)} (S-K)={self.fmt(sk_mid)} gap={self.fmt(parity_mid)} "
                    f"case1:[({call_bid if call_bid is not None else 'NA'}-{put_ask if put_ask is not None else 'NA'})-({b_ask}-{strike})]={self.fmt(edge1)} "
                    f"case2:[({b_bid}-{strike})-({call_ask if call_ask is not None else 'NA'}-{put_bid if put_bid is not None else 'NA'})]={self.fmt(edge2)}"
                )
            if best is not None:
                edge, strike, direction, legs = best
                print(f"[EDGE] strike={strike} direction={direction} edge={edge}")
                await self.submit_bundle(strike, direction, legs)
            elif debug_rows and time.monotonic() - self.last_no_trade_at >= 1.0:
                self.last_no_trade_at = time.monotonic()
                best_live = max(available, default=None, key=lambda item: item[0])
                summary = (
                    f" best_live=strike={best_live[1]} direction={best_live[2]} edge={best_live[0]:.1f}"
                    if best_live is not None else
                    " best_live=NA"
                )
                print(f"[NO-TRADE] threshold={self.EDGE_THRESHOLD}{summary}")
                for row in debug_rows:
                    print(f"[PCP-DEBUG] {row}")

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.relevant_symbols():
            await self.check_pcp()

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

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print(f"[REJECTED] order={order_id}: {reason}")
        self.clear_bundle_leg(order_id)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        if not success:
            print(f"[CANCEL-FAIL] order={order_id}: {error}")
            return
        self.clear_bundle_leg(order_id)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        return

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
    client = PCPBot(
        os.getenv("UTC_HOST", "34.197.188.76:3333"),
        os.getenv("UTC_USERNAME", "uiuc"),
        os.getenv("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
