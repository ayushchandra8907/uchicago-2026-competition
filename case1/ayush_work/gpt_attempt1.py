from typing import Optional, Dict, Tuple, Set
import asyncio
import math
import time

from utcxchangelib import XChangeClient, Side


class MyXchangeClient(XChangeClient):
    """
    Northwestern-style aggressive market maker adapted to XChange.

    Core behavior:
    - improve best bid / ask by one tick when possible
    - skew quotes by inventory
    - reduce size near position limits
    - increase size on inventory-reducing side
    - cancel and replace aggressively
    - emergency unwind when position gets too large

    This is intended as a practical benchmark bot, not a final competition bot.
    """

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

        # -----------------------------
        # Universe
        # -----------------------------
        self.PREFERRED_SYMBOLS = ["A", "C", "ETF", "B"]
        self.mm_symbols: Set[str] = set()

        # -----------------------------
        # Hyperparameters
        # -----------------------------
        self.TICK_SIZE = 1

        self.BASE_ORDER_SIZE = 2
        self.MAX_POSITION = 20
        self.POSITION_SOFT_STOP_FRAC = 0.70

        # Northwestern-style skew:
        # inv_adj = inventory * spread_improvement * inventory_skew
        # In integer-tick markets, keep this small but meaningful.
        self.SPREAD_IMPROVEMENT = 1
        self.INVENTORY_SKEW = 0.35

        # Requote cadence
        self.QUOTE_FREQUENCY = 8           # every N book updates per symbol
        self.MIN_REQUOTE_SECS = 0.15       # don't thrash too hard
        self.REFRESH_FAILSAFE_SECS = 0.40  # periodic refresh if updates slow down

        # Emergency controls
        self.EMERGENCY_UNWIND_FRAC = 1.0   # trigger when abs(pos) > MAX_POSITION
        self.UNWIND_FRACTION = 0.5         # unwind about half of excess

        # -----------------------------
        # State
        # -----------------------------
        self.best_bid: Dict[str, int] = {}
        self.best_ask: Dict[str, int] = {}

        # symbol -> {"bid": (oid, px, qty), "ask": (oid, px, qty)}
        self.active_orders: Dict[str, Dict[str, Optional[Tuple[str, int, int]]]] = {}

        self.update_counter: Dict[str, int] = {}
        self.last_quote_ts: Dict[str, float] = {}
        self.last_fill_ts: Dict[str, float] = {}

        self._book_event = asyncio.Event()

    # =========================================================
    # Helpers
    # =========================================================
    def _now(self) -> float:
        return time.time()

    def _inventory(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def _is_plain_symbol(self, symbol: str) -> bool:
        s = symbol.upper().strip()
        return s in {"A", "B", "C", "ETF"}

    def _ensure_symbol_state(self, symbol: str) -> None:
        if symbol not in self.best_bid:
            self.best_bid[symbol] = 0
        if symbol not in self.best_ask:
            self.best_ask[symbol] = 0
        if symbol not in self.active_orders:
            self.active_orders[symbol] = {"bid": None, "ask": None}
        if symbol not in self.update_counter:
            self.update_counter[symbol] = 0
        if symbol not in self.last_quote_ts:
            self.last_quote_ts[symbol] = 0.0
        if symbol not in self.last_fill_ts:
            self.last_fill_ts[symbol] = 0.0

    def _top_of_book(self, symbol: str) -> Optional[Tuple[int, int]]:
        """
        Updates best bid/ask from full book and returns (best_bid, best_ask).
        """
        book = self.order_books.get(symbol)
        if not book:
            return None

        bids = [(p, q) for p, q in book.bids.items() if q and q > 0]
        asks = [(p, q) for p, q in book.asks.items() if q and q > 0]

        if not bids or not asks:
            return None

        best_bid_px = max(bids, key=lambda x: x[0])[0]
        best_ask_px = min(asks, key=lambda x: x[0])[0]

        if best_ask_px <= best_bid_px:
            return None

        self.best_bid[symbol] = int(best_bid_px)
        self.best_ask[symbol] = int(best_ask_px)
        return self.best_bid[symbol], self.best_ask[symbol]

    def _discover_symbols(self) -> None:
        available = sorted(self.order_books.keys())

        chosen = []
        for s in self.PREFERRED_SYMBOLS:
            if s in available and self._is_plain_symbol(s):
                chosen.append(s)

        if not chosen:
            for s in available:
                if self._is_plain_symbol(s):
                    chosen.append(s)

        self.mm_symbols = set(chosen)
        for s in self.mm_symbols:
            self._ensure_symbol_state(s)

        print("Detected books:", available)
        print("Trading symbols:", sorted(self.mm_symbols))

    def _get_cancel_fn(self):
        return getattr(self, "cancel_order", None) or getattr(self, "cancel", None)

    async def _cancel_order_safe(self, order_id: str) -> None:
        cancel_fn = self._get_cancel_fn()
        if cancel_fn is None:
            return
        try:
            await cancel_fn(order_id)
        except Exception:
            pass

    async def _place_limit(self, symbol: str, side: Side, qty: int, px: int, ioc: bool = False) -> Optional[str]:
        """
        Adapts to place_order() signature available in utcxchangelib.
        If IOC isn't supported in your library version, it will ignore it.
        """
        try:
            try:
                maybe_id = await self.place_order(symbol, qty, side, px, ioc=ioc)
            except TypeError:
                maybe_id = await self.place_order(symbol, qty, side, px)

            if isinstance(maybe_id, str):
                return maybe_id
        except Exception:
            return None

        # fallback infer from open orders
        await asyncio.sleep(0.03)
        for oid, order in self.open_orders.items():
            try:
                if order.symbol == symbol and order.side == side and order.price == px:
                    return oid
            except Exception:
                pass
        return None

    async def _cancel_side(self, symbol: str, side_name: str) -> None:
        current = self.active_orders[symbol][side_name]
        if current is None:
            return
        oid, _, _ = current
        await self._cancel_order_safe(oid)
        self.active_orders[symbol][side_name] = None

    async def _cancel_all(self, symbol: str) -> None:
        await self._cancel_side(symbol, "bid")
        await self._cancel_side(symbol, "ask")

    def _calculate_size(self, symbol: str, side: Side) -> int:
        inv = self._inventory(symbol)

        soft_limit = self.MAX_POSITION * self.POSITION_SOFT_STOP_FRAC

        if side == Side.BUY and inv >= soft_limit:
            return 0
        if side == Side.SELL and inv <= -soft_limit:
            return 0

        factor = max(0.2, 1.0 - abs(inv) / float(self.MAX_POSITION))

        # increase size on inventory-reducing side
        if (side == Side.SELL and inv > 0) or (side == Side.BUY and inv < 0):
            factor *= 1.5

        qty = max(1, int(round(self.BASE_ORDER_SIZE * factor)))
        return qty

    async def _emergency_unwind(self, symbol: str) -> None:
        await self._cancel_all(symbol)

        inv = self._inventory(symbol)
        best_b = self.best_bid.get(symbol, 0)
        best_a = self.best_ask.get(symbol, 0)

        if best_b == 0 or best_a == 0 or best_a <= best_b:
            return

        if inv > self.MAX_POSITION * self.POSITION_SOFT_STOP_FRAC:
            # sell inventory down aggressively near bid
            excess = inv - int(self.MAX_POSITION * 0.5)
            size = max(1, min(int(abs(inv) * self.UNWIND_FRACTION), excess if excess > 0 else 1))
            px = max(1, best_b - self.TICK_SIZE)
            await self._place_limit(symbol, Side.BUY, size, px, ioc=True)

        elif inv < -self.MAX_POSITION * self.POSITION_SOFT_STOP_FRAC:
            # buy inventory back aggressively near ask
            excess = abs(inv) - int(self.MAX_POSITION * 0.5)
            size = max(1, min(int(abs(inv) * self.UNWIND_FRACTION), excess if excess > 0 else 1))
            px = best_a + self.TICK_SIZE
            await self._place_limit(symbol, Side.SELL, size, px, ioc=True)

    async def _aggressive_quote(self, symbol: str) -> None:
        self._ensure_symbol_state(symbol)

        tob = self._top_of_book(symbol)
        if tob is None:
            return

        best_b, best_a = tob
        if best_b == 0 or best_a == 0 or best_a <= best_b:
            return

        now = self._now()
        if now - self.last_quote_ts[symbol] < self.MIN_REQUOTE_SECS:
            return

        inv = self._inventory(symbol)

        # Northwestern logic adapted to integer ticks
        # inv_adj = inventory * spread_improvement * inventory_skew
        inv_adj = inv * self.SPREAD_IMPROVEMENT * self.INVENTORY_SKEW

        our_bid = int(round(best_b + self.TICK_SIZE - inv_adj))
        our_ask = int(round(best_a - self.TICK_SIZE - inv_adj))

        # maintain valid spread
        if our_ask <= our_bid:
            mid = (best_b + best_a) / 2.0
            our_bid = max(1, int(math.floor(mid - self.TICK_SIZE)))
            our_ask = int(math.ceil(mid + self.TICK_SIZE))

        # keep bid < ask no matter what
        if our_ask <= our_bid:
            return

        bid_size = self._calculate_size(symbol, Side.BUY)
        ask_size = self._calculate_size(symbol, Side.SELL)

        # update aggressive bid
        if bid_size > 0:
            current = self.active_orders[symbol]["bid"]
            if current is None or current[1] != our_bid or current[2] != bid_size:
                if current is not None:
                    await self._cancel_side(symbol, "bid")
                oid = await self._place_limit(symbol, Side.SELL, bid_size, our_bid)
                if oid is not None:
                    self.active_orders[symbol]["bid"] = (oid, our_bid, bid_size)

        else:
            await self._cancel_side(symbol, "bid")

        # update aggressive ask
        if ask_size > 0:
            current = self.active_orders[symbol]["ask"]
            if current is None or current[1] != our_ask or current[2] != ask_size:
                if current is not None:
                    await self._cancel_side(symbol, "ask")
                oid = await self._place_limit(symbol, Side.BUY, ask_size, our_ask)
                if oid is not None:
                    self.active_orders[symbol]["ask"] = (oid, our_ask, ask_size)

        else:
            await self._cancel_side(symbol, "ask")

        self.last_quote_ts[symbol] = now

    # =========================================================
    # Exchange callbacks
    # =========================================================
    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        for symbol in self.active_orders:
            for side_name in ("bid", "ask"):
                current = self.active_orders[symbol][side_name]
                if current is not None and current[0] == order_id:
                    self.active_orders[symbol][side_name] = None

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        """
        Clear filled order state and re-quote quickly.
        """
        for symbol in self.active_orders:
            matched = False
            for side_name in ("bid", "ask"):
                current = self.active_orders[symbol][side_name]
                if current is not None and current[0] == order_id:
                    self.active_orders[symbol][side_name] = None
                    matched = True

            if matched:
                self.last_fill_ts[symbol] = self._now()

                if abs(self._inventory(symbol)) > self.MAX_POSITION * self.EMERGENCY_UNWIND_FRAC:
                    await self._emergency_unwind(symbol)
                else:
                    self._book_event.set()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        for symbol in self.active_orders:
            for side_name in ("bid", "ask"):
                current = self.active_orders[symbol][side_name]
                if current is not None and current[0] == order_id:
                    self.active_orders[symbol][side_name] = None

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # No extra trade-msg logic for benchmark.
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol not in self.mm_symbols and self.mm_symbols:
            return

        self._ensure_symbol_state(symbol)

        # refresh cached top of book
        self._top_of_book(symbol)

        self.update_counter[symbol] += 1

        # Northwestern-style quote cadence: every N updates
        if self.update_counter[symbol] % self.QUOTE_FREQUENCY == 0:
            self._book_event.set()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # Ignore swaps for this benchmark
        pass

    # =========================================================
    # Main trading loop
    # =========================================================
    async def trade(self):
        await asyncio.sleep(5.0)
        self._discover_symbols()

        if not self.mm_symbols:
            print("No suitable symbols found for trading.")
            return

        while True:
            try:
                await asyncio.wait_for(self._book_event.wait(), timeout=self.REFRESH_FAILSAFE_SECS)
            except asyncio.TimeoutError:
                pass
            self._book_event.clear()

            for symbol in sorted(self.mm_symbols):
                try:
                    # always refresh top of book before quote
                    self._top_of_book(symbol)

                    # if inventory is too large, prioritize unwind
                    if abs(self._inventory(symbol)) > self.MAX_POSITION * self.EMERGENCY_UNWIND_FRAC:
                        await self._emergency_unwind(symbol)
                    else:
                        await self._aggressive_quote(symbol)

                except Exception as e:
                    print(f"quote error for {symbol}: {e}")

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


""" CONNECTING TO EXCHANGE """

async def main():
    SERVER = "34.197.188.76:3333"
    my_client = MyXchangeClient(SERVER, "uiuc", "mesa-lynx-octopus")
    await my_client.start()


if __name__ == "__main__":
    asyncio.run(main())