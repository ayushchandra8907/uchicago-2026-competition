import asyncio
from concurrent.futures import Future
from threading import Lock

from pynput import keyboard
from utcxchangelib import Side, XChangeClient


# Exchange connection settings.
SERVER = "uchicago.exchange:3333"
USERNAME = "uiuc"
PASSWORD = "mesa-lynx-octopus"

# Simple macro order size.
ORDER_QTY = 40

# Keep the same hotkey mapping as the old macro.
POS_BUY_HIKE = ("R_HIKE", Side.BUY)
POS_SELL_HIKE = ("R_HIKE", Side.SELL)
POS_BUY_HOLD = ("R_HOLD", Side.BUY)
POS_SELL_HOLD = ("R_HOLD", Side.SELL)
POS_BUY_CUT = ("R_CUT", Side.BUY)
POS_SELL_CUT = ("R_CUT", Side.SELL)

KEY_TO_ACTION = {
    "e": POS_BUY_HIKE,
    "o": POS_SELL_HIKE,
    "r": POS_BUY_HOLD,
    "p": POS_SELL_HOLD,
    "w": POS_BUY_CUT,
    "i": POS_SELL_CUT,
}


class MacroXChangeClient(XChangeClient):
    """Small keyboard-driven order macro for the prediction market contracts."""

    def __init__(self, host: str, username: str, password: str):
        tracked_symbols = ["R_HIKE", "R_HOLD", "R_CUT"]
        super().__init__(host, username, password, symbols=tracked_symbols)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.listener: keyboard.Listener | None = None
        self.stop_event = asyncio.Event()
        self.submit_lock = asyncio.Lock()
        self.press_lock = Lock()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: str | None = None) -> None:
        return None

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        return None

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        return None

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release) -> None:
        pass

    def _start_listener(self) -> None:
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.stop_event.set)
            return False

        char = getattr(key, "char", None)
        if char is None:
            return None

        char = char.lower()
        if char not in KEY_TO_ACTION:
            return None

        symbol, side = KEY_TO_ACTION[char]
        with self.press_lock:
            if self.loop is None:
                return None

            future: Future = asyncio.run_coroutine_threadsafe(
                self.submit_macro_order(symbol, side),
                self.loop,
            )
            future.add_done_callback(self._log_future_exception)

        return None

    @staticmethod
    def _log_future_exception(future: Future) -> None:
        future.exception()

    async def submit_macro_order(self, symbol: str, side: Side) -> None:
        async with self.submit_lock:
            await self.place_order(symbol, ORDER_QTY, side, None)

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._start_listener()

        connect_task = asyncio.create_task(self.connect())
        stop_task = asyncio.create_task(self.stop_event.wait())

        try:
            done, pending = await asyncio.wait(
                {connect_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if connect_task in done:
                await connect_task
            else:
                connect_task.cancel()
                try:
                    await connect_task
                except asyncio.CancelledError:
                    pass
        finally:
            if self.listener is not None:
                self.listener.stop()


async def main() -> None:
    client = MacroXChangeClient(SERVER, USERNAME, PASSWORD)
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
