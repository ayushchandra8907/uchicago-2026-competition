from __future__ import annotations

import unittest

from case1.ayush_work.marketA_v1.config import build_app_config
from case1.ayush_work.marketA_v1.live.a_bot import ABot


class RecordingBot(ABot):
    def __init__(self):
        super().__init__("127.0.0.1:0", "user", "pass", config=build_app_config())
        self.placed_orders: list[tuple[str, int, str, int]] = []
        self.cancelled_orders: list[str] = []
        self._order_counter = 0

    async def place_order(self, symbol: str, qty: int, side, px: int = None) -> str:  # type: ignore[override]
        self._order_counter += 1
        side_name = "BUY" if int(side.value) == 1 else "SELL"
        self.placed_orders.append((symbol, qty, side_name, px))
        return f"oid-{self._order_counter}"

    async def cancel_order(self, order_id: str) -> None:  # type: ignore[override]
        self.cancelled_orders.append(order_id)


class LiveBotTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_ignores_non_a_book_updates_and_only_places_a_orders(self) -> None:
        bot = RecordingBot()
        bot.order_books["A"].bids = {99: 20}
        bot.order_books["A"].asks = {101: 20}

        await bot.bot_handle_book_update("B")
        self.assertEqual(bot.placed_orders, [])

        await bot.bot_handle_book_update("A")

        self.assertTrue(bot.placed_orders)
        self.assertTrue(all(symbol == "A" for symbol, *_ in bot.placed_orders))


if __name__ == "__main__":
    unittest.main()
