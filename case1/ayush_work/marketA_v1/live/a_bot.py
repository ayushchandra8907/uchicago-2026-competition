from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Optional

from utcxchangelib import Side, XChangeClient

from case1.ayush_work.marketA_v1.config import AppConfig, load_app_config
from case1.ayush_work.marketA_v1.execution import RestingOrder
from case1.ayush_work.marketA_v1.market_maker import StrategyEngine
from case1.ayush_work.marketA_v1.models import BookState, MarketEvent, NewsState, OrderIntent, QuoteIntent, TradeState


LOGGER = logging.getLogger("a-only-live-bot")
LOGGER.setLevel(logging.INFO)


@dataclass
class LiveOrderState:
    order_id: str
    side: str
    px: int
    qty: int
    placed_time_ms: float
    aggressive: bool
    mode: str
    remaining_qty: int
    cancel_pending: bool = False


class ABot(XChangeClient):
    def __init__(self, host: str, username: str, password: str, *, config: AppConfig) -> None:
        super().__init__(host, username, password, silent=True, symbols=["A"])
        self.config = config
        self.engine = StrategyEngine(config)
        self.event_seq = 0
        self.last_local_fill_time_ms = -1e9
        self.orders_by_id: dict[str, LiveOrderState] = {}
        self.resting_by_side: dict[str, LiveOrderState] = {}
        self.desired_by_side: dict[str, OrderIntent | None] = {"BUY": None, "SELL": None}

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        order = self.orders_by_id.get(order_id)
        if order is None:
            return
        if success:
            self.orders_by_id.pop(order_id, None)
            if self.resting_by_side.get(order.side) is order:
                self.resting_by_side.pop(order.side, None)
            desired = self.desired_by_side.get(order.side)
            if desired is not None:
                await self._place_passive_order(desired, self.engine.state.mode, self._now_ms())
        else:
            order.cancel_pending = False
            LOGGER.warning("Cancel failed for %s: %s", order_id, error)

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        order = self.orders_by_id.get(order_id)
        if order is None:
            return
        now_ms = self._now_ms()
        order.remaining_qty = max(0, order.remaining_qty - qty)
        self.last_local_fill_time_ms = now_ms
        self.engine.on_fill(
            side=order.side,  # type: ignore[arg-type]
            qty=qty,
            price_px=price,
            aggressive=order.aggressive,
            mode=order.mode,  # type: ignore[arg-type]
            time_ms=now_ms,
        )
        if order.remaining_qty == 0:
            self.orders_by_id.pop(order_id, None)
            if self.resting_by_side.get(order.side) is order:
                self.resting_by_side.pop(order.side, None)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        order = self.orders_by_id.pop(order_id, None)
        if order is not None and self.resting_by_side.get(order.side) is order:
            self.resting_by_side.pop(order.side, None)
        LOGGER.warning("Order rejected %s: %s", order_id, reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol != "A":
            return
        await self._handle_strategy_event(
            MarketEvent(
                kind="trade",
                session_id="live",
                seq=self._next_seq(),
                time_ms=self._now_ms(),
                trade=TradeState(price_px=price, qty=qty),
            )
        )

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol != "A":
            return
        book = self._book_snapshot("A")
        if book is None:
            return
        await self._handle_strategy_event(
            MarketEvent(
                kind="book",
                session_id="live",
                seq=self._next_seq(),
                time_ms=self._now_ms(),
                book=book,
            )
        )

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return None

    async def bot_handle_news(self, news_release: dict):
        news = NewsState(
            kind=str(news_release.get("kind") or "unstructured"),
            symbol=news_release.get("symbol"),
            structured_subtype=(news_release.get("new_data") or {}).get("structured_subtype"),
            earnings_asset=(news_release.get("new_data") or {}).get("asset"),
            earnings_value=(news_release.get("new_data") or {}).get("value"),
            content=(news_release.get("new_data") or {}).get("content"),
            raw=dict(news_release),
        )
        await self._handle_strategy_event(
            MarketEvent(
                kind="news",
                session_id="live",
                seq=self._next_seq(),
                time_ms=self._now_ms(),
                news=news,
            )
        )

    async def _handle_strategy_event(self, event: MarketEvent) -> None:
        now_ms = float(event.time_ms)
        self._sync_exchange_positions_if_safe(now_ms)
        await self._cancel_stale_aggressive_orders(now_ms)
        intent = self.engine.on_event(event)
        if intent is None:
            return
        await self._reconcile_intent(intent, now_ms)

    async def _reconcile_intent(self, intent: QuoteIntent, now_ms: float) -> None:
        self.desired_by_side["BUY"] = intent.bid
        self.desired_by_side["SELL"] = intent.ask
        view = {
            side: RestingOrder(
                side=state.side,  # type: ignore[arg-type]
                px=state.px,
                qty=state.qty,
                remaining_qty=state.remaining_qty,
                placed_time_ms=state.placed_time_ms,
            )
            for side, state in self.resting_by_side.items()
            if not state.cancel_pending
        }
        decision = self.engine.quote_synchronizer.sync(view, intent, now_ms)

        for cancel in decision.cancels:
            live_order = self.resting_by_side.get(cancel.side)
            if live_order is None or live_order.cancel_pending:
                continue
            live_order.cancel_pending = True
            await self.cancel_order(live_order.order_id)

        for placement in decision.placements:
            existing = self.resting_by_side.get(placement.side)
            if existing is None:
                await self._place_passive_order(placement, intent.mode, now_ms)

        for action in decision.aggressive_actions:
            await self._place_aggressive_order(action, intent.mode, now_ms)

    async def _place_passive_order(self, intent: OrderIntent, mode: str, now_ms: float) -> None:
        order_id = await self.place_order("A", intent.qty, _to_side_enum(intent.side), intent.px)
        state = LiveOrderState(
            order_id=order_id,
            side=intent.side,
            px=intent.px,
            qty=intent.qty,
            remaining_qty=intent.qty,
            placed_time_ms=now_ms,
            aggressive=False,
            mode=mode,
        )
        self.orders_by_id[order_id] = state
        self.resting_by_side[intent.side] = state

    async def _place_aggressive_order(self, intent: OrderIntent, mode: str, now_ms: float) -> None:
        order_id = await self.place_order("A", intent.qty, _to_side_enum(intent.side), intent.px)
        self.orders_by_id[order_id] = LiveOrderState(
            order_id=order_id,
            side=intent.side,
            px=intent.px,
            qty=intent.qty,
            remaining_qty=intent.qty,
            placed_time_ms=now_ms,
            aggressive=True,
            mode=mode,
        )

    async def _cancel_stale_aggressive_orders(self, now_ms: float) -> None:
        stale_after_ms = max(500, self.config.strategy.requote_cooldown_ms * 2)
        for order in list(self.orders_by_id.values()):
            if not order.aggressive or order.cancel_pending:
                continue
            if now_ms - order.placed_time_ms < stale_after_ms:
                continue
            order.cancel_pending = True
            await self.cancel_order(order.order_id)

    def _sync_exchange_positions_if_safe(self, now_ms: float) -> None:
        if now_ms - self.last_local_fill_time_ms <= 1_000.0:
            return
        exchange_inventory = int(self.positions.get("A", self.engine.state.inventory))
        exchange_cash = float(self.positions.get("cash", self.engine.state.cash))
        if exchange_inventory != self.engine.state.inventory or exchange_cash != self.engine.state.cash:
            self.engine.sync_inventory(exchange_inventory, exchange_cash)

    def _book_snapshot(self, symbol: str) -> BookState | None:
        book = self.order_books.get(symbol)
        if book is None:
            return None
        bid_levels = tuple(sorted(((int(px), int(qty)) for px, qty in book.bids.items() if qty > 0), reverse=True)[:10])
        ask_levels = tuple(sorted(((int(px), int(qty)) for px, qty in book.asks.items() if qty > 0))[:10])
        best_bid_px, best_bid_qty = bid_levels[0] if bid_levels else (None, None)
        best_ask_px, best_ask_qty = ask_levels[0] if ask_levels else (None, None)
        return BookState(
            best_bid_px=best_bid_px,
            best_bid_qty=best_bid_qty,
            best_ask_px=best_ask_px,
            best_ask_qty=best_ask_qty,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )

    def _next_seq(self) -> int:
        self.event_seq += 1
        return self.event_seq

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic_ns() / 1_000_000.0

    async def start(self) -> None:
        await self.connect()


def _to_side_enum(side: str) -> Side:
    return Side.BUY if side == "BUY" else Side.SELL


def load_exchange_credentials(config: AppConfig) -> tuple[str, str, str]:
    local_config = config.paths.project_root / "local_config.json"
    payload = json.loads(local_config.read_text(encoding="utf-8"))
    return payload["host"], payload["username"], payload["password"]


async def run_live_bot(config_path: str | None = None) -> None:
    config = load_app_config(config_path)
    host, username, password = load_exchange_credentials(config)
    bot = ABot(host, username, password, config=config)
    await bot.start()


def main() -> None:
    asyncio.run(run_live_bot())


if __name__ == "__main__":
    main()
