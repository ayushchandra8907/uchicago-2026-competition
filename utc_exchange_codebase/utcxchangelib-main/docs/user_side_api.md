# User-Side Exchange API Guide

This library exposes one main class: `XChangeClient`. You use it in two ways:

1. You call its public methods to connect and send requests to the exchange.
2. You subclass it and override `bot_handle_*` methods to react to exchange events.

The implementation lives in [utcxchangelib/xchange_client.py](/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/utc_exchange_codebase/utcxchangelib-main/utcxchangelib/xchange_client.py).

## Mental Model

The flow is:

1. Create a subclass of `XChangeClient`.
2. Override the handler methods you care about.
3. Start any background trading task you want.
4. Call `await self.connect()` to authenticate and begin the message loop.
5. Use methods like `place_order`, `cancel_order`, and `place_swap_order` to send requests.
6. Read local state from `self.positions`, `self.order_books`, and `self.open_orders`.

The base client already:

- opens the gRPC stream,
- authenticates with username and password,
- tracks open orders,
- keeps local order books,
- keeps local positions,
- converts incoming exchange messages into higher-level callbacks.

## Functions You Call Yourself

### `__init__(host, username, password, silent=False, symbols=None, swap_map=None)`

Creates the client and initializes local state.

- `host`: exchange server address, for example `127.0.0.1:3333`
- `username`, `password`: exchange credentials
- `silent`: reduces logging noise when `True`
- `symbols`: optional custom symbol universe
- `swap_map`: optional custom swap definitions

Important side effects:

- `self.positions` starts with zero inventory for each symbol plus `cash`
- `self.order_books` starts with an empty `OrderBook` per symbol
- `self.open_orders` starts empty
- `self.connected` starts as `False`

### `connect()`

Authenticates to the exchange and starts the main receive loop.

What it does:

- opens a gRPC connection,
- sends your username/password,
- continuously reads exchange messages,
- dispatches each message to the correct handler.

Important detail:

- this method does not return during normal operation; it is the main event loop.
- in practice, users usually create a task for their trading logic and then `await self.connect()`.

### `place_order(symbol, qty, side, px=None) -> str`

Submits an order to the exchange and returns the generated `order_id`.

Parameters:

- `symbol`: instrument name
- `qty`: quantity to buy or sell
- `side`: either `Side.BUY`, `Side.SELL`, or a string such as `"buy"` / `"sell"`
- `px`: limit price; if omitted, the library sends a market order

Behavior:

- if `px` is provided, the request is a limit order
- if `px` is `None`, the request is a market order
- the order is inserted into `self.open_orders` immediately after the request is written to the stream

`self.open_orders[order_id]` stores:

- the original protobuf order request,
- remaining quantity,
- whether the order was a market order

### `place_swap_order(swap, qty)`

Sends a swap request to the exchange.

Typical default swaps:

- `toETF`: consumes `A + B + C` and creates `ETF`
- `fromETF`: consumes `ETF` and creates `A + B + C`

The exact meaning depends on `self.swap_map`.

### `cancel_order(order_id)`

Requests cancellation of a previously submitted order.

Important detail:

- this only sends the cancel request.
- you learn whether the cancel actually succeeded later through `bot_handle_cancel_response(...)`.

## State You Read From Your Bot

### `self.positions`

A dictionary-like object mapping symbol to current position, plus a `cash` entry.

Examples:

- `self.positions["A"]`
- `self.positions["ETF"]`
- `self.positions["cash"]`

This state is refreshed from:

- full position snapshots,
- incremental position updates,
- incremental cash updates.

### `self.order_books`

A mapping from symbol to `OrderBook`.

Each `OrderBook` has:

- `bids`: `{price: quantity}`
- `asks`: `{price: quantity}`

This state is refreshed from:

- full book snapshots,
- incremental book updates.

### `self.open_orders`

Tracks orders that the client still considers active.

This is useful for:

- deciding what to cancel,
- checking remaining size,
- connecting fills and rejections back to your strategy state.

## Callback Hooks You Override

These are the main user-side extension points. The base implementation does nothing, so your subclass should override the ones your strategy needs.

### `bot_handle_book_update(symbol)`

Called after the local book for `symbol` has already been updated.

Use this when you want to react to:

- a full snapshot,
- an incremental size update,
- best bid/ask changes,
- book imbalance changes.

Read data from `self.order_books[symbol]`.

### `bot_handle_trade_msg(symbol, price, qty)`

Called when the exchange reports a trade print.

Use this for:

- last-trade tracking,
- tape-based signals,
- volume monitoring.

This does not mean your own order was filled; that is handled separately by `bot_handle_order_fill`.

### `bot_handle_order_fill(order_id, qty, price)`

Called when one of your tracked orders gets filled.

What is already true before this callback runs:

- the fill has been matched to `self.open_orders[order_id]`
- the order's remaining quantity in `self.open_orders` has been reduced

Use this for:

- inventory/risk logic,
- internal PnL tracking,
- follow-up hedging actions,
- logging execution quality.

Important caveat:

- this method receives only the fill details, not the symbol or side directly.
- if you need them, look them up in `self.open_orders[order_id][0]` before the order disappears.

### `bot_handle_order_rejected(order_id, reason)`

Called when the exchange rejects one of your submitted orders.

Use this for:

- debugging invalid orders,
- retry logic,
- risk throttling,
- logging rejection reasons.

After this callback, the order is removed from `self.open_orders`.

### `bot_handle_cancel_response(order_id, success, error)`

Called when the exchange responds to a cancel request.

Parameters:

- `order_id`: order you tried to cancel
- `success`: `True` if the cancel succeeded
- `error`: failure reason if the cancel failed

Use this for:

- confirming whether an order is still live,
- updating strategy state,
- retrying or replacing orders.

If cancel succeeds, the client removes the order from `self.open_orders` after calling this hook.

### `bot_handle_swap_response(swap, qty, success)`

Called when the exchange responds to a swap request.

Use this for:

- confirming ETF creation/redemption,
- handling custom swap products,
- updating strategy state tied to basket conversions.

The `swap` name is the exchange-facing swap identifier.

### `bot_handle_news(news_release)`

Called when the exchange publishes news.

The callback receives a normalized dictionary with this shape:

```python
{
    "tick": int,
    "kind": "structured" | "unstructured",
    "symbol": str | None,
    "new_data": {...},
}
```

For structured news:

- earnings:

```python
{
    "value": float,
    "asset": str,
    "structured_subtype": "earnings",
}
```

- CPI print:

```python
{
    "forecast": float,
    "actual": float,
    "structured_subtype": "cpi_print",
}
```

For unstructured news:

```python
{
    "content": str,
    "type": str,
}
```

Use this for event-driven strategies and news parsing.

### `bot_handle_market_resolved(market_id, winning_symbol, tick)`

Called when a prediction market resolves.

Use this for:

- cleaning up prediction-market logic,
- recording outcome labels,
- updating strategy state after resolution.

### `bot_handle_settlement_payout(user, market_id, amount, tick)`

Called when the exchange sends a settlement payout message.

Use this for:

- reconciling realized payouts,
- recording event-driven cash flows,
- post-settlement accounting.

## Internal Functions You Usually Do Not Override

The following methods are part of the library's internal message plumbing, not the normal user extension surface:

- `handle_trade_msg`
- `handle_order_fill`
- `handle_order_rejected`
- `handle_cancel_response`
- `handle_swap_response`
- `handle_book_snapshot`
- `handle_book_update`
- `handle_position_snapshot`
- `handle_news_message`
- `handle_authenticate_response`
- `process_message`
- `_ensure_symbol`

These methods maintain local state and then call your `bot_handle_*` hook. In normal usage, override the hook, not the internal handler.

## Minimal Usage Pattern

```python
from typing import Optional
import asyncio

from utcxchangelib import XChangeClient, Side


class MyClient(XChangeClient):
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("fill", order_id, qty, price)

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str],
    ) -> None:
        print("cancel response", order_id, success, error)

    async def trade(self):
        await asyncio.sleep(1)
        order_id = await self.place_order("A", 10, Side.BUY, 100)
        await self.cancel_order(order_id)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()
```

## Practical Notes

- `connect()` is the long-running receive loop, so call it once.
- `place_order()` only confirms that the request was sent, not that it filled.
- `cancel_order()` only confirms that the cancel request was sent, not that the order was canceled.
- Fills, rejections, cancels, news, and book changes arrive asynchronously through callbacks.
- The local client state is intended to be your strategy's read model.
