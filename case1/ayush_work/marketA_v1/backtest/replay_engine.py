from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..config import AppConfig
from ..execution import RestingOrder
from ..market_maker import StrategyEngine
from ..models import BacktestResult, BookState, FillRecord, MarketEvent, ModeName, OrderIntent, SessionData, Side


@dataclass(frozen=True)
class ReplayArtifacts:
    result: BacktestResult
    equity_curve: tuple[tuple[float, float], ...]


def run_session_backtest(session: SessionData, config: AppConfig) -> ReplayArtifacts:
    engine = StrategyEngine(config)
    resting_orders: dict[Side, RestingOrder] = {}
    fills: list[FillRecord] = []
    pnl_by_mode: defaultdict[str, float] = defaultdict(float)
    inventory_path: list[tuple[float, int]] = []
    equity_curve: list[tuple[float, float]] = []

    current_book: BookState | None = None
    last_trade_px: int | None = None

    for event in session.events:
        passive_fills = _simulate_passive_fills(event, resting_orders, current_book)
        for fill in passive_fills:
            engine.on_fill(
                side=fill.side,
                qty=fill.qty,
                price_px=fill.px,
                aggressive=False,
                mode=fill.mode,
                time_ms=fill.time_ms,
            )
            fills.append(fill)
            pnl_by_mode[fill.mode] += fill.edge_px

        if event.trade is not None:
            last_trade_px = event.trade.price_px
        if event.book is not None:
            current_book = event.book

        intent = engine.on_event(event)
        if intent is not None:
            decision = engine.quote_synchronizer.sync(resting_orders, intent, event.time_ms)
            for cancel in decision.cancels:
                resting_orders.pop(cancel.side, None)
            for action in decision.aggressive_actions:
                fill = _execute_aggressive_action(action, current_book, event.time_ms, intent.mode, engine.state.fair_px)
                if fill is None:
                    continue
                engine.on_fill(
                    side=fill.side,
                    qty=fill.qty,
                    price_px=fill.px,
                    aggressive=True,
                    mode=fill.mode,
                    time_ms=fill.time_ms,
                )
                fills.append(fill)
                pnl_by_mode[fill.mode] += fill.edge_px
            for placement in decision.placements:
                order = _create_resting_order(placement, current_book, event.time_ms, intent.mode)
                resting_orders[placement.side] = order

        mark_px = _mark_price(current_book, last_trade_px, engine.state.fair_px)
        equity = engine.state.cash + (engine.state.inventory * mark_px)
        if (
            not inventory_path
            or inventory_path[-1][1] != engine.state.inventory
            or event.time_ms - inventory_path[-1][0] >= 1_000.0
        ):
            inventory_path.append((event.time_ms, engine.state.inventory))
        equity_curve.append((event.time_ms, equity))

    final_mark_px = _mark_price(current_book, last_trade_px, engine.state.fair_px)
    total_pnl = engine.state.cash + (engine.state.inventory * final_mark_px)
    result = BacktestResult(
        session_id=session.session_id,
        final_inventory=engine.state.inventory,
        final_cash=engine.state.cash,
        mark_px=final_mark_px,
        total_pnl=total_pnl,
        max_drawdown=_max_drawdown(equity_curve),
        passive_fill_count=sum(1 for fill in fills if not fill.aggressive),
        aggressive_fill_count=sum(1 for fill in fills if fill.aggressive),
        pnl_by_mode=dict(pnl_by_mode),
        inventory_path=tuple(inventory_path),
        fills=tuple(fills),
    )
    return ReplayArtifacts(result=result, equity_curve=tuple(equity_curve))


def _create_resting_order(intent: OrderIntent, book: BookState | None, now_ms: float, mode: ModeName) -> RestingOrder:
    queue_ahead_qty = 0
    if book is not None:
        visible_qty = book.level_qty(intent.side, intent.px)
        if visible_qty is not None:
            queue_ahead_qty = int(visible_qty)
        elif intent.side == "BUY" and book.best_bid_px is not None and intent.px <= book.best_bid_px:
            queue_ahead_qty = int(book.best_bid_qty or 0)
        elif intent.side == "SELL" and book.best_ask_px is not None and intent.px >= book.best_ask_px:
            queue_ahead_qty = int(book.best_ask_qty or 0)
    return RestingOrder(
        side=intent.side,
        px=intent.px,
        qty=intent.qty,
        placed_time_ms=now_ms,
        queue_ahead_qty=queue_ahead_qty,
        aggressive=False,
        mode=mode,
    )


def _simulate_passive_fills(
    event: MarketEvent,
    resting_orders: dict[Side, RestingOrder],
    previous_book: BookState | None,
) -> list[FillRecord]:
    fills: list[FillRecord] = []
    if not resting_orders:
        return fills

    if event.trade is not None:
        fills.extend(_trade_based_fills(event.time_ms, event.trade.price_px, event.trade.qty, previous_book, resting_orders))
    if event.book is not None and previous_book is not None:
        fills.extend(_book_move_fills(event.time_ms, previous_book, event.book, resting_orders))

    for fill in fills:
        order = resting_orders.get(fill.side)
        if order is not None and order.remaining_qty == 0:
            resting_orders.pop(fill.side, None)
    return fills


def _trade_based_fills(
    time_ms: float,
    trade_price_px: int,
    trade_qty: int,
    book: BookState | None,
    resting_orders: dict[Side, RestingOrder],
) -> list[FillRecord]:
    if book is None:
        return []
    fills: list[FillRecord] = []
    aggressor = "UNKNOWN"
    if book.best_ask_px is not None and trade_price_px >= book.best_ask_px:
        aggressor = "BUY"
    elif book.best_bid_px is not None and trade_price_px <= book.best_bid_px:
        aggressor = "SELL"
    elif book.mid_px is not None:
        if trade_price_px > book.mid_px:
            aggressor = "BUY"
        elif trade_price_px < book.mid_px:
            aggressor = "SELL"

    if aggressor == "SELL":
        order = resting_orders.get("BUY")
        if order is not None and order.px >= trade_price_px:
            fill_qty = _consume_queue(order, trade_qty)
            if fill_qty > 0:
                fills.append(_make_fill_record(order, time_ms, fill_qty, _mark_price(book, trade_price_px, None)))
    elif aggressor == "BUY":
        order = resting_orders.get("SELL")
        if order is not None and order.px <= trade_price_px:
            fill_qty = _consume_queue(order, trade_qty)
            if fill_qty > 0:
                fills.append(_make_fill_record(order, time_ms, fill_qty, _mark_price(book, trade_price_px, None)))
    return fills


def _book_move_fills(
    time_ms: float,
    previous_book: BookState,
    current_book: BookState,
    resting_orders: dict[Side, RestingOrder],
) -> list[FillRecord]:
    fills: list[FillRecord] = []
    for side in ("BUY", "SELL"):
        order = resting_orders.get(side)
        if order is None:
            continue
        previous_qty = previous_book.level_qty(side, order.px)
        if previous_qty is None:
            previous_qty = order.queue_ahead_qty + int(order.remaining_qty or 0)
        current_qty = current_book.level_qty(side, order.px)
        removed = 0
        if current_qty is None:
            crossed = False
            if side == "BUY":
                crossed = current_book.best_bid_px is None or current_book.best_bid_px < order.px
            else:
                crossed = current_book.best_ask_px is None or current_book.best_ask_px > order.px
            if crossed:
                removed = previous_qty
        elif previous_qty > current_qty:
            removed = previous_qty - current_qty
        if removed <= 0:
            continue
        fill_qty = _consume_queue(order, removed)
        if fill_qty > 0:
            fills.append(_make_fill_record(order, time_ms, fill_qty, _mark_price(current_book, None, None)))
    return fills


def _consume_queue(order: RestingOrder, removed_qty: int) -> int:
    remaining = int(removed_qty)
    if order.queue_ahead_qty > 0:
        queue_consumed = min(order.queue_ahead_qty, remaining)
        order.queue_ahead_qty -= queue_consumed
        remaining -= queue_consumed
    if remaining <= 0:
        return 0
    fill_qty = min(int(order.remaining_qty or 0), remaining)
    order.remaining_qty = int(order.remaining_qty or 0) - fill_qty
    return fill_qty


def _execute_aggressive_action(
    action: OrderIntent,
    book: BookState | None,
    time_ms: float,
    mode: ModeName,
    fair_px: int | None,
) -> FillRecord | None:
    if book is None:
        return None
    if action.side == "BUY":
        if book.best_ask_px is None or action.px < book.best_ask_px:
            return None
        fill_px = int(book.best_ask_px)
        fill_qty = min(action.qty, int(book.best_ask_qty or action.qty))
    else:
        if book.best_bid_px is None or action.px > book.best_bid_px:
            return None
        fill_px = int(book.best_bid_px)
        fill_qty = min(action.qty, int(book.best_bid_qty or action.qty))
    if fill_qty <= 0:
        return None
    return FillRecord(
        time_ms=time_ms,
        side=action.side,
        px=fill_px,
        qty=fill_qty,
        aggressive=True,
        mode=mode,
        edge_px=_edge_px(action.side, fill_px, _mark_price(book, None, fair_px), fill_qty),
    )


def _make_fill_record(order: RestingOrder, time_ms: float, qty: int, mark_px: float) -> FillRecord:
    return FillRecord(
        time_ms=time_ms,
        side=order.side,
        px=order.px,
        qty=qty,
        aggressive=False,
        mode=order.mode,  # type: ignore[arg-type]
        edge_px=_edge_px(order.side, order.px, mark_px, qty),
    )


def _edge_px(side: Side, px: int, mark_px: float, qty: int) -> float:
    if side == "BUY":
        return (mark_px - px) * qty
    return (px - mark_px) * qty


def _mark_price(book: BookState | None, last_trade_px: int | None, fair_px: int | None) -> float:
    if book is not None and book.mid_px is not None:
        return float(book.mid_px)
    if last_trade_px is not None:
        return float(last_trade_px)
    if fair_px is not None:
        return float(fair_px)
    return 0.0


def _max_drawdown(equity_curve: list[tuple[float, float]]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    max_drawdown = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown
