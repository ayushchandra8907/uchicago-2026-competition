from __future__ import annotations

from collections import deque
import math

from .models import BookState, FeatureSnapshot, Side, TradeAggressor, TradeState


def compute_microprice(book: BookState | None) -> float | None:
    if book is None:
        return None
    if None in (book.best_bid_px, book.best_bid_qty, book.best_ask_px, book.best_ask_qty):
        return None
    denom = int(book.best_bid_qty) + int(book.best_ask_qty)
    if denom <= 0:
        return None
    return ((int(book.best_ask_px) * int(book.best_bid_qty)) + (int(book.best_bid_px) * int(book.best_ask_qty))) / denom


def compute_imbalance(book: BookState | None) -> float | None:
    if book is None or book.best_bid_qty is None or book.best_ask_qty is None:
        return None
    denom = book.best_bid_qty + book.best_ask_qty
    if denom <= 0:
        return None
    return (book.best_bid_qty - book.best_ask_qty) / denom


def infer_trade_aggressor(trade: TradeState, book: BookState | None) -> TradeAggressor:
    if book is None:
        return trade.aggressor
    if book.best_ask_px is not None and trade.price_px >= book.best_ask_px:
        return "BUY"
    if book.best_bid_px is not None and trade.price_px <= book.best_bid_px:
        return "SELL"
    mid_px = book.mid_px
    if mid_px is None:
        return trade.aggressor
    if trade.price_px > mid_px:
        return "BUY"
    if trade.price_px < mid_px:
        return "SELL"
    return trade.aggressor


class FeatureEngine:
    def __init__(self) -> None:
        self.current_book: BookState | None = None
        self.last_trade_px: int | None = None
        self._mid_history: deque[tuple[float, float]] = deque()
        self._trade_history: deque[tuple[float, int, int]] = deque()
        self._fill_history: deque[tuple[float, int]] = deque()

    def on_book(self, book: BookState, time_ms: float) -> None:
        self.current_book = book
        if book.mid_px is not None:
            self._mid_history.append((time_ms, float(book.mid_px)))
        self._trim(time_ms)

    def on_trade(self, trade: TradeState, time_ms: float) -> TradeState:
        aggressor = infer_trade_aggressor(trade, self.current_book)
        signed_qty = 0
        if aggressor == "BUY":
            signed_qty = int(trade.qty)
        elif aggressor == "SELL":
            signed_qty = -int(trade.qty)
        self._trade_history.append((time_ms, int(trade.qty), signed_qty))
        self.last_trade_px = int(trade.price_px)
        self._trim(time_ms)
        return TradeState(price_px=trade.price_px, qty=trade.qty, aggressor=aggressor)

    def on_fill(self, side: Side, qty: int, time_ms: float) -> None:
        signed_qty = int(qty) if side == "BUY" else -int(qty)
        self._fill_history.append((time_ms, signed_qty))
        self._trim(time_ms)

    def snapshot(self, time_ms: float, reference_fair_px: int | None) -> FeatureSnapshot:
        self._trim(time_ms)
        mid_px = self.current_book.mid_px if self.current_book else None
        microprice_px = compute_microprice(self.current_book)
        quote_to_fair_px = None if mid_px is None or reference_fair_px is None else mid_px - reference_fair_px
        trade_count_1s, trade_volume_1s, trade_pressure_1s = self._trade_stats(time_ms, 1_000.0)
        _, _, trade_pressure_5s = self._trade_stats(time_ms, 5_000.0)
        fill_pressure_2s = self._fill_pressure(time_ms, 2_000.0)
        return FeatureSnapshot(
            time_ms=time_ms,
            book=self.current_book,
            last_trade_px=self.last_trade_px,
            mid_px=mid_px,
            spread_px=self.current_book.spread_px if self.current_book else None,
            microprice_px=microprice_px,
            imbalance=compute_imbalance(self.current_book),
            realized_vol_1s=self._realized_vol(time_ms, 1_000.0),
            realized_vol_5s=self._realized_vol(time_ms, 5_000.0),
            trade_count_1s=trade_count_1s,
            trade_volume_1s=trade_volume_1s,
            trade_pressure_1s=trade_pressure_1s,
            trade_pressure_5s=trade_pressure_5s,
            fill_pressure_2s=fill_pressure_2s,
            reference_fair_px=reference_fair_px,
            quote_to_fair_px=quote_to_fair_px,
        )

    def _trim(self, time_ms: float) -> None:
        min_time = time_ms - 10_000.0
        while self._mid_history and self._mid_history[0][0] < min_time:
            self._mid_history.popleft()
        while self._trade_history and self._trade_history[0][0] < min_time:
            self._trade_history.popleft()
        while self._fill_history and self._fill_history[0][0] < min_time:
            self._fill_history.popleft()

    def _realized_vol(self, time_ms: float, window_ms: float) -> float | None:
        mids = [mid for ts, mid in self._mid_history if ts >= time_ms - window_ms]
        if len(mids) < 3:
            return None
        returns: list[float] = []
        previous = mids[0]
        for mid in mids[1:]:
            if previous > 0 and mid > 0:
                returns.append(math.log(mid / previous))
            previous = mid
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(max(variance, 0.0))

    def _trade_stats(self, time_ms: float, window_ms: float) -> tuple[int, int, float]:
        count = 0
        volume = 0
        signed = 0
        for ts, qty, signed_qty in reversed(self._trade_history):
            if ts < time_ms - window_ms:
                break
            count += 1
            volume += qty
            signed += signed_qty
        pressure = 0.0 if volume <= 0 else signed / volume
        return count, volume, pressure

    def _fill_pressure(self, time_ms: float, window_ms: float) -> float:
        signed = 0
        total = 0
        for ts, signed_qty in reversed(self._fill_history):
            if ts < time_ms - window_ms:
                break
            signed += signed_qty
            total += abs(signed_qty)
        return 0.0 if total <= 0 else signed / total
