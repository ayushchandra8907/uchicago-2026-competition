from __future__ import annotations

from collections import deque

from ..config import StrategyConfig
from ..core.types import Decision, DesiredOrder, ModeState, NewsEvent, StrategySnapshot


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class CMarketMakerStrategy:
    """Simple passive market maker for stock C that leans against bursty trade flow."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mode: ModeState = "IDLE"
        self.fair_value: int | None = None
        self.microprice: float | None = None
        self.recent_trade_signals: deque[tuple[int, int]] = deque()
        self.recent_trade_count: int = 0
        self.flow_score: int = 0
        self.burst_active: bool = False
        self.quote_side: str | None = None
        self.quote_px: int | None = None
        self.quote_reason: str | None = None

    def on_book(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="book")

    def on_trade(self, snapshot: StrategySnapshot) -> Decision:
        self._record_trade(snapshot)
        return self._evaluate(snapshot, trigger="trade")

    def on_fill(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="fill")

    def on_timer(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="timer")

    def on_news(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> Decision:
        return self._cancel_or_observe(snapshot, reason="ignoring news for stock C market making")

    def export_state(self) -> dict[str, int | float | str | bool | None]:
        return {
            "mode": self.mode,
            "active_signal_kind": "stock_c_market_maker",
            "fair_value": self.fair_value,
            "mm_microprice": None if self.microprice is None else round(self.microprice, 3),
            "mm_recent_trade_count": self.recent_trade_count,
            "mm_flow_score": self.flow_score,
            "mm_burst_active": self.burst_active,
            "mm_quote_side": self.quote_side,
            "mm_quote_px": self.quote_px,
            "mm_quote_reason": self.quote_reason,
        }

    def _evaluate(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        book = snapshot.book
        if book.best_bid is None or book.best_ask is None or book.mid is None:
            self.mode = "IDLE"
            self._update_state(snapshot, quote_side=None, quote_px=None, quote_reason="waiting for a two-sided C book")
            return self._cancel_or_observe(snapshot, reason="waiting for a two-sided C book")

        self._trim_trade_signals(snapshot.now_ms)
        self.microprice = self._microprice(snapshot)
        fair = float(self.microprice if self.microprice is not None else book.mid) + float(self.config.maker_fair_value_offset_ticks)
        self.fair_value = int(round(fair))
        self.recent_trade_count = len(self.recent_trade_signals)
        self.flow_score = sum(signal for _, signal in self.recent_trade_signals)
        self.burst_active = (
            self.recent_trade_count >= int(self.config.maker_min_recent_trade_count)
            and abs(self.flow_score) >= int(self.config.maker_flow_trigger)
        )

        inventory = int(snapshot.inventory)
        spread = int(book.spread or 0)

        if abs(inventory) >= int(self.config.maker_aggressive_flatten_inventory):
            self.mode = "UNWIND"
            return self._aggressive_inventory_flatten(snapshot, reason=f"{trigger}: inventory beyond aggressive flatten threshold")

        if spread < int(self.config.maker_min_spread_ticks):
            if inventory > 0:
                self.mode = "MARKET_MAKING_STUB"
                return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=min(inventory, self._quote_qty(inventory)), reason="tight spread; recycling long inventory")
            if inventory < 0:
                self.mode = "MARKET_MAKING_STUB"
                return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=min(abs(inventory), self._quote_qty(inventory)), reason="tight spread; recycling short inventory")
            self.mode = "IDLE"
            self._update_state(snapshot, quote_side=None, quote_px=None, quote_reason="spread too tight for stock C market making")
            return self._cancel_or_observe(snapshot, reason="spread too tight for stock C market making")

        soft_limit = int(self.config.maker_inventory_soft_limit)
        hard_limit = int(self.config.maker_inventory_hard_limit)
        quote_qty = self._quote_qty(inventory)

        if inventory >= soft_limit:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=min(inventory, quote_qty), reason="inventory skewed long; offering stock C")
        if inventory <= -soft_limit:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=min(abs(inventory), quote_qty), reason="inventory skewed short; bidding for stock C")

        if self.burst_active:
            if self.flow_score >= int(self.config.maker_flow_trigger) and inventory > -hard_limit:
                self.mode = "MARKET_MAKING_STUB"
                return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=quote_qty, reason="buy burst in C; leaning offer into troll flow")
            if self.flow_score <= -int(self.config.maker_flow_trigger) and inventory < hard_limit:
                self.mode = "MARKET_MAKING_STUB"
                return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=quote_qty, reason="sell burst in C; leaning bid into troll flow")

        if inventory > 0:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=min(inventory, quote_qty), reason="no burst; recycling long C inventory")
        if inventory < 0:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=min(abs(inventory), quote_qty), reason="no burst; recycling short C inventory")

        self.mode = "IDLE"
        self._update_state(snapshot, quote_side=None, quote_px=None, quote_reason="waiting for a bursty spread in C")
        return self._cancel_or_observe(snapshot, reason="waiting for a bursty spread in C")

    def _record_trade(self, snapshot: StrategySnapshot) -> None:
        if snapshot.last_trade_px is None:
            return
        book = snapshot.book
        if book.best_bid is None or book.best_ask is None or book.mid is None:
            return
        trade_px = float(snapshot.last_trade_px)
        if trade_px >= float(book.best_ask.px):
            signal = 1
        elif trade_px <= float(book.best_bid.px):
            signal = -1
        else:
            signal = _sign(trade_px - float(book.mid))
        if signal == 0:
            return
        self.recent_trade_signals.append((int(snapshot.now_ms), signal))
        self._trim_trade_signals(snapshot.now_ms)

    def _trim_trade_signals(self, now_ms: int) -> None:
        cutoff = int(now_ms) - int(self.config.maker_trade_window_ms)
        while self.recent_trade_signals and self.recent_trade_signals[0][0] < cutoff:
            self.recent_trade_signals.popleft()

    @staticmethod
    def _microprice(snapshot: StrategySnapshot) -> float | None:
        book = snapshot.book
        if book.best_bid is None or book.best_ask is None:
            return book.mid
        total_qty = int(book.best_bid.qty) + int(book.best_ask.qty)
        if total_qty <= 0:
            return book.mid
        return (
            (float(book.best_bid.px) * float(book.best_ask.qty))
            + (float(book.best_ask.px) * float(book.best_bid.qty))
        ) / float(total_qty)

    def _quote_qty(self, inventory: int) -> int:
        hard_limit = max(1, int(self.config.maker_inventory_hard_limit))
        base_qty = max(1, int(self.config.maker_quote_qty))
        distance = max(0.0, 1.0 - (abs(int(inventory)) / float(hard_limit)))
        scaled = int(round(base_qty * max(0.35, distance)))
        return max(1, min(base_qty, scaled))

    def _quote(self, snapshot: StrategySnapshot, *, side: str, px: int, qty: int, reason: str) -> Decision:
        self._update_state(snapshot, quote_side=side, quote_px=px, quote_reason=reason)
        direction = 1 if side == "BUY" else -1
        target_inventory = int(snapshot.inventory) + (direction * int(qty))
        return Decision(
            mode="MARKET_MAKING_STUB",
            target_inventory=target_inventory,
            desired_order=DesiredOrder(
                side=side,
                px=int(px),
                qty=max(1, int(qty)),
                aggressive=False,
                intent="stock_c_market_make",
                reason=reason,
                symbol=self.config.symbol,
            ),
            cancel_all=False,
            observe_only=False,
            reason=reason,
            fair_value=self.fair_value,
        )

    def _aggressive_inventory_flatten(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        book = snapshot.book
        inventory = int(snapshot.inventory)
        if inventory > 0 and book.best_bid is not None:
            self._update_state(snapshot, quote_side="SELL", quote_px=book.best_bid.px, quote_reason=reason)
            return Decision(
                mode="UNWIND",
                target_inventory=0,
                desired_order=DesiredOrder(
                    side="SELL",
                    px=book.best_bid.px,
                    qty=abs(inventory),
                    aggressive=True,
                    intent="stock_c_inventory_flatten",
                    reason=reason,
                    symbol=self.config.symbol,
                ),
                cancel_all=True,
                observe_only=False,
                reason=reason,
                fair_value=self.fair_value,
            )
        if inventory < 0 and book.best_ask is not None:
            self._update_state(snapshot, quote_side="BUY", quote_px=book.best_ask.px, quote_reason=reason)
            return Decision(
                mode="UNWIND",
                target_inventory=0,
                desired_order=DesiredOrder(
                    side="BUY",
                    px=book.best_ask.px,
                    qty=abs(inventory),
                    aggressive=True,
                    intent="stock_c_inventory_flatten",
                    reason=reason,
                    symbol=self.config.symbol,
                ),
                cancel_all=True,
                observe_only=False,
                reason=reason,
                fair_value=self.fair_value,
            )
        return self._cancel_or_observe(snapshot, reason=reason)

    def _cancel_or_observe(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        cancel_live = bool(snapshot.open_orders)
        return Decision(
            mode=self.mode,
            target_inventory=int(snapshot.inventory),
            desired_order=None,
            cancel_all=cancel_live,
            observe_only=True,
            reason=reason,
            fair_value=self.fair_value,
        )

    def _update_state(self, snapshot: StrategySnapshot, *, quote_side: str | None, quote_px: int | None, quote_reason: str | None) -> None:
        self.quote_side = quote_side
        self.quote_px = quote_px
        self.quote_reason = quote_reason
        if self.fair_value is None and snapshot.book.mid is not None:
            self.fair_value = int(round(float(snapshot.book.mid)))
