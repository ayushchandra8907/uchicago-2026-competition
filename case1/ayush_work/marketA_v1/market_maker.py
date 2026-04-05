from __future__ import annotations

from dataclasses import replace
import math

from .config import AppConfig
from .execution import QuoteSynchronizer
from .fair_value import FairValueModel
from .features import FeatureEngine
from .models import BookState, FeatureSnapshot, MarketEvent, ModeName, OrderIntent, QuoteIntent, Side, StrategyState
from .regimes import RegimeManager
from .risk import RiskManager


class StrategyEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.features = FeatureEngine()
        self.fair_value_model = FairValueModel(
            pe_ratio=config.strategy.initial_pe_ratio,
            price_scale=config.strategy.price_scale,
        )
        self.state = StrategyState()
        self.risk = RiskManager(config.risk)
        self.regimes = RegimeManager(config.strategy, config.risk)
        self.quote_synchronizer = QuoteSynchronizer(config.strategy.requote_cooldown_ms)
        self.last_snapshot: FeatureSnapshot | None = None

    def on_event(self, event: MarketEvent) -> QuoteIntent | None:
        now_ms = float(event.time_ms)
        if event.book is not None:
            self.features.on_book(event.book, now_ms)
            if event.book.mid_px is not None:
                self.state.last_mid_px = event.book.mid_px

        if event.trade is not None:
            trade = self.features.on_trade(event.trade, now_ms)
            event = replace(event, trade=trade)

        if event.news is not None and event.news.kind == "structured" and event.news.structured_subtype == "earnings":
            if event.news.earnings_asset == "A" and event.news.earnings_value is not None:
                new_fair_px = self.fair_value_model.update_earnings(event.news.earnings_value)
                old_fair_px = self.state.fair_px
                self.state.latest_earnings = event.news.earnings_value
                self.state.fair_px = new_fair_px
                self.state.last_earnings_time_ms = now_ms
                if old_fair_px is None:
                    self.state.last_earnings_direction = 1 if new_fair_px >= int(round(self.state.last_mid_px or new_fair_px)) else -1
                else:
                    self.state.last_earnings_direction = 1 if new_fair_px >= old_fair_px else -1

        reference_fair_px = self.fair_value_model.reference_fair_px(self.features.current_book.mid_px if self.features.current_book else None)
        snapshot = self.features.snapshot(now_ms, reference_fair_px)
        self.last_snapshot = snapshot

        if snapshot.book is None or snapshot.mid_px is None:
            return None

        mode = self.regimes.determine_mode(self.state, snapshot)
        self.state.mode = mode
        self.state.fair_px = reference_fair_px

        if mode == "EARNINGS_SHOCK":
            return self._shock_quotes(snapshot)
        if mode == "OVERSHOOT_FADE":
            return self._overshoot_quotes(snapshot)
        if mode == "INVENTORY_UNWIND":
            return self._unwind_quotes(snapshot)
        if mode == "RISK_OFF":
            return self._risk_off_quotes(snapshot)
        return self._normal_quotes(snapshot)

    def on_fill(self, *, side: Side, qty: int, price_px: int, aggressive: bool, mode: ModeName, time_ms: float) -> None:
        signed_qty = qty if side == "BUY" else -qty
        self.state.inventory += signed_qty
        cash_delta = -price_px * qty if side == "BUY" else price_px * qty
        self.state.cash += cash_delta
        self.features.on_fill(side, qty, time_ms)
        if aggressive:
            if side == "BUY":
                self.state.aggressive_buy_fills += qty
            else:
                self.state.aggressive_sell_fills += qty
        else:
            if side == "BUY":
                self.state.passive_buy_fills += qty
            else:
                self.state.passive_sell_fills += qty
        self.state.mode = mode

    def sync_inventory(self, inventory: int, cash: float | None = None) -> None:
        self.state.inventory = int(inventory)
        if cash is not None:
            self.state.cash = float(cash)

    def _normal_quotes(self, snapshot: FeatureSnapshot) -> QuoteIntent:
        signal_fair_px = self._signal_fair_px(snapshot)
        reservation_px = int(round(signal_fair_px - (self.config.strategy.inventory_penalty * self.state.inventory)))
        half_spread_px = self._half_spread_px(snapshot, signal_fair_px)
        aggressive_actions = self._default_aggressive_actions(snapshot, signal_fair_px, half_spread_px, mode="NORMAL_MM")
        bid, ask = self._passive_two_sided_quotes(
            snapshot.book,
            reservation_px=reservation_px,
            half_spread_px=half_spread_px,
            mode="NORMAL_MM",
            base_qty=self.config.strategy.normal_order_size,
        )
        return QuoteIntent(
            mode="NORMAL_MM",
            reference_fair_px=snapshot.reference_fair_px,
            reservation_px=reservation_px,
            bid=bid,
            ask=ask,
            aggressive_actions=aggressive_actions,
            reason="inventory-aware baseline market making",
        )

    def _shock_quotes(self, snapshot: FeatureSnapshot) -> QuoteIntent:
        signal_fair_px = self._signal_fair_px(snapshot)
        bias_px = self.state.last_earnings_direction * max(2, self._half_spread_px(snapshot, signal_fair_px) // 2)
        reservation_px = int(round(signal_fair_px + bias_px - (self.config.strategy.inventory_penalty * self.state.inventory * 0.7)))
        half_spread_px = max(self.config.strategy.min_half_spread_px, self._half_spread_px(snapshot, signal_fair_px) - 1)
        aggressive_actions = self._default_aggressive_actions(
            snapshot,
            signal_fair_px,
            half_spread_px,
            mode="EARNINGS_SHOCK",
            edge_px=self.config.strategy.shock_aggressive_edge_px,
            qty=self.config.strategy.shock_order_size,
        )
        bid, ask = self._passive_two_sided_quotes(
            snapshot.book,
            reservation_px=reservation_px,
            half_spread_px=half_spread_px,
            mode="EARNINGS_SHOCK",
            base_qty=self.config.strategy.shock_order_size,
            directional_bias=self.state.last_earnings_direction,
        )
        return QuoteIntent(
            mode="EARNINGS_SHOCK",
            reference_fair_px=snapshot.reference_fair_px,
            reservation_px=reservation_px,
            bid=bid,
            ask=ask,
            aggressive_actions=aggressive_actions,
            reason="post-earnings price discovery",
        )

    def _overshoot_quotes(self, snapshot: FeatureSnapshot) -> QuoteIntent:
        signal_fair_px = self._signal_fair_px(snapshot)
        reservation_px = int(round(snapshot.reference_fair_px or signal_fair_px))
        half_spread_px = max(self.config.strategy.min_half_spread_px, self._half_spread_px(snapshot, signal_fair_px) - 1)
        aggressive_actions: list[OrderIntent] = []
        book = snapshot.book
        qty = self._quote_qty(
            side="SELL" if self.state.last_earnings_direction > 0 else "BUY",
            base_qty=self.config.strategy.unwind_order_size,
            mode="OVERSHOOT_FADE",
            resting_open_qty=0,
        )
        if self.state.last_earnings_direction > 0 and book.best_bid_px is not None:
            if book.best_bid_px >= reservation_px + self.config.strategy.overshoot_fade_edge_px and qty > 0:
                aggressive_actions.append(
                    OrderIntent(
                        side="SELL",
                        px=int(book.best_bid_px),
                        qty=min(qty, int(book.best_bid_qty or qty)),
                        aggressive=True,
                        reason="fade upside overshoot above fair",
                    )
                )
        elif self.state.last_earnings_direction < 0 and book.best_ask_px is not None:
            if book.best_ask_px <= reservation_px - self.config.strategy.overshoot_fade_edge_px and qty > 0:
                aggressive_actions.append(
                    OrderIntent(
                        side="BUY",
                        px=int(book.best_ask_px),
                        qty=min(qty, int(book.best_ask_qty or qty)),
                        aggressive=True,
                        reason="fade downside overshoot below fair",
                    )
                )
        bid, ask = self._passive_two_sided_quotes(
            book,
            reservation_px=reservation_px,
            half_spread_px=half_spread_px,
            mode="OVERSHOOT_FADE",
            base_qty=self.config.strategy.unwind_order_size,
            directional_bias=-self.state.last_earnings_direction,
        )
        return QuoteIntent(
            mode="OVERSHOOT_FADE",
            reference_fair_px=snapshot.reference_fair_px,
            reservation_px=reservation_px,
            bid=bid,
            ask=ask,
            aggressive_actions=tuple(aggressive_actions),
            reason="fade post-earnings overshoot back toward fair",
        )

    def _unwind_quotes(self, snapshot: FeatureSnapshot) -> QuoteIntent:
        book = snapshot.book
        fair_px = int(round(snapshot.reference_fair_px or snapshot.mid_px or 0))
        aggressive_actions: list[OrderIntent] = []
        bid: OrderIntent | None = None
        ask: OrderIntent | None = None
        if self.state.inventory > 0:
            qty = self._quote_qty(side="SELL", base_qty=self.config.strategy.unwind_order_size, mode="INVENTORY_UNWIND", resting_open_qty=0)
            if book.best_bid_px is not None and book.best_bid_px >= fair_px and qty > 0:
                aggressive_actions.append(
                    OrderIntent(
                        side="SELL",
                        px=int(book.best_bid_px),
                        qty=min(qty, int(book.best_bid_qty or qty)),
                        aggressive=True,
                        reason="reduce long inventory into bid at or above fair",
                    )
                )
            passive_qty = self._quote_qty(side="SELL", base_qty=self.config.strategy.unwind_order_size, mode="INVENTORY_UNWIND", resting_open_qty=0)
            if passive_qty > 0:
                ask_px = max(int(fair_px), int((book.best_bid_px or fair_px) + 1))
                ask = OrderIntent(side="SELL", px=ask_px, qty=passive_qty, aggressive=False, reason="one-sided long unwind quote")
        elif self.state.inventory < 0:
            qty = self._quote_qty(side="BUY", base_qty=self.config.strategy.unwind_order_size, mode="INVENTORY_UNWIND", resting_open_qty=0)
            if book.best_ask_px is not None and book.best_ask_px <= fair_px and qty > 0:
                aggressive_actions.append(
                    OrderIntent(
                        side="BUY",
                        px=int(book.best_ask_px),
                        qty=min(qty, int(book.best_ask_qty or qty)),
                        aggressive=True,
                        reason="reduce short inventory into ask at or below fair",
                    )
                )
            passive_qty = self._quote_qty(side="BUY", base_qty=self.config.strategy.unwind_order_size, mode="INVENTORY_UNWIND", resting_open_qty=0)
            if passive_qty > 0:
                bid_px = min(int(fair_px), int((book.best_ask_px or fair_px) - 1))
                bid = OrderIntent(side="BUY", px=bid_px, qty=passive_qty, aggressive=False, reason="one-sided short unwind quote")
        return QuoteIntent(
            mode="INVENTORY_UNWIND",
            reference_fair_px=snapshot.reference_fair_px,
            reservation_px=fair_px,
            bid=bid,
            ask=ask,
            aggressive_actions=tuple(aggressive_actions),
            reason="inventory unwind mode",
        )

    def _risk_off_quotes(self, snapshot: FeatureSnapshot) -> QuoteIntent:
        unwind_intent = self._unwind_quotes(snapshot)
        return QuoteIntent(
            mode="RISK_OFF",
            reference_fair_px=unwind_intent.reference_fair_px,
            reservation_px=unwind_intent.reservation_px,
            bid=unwind_intent.bid,
            ask=unwind_intent.ask,
            aggressive_actions=unwind_intent.aggressive_actions,
            reason="hard risk-off inventory flattening",
        )

    def _signal_fair_px(self, snapshot: FeatureSnapshot) -> int:
        base_fair = snapshot.reference_fair_px or int(round(snapshot.mid_px or 0.0))
        micro_offset = 0.0
        if snapshot.microprice_px is not None and snapshot.mid_px is not None:
            micro_offset = self.config.strategy.microprice_alpha * (snapshot.microprice_px - snapshot.mid_px)
        trade_offset = self.config.strategy.trade_pressure_alpha * snapshot.trade_pressure_1s * max(snapshot.spread_px or 1.0, 1.0)
        return int(round(base_fair + micro_offset + trade_offset))

    def _half_spread_px(self, snapshot: FeatureSnapshot, signal_fair_px: int) -> int:
        market_half_spread = max((snapshot.spread_px or 0.0) / 2.0, float(self.config.strategy.min_half_spread_px))
        vol_bps = (snapshot.realized_vol_5s or 0.0) * 10_000.0
        half_spread = (
            self.config.strategy.base_half_spread_px
            + (self.config.strategy.market_spread_weight * market_half_spread)
            + (self.config.strategy.vol_widening * vol_bps / 100.0)
            + (self.config.strategy.toxicity_widening * abs(snapshot.trade_pressure_1s))
            + (self.config.strategy.fill_widening * abs(snapshot.fill_pressure_2s))
        )
        if snapshot.quote_to_fair_px is not None and abs(snapshot.quote_to_fair_px) >= max(2, self.config.strategy.aggressive_edge_px // 2):
            half_spread += 1.0
        return max(self.config.strategy.min_half_spread_px, int(round(half_spread)))

    def _default_aggressive_actions(
        self,
        snapshot: FeatureSnapshot,
        signal_fair_px: int,
        half_spread_px: int,
        *,
        mode: ModeName,
        edge_px: int | None = None,
        qty: int | None = None,
    ) -> tuple[OrderIntent, ...]:
        book = snapshot.book
        edge_px = edge_px or self.config.strategy.aggressive_edge_px
        qty = qty or self.config.strategy.normal_order_size
        actions: list[OrderIntent] = []
        buy_qty = self._quote_qty(side="BUY", base_qty=qty, mode=mode, resting_open_qty=0)
        sell_qty = self._quote_qty(side="SELL", base_qty=qty, mode=mode, resting_open_qty=0)
        if book.best_ask_px is not None and buy_qty > 0 and book.best_ask_px <= signal_fair_px - edge_px:
            actions.append(
                OrderIntent(
                    side="BUY",
                    px=int(book.best_ask_px),
                    qty=min(buy_qty, int(book.best_ask_qty or buy_qty)),
                    aggressive=True,
                    reason="cross stale ask below estimated fair",
                )
            )
        if book.best_bid_px is not None and sell_qty > 0 and book.best_bid_px >= signal_fair_px + edge_px:
            actions.append(
                OrderIntent(
                    side="SELL",
                    px=int(book.best_bid_px),
                    qty=min(sell_qty, int(book.best_bid_qty or sell_qty)),
                    aggressive=True,
                    reason="cross stale bid above estimated fair",
                )
            )
        return tuple(actions)

    def _passive_two_sided_quotes(
        self,
        book: BookState,
        *,
        reservation_px: int,
        half_spread_px: int,
        mode: ModeName,
        base_qty: int,
        directional_bias: int = 0,
    ) -> tuple[OrderIntent | None, OrderIntent | None]:
        bid_px = reservation_px - half_spread_px + min(0, directional_bias)
        ask_px = reservation_px + half_spread_px + max(0, directional_bias)
        if book.best_ask_px is not None:
            bid_px = min(bid_px, int(book.best_ask_px) - 1)
        if book.best_bid_px is not None:
            ask_px = max(ask_px, int(book.best_bid_px) + 1)

        resting_open_qty = 0
        bid_qty = self._quote_qty(side="BUY", base_qty=base_qty, mode=mode, resting_open_qty=resting_open_qty)
        ask_qty = self._quote_qty(side="SELL", base_qty=base_qty, mode=mode, resting_open_qty=resting_open_qty + bid_qty)
        bid = None
        ask = None
        if book.best_ask_px is not None and bid_px < book.best_ask_px and bid_qty > 0:
            bid = OrderIntent(side="BUY", px=int(bid_px), qty=bid_qty, aggressive=False, reason=f"{mode.lower()} bid")
        if book.best_bid_px is not None and ask_px > book.best_bid_px and ask_qty > 0:
            ask = OrderIntent(side="SELL", px=int(ask_px), qty=ask_qty, aggressive=False, reason=f"{mode.lower()} ask")
        if bid is not None and ask is not None and bid.px >= ask.px:
            if self.state.inventory >= 0:
                bid = None
            else:
                ask = None
        return bid, ask

    def _quote_qty(self, *, side: Side, base_qty: int, mode: ModeName, resting_open_qty: int) -> int:
        soft_limit = self.risk.soft_limit(mode)
        inventory_ratio = min(1.0, abs(self.state.inventory) / max(soft_limit, 1))
        qty = int(round(base_qty * (1.0 - (0.6 * inventory_ratio))))
        if (side == "BUY" and self.state.inventory < 0) or (side == "SELL" and self.state.inventory > 0):
            qty = int(math.ceil(qty * 1.25))
        qty = max(1, qty)
        return self.risk.clamp_order_qty(
            qty,
            side=side,
            inventory=self.state.inventory,
            resting_open_qty=resting_open_qty,
            mode=mode,
        )
