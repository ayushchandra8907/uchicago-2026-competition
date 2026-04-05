from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
import time
from typing import Iterable, Literal

from a_bot_config import AConfig, RiskConfig


SideName = Literal["BUY", "SELL"]
ModeName = Literal[
    "OPENING_MICRO_MM",
    "PRE_NEWS_PULLBACK",
    "POST_NEWS_SHOCK",
    "UNWIND",
    "STEADY_MM",
]

TICKS_PER_SECOND = 5
DAY_TICKS = 450
DAY_MS = 90_000
EARNINGS_TICKS = (110, 440)
TICK_MS = 200


@dataclass(frozen=True)
class BookLevel:
    px: int
    qty: int


@dataclass(frozen=True)
class BookSnapshot:
    best_bid: BookLevel | None = None
    best_ask: BookLevel | None = None

    @classmethod
    def from_order_book(cls, book) -> "BookSnapshot":
        bids = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
        asks = [(int(px), int(qty)) for px, qty in book.asks.items() if qty > 0]
        best_bid_level = max(bids, key=lambda level: level[0]) if bids else None
        best_ask_level = min(asks, key=lambda level: level[0]) if asks else None
        best_bid = None if best_bid_level is None else BookLevel(px=best_bid_level[0], qty=best_bid_level[1])
        best_ask = None if best_ask_level is None else BookLevel(px=best_ask_level[0], qty=best_ask_level[1])
        return cls(best_bid=best_bid, best_ask=best_ask)

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.px + self.best_ask.px) / 2.0

    @property
    def spread(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask.px - self.best_bid.px


@dataclass(frozen=True)
class DesiredOrder:
    side: SideName
    px: int
    qty: int
    aggressive: bool = False
    reason: str = ""


@dataclass(frozen=True)
class QuotePlan:
    mode: ModeName
    bid: DesiredOrder | None
    ask: DesiredOrder | None
    aggressive_actions: tuple[DesiredOrder, ...]
    observe_only: bool
    reason: str


@dataclass
class ManagedOrder:
    order_id: str
    side: SideName
    px: int
    qty: int
    remaining_qty: int
    submitted_ms: int
    aggressive: bool = False
    cancel_pending: bool = False
    restored: bool = False

    @property
    def is_active(self) -> bool:
        return self.remaining_qty > 0 and not self.cancel_pending


@dataclass(frozen=True)
class CancelCommand:
    order_id: str
    side: SideName


@dataclass(frozen=True)
class PlaceCommand:
    side: SideName
    px: int
    qty: int
    aggressive: bool
    reason: str


@dataclass(frozen=True)
class SyncActions:
    cancels: tuple[CancelCommand, ...]
    placements: tuple[PlaceCommand, ...]


@dataclass(frozen=True)
class NewsReaction:
    relevant: bool
    fair_value_updated: bool
    earnings_value: float | None = None
    old_fair_value: int | None = None
    new_fair_value: int | None = None
    shock_direction: int = 0
    shock_threshold: int | None = None
    tick: int | None = None


class AValuationModel:
    """Deterministic A valuation from earnings, fixed P/E, and exchange price scaling."""

    def __init__(
        self,
        a_config: AConfig,
        *,
        initial_fair_value: int | None = None,
        initial_earnings_value: float | None = None,
    ):
        self.pe_ratio = float(a_config.pe_ratio)
        self.price_scale = int(a_config.price_scale)
        self.fair_value = initial_fair_value
        self.last_earnings_value = initial_earnings_value
        self.last_source = "journal" if initial_fair_value is not None else "none"

    @property
    def has_fair_value(self) -> bool:
        return self.fair_value is not None

    def fair_from_earnings(self, earnings_value: float) -> int:
        return round(float(earnings_value) * self.pe_ratio * self.price_scale)

    def on_earnings_release(self, earnings_value: float) -> tuple[int | None, int]:
        old_fair = self.fair_value
        self.last_earnings_value = float(earnings_value)
        self.fair_value = self.fair_from_earnings(self.last_earnings_value)
        self.last_source = "earnings"
        return old_fair, self.fair_value


class QuoteEngine:
    """Converts the current A mode, fair value, and book into desired orders."""

    def __init__(self, a_config: AConfig, risk: RiskConfig):
        self.a_config = a_config
        self.risk = risk

    def compute_quotes(
        self,
        mode: ModeName,
        fair_value: int | None,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        can_trade: bool,
        reason_if_blocked: str,
        shock_direction: int = 0,
        shock_threshold: int | None = None,
    ) -> QuotePlan:
        if not can_trade:
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason=reason_if_blocked,
            )

        if mode == "PRE_NEWS_PULLBACK":
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=False,
                reason="flat before scheduled A earnings",
            )

        if mode == "OPENING_MICRO_MM":
            return self._opening_quotes(mode, inventory, book, buy_exposure, sell_exposure)

        if fair_value is None:
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for the first usable A fair value.",
            )

        if mode == "POST_NEWS_SHOCK":
            return self._shock_quotes(
                mode,
                fair_value,
                inventory,
                book,
                buy_exposure,
                sell_exposure,
                shock_direction=shock_direction,
                shock_threshold=shock_threshold or self.a_config.shock_take_min_edge,
            )

        if mode == "UNWIND":
            return self._unwind_quotes(mode, fair_value, inventory, book, buy_exposure, sell_exposure)

        return self._steady_quotes(mode, fair_value, inventory, book, buy_exposure, sell_exposure)

    def _opening_quotes(
        self,
        mode: ModeName,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
    ) -> QuotePlan:
        mid = book.mid
        spread = book.spread
        if mid is None or spread is None or spread < self.a_config.opening_min_book_spread:
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=False,
                reason="waiting for a wide enough opening spread",
            )

        bid_px = int(floor(mid - self.a_config.opening_half_spread_ticks))
        ask_px = int(ceil(mid + self.a_config.opening_half_spread_ticks))
        bid_px, ask_px = self._clamp_inside_book(bid_px, ask_px, book)

        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=self.a_config.opening_max_position,
        )
        desired_bid = None
        desired_ask = None
        if allowed_buy > 0:
            desired_bid = DesiredOrder(
                side="BUY",
                px=bid_px,
                qty=min(self.a_config.opening_quote_size, allowed_buy),
                aggressive=False,
                reason="opening micro-mm bid around live mid",
            )
        if allowed_sell > 0:
            desired_ask = DesiredOrder(
                side="SELL",
                px=ask_px,
                qty=min(self.a_config.opening_quote_size, allowed_sell),
                aggressive=False,
                reason="opening micro-mm ask around live mid",
            )

        return QuotePlan(
            mode=mode,
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=(),
            observe_only=False,
            reason="opening micro-mm",
        )

    def _steady_quotes(
        self,
        mode: ModeName,
        fair_value: int,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=self.a_config.steady_max_position,
        )
        take_edge = max(2, self.a_config.steady_half_spread_ticks + 1)

        if book.best_ask is not None and allowed_buy > 0 and book.best_ask.px <= fair_value - take_edge:
            aggressive_actions.append(
                DesiredOrder(
                    side="BUY",
                    px=book.best_ask.px,
                    qty=min(self.a_config.steady_quote_size, allowed_buy, book.best_ask.qty),
                    aggressive=True,
                    reason="steady-state buy through stale ask below fair",
                )
            )
        if book.best_bid is not None and allowed_sell > 0 and book.best_bid.px >= fair_value + take_edge:
            aggressive_actions.append(
                DesiredOrder(
                    side="SELL",
                    px=book.best_bid.px,
                    qty=min(self.a_config.steady_quote_size, allowed_sell, book.best_bid.qty),
                    aggressive=True,
                    reason="steady-state sell through stale bid above fair",
                )
            )

        reservation = fair_value - (self.a_config.steady_inventory_skew * inventory)
        bid_px = int(floor(reservation - self.a_config.steady_half_spread_ticks))
        ask_px = int(ceil(reservation + self.a_config.steady_half_spread_ticks))
        bid_px, ask_px = self._clamp_inside_book(bid_px, ask_px, book)

        aggressive_sides = {action.side for action in aggressive_actions}
        desired_bid = None
        desired_ask = None
        if allowed_buy > 0 and "BUY" not in aggressive_sides:
            desired_bid = DesiredOrder(
                side="BUY",
                px=bid_px,
                qty=min(self.a_config.steady_quote_size, allowed_buy),
                aggressive=False,
                reason="steady-state bid around exact fair",
            )
        if allowed_sell > 0 and "SELL" not in aggressive_sides:
            desired_ask = DesiredOrder(
                side="SELL",
                px=ask_px,
                qty=min(self.a_config.steady_quote_size, allowed_sell),
                aggressive=False,
                reason="steady-state ask around exact fair",
            )

        return QuotePlan(
            mode=mode,
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=tuple(aggressive_actions),
            observe_only=False,
            reason="steady mm",
        )

    def _shock_quotes(
        self,
        mode: ModeName,
        fair_value: int,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        shock_direction: int,
        shock_threshold: int,
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=self.a_config.shock_max_position,
        )

        desired_bid = None
        desired_ask = None
        half_spread = self.a_config.steady_half_spread_ticks

        if shock_direction > 0:
            if book.best_ask is not None and allowed_buy > 0 and book.best_ask.px <= fair_value - shock_threshold:
                aggressive_actions.append(
                    DesiredOrder(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(self.a_config.shock_quote_size, allowed_buy, book.best_ask.qty),
                        aggressive=True,
                        reason="earnings upside shock buy through stale asks",
                    )
                )
            if allowed_sell > 0:
                ask_px = int(ceil(fair_value + half_spread))
                _, ask_px = self._clamp_inside_book(fair_value - 1, ask_px, book)
                desired_ask = DesiredOrder(
                    side="SELL",
                    px=ask_px,
                    qty=min(self.a_config.shock_quote_size, allowed_sell),
                    aggressive=False,
                    reason="post-shock ask to unwind long inventory",
                )
        elif shock_direction < 0:
            if book.best_bid is not None and allowed_sell > 0 and book.best_bid.px >= fair_value + shock_threshold:
                aggressive_actions.append(
                    DesiredOrder(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(self.a_config.shock_quote_size, allowed_sell, book.best_bid.qty),
                        aggressive=True,
                        reason="earnings downside shock sell through stale bids",
                    )
                )
            if allowed_buy > 0:
                bid_px = int(floor(fair_value - half_spread))
                bid_px, _ = self._clamp_inside_book(bid_px, fair_value + 1, book)
                desired_bid = DesiredOrder(
                    side="BUY",
                    px=bid_px,
                    qty=min(self.a_config.shock_quote_size, allowed_buy),
                    aggressive=False,
                    reason="post-shock bid to unwind short inventory",
                )
        else:
            return self._steady_quotes("STEADY_MM", fair_value, inventory, book, buy_exposure, sell_exposure)

        return QuotePlan(
            mode=mode,
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=tuple(aggressive_actions),
            observe_only=False,
            reason="post-news shock",
        )

    def _unwind_quotes(
        self,
        mode: ModeName,
        fair_value: int,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=self.a_config.steady_max_position,
        )

        if inventory > 0:
            if book.best_bid is not None and allowed_sell > 0 and book.best_bid.px >= fair_value:
                aggressive_actions.append(
                    DesiredOrder(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(self.a_config.steady_quote_size, allowed_sell, book.best_bid.qty),
                        aggressive=True,
                        reason="unwind long inventory into rich bid",
                    )
                )
            ask_px = int(ceil((fair_value - (self.a_config.unwind_inventory_skew * inventory)) + 1))
            _, ask_px = self._clamp_inside_book(fair_value - 1, ask_px, book)
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=DesiredOrder(
                    side="SELL",
                    px=ask_px,
                    qty=min(self.a_config.steady_quote_size, allowed_sell),
                    aggressive=False,
                    reason="unwind ask to reduce long inventory",
                )
                if allowed_sell > 0
                else None,
                aggressive_actions=tuple(aggressive_actions),
                observe_only=False,
                reason="unwind long inventory",
            )

        if inventory < 0:
            if book.best_ask is not None and allowed_buy > 0 and book.best_ask.px <= fair_value:
                aggressive_actions.append(
                    DesiredOrder(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(self.a_config.steady_quote_size, allowed_buy, book.best_ask.qty),
                        aggressive=True,
                        reason="unwind short inventory into cheap ask",
                    )
                )
            bid_px = int(floor((fair_value - (self.a_config.unwind_inventory_skew * inventory)) - 1))
            bid_px, _ = self._clamp_inside_book(bid_px, fair_value + 1, book)
            return QuotePlan(
                mode=mode,
                bid=DesiredOrder(
                    side="BUY",
                    px=bid_px,
                    qty=min(self.a_config.steady_quote_size, allowed_buy),
                    aggressive=False,
                    reason="unwind bid to reduce short inventory",
                )
                if allowed_buy > 0
                else None,
                ask=None,
                aggressive_actions=tuple(aggressive_actions),
                observe_only=False,
                reason="unwind short inventory",
            )

        return self._steady_quotes("STEADY_MM", fair_value, inventory, book, buy_exposure, sell_exposure)

    @staticmethod
    def _allowed_size(
        inventory: int,
        buy_exposure: int,
        sell_exposure: int,
        *,
        cap: int,
    ) -> tuple[int, int]:
        allowed_buy = max(0, cap - (inventory + buy_exposure))
        allowed_sell = max(0, cap + inventory - sell_exposure)
        return allowed_buy, allowed_sell

    @staticmethod
    def _clamp_inside_book(bid_px: int, ask_px: int, book: BookSnapshot) -> tuple[int, int]:
        if book.best_ask is not None:
            bid_px = min(bid_px, book.best_ask.px - 1)
        if book.best_bid is not None:
            ask_px = max(ask_px, book.best_bid.px + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1
        return bid_px, ask_px


class OrderManager:
    """Tracks one managed order per side and decides when to cancel/replace."""

    def __init__(self, symbol: str, risk: RiskConfig):
        self.symbol = symbol
        self.risk = risk
        self.orders: dict[str, ManagedOrder] = {}
        self.live_by_side: dict[SideName, str | None] = {"BUY": None, "SELL": None}
        self.last_action_ms: dict[SideName, int] = {"BUY": 0, "SELL": 0}

    def restore_order(self, order: ManagedOrder) -> None:
        self.orders[order.order_id] = order
        self.live_by_side[order.side] = order.order_id

    def note_submitted(
        self,
        order_id: str,
        side: SideName,
        px: int,
        qty: int,
        now_ms: int,
        aggressive: bool = False,
        restored: bool = False,
    ) -> ManagedOrder:
        order = ManagedOrder(
            order_id=order_id,
            side=side,
            px=px,
            qty=qty,
            remaining_qty=qty,
            submitted_ms=now_ms,
            aggressive=aggressive,
            restored=restored,
        )
        self.orders[order_id] = order
        self.live_by_side[side] = order_id
        self.last_action_ms[side] = now_ms
        return order

    def live_order(self, side: SideName) -> ManagedOrder | None:
        order_id = self.live_by_side.get(side)
        if order_id is None:
            return None
        return self.orders.get(order_id)

    def buy_exposure(self) -> int:
        order = self.live_order("BUY")
        return 0 if order is None or order.cancel_pending else order.remaining_qty

    def sell_exposure(self) -> int:
        order = self.live_order("SELL")
        return 0 if order is None or order.cancel_pending else order.remaining_qty

    def mark_cancel_requested(self, order_id: str, now_ms: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.cancel_pending = True
        self.last_action_ms[order.side] = now_ms
        return order

    def handle_fill(self, order_id: str, qty: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.remaining_qty = max(0, order.remaining_qty - qty)
        if order.remaining_qty == 0:
            self._drop_order(order_id)
        return order

    def handle_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        if success:
            self._drop_order(order_id)
            return order
        order.cancel_pending = False
        return order

    def handle_rejection(self, order_id: str) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        self._drop_order(order_id)
        return order

    def forget_order(self, order_id: str) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        self._drop_order(order_id)
        return order

    def has_stale_quote(self, now_ms: int) -> bool:
        for side in ("BUY", "SELL"):
            order = self.live_order(side)
            if order is None or order.cancel_pending:
                continue
            if now_ms - order.submitted_ms >= self.risk.stale_quote_ms:
                return True
        return False

    def build_actions(self, plan: QuotePlan, now_ms: int) -> SyncActions:
        desired_by_side: dict[SideName, DesiredOrder | None] = {
            "BUY": plan.bid,
            "SELL": plan.ask,
        }
        for aggressive in plan.aggressive_actions:
            desired_by_side[aggressive.side] = aggressive

        cancels: list[CancelCommand] = []
        placements: list[PlaceCommand] = []

        for side in ("BUY", "SELL"):
            desired = desired_by_side[side]
            live = self.live_order(side)
            if live is not None and live.cancel_pending:
                continue

            needs_reprice = (
                live is not None
                and desired is not None
                and (
                    live.px != desired.px
                    or live.remaining_qty != desired.qty
                    or live.aggressive != desired.aggressive
                )
            )

            if live is not None and desired is None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    cancels.append(CancelCommand(order_id=live.order_id, side=side))
                continue

            if live is not None and needs_reprice:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    # Cancel first and let the next event loop place the replacement.
                    # This keeps the bot from doubling exposure on a single side.
                    cancels.append(CancelCommand(order_id=live.order_id, side=side))
                continue

            if live is None and desired is not None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    placements.append(
                        PlaceCommand(
                            side=side,
                            px=desired.px,
                            qty=desired.qty,
                            aggressive=desired.aggressive,
                            reason=desired.reason,
                        )
                    )

        return SyncActions(cancels=tuple(cancels), placements=tuple(placements))

    def restored_orders(self) -> Iterable[ManagedOrder]:
        return (order for order in self.orders.values() if order.restored)

    def _drop_order(self, order_id: str) -> None:
        order = self.orders.pop(order_id, None)
        if order is None:
            return
        if self.live_by_side.get(order.side) == order_id:
            self.live_by_side[order.side] = None


class MarketAStrategy:
    """Owns the valuation model, schedule tracking, quote engine, and order manager for A."""

    def __init__(
        self,
        a_config: AConfig,
        risk: RiskConfig,
        restored_orders: Iterable[ManagedOrder] = (),
        recovered_fair_value: int | None = None,
        recovered_earnings_value: float | None = None,
    ):
        starting_fair = recovered_fair_value
        if starting_fair is None:
            starting_fair = a_config.initial_fair_value

        self.a_config = a_config
        self.risk = risk
        self.valuation = AValuationModel(
            a_config=a_config,
            initial_fair_value=starting_fair,
            initial_earnings_value=recovered_earnings_value,
        )
        self.quote_engine = QuoteEngine(a_config, risk)
        self.order_manager = OrderManager(symbol="A", risk=risk)
        self.inventory = 0
        self.book = BookSnapshot()
        self.mode: ModeName = "OPENING_MICRO_MM"
        self.startup_ms = self._now_ms()
        self.tick_anchor_tick: int | None = None
        self.tick_anchor_ms: int | None = None
        self.last_news_tick: int | None = None
        self.current_round_earnings_seen = recovered_fair_value is not None
        self.shock_started_ms: int | None = None
        self.shock_reference_fair: int | None = None
        self.shock_target_fair: int | None = None
        self.shock_direction = 0
        self.shock_threshold: int | None = None
        self.recovery_pending: set[str] = set()
        self.recovery_active = False

        for order in restored_orders:
            self.order_manager.restore_order(order)
            self.recovery_pending.add(order.order_id)

        if self.recovery_pending:
            self.recovery_active = True

    @property
    def fair_value(self) -> int | None:
        return self.valuation.fair_value

    @property
    def last_earnings_value(self) -> float | None:
        return self.valuation.last_earnings_value

    def set_inventory(self, inventory: int) -> None:
        self.inventory = int(inventory)

    def on_book_update(self, symbol: str, book) -> bool:
        return self.on_book_update_at(symbol, book, self._now_ms())

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != "A":
            return False
        self.book = BookSnapshot.from_order_book(book)
        self.mode = self._determine_mode(now_ms)
        return True

    def on_news(self, news_release: dict, now_ms: int) -> NewsReaction:
        tick = news_release.get("tick")
        if isinstance(tick, int):
            self.tick_anchor_tick = tick
            self.tick_anchor_ms = now_ms
            self.last_news_tick = tick

        if not self._handles_a_earnings(news_release):
            self.mode = self._determine_mode(now_ms)
            return NewsReaction(relevant=False, fair_value_updated=False, tick=tick if isinstance(tick, int) else None)

        earnings_value = float(news_release["new_data"]["value"])
        self.current_round_earnings_seen = True
        old_fair, new_fair = self.valuation.on_earnings_release(earnings_value)
        reference_fair = old_fair
        if reference_fair is None:
            mid = self.book.mid
            if mid is not None:
                reference_fair = round(mid)
        if reference_fair is None:
            reference_fair = new_fair

        move_size = abs(new_fair - reference_fair)
        self.shock_reference_fair = reference_fair
        self.shock_target_fair = new_fair
        self.shock_direction = 1 if new_fair > reference_fair else -1 if new_fair < reference_fair else 0
        self.shock_threshold = max(
            self.a_config.shock_take_min_edge,
            round(self.a_config.shock_take_fraction * move_size),
        )
        self.shock_started_ms = now_ms
        self.mode = self._determine_mode(now_ms)

        return NewsReaction(
            relevant=True,
            fair_value_updated=True,
            earnings_value=earnings_value,
            old_fair_value=old_fair,
            new_fair_value=new_fair,
            shock_direction=self.shock_direction,
            shock_threshold=self.shock_threshold,
            tick=tick if isinstance(tick, int) else None,
        )

    def on_fill(self, order_id: str, qty: int, price: int) -> ManagedOrder | None:
        order = self.order_manager.handle_fill(order_id, qty)
        if order is None:
            return None
        signed_qty = qty if order.side == "BUY" else -qty
        self.inventory += signed_qty
        if order_id in self.recovery_pending and order.remaining_qty == 0:
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.order_manager.handle_cancel_response(order_id, success)
        if order is None:
            return None
        if order_id in self.recovery_pending:
            if not success:
                # After a restart there is no open-order snapshot, so we treat a failed
                # recovery cancel as unresolved local state and stop managing it.
                self.order_manager.forget_order(order_id)
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_rejection(self, order_id: str) -> ManagedOrder | None:
        order = self.order_manager.handle_rejection(order_id)
        if order is None:
            return None
        if order_id in self.recovery_pending:
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_recovery_complete(self) -> None:
        self.recovery_active = False
        self.recovery_pending.clear()

    def recovery_orders_to_cancel(self) -> list[ManagedOrder]:
        return [
            order
            for order in self.order_manager.restored_orders()
            if order.order_id in self.recovery_pending and not order.cancel_pending
        ]

    def _maybe_finish_recovery(self) -> None:
        if self.recovery_active and not self.recovery_pending:
            self.on_recovery_complete()

    def compute_quotes(self, now_ms: int | None = None) -> QuotePlan:
        if now_ms is None:
            now_ms = self._now_ms()
        self.mode = self._determine_mode(now_ms)
        reason_if_blocked = "Waiting for recovered A orders to be cancelled."
        return self.quote_engine.compute_quotes(
            mode=self.mode,
            fair_value=self.valuation.fair_value,
            inventory=self.inventory,
            book=self.book,
            buy_exposure=self.order_manager.buy_exposure(),
            sell_exposure=self.order_manager.sell_exposure(),
            can_trade=not self.recovery_active,
            reason_if_blocked=reason_if_blocked,
            shock_direction=self.shock_direction,
            shock_threshold=self.shock_threshold,
        )

    def ms_until_next_scheduled_earnings(self, now_ms: int | None = None) -> int | None:
        if now_ms is None:
            now_ms = self._now_ms()
        if self.tick_anchor_tick is not None and self.tick_anchor_ms is not None:
            estimated_day_tick = self._estimated_day_tick(now_ms)
            if estimated_day_tick is None:
                return None
            next_tick = self._next_earnings_day_tick(estimated_day_tick)
            delta_ticks = (next_tick - estimated_day_tick) % DAY_TICKS
            if delta_ticks == 0:
                return 0
            return round(delta_ticks * TICK_MS)

        if not self.a_config.startup_assume_fresh_round:
            return None

        day_elapsed_ms = max(0, (now_ms - self.startup_ms) % DAY_MS)
        for earnings_ms in (22_000, 88_000):
            if day_elapsed_ms <= earnings_ms:
                return earnings_ms - day_elapsed_ms
        return DAY_MS - day_elapsed_ms + 22_000

    def _determine_mode(self, now_ms: int) -> ModeName:
        if self._shock_active(now_ms):
            return "POST_NEWS_SHOCK"

        until_next_earnings = self.ms_until_next_scheduled_earnings(now_ms)
        if until_next_earnings is not None and until_next_earnings <= self.a_config.pre_news_pullback_ms:
            return "PRE_NEWS_PULLBACK"

        if self.valuation.fair_value is None:
            return "OPENING_MICRO_MM"

        if self._should_unwind():
            return "UNWIND"

        return "STEADY_MM"

    def _shock_active(self, now_ms: int) -> bool:
        if self.shock_started_ms is None or self.shock_target_fair is None:
            return False
        if now_ms - self.shock_started_ms < self.a_config.shock_window_ms:
            return True
        return False

    def _should_unwind(self) -> bool:
        if self.shock_started_ms is None:
            return False
        if self.inventory == 0:
            return False
        return abs(self.inventory) > self.a_config.unwind_flatten_threshold

    @staticmethod
    def _handles_a_earnings(news_release: dict) -> bool:
        if news_release.get("kind") != "structured":
            return False
        new_data = news_release.get("new_data") or {}
        if new_data.get("structured_subtype") != "earnings":
            return False
        asset = str(new_data.get("asset") or news_release.get("symbol") or "").upper()
        return asset == "A"

    @staticmethod
    def _next_earnings_day_tick(current_day_tick: float) -> int:
        for earnings_tick in EARNINGS_TICKS:
            if current_day_tick < earnings_tick:
                return earnings_tick
        return EARNINGS_TICKS[0]

    def _estimated_day_tick(self, now_ms: int) -> float | None:
        if self.tick_anchor_tick is None or self.tick_anchor_ms is None:
            return None
        elapsed_ms = max(0, now_ms - self.tick_anchor_ms)
        elapsed_ticks = elapsed_ms / TICK_MS
        return (self.tick_anchor_tick + elapsed_ticks) % DAY_TICKS

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
