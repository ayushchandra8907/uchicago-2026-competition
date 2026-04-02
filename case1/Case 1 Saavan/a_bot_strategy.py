from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from statistics import median
import time
from typing import Iterable, Literal

from a_bot_config import AConfig, RiskConfig


SideName = Literal["BUY", "SELL"]


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


@dataclass(frozen=True)
class DesiredOrder:
    side: SideName
    px: int
    qty: int
    aggressive: bool = False
    reason: str = ""


@dataclass(frozen=True)
class QuotePlan:
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


@dataclass
class PECalibrationJob:
    earnings_value: float
    started_ms: int
    sample_start_ms: int
    finalize_ms: int
    samples: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class NewsReaction:
    relevant: bool
    fair_value_updated: bool
    started_pe_calibration: bool
    earnings_value: float | None = None


@dataclass(frozen=True)
class LearnedPEUpdate:
    status: str
    implied_pe: float
    trusted_pe: float
    trusted_confidence: int
    candidate_pe: float | None
    candidate_confidence: int
    reference_price: float
    sample_count: int
    trading_ready: bool


class AValuationModel:
    """Keeps the latest fair value estimate for A and learns a hidden constant P/E ratio."""

    def __init__(
        self,
        a_config: AConfig,
        *,
        learned_pe_ratio: float | None = None,
        learned_pe_confidence: int = 0,
        initial_fair_value: int | None = None,
    ):
        self.configured_pe_ratio = a_config.pe_ratio
        self.trusted_pe_ratio = learned_pe_ratio
        self.trusted_pe_confidence = max(0, int(learned_pe_confidence))
        self.min_learned_confidence = max(1, int(a_config.pe_learning_min_confidence))
        self.consistency_tolerance = max(0.0, float(a_config.pe_learning_consistency_tolerance))
        self.replacement_confirmations = max(1, int(a_config.pe_replacement_confirmations))
        self.candidate_pe_ratio: float | None = None
        self.candidate_pe_confidence = 0
        self.fair_value = initial_fair_value
        self.last_earnings_value: float | None = None
        self.last_source = "config" if initial_fair_value is not None else "none"

    @property
    def has_fair_value(self) -> bool:
        return self.fair_value is not None

    def set_fair_value(self, fair_value: int, source: str) -> None:
        self.fair_value = int(fair_value)
        self.last_source = source

    @property
    def effective_pe_ratio(self) -> float | None:
        if self.configured_pe_ratio is not None:
            return self.configured_pe_ratio
        return self.trusted_pe_ratio

    @property
    def can_trade_from_pe(self) -> bool:
        if self.configured_pe_ratio is not None:
            return True
        return self.trusted_pe_ratio is not None and self.trusted_pe_confidence >= self.min_learned_confidence

    @property
    def pe_status_reason(self) -> str | None:
        if self.configured_pe_ratio is not None:
            return None
        if self.trusted_pe_ratio is None:
            return "No learned A P/E ratio yet; waiting for A earnings to calibrate."
        if self.trusted_pe_confidence < self.min_learned_confidence:
            return (
                f"Learned A P/E={self.trusted_pe_ratio:.4f} "
                f"with confidence {self.trusted_pe_confidence}/{self.min_learned_confidence}; waiting for another confirming earnings event."
            )
        return None

    def on_earnings_release(self, earnings_value: float) -> bool:
        self.last_earnings_value = float(earnings_value)
        pe_ratio = self.effective_pe_ratio
        if pe_ratio is None:
            return False
        self.set_fair_value(round(pe_ratio * self.last_earnings_value), source="earnings")
        return True

    def apply_learned_pe(
        self,
        implied_pe: float,
        *,
        reference_price: float,
        sample_count: int,
    ) -> LearnedPEUpdate:
        implied_pe = float(implied_pe)
        if self.configured_pe_ratio is not None:
            trusted_pe = self.configured_pe_ratio
            trusted_confidence = self.min_learned_confidence
            status = "configured"
        else:
            if not self.can_trade_from_pe:
                if self.trusted_pe_ratio is None:
                    self.trusted_pe_ratio = implied_pe
                    self.trusted_pe_confidence = 1
                    status = "bootstrap_started"
                else:
                    relative_diff = self._relative_difference(implied_pe, self.trusted_pe_ratio)
                    if relative_diff <= self.consistency_tolerance:
                        weight = min(max(self.trusted_pe_confidence, 1), 5)
                        self.trusted_pe_ratio = ((self.trusted_pe_ratio * weight) + implied_pe) / (weight + 1)
                        self.trusted_pe_confidence = min(self.trusted_pe_confidence + 1, 10)
                        status = "bootstrap_confirmed"
                    else:
                        self.trusted_pe_ratio = implied_pe
                        self.trusted_pe_confidence = 1
                        status = "bootstrap_reset"
                self.candidate_pe_ratio = None
                self.candidate_pe_confidence = 0
            else:
                relative_diff = self._relative_difference(implied_pe, self.trusted_pe_ratio)
                if relative_diff <= self.consistency_tolerance:
                    weight = min(max(self.trusted_pe_confidence, 1), 5)
                    self.trusted_pe_ratio = ((self.trusted_pe_ratio * weight) + implied_pe) / (weight + 1)
                    self.trusted_pe_confidence = min(self.trusted_pe_confidence + 1, 10)
                    self.candidate_pe_ratio = None
                    self.candidate_pe_confidence = 0
                    status = "trusted_confirmed"
                else:
                    if self.candidate_pe_ratio is None:
                        self.candidate_pe_ratio = implied_pe
                        self.candidate_pe_confidence = 1
                        status = "candidate_started"
                    else:
                        candidate_diff = self._relative_difference(implied_pe, self.candidate_pe_ratio)
                        if candidate_diff <= self.consistency_tolerance:
                            weight = min(max(self.candidate_pe_confidence, 1), 5)
                            self.candidate_pe_ratio = ((self.candidate_pe_ratio * weight) + implied_pe) / (weight + 1)
                            self.candidate_pe_confidence += 1
                            status = "candidate_confirmed"
                            if self.candidate_pe_confidence >= self.replacement_confirmations:
                                self.trusted_pe_ratio = self.candidate_pe_ratio
                                self.trusted_pe_confidence = max(
                                    self.min_learned_confidence,
                                    self.candidate_pe_confidence,
                                )
                                self.candidate_pe_ratio = None
                                self.candidate_pe_confidence = 0
                                status = "candidate_promoted"
                        else:
                            self.candidate_pe_ratio = implied_pe
                            self.candidate_pe_confidence = 1
                            status = "candidate_restarted"
            trusted_pe = float(self.trusted_pe_ratio)
            trusted_confidence = int(self.trusted_pe_confidence)

        if trusted_pe is not None and self.last_earnings_value is not None:
            source = "learned_pe" if self.configured_pe_ratio is None else "configured"
            self.set_fair_value(round(trusted_pe * self.last_earnings_value), source=source)

        return LearnedPEUpdate(
            status=status,
            implied_pe=implied_pe,
            trusted_pe=float(trusted_pe),
            trusted_confidence=int(trusted_confidence),
            candidate_pe=self.candidate_pe_ratio,
            candidate_confidence=int(self.candidate_pe_confidence),
            reference_price=float(reference_price),
            sample_count=int(sample_count),
            trading_ready=self.can_trade_from_pe,
        )

    @property
    def learned_pe_ratio(self) -> float | None:
        return self.trusted_pe_ratio

    @property
    def learned_pe_confidence(self) -> int:
        return self.trusted_pe_confidence

    @staticmethod
    def _relative_difference(left: float, right: float) -> float:
        return abs(left - right) / max(abs(right), 1e-9)

    def handles_news(self, news_release: dict) -> bool:
        if news_release.get("kind") != "structured":
            return False
        new_data = news_release.get("new_data") or {}
        if new_data.get("structured_subtype") != "earnings":
            return False
        asset = str(new_data.get("asset") or news_release.get("symbol") or "").upper()
        return asset == "A"


class QuoteEngine:
    """Converts the latest fair value, inventory, and book into target quotes."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def compute_quotes(
        self,
        fair_value: int | None,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        can_trade: bool,
        reason_if_blocked: str,
        quote_size: int | None = None,
    ) -> QuotePlan:
        if not can_trade:
            return QuotePlan(
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason=reason_if_blocked,
            )

        if fair_value is None:
            return QuotePlan(
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for the first usable A fair value.",
            )

        # Treat current inventory plus same-side resting orders as committed exposure.
        quote_size = self.risk.quote_size if quote_size is None else int(quote_size)
        allowed_buy = max(0, self.risk.max_position - (inventory + buy_exposure))
        allowed_sell = max(0, self.risk.max_position + inventory - sell_exposure)

        # Positive inventory should push the reservation price lower so the bot
        # naturally leans toward selling inventory back down, and vice versa.
        reservation = fair_value - (self.risk.inventory_skew * inventory)
        bid_px = int(floor(reservation - self.risk.min_edge))
        ask_px = int(ceil(reservation + self.risk.min_edge))

        if book.best_ask is not None:
            bid_px = min(bid_px, book.best_ask.px - 1)
        if book.best_bid is not None:
            ask_px = max(ask_px, book.best_bid.px + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        aggressive_actions: list[DesiredOrder] = []
        desired_bid: DesiredOrder | None = None
        desired_ask: DesiredOrder | None = None

        if book.best_ask is not None and allowed_buy > 0:
            if book.best_ask.px <= fair_value - self.risk.take_edge:
                aggressive_actions.append(
                    DesiredOrder(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(quote_size, allowed_buy, book.best_ask.qty),
                        aggressive=True,
                        reason="best ask is cheap versus fair value",
                    )
                )

        if book.best_bid is not None and allowed_sell > 0:
            if book.best_bid.px >= fair_value + self.risk.take_edge:
                aggressive_actions.append(
                    DesiredOrder(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(quote_size, allowed_sell, book.best_bid.qty),
                        aggressive=True,
                        reason="best bid is rich versus fair value",
                    )
                )

        aggressive_sides = {action.side for action in aggressive_actions}
        if allowed_buy > 0 and "BUY" not in aggressive_sides:
            desired_bid = DesiredOrder(
                side="BUY",
                px=bid_px,
                qty=min(quote_size, allowed_buy),
                aggressive=False,
                reason="passive buy quote around reservation price",
            )
        if allowed_sell > 0 and "SELL" not in aggressive_sides:
            desired_ask = DesiredOrder(
                side="SELL",
                px=ask_px,
                qty=min(quote_size, allowed_sell),
                aggressive=False,
                reason="passive sell quote around reservation price",
            )

        return QuotePlan(
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=tuple(aggressive_actions),
            observe_only=False,
            reason="ready",
        )


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
    """Owns the valuation model, quote engine, and order manager for market A."""

    def __init__(
        self,
        a_config: AConfig,
        risk: RiskConfig,
        restored_orders: Iterable[ManagedOrder] = (),
        recovered_fair_value: int | None = None,
        learned_pe_ratio: float | None = None,
        learned_pe_confidence: int = 0,
    ):
        starting_fair = recovered_fair_value
        if starting_fair is None:
            starting_fair = a_config.initial_fair_value

        self.a_config = a_config
        self.risk = risk
        self.valuation = AValuationModel(
            a_config=a_config,
            learned_pe_ratio=learned_pe_ratio,
            learned_pe_confidence=learned_pe_confidence,
            initial_fair_value=starting_fair,
        )
        if recovered_fair_value is not None:
            self.valuation.last_source = "journal"

        self.quote_engine = QuoteEngine(risk)
        self.order_manager = OrderManager(symbol="A", risk=risk)
        self.inventory = 0
        self.book = BookSnapshot()
        self.current_round_earnings_seen = False
        self.pending_pe_calibration: PECalibrationJob | None = None
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
    def effective_pe_ratio(self) -> float | None:
        return self.valuation.effective_pe_ratio

    @property
    def learned_pe_ratio(self) -> float | None:
        return self.valuation.learned_pe_ratio

    @property
    def learned_pe_confidence(self) -> int:
        return self.valuation.learned_pe_confidence

    def set_inventory(self, inventory: int) -> None:
        self.inventory = int(inventory)

    def on_book_update(self, symbol: str, book) -> bool:
        return self.on_book_update_at(symbol, book, self._now_ms())

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != "A":
            return False
        self.book = BookSnapshot.from_order_book(book)
        self._capture_mid_sample(now_ms)
        return True

    def on_news(self, news_release: dict, now_ms: int) -> NewsReaction:
        if not self.valuation.handles_news(news_release):
            return NewsReaction(relevant=False, fair_value_updated=False, started_pe_calibration=False)

        earnings_value = float(news_release["new_data"]["value"])
        self.current_round_earnings_seen = True
        fair_updated = self.valuation.on_earnings_release(earnings_value)
        started_pe_calibration = False

        if self.valuation.configured_pe_ratio is None:
            self.pending_pe_calibration = PECalibrationJob(
                earnings_value=earnings_value,
                started_ms=now_ms,
                sample_start_ms=now_ms + self.a_config.pe_learning_delay_ms,
                finalize_ms=now_ms + self.a_config.pe_learning_delay_ms + self.a_config.pe_learning_sample_window_ms,
            )
            started_pe_calibration = True

        return NewsReaction(
            relevant=True,
            fair_value_updated=fair_updated,
            started_pe_calibration=started_pe_calibration,
            earnings_value=earnings_value,
        )

    def maybe_finalize_pe_learning(self, now_ms: int) -> LearnedPEUpdate | None:
        if self.pending_pe_calibration is None:
            return None

        self._capture_mid_sample(now_ms)
        pending = self.pending_pe_calibration
        if now_ms < pending.finalize_ms:
            return None
        if len(pending.samples) < self.a_config.pe_learning_min_samples:
            return None
        if pending.earnings_value == 0:
            self.pending_pe_calibration = None
            return None

        reference_price = float(median(pending.samples))
        implied_pe = reference_price / pending.earnings_value
        update = self.valuation.apply_learned_pe(
            implied_pe,
            reference_price=reference_price,
            sample_count=len(pending.samples),
        )
        self.pending_pe_calibration = None
        return update

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

    def compute_quotes(self) -> QuotePlan:
        if not self.valuation.can_trade_from_pe:
            return self.quote_engine.compute_quotes(
                fair_value=None,
                inventory=self.inventory,
                book=self.book,
                buy_exposure=self.order_manager.buy_exposure(),
                sell_exposure=self.order_manager.sell_exposure(),
                can_trade=False,
                reason_if_blocked=self.valuation.pe_status_reason or "A valuation is not ready yet.",
            )
        reason_if_blocked = "Waiting for recovered A orders to be cancelled."
        return self.quote_engine.compute_quotes(
            fair_value=self.valuation.fair_value,
            inventory=self.inventory,
            book=self.book,
            buy_exposure=self.order_manager.buy_exposure(),
            sell_exposure=self.order_manager.sell_exposure(),
            can_trade=not self.recovery_active,
            reason_if_blocked=reason_if_blocked,
            quote_size=self._active_quote_size(),
        )

    def _maybe_finish_recovery(self) -> None:
        if self.recovery_active and not self.recovery_pending:
            self.on_recovery_complete()

    def _active_quote_size(self) -> int:
        if self.valuation.configured_pe_ratio is not None:
            return self.risk.quote_size
        if self.current_round_earnings_seen:
            return self.risk.quote_size
        if self.valuation.can_trade_from_pe:
            return max(1, self.risk.quote_size // 2)
        return self.risk.quote_size

    def _capture_mid_sample(self, now_ms: int | None = None) -> None:
        if self.pending_pe_calibration is None:
            return
        if now_ms is None:
            now_ms = self._now_ms()
        if now_ms < self.pending_pe_calibration.sample_start_ms:
            return
        current_mid = self._current_mid()
        if current_mid is None:
            return
        self.pending_pe_calibration.samples.append(current_mid)

    def _current_mid(self) -> float | None:
        if self.book.best_bid is None or self.book.best_ask is None:
            return None
        return (self.book.best_bid.px + self.book.best_ask.px) / 2.0

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
