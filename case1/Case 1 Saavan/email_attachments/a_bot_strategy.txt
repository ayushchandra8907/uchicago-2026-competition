from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, floor, sqrt
from statistics import median
import time
from typing import Iterable, Literal

from a_bot_config import AConfig, RiskConfig


SideName = Literal["BUY", "SELL"]
OverlayName = Literal["earnings", "mm"]
ModeName = Literal[
    "OPENING_MICRO_MM",
    "PRE_NEWS_PULLBACK",
    "POST_EARNINGS_SHOCK",
    "MULTIPLIER_DISCOVERY",
    "NEWS_CAUTIOUS_MM",
    "UNWIND",
    "STEADY_MM",
]

DAY_TICKS = 450
DAY_MS = 90_000
EARNINGS_TICKS = (150, 300)
TICK_MS = 200


@dataclass(frozen=True)
class BookLevel:
    px: int
    qty: int


@dataclass(frozen=True)
class BookSnapshot:
    best_bid: BookLevel | None = None
    best_ask: BookLevel | None = None
    bid_levels: tuple[BookLevel, ...] = ()
    ask_levels: tuple[BookLevel, ...] = ()

    @classmethod
    def from_order_book(cls, book, depth_levels: int = 10) -> "BookSnapshot":
        bids = [(int(px), int(qty)) for px, qty in book.bids.items() if qty > 0]
        asks = [(int(px), int(qty)) for px, qty in book.asks.items() if qty > 0]
        best_bid_level = max(bids, key=lambda level: level[0]) if bids else None
        best_ask_level = min(asks, key=lambda level: level[0]) if asks else None
        best_bid = None if best_bid_level is None else BookLevel(px=best_bid_level[0], qty=best_bid_level[1])
        best_ask = None if best_ask_level is None else BookLevel(px=best_ask_level[0], qty=best_ask_level[1])
        sorted_bids = tuple(BookLevel(px=px, qty=qty) for px, qty in sorted(bids, key=lambda level: level[0], reverse=True)[:depth_levels])
        sorted_asks = tuple(BookLevel(px=px, qty=qty) for px, qty in sorted(asks, key=lambda level: level[0])[:depth_levels])
        return cls(best_bid=best_bid, best_ask=best_ask, bid_levels=sorted_bids, ask_levels=sorted_asks)

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

    @property
    def microprice(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        total_qty = self.best_bid.qty + self.best_ask.qty
        if total_qty <= 0:
            return None
        return ((self.best_ask.px * self.best_bid.qty) + (self.best_bid.px * self.best_ask.qty)) / total_qty

    @property
    def top_of_book_imbalance(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        total_qty = self.best_bid.qty + self.best_ask.qty
        if total_qty <= 0:
            return None
        return (self.best_bid.qty - self.best_ask.qty) / total_qty


@dataclass(frozen=True)
class DesiredOrder:
    side: SideName
    px: int
    qty: int
    overlay: OverlayName = "mm"
    aggressive: bool = False
    reason: str = ""
    intent: str = ""
    mode_at_submit: str = ""
    evaluation_reason: str = ""


@dataclass(frozen=True)
class QuotePlan:
    mode: ModeName
    bid: DesiredOrder | None
    ask: DesiredOrder | None
    aggressive_actions: tuple[DesiredOrder, ...]
    observe_only: bool
    reason: str


@dataclass(frozen=True)
class OverlayPlan:
    overlay: OverlayName
    bid: DesiredOrder | None
    ask: DesiredOrder | None
    aggressive_actions: tuple[DesiredOrder, ...]
    reason: str


@dataclass
class ManagedOrder:
    order_id: str
    side: SideName
    px: int
    qty: int
    remaining_qty: int
    submitted_ms: int
    overlay: OverlayName = "mm"
    aggressive: bool = False
    cancel_pending: bool = False
    restored: bool = False
    intent: str = ""
    mode_at_submit: str = ""
    evaluation_reason: str = ""

    @property
    def is_active(self) -> bool:
        return self.remaining_qty > 0 and not self.cancel_pending


@dataclass(frozen=True)
class CancelCommand:
    order_id: str
    side: SideName
    reason: str


@dataclass(frozen=True)
class PlaceCommand:
    side: SideName
    px: int
    qty: int
    overlay: OverlayName
    aggressive: bool
    reason: str
    intent: str
    mode_at_submit: str
    evaluation_reason: str


@dataclass(frozen=True)
class SyncActions:
    cancels: tuple[CancelCommand, ...]
    placements: tuple[PlaceCommand, ...]


@dataclass(frozen=True)
class NewsReaction:
    relevant: bool
    fair_value_updated: bool
    note: str | None = None
    earnings_value: float | None = None
    old_fair_value: int | None = None
    new_fair_value: int | None = None
    shock_direction: int = 0
    shock_threshold: int | None = None
    tick: int | None = None


@dataclass(frozen=True)
class LearningEvent:
    status: Literal["trusted", "updated", "candidate", "replaced", "skipped"]
    estimate: float | None
    trusted_multiplier: float | None
    confidence: int
    method: str
    settled_mid: float | None
    reason: str
    tolerance: float | None = None


@dataclass
class DiscoveryWindow:
    started_ms: int
    min_lock_ms: int
    max_lock_ms: int
    next_sample_ms: int
    earnings_value: float
    samples: list[float] = field(default_factory=list)
    invalidated_reason: str | None = None


@dataclass(frozen=True)
class MultiplierUpdateResult:
    status: Literal["trusted", "updated", "candidate", "replaced"]
    estimate: float
    trusted_multiplier: float
    confidence: int
    tolerance: float
    reason: str


class AValuationModel:
    """Round-specific A valuation learned as price-units per unit of earnings."""

    def __init__(
        self,
        a_config: AConfig,
        *,
        initial_multiplier: float | None = None,
        initial_confidence: int = 0,
        initial_fair_value: int | None = None,
        initial_earnings_value: float | None = None,
    ):
        self.a_config = a_config
        self.trusted_multiplier = float(initial_multiplier) if initial_multiplier is not None else None
        self.multiplier_confidence = int(initial_confidence) if self.trusted_multiplier is not None else 0
        self.candidate_multiplier: float | None = None
        self.candidate_hits = 0
        self.fair_value = initial_fair_value
        self.last_earnings_value = initial_earnings_value
        self.last_source = "journal" if initial_fair_value is not None else "none"

        if self.fair_value is None and self.trusted_multiplier is not None and initial_earnings_value is not None:
            self.fair_value = self.fair_from_earnings(initial_earnings_value, multiplier=self.trusted_multiplier)
            self.last_source = "journal_multiplier"

    @property
    def has_fair_value(self) -> bool:
        return self.fair_value is not None

    @property
    def has_multiplier(self) -> bool:
        return self.trusted_multiplier is not None

    def fair_from_earnings(self, earnings_value: float, *, multiplier: float | None = None) -> int | None:
        chosen_multiplier = self.trusted_multiplier if multiplier is None else multiplier
        if chosen_multiplier is None:
            return None
        return round(float(earnings_value) * chosen_multiplier)

    def on_structured_earnings(self, earnings_value: float) -> tuple[int | None, int | None]:
        old_fair = self.fair_value
        self.last_earnings_value = float(earnings_value)
        self.fair_value = self.fair_from_earnings(self.last_earnings_value)
        self.last_source = "trusted_multiplier" if self.fair_value is not None else "earnings_pending"
        return old_fair, self.fair_value

    def current_tolerance(self) -> float | None:
        if self.trusted_multiplier is None:
            return None
        scaled_fraction = self.a_config.calibration_tolerance_fraction / max(1.0, sqrt(max(1, self.multiplier_confidence)))
        fraction = max(self.a_config.calibration_min_tolerance_fraction, scaled_fraction)
        return max(8.0, self.trusted_multiplier * fraction)

    def absorb_estimate(self, estimate: float) -> MultiplierUpdateResult:
        estimate = float(estimate)
        if estimate <= 0:
            raise ValueError("Multiplier estimates must be positive.")

        if self.trusted_multiplier is None:
            self.trusted_multiplier = estimate
            self.multiplier_confidence = 1
            self.candidate_multiplier = None
            self.candidate_hits = 0
            if self.last_earnings_value is not None:
                self.fair_value = self.fair_from_earnings(self.last_earnings_value)
            self.last_source = "learned_multiplier"
            return MultiplierUpdateResult(
                status="trusted",
                estimate=estimate,
                trusted_multiplier=self.trusted_multiplier,
                confidence=self.multiplier_confidence,
                tolerance=self.current_tolerance() or 0.0,
                reason="Bootstrapped the round multiplier from settled post-earnings price levels.",
            )

        tolerance = self.current_tolerance() or 0.0
        if abs(estimate - self.trusted_multiplier) <= tolerance:
            blended = ((self.trusted_multiplier * self.multiplier_confidence) + estimate) / (self.multiplier_confidence + 1)
            self.trusted_multiplier = blended
            self.multiplier_confidence += 1
            self.candidate_multiplier = None
            self.candidate_hits = 0
            if self.last_earnings_value is not None:
                self.fair_value = self.fair_from_earnings(self.last_earnings_value)
            self.last_source = "learned_multiplier"
            return MultiplierUpdateResult(
                status="updated",
                estimate=estimate,
                trusted_multiplier=self.trusted_multiplier,
                confidence=self.multiplier_confidence,
                tolerance=self.current_tolerance() or tolerance,
                reason="Reconfirmed the round multiplier inside tolerance and tightened the estimate.",
            )

        if self.candidate_multiplier is None or abs(estimate - self.candidate_multiplier) > tolerance:
            self.candidate_multiplier = estimate
            self.candidate_hits = 1
            return MultiplierUpdateResult(
                status="candidate",
                estimate=estimate,
                trusted_multiplier=self.trusted_multiplier,
                confidence=self.multiplier_confidence,
                tolerance=tolerance,
                reason="Observed a divergent multiplier candidate; waiting for another structured-earnings confirmation before replacing the trusted value.",
            )

        self.candidate_multiplier = ((self.candidate_multiplier * self.candidate_hits) + estimate) / (self.candidate_hits + 1)
        self.candidate_hits += 1
        if self.candidate_hits < self.a_config.candidate_confirmations:
            return MultiplierUpdateResult(
                status="candidate",
                estimate=estimate,
                trusted_multiplier=self.trusted_multiplier,
                confidence=self.multiplier_confidence,
                tolerance=tolerance,
                reason="Divergent candidate repeated but still needs one more confirmation before replacing the trusted value.",
            )

        self.trusted_multiplier = self.candidate_multiplier
        self.multiplier_confidence = 1
        self.candidate_multiplier = None
        self.candidate_hits = 0
        if self.last_earnings_value is not None:
            self.fair_value = self.fair_from_earnings(self.last_earnings_value)
        self.last_source = "replaced_multiplier"
        return MultiplierUpdateResult(
            status="replaced",
            estimate=estimate,
            trusted_multiplier=self.trusted_multiplier,
            confidence=self.multiplier_confidence,
            tolerance=self.current_tolerance() or tolerance,
            reason="Repeated divergent estimates replaced the trusted round multiplier.",
        )


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
        unwind_aggressive: bool = False,
    ) -> QuotePlan:
        if not can_trade:
            return QuotePlan(mode=mode, bid=None, ask=None, aggressive_actions=(), observe_only=True, reason=reason_if_blocked)

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

        if mode in {"MULTIPLIER_DISCOVERY", "NEWS_CAUTIOUS_MM"}:
            return self._cautious_quotes(mode, fair_value, inventory, book, buy_exposure, sell_exposure)

        if fair_value is None:
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for a clean A multiplier estimate.",
            )

        if mode == "POST_EARNINGS_SHOCK":
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
            return self._unwind_quotes(
                mode,
                fair_value,
                inventory,
                book,
                buy_exposure,
                sell_exposure,
                aggressive_allowed=unwind_aggressive,
            )

        return self._steady_quotes(mode, fair_value, inventory, book, buy_exposure, sell_exposure)

    @staticmethod
    def _desired(
        *,
        side: SideName,
        px: int,
        qty: int,
        overlay: OverlayName,
        aggressive: bool,
        reason: str,
        intent: str,
        mode: ModeName,
    ) -> DesiredOrder:
        return DesiredOrder(
            side=side,
            px=px,
            qty=qty,
            overlay=overlay,
            aggressive=aggressive,
            reason=reason,
            intent=intent,
            mode_at_submit=mode,
            evaluation_reason=reason,
        )

    def cap_for_mode(self, mode: ModeName) -> int:
        if mode == "OPENING_MICRO_MM":
            return self.a_config.opening_max_position
        if mode == "MULTIPLIER_DISCOVERY":
            return self.a_config.discovery_max_position
        if mode == "NEWS_CAUTIOUS_MM":
            return self.a_config.news_caution_max_position
        if mode == "POST_EARNINGS_SHOCK":
            return self.a_config.shock_max_position
        if mode == "PRE_NEWS_PULLBACK":
            return 0
        return self.a_config.steady_max_position

    @staticmethod
    def clamp_allowed_sizes(
        overlay_allowed: tuple[int, int],
        global_allowed: tuple[int, int],
    ) -> tuple[int, int]:
        return min(overlay_allowed[0], global_allowed[0]), min(overlay_allowed[1], global_allowed[1])

    def allowed_size_for_mode(
        self,
        mode: ModeName,
        inventory: int,
        buy_exposure: int,
        sell_exposure: int,
    ) -> tuple[int, int]:
        cap = self.cap_for_mode(mode)
        if cap <= 0:
            return 0, 0
        return self._allowed_size(inventory, buy_exposure, sell_exposure, cap=cap)

    def _opening_quotes(
        self,
        mode: ModeName,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        overlay: OverlayName = "mm",
        cap_override: int | None = None,
        quote_size_override: int | None = None,
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
        quote_size = self.a_config.opening_quote_size if quote_size_override is None else max(0, int(quote_size_override))
        cap = self.a_config.opening_max_position if cap_override is None else max(0, int(cap_override))
        allowed_buy, allowed_sell = self._allowed_size(inventory, buy_exposure, sell_exposure, cap=cap)

        return QuotePlan(
            mode=mode,
            bid=self._desired(
                side="BUY",
                px=bid_px,
                qty=min(quote_size, allowed_buy),
                overlay=overlay,
                aggressive=False,
                reason="opening micro-mm bid around live mid",
                intent="opening_mm",
                mode=mode,
            )
            if allowed_buy > 0
            else None,
            ask=self._desired(
                side="SELL",
                px=ask_px,
                qty=min(quote_size, allowed_sell),
                overlay=overlay,
                aggressive=False,
                reason="opening micro-mm ask around live mid",
                intent="opening_mm",
                mode=mode,
            )
            if allowed_sell > 0
            else None,
            aggressive_actions=(),
            observe_only=False,
            reason="opening micro-mm",
        )

    def _cautious_quotes(
        self,
        mode: ModeName,
        fair_value: int | None,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        overlay: OverlayName = "mm",
        max_position_override: int | None = None,
        quote_size_override: int | None = None,
        half_spread_override: int | None = None,
    ) -> QuotePlan:
        anchor = book.mid
        if anchor is None and fair_value is not None:
            anchor = float(fair_value)
        if anchor is None:
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="waiting for a usable A anchor during cautious quoting",
            )

        if mode == "MULTIPLIER_DISCOVERY":
            half_spread_ticks = self.a_config.discovery_half_spread_ticks
            quote_size = self.a_config.discovery_quote_size
            max_position = self.a_config.discovery_max_position
            intent = "multiplier_discovery_mm"
            reason = "tiny quotes while the post-earnings multiplier is still settling"
        else:
            half_spread_ticks = self.a_config.news_caution_half_spread_ticks
            quote_size = self.a_config.news_caution_quote_size
            max_position = self.a_config.news_caution_max_position
            intent = "news_cautious_mm"
            reason = "tiny quotes while A-specific news risk is elevated"

        if half_spread_override is not None:
            half_spread_ticks = max(0, int(half_spread_override))
        if quote_size_override is not None:
            quote_size = max(0, int(quote_size_override))
        if max_position_override is not None:
            max_position = max(0, int(max_position_override))

        bid_px = int(floor(anchor - half_spread_ticks))
        ask_px = int(ceil(anchor + half_spread_ticks))
        if book.best_bid is not None:
            bid_px = min(bid_px, book.best_bid.px - 1)
        if book.best_ask is not None:
            ask_px = max(ask_px, book.best_ask.px + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=max_position,
        )

        return QuotePlan(
            mode=mode,
            bid=self._desired(
                side="BUY",
                px=bid_px,
                qty=min(quote_size, allowed_buy),
                overlay=overlay,
                aggressive=False,
                reason=reason,
                intent=intent,
                mode=mode,
            )
            if allowed_buy > 0
            else None,
            ask=self._desired(
                side="SELL",
                px=ask_px,
                qty=min(quote_size, allowed_sell),
                overlay=overlay,
                aggressive=False,
                reason=reason,
                intent=intent,
                mode=mode,
            )
            if allowed_sell > 0
            else None,
            aggressive_actions=(),
            observe_only=False,
            reason="cautious quoting",
        )

    def _steady_passive_qty(self, side: SideName, inventory: int, allowed: int, *, quote_size: int | None = None) -> int:
        if allowed <= 0:
            return 0
        base_quote_size = self.a_config.steady_quote_size if quote_size is None else max(0, int(quote_size))
        qty = min(base_quote_size, allowed)
        worsening_side: SideName | None = None
        if inventory > 0:
            worsening_side = "BUY"
        elif inventory < 0:
            worsening_side = "SELL"
        if side != worsening_side:
            return qty

        abs_inventory = abs(inventory)
        if abs_inventory >= self.a_config.unwind_exit_position or abs_inventory >= self.a_config.steady_passive_reduce_full:
            return min(1, allowed)
        if abs_inventory >= self.a_config.steady_passive_reduce_start:
            return min(max(1, base_quote_size - 1), allowed)
        return qty

    def _steady_quotes(
        self,
        mode: ModeName,
        fair_value: int,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        overlay: OverlayName = "mm",
        cap_override: int | None = None,
        quote_size_override: int | None = None,
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        quote_size = self.a_config.steady_quote_size if quote_size_override is None else max(0, int(quote_size_override))
        cap = self.a_config.steady_max_position if cap_override is None else max(0, int(cap_override))
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=cap,
        )

        take_edge = max(self.a_config.steady_take_min_edge, self.a_config.steady_half_spread_ticks)
        if abs(inventory) >= self.a_config.steady_take_inventory_guard:
            take_edge = max(take_edge, self.a_config.steady_take_large_inventory_edge)
        allow_aggressive_takes = abs(inventory) < self.a_config.unwind_exit_position
        can_take_buy = allow_aggressive_takes and inventory < self.a_config.steady_take_inventory_guard
        can_take_sell = allow_aggressive_takes and inventory > -self.a_config.steady_take_inventory_guard

        if can_take_buy and book.best_ask is not None and allowed_buy > 0 and book.best_ask.px <= fair_value - take_edge:
            aggressive_actions.append(
                self._desired(
                    side="BUY",
                    px=book.best_ask.px,
                    qty=min(quote_size, allowed_buy, book.best_ask.qty),
                    overlay=overlay,
                    aggressive=True,
                    reason="steady-state buy through stale ask below fair",
                    intent="steady_take",
                    mode=mode,
                )
            )
        if can_take_sell and book.best_bid is not None and allowed_sell > 0 and book.best_bid.px >= fair_value + take_edge:
            aggressive_actions.append(
                self._desired(
                    side="SELL",
                    px=book.best_bid.px,
                    qty=min(quote_size, allowed_sell, book.best_bid.qty),
                    overlay=overlay,
                    aggressive=True,
                    reason="steady-state sell through stale bid above fair",
                    intent="steady_take",
                    mode=mode,
                )
            )

        reservation = fair_value - (self.a_config.steady_inventory_skew * inventory)
        bid_px = int(floor(reservation - self.a_config.steady_half_spread_ticks))
        ask_px = int(ceil(reservation + self.a_config.steady_half_spread_ticks))
        bid_px, ask_px = self._clamp_inside_book(bid_px, ask_px, book)
        aggressive_sides = {action.side for action in aggressive_actions}
        passive_bid_qty = self._steady_passive_qty("BUY", inventory, allowed_buy, quote_size=quote_size)
        passive_ask_qty = self._steady_passive_qty("SELL", inventory, allowed_sell, quote_size=quote_size)

        return QuotePlan(
            mode=mode,
            bid=self._desired(
                side="BUY",
                px=bid_px,
                qty=passive_bid_qty,
                overlay=overlay,
                aggressive=False,
                reason="steady-state bid around learned fair",
                intent="steady_mm_passive",
                mode=mode,
            )
            if passive_bid_qty > 0 and "BUY" not in aggressive_sides
            else None,
            ask=self._desired(
                side="SELL",
                px=ask_px,
                qty=passive_ask_qty,
                overlay=overlay,
                aggressive=False,
                reason="steady-state ask around learned fair",
                intent="steady_mm_passive",
                mode=mode,
            )
            if passive_ask_qty > 0 and "SELL" not in aggressive_sides
            else None,
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
        overlay: OverlayName = "earnings",
        cap_override: int | None = None,
        quote_size_override: int | None = None,
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        quote_size = self.a_config.shock_quote_size if quote_size_override is None else max(0, int(quote_size_override))
        cap = self.a_config.shock_max_position if cap_override is None else max(0, int(cap_override))
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=cap,
        )
        desired_bid = None
        desired_ask = None
        half_spread = self.a_config.steady_half_spread_ticks

        if shock_direction > 0:
            if book.best_ask is not None and allowed_buy > 0 and book.best_ask.px <= fair_value - shock_threshold:
                aggressive_actions.append(
                    self._desired(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(quote_size, allowed_buy, book.best_ask.qty),
                        overlay=overlay,
                        aggressive=True,
                        reason="earnings upside shock buy through stale asks",
                        intent="post_earnings_shock_take",
                        mode=mode,
                    )
                )
            if allowed_sell > 0:
                ask_px = int(ceil(fair_value + half_spread))
                _, ask_px = self._clamp_inside_book(fair_value - 1, ask_px, book)
                desired_ask = self._desired(
                    side="SELL",
                    px=ask_px,
                    qty=min(quote_size, allowed_sell),
                    overlay=overlay,
                    aggressive=False,
                    reason="post-earnings ask to unwind long inventory",
                    intent="post_earnings_shock_unwind",
                    mode=mode,
                )
        elif shock_direction < 0:
            if book.best_bid is not None and allowed_sell > 0 and book.best_bid.px >= fair_value + shock_threshold:
                aggressive_actions.append(
                    self._desired(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(quote_size, allowed_sell, book.best_bid.qty),
                        overlay=overlay,
                        aggressive=True,
                        reason="earnings downside shock sell through stale bids",
                        intent="post_earnings_shock_take",
                        mode=mode,
                    )
                )
            if allowed_buy > 0:
                bid_px = int(floor(fair_value - half_spread))
                bid_px, _ = self._clamp_inside_book(bid_px, fair_value + 1, book)
                desired_bid = self._desired(
                    side="BUY",
                    px=bid_px,
                    qty=min(quote_size, allowed_buy),
                    overlay=overlay,
                    aggressive=False,
                    reason="post-earnings bid to unwind short inventory",
                    intent="post_earnings_shock_unwind",
                    mode=mode,
                )
        else:
            return self._steady_quotes("STEADY_MM", fair_value, inventory, book, buy_exposure, sell_exposure)

        return QuotePlan(
            mode=mode,
            bid=desired_bid,
            ask=desired_ask,
            aggressive_actions=tuple(aggressive_actions),
            observe_only=False,
            reason="post-earnings shock",
        )

    def _unwind_quotes(
        self,
        mode: ModeName,
        fair_value: int,
        inventory: int,
        book: BookSnapshot,
        buy_exposure: int,
        sell_exposure: int,
        *,
        aggressive_allowed: bool,
        overlay: OverlayName = "earnings",
        cap_override: int | None = None,
        quote_size_override: int | None = None,
        aggressive_intent: str = "unwind",
        passive_intent: str = "unwind",
    ) -> QuotePlan:
        aggressive_actions: list[DesiredOrder] = []
        quote_size = self.a_config.steady_quote_size if quote_size_override is None else max(0, int(quote_size_override))
        cap = self.a_config.steady_max_position if cap_override is None else max(0, int(cap_override))
        allowed_buy, allowed_sell = self._allowed_size(
            inventory,
            buy_exposure,
            sell_exposure,
            cap=cap,
        )
        passive_take_edge = max(self.a_config.steady_take_large_inventory_edge, self.a_config.steady_half_spread_ticks)

        if inventory > 0:
            if (
                book.best_bid is not None
                and allowed_sell > 0
                and (
                    (aggressive_allowed and book.best_bid.px >= fair_value)
                    or (not aggressive_allowed and book.best_bid.px >= fair_value + passive_take_edge)
                )
            ):
                aggressive_actions.append(
                    self._desired(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=min(quote_size, allowed_sell, book.best_bid.qty),
                        overlay=overlay,
                        aggressive=True,
                        reason="unwind long inventory into rich bid",
                        intent=aggressive_intent,
                        mode=mode,
                    )
                )
            ask_px = int(ceil((fair_value - (self.a_config.unwind_inventory_skew * inventory)) + 1))
            _, ask_px = self._clamp_inside_book(fair_value - 1, ask_px, book)
            return QuotePlan(
                mode=mode,
                bid=None,
                ask=self._desired(
                    side="SELL",
                    px=ask_px,
                    qty=min(quote_size, allowed_sell),
                    overlay=overlay,
                    aggressive=False,
                    reason="unwind ask to reduce long inventory",
                    intent=passive_intent,
                    mode=mode,
                )
                if allowed_sell > 0
                else None,
                aggressive_actions=tuple(aggressive_actions),
                observe_only=False,
                reason="unwind long inventory",
            )

        if inventory < 0:
            if (
                book.best_ask is not None
                and allowed_buy > 0
                and (
                    (aggressive_allowed and book.best_ask.px <= fair_value)
                    or (not aggressive_allowed and book.best_ask.px <= fair_value - passive_take_edge)
                )
            ):
                aggressive_actions.append(
                    self._desired(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=min(quote_size, allowed_buy, book.best_ask.qty),
                        overlay=overlay,
                        aggressive=True,
                        reason="unwind short inventory into cheap ask",
                        intent=aggressive_intent,
                        mode=mode,
                    )
                )
            bid_px = int(floor((fair_value - (self.a_config.unwind_inventory_skew * inventory)) - 1))
            bid_px, _ = self._clamp_inside_book(bid_px, fair_value + 1, book)
            return QuotePlan(
                mode=mode,
                bid=self._desired(
                    side="BUY",
                    px=bid_px,
                    qty=min(quote_size, allowed_buy),
                    overlay=overlay,
                    aggressive=False,
                    reason="unwind bid to reduce short inventory",
                    intent=passive_intent,
                    mode=mode,
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
        overlay: OverlayName = "mm",
        aggressive: bool = False,
        restored: bool = False,
        intent: str = "",
        mode_at_submit: str = "",
        evaluation_reason: str = "",
    ) -> ManagedOrder:
        order = ManagedOrder(
            order_id=order_id,
            side=side,
            px=px,
            qty=qty,
            remaining_qty=qty,
            submitted_ms=now_ms,
            overlay=overlay,
            aggressive=aggressive,
            restored=restored,
            intent=intent,
            mode_at_submit=mode_at_submit,
            evaluation_reason=evaluation_reason,
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

    def buy_exposure(self, overlay: OverlayName | None = None) -> int:
        order = self.live_order("BUY")
        if order is None or order.cancel_pending:
            return 0
        if overlay is not None and order.overlay != overlay:
            return 0
        return order.remaining_qty

    def sell_exposure(self, overlay: OverlayName | None = None) -> int:
        order = self.live_order("SELL")
        if order is None or order.cancel_pending:
            return 0
        if overlay is not None and order.overlay != overlay:
            return 0
        return order.remaining_qty

    def mark_cancel_requested(self, order_id: str, now_ms: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.cancel_pending = True
        self.last_action_ms[order.side] = now_ms
        return order

    def live_orders_snapshot(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for side in ("BUY", "SELL"):
            order = self.live_order(side)
            if order is None:
                continue
            rows.append(
                {
                    "order_id": order.order_id,
                    "side": order.side,
                    "px": order.px,
                    "qty": order.qty,
                    "remaining_qty": order.remaining_qty,
                    "submitted_ms": order.submitted_ms,
                    "overlay": order.overlay,
                    "aggressive": order.aggressive,
                    "cancel_pending": order.cancel_pending,
                    "restored": order.restored,
                    "intent": order.intent,
                    "mode_at_submit": order.mode_at_submit,
                    "evaluation_reason": order.evaluation_reason,
                }
            )
        return rows

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

            if live is not None and desired is None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    cancels.append(CancelCommand(order_id=live.order_id, side=side, reason="desired order removed for this side"))
                continue

            if live is not None and desired is not None:
                price_gap = abs(live.px - desired.px)
                stale = now_ms - live.submitted_ms >= self.risk.stale_quote_ms
                needs_reprice = (
                    live.remaining_qty != desired.qty
                    or live.overlay != desired.overlay
                    or live.aggressive != desired.aggressive
                    or stale
                    or live.aggressive
                    or desired.aggressive
                    or price_gap >= self.risk.passive_reprice_threshold_ticks
                )
                if needs_reprice and now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    cancel_reasons: list[str] = []
                    if live.remaining_qty != desired.qty:
                        cancel_reasons.append("qty changed")
                    if live.overlay != desired.overlay:
                        cancel_reasons.append("overlay changed")
                    if live.aggressive != desired.aggressive:
                        cancel_reasons.append("aggressiveness changed")
                    if stale:
                        cancel_reasons.append("quote became stale")
                    if price_gap >= self.risk.passive_reprice_threshold_ticks:
                        cancel_reasons.append(f"desired price moved by {price_gap} ticks")
                    if live.aggressive:
                        cancel_reasons.append("existing order is aggressive")
                    if desired.aggressive:
                        cancel_reasons.append("new order is aggressive")
                    cancels.append(
                        CancelCommand(
                            order_id=live.order_id,
                            side=side,
                            reason=", ".join(cancel_reasons) if cancel_reasons else "reprice",
                        )
                    )
                continue

            if live is None and desired is not None:
                if now_ms - self.last_action_ms[side] >= self.risk.reprice_cooldown_ms:
                    placements.append(
                        PlaceCommand(
                            side=side,
                            px=desired.px,
                            qty=desired.qty,
                            overlay=desired.overlay,
                            aggressive=desired.aggressive,
                            reason=desired.reason,
                            intent=desired.intent,
                            mode_at_submit=desired.mode_at_submit,
                            evaluation_reason=desired.evaluation_reason,
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
    """Owns multiplier learning, schedule tracking, quote generation, and recovery for A."""

    def __init__(
        self,
        a_config: AConfig,
        risk: RiskConfig,
        restored_orders: Iterable[ManagedOrder] = (),
        recovered_multiplier: float | None = None,
        recovered_multiplier_confidence: int = 0,
        recovered_fair_value: int | None = None,
        recovered_earnings_value: float | None = None,
        book_depth_levels: int = 10,
    ):
        starting_fair = recovered_fair_value if recovered_fair_value is not None else a_config.initial_fair_value
        starting_multiplier = recovered_multiplier if recovered_multiplier is not None else a_config.initial_multiplier
        self.a_config = a_config
        self.risk = risk
        self.book_depth_levels = max(1, book_depth_levels)
        self.valuation = AValuationModel(
            a_config=a_config,
            initial_multiplier=starting_multiplier,
            initial_confidence=recovered_multiplier_confidence,
            initial_fair_value=starting_fair,
            initial_earnings_value=recovered_earnings_value,
        )
        self.quote_engine = QuoteEngine(a_config, risk)
        self.order_manager = OrderManager(symbol="A", risk=risk)
        self.inventory = 0
        self.earnings_position = 0
        self.mm_position = 0
        self.book = BookSnapshot()
        self.mode: ModeName = "OPENING_MICRO_MM"
        self.startup_ms = self._now_ms()
        self.tick_anchor_tick: int | None = None
        self.tick_anchor_ms: int | None = None
        self.last_news_tick: int | None = None
        self.discovery_window: DiscoveryWindow | None = None
        self.news_caution_active = False
        self.shock_started_ms: int | None = None
        self.shock_direction = 0
        self.shock_threshold: int | None = None
        self.shock_target_fair: int | None = None
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.learning_events: list[LearningEvent] = []
        self.recovery_pending: set[str] = set()
        self.recovery_active = False
        self.unwind_active = False
        self.unwind_aggressive_active = False

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

    @property
    def trusted_multiplier(self) -> float | None:
        return self.valuation.trusted_multiplier

    @property
    def multiplier_confidence(self) -> int:
        return self.valuation.multiplier_confidence

    def set_inventory(self, inventory: int) -> None:
        inventory = int(inventory)
        self.inventory = inventory
        if (
            abs(inventory) >= self.a_config.unwind_entry_position
            or self.unwind_active
            or abs(self.earnings_position) > 0
            or self.mode in {"PRE_NEWS_PULLBACK", "POST_EARNINGS_SHOCK", "UNWIND"}
        ):
            self.earnings_position = inventory
            self.mm_position = 0
        else:
            self.mm_position = inventory
            self.earnings_position = 0
        self._ensure_position_invariant()
        self._refresh_unwind_state()

    def sync_inventory_from_exchange(self, inventory: int) -> None:
        inventory = int(inventory)
        delta = inventory - self.inventory
        self.inventory = inventory
        if delta == 0:
            self._ensure_position_invariant()
            self._refresh_unwind_state()
            return
        preferred_overlay = self._preferred_overlay_for_inventory_delta(inventory, delta)
        if preferred_overlay == "earnings":
            self.earnings_position += delta
        else:
            self.mm_position += delta
        self._ensure_position_invariant()
        self._refresh_unwind_state()

    def overlay_position(self, overlay: OverlayName) -> int:
        return self.earnings_position if overlay == "earnings" else self.mm_position

    def overlay_budgets(self, mode: ModeName | None = None) -> tuple[int, int, bool]:
        active_mode = self.mode if mode is None else mode
        budget_shift_active = active_mode in {"PRE_NEWS_PULLBACK", "POST_EARNINGS_SHOCK"}
        if budget_shift_active:
            return self.a_config.earnings_shift_budget, self.a_config.mm_shift_budget, True
        return self.a_config.earnings_base_budget, self.a_config.mm_base_budget, False

    def overlay_exposures(self, overlay: OverlayName) -> tuple[int, int]:
        return self.order_manager.buy_exposure(overlay), self.order_manager.sell_exposure(overlay)

    def global_allowed_size(self) -> tuple[int, int]:
        return self.quote_engine._allowed_size(
            self.inventory,
            self.order_manager.buy_exposure(),
            self.order_manager.sell_exposure(),
            cap=self.a_config.total_position_limit,
        )

    def overlay_allowed_size(self, overlay: OverlayName, *, mode_cap: int, budget: int) -> tuple[int, int]:
        cap = max(0, min(mode_cap, budget))
        overlay_buy_exposure, overlay_sell_exposure = self.overlay_exposures(overlay)
        overlay_allowed = self.quote_engine._allowed_size(
            self.overlay_position(overlay),
            overlay_buy_exposure,
            overlay_sell_exposure,
            cap=cap,
        )
        return self.quote_engine.clamp_allowed_sizes(overlay_allowed, self.global_allowed_size())

    def drain_learning_events(self) -> list[LearningEvent]:
        events = list(self.learning_events)
        self.learning_events.clear()
        return events

    def on_book_update(self, symbol: str, book) -> bool:
        return self.on_book_update_at(symbol, book, self._now_ms())

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != "A":
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        self._advance_discovery(now_ms)
        self.mode = self._determine_mode(now_ms)
        return True

    def on_news(self, news_release: dict, now_ms: int) -> NewsReaction:
        tick = news_release.get("tick")
        if isinstance(tick, int):
            self.tick_anchor_tick = tick
            self.tick_anchor_ms = now_ms
            self.last_news_tick = tick

        if self._is_a_unstructured_news(news_release):
            self.news_caution_active = True
            if self.discovery_window is not None and self.discovery_window.invalidated_reason is None:
                self.discovery_window.invalidated_reason = (
                    "calibration contaminated because unstructured A news arrived before the new multiplier locked"
                )
            self.mode = self._determine_mode(now_ms)
            return NewsReaction(
                relevant=True,
                fair_value_updated=False,
                note="Detected unstructured A news; switching into cautious quoting until the next structured A earnings reset.",
                tick=tick if isinstance(tick, int) else None,
            )

        if not self._handles_a_earnings(news_release):
            self.mode = self._determine_mode(now_ms)
            return NewsReaction(relevant=False, fair_value_updated=False, tick=tick if isinstance(tick, int) else None)

        earnings_value = float(news_release["new_data"]["value"])
        old_fair, new_fair = self.valuation.on_structured_earnings(earnings_value)
        self.news_caution_active = False
        self.discovery_window = DiscoveryWindow(
            started_ms=now_ms,
            min_lock_ms=now_ms + self.a_config.calibration_min_delay_ms,
            max_lock_ms=now_ms + self.a_config.calibration_max_delay_ms,
            next_sample_ms=now_ms + self.a_config.calibration_min_delay_ms,
            earnings_value=earnings_value,
        )

        if new_fair is not None:
            reference_fair = old_fair
            if reference_fair is None and self.book.mid is not None:
                reference_fair = round(self.book.mid)
            if reference_fair is None:
                reference_fair = new_fair
            move_size = abs(new_fair - reference_fair)
            self.shock_started_ms = now_ms
            self.shock_direction = 1 if new_fair > reference_fair else -1 if new_fair < reference_fair else 0
            self.shock_threshold = max(self.a_config.shock_take_min_edge, round(self.a_config.shock_take_fraction * move_size))
            self.shock_target_fair = new_fair
            note = (
                f"A earnings tick={tick} moved provisional fair from {old_fair} to {new_fair} "
                f"on earnings={earnings_value}; shock_direction={self.shock_direction} threshold={self.shock_threshold}"
            )
            fair_updated = True
        else:
            self.shock_started_ms = None
            self.shock_direction = 0
            self.shock_threshold = None
            self.shock_target_fair = None
            note = (
                f"Structured A earnings arrived with earnings={earnings_value}; "
                f"starting multiplier discovery after {self.a_config.calibration_min_delay_ms}ms."
            )
            fair_updated = False

        self.mode = self._determine_mode(now_ms)
        return NewsReaction(
            relevant=True,
            fair_value_updated=fair_updated,
            note=note,
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
        if order.overlay == "earnings":
            self.earnings_position += signed_qty
        else:
            self.mm_position += signed_qty
        self._ensure_position_invariant()
        self._refresh_unwind_state()
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms()
        if order_id in self.recovery_pending and order.remaining_qty == 0:
            self.recovery_pending.discard(order_id)
            self._maybe_finish_recovery()
        return order

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = self._now_ms() if now_ms is None else int(now_ms)

    def on_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.order_manager.handle_cancel_response(order_id, success)
        if order is None:
            return None
        if order_id in self.recovery_pending:
            if not success:
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

    def compute_quotes(self, now_ms: int | None = None) -> QuotePlan:
        if now_ms is None:
            now_ms = self._now_ms()
        self._advance_discovery(now_ms)
        self.mode = self._determine_mode(now_ms)
        if self.recovery_active:
            return QuotePlan(
                mode=self.mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for recovered A orders to be cancelled.",
            )

        earnings_plan = self._compute_earnings_overlay_plan(self.mode)
        mm_plan = self._compute_mm_overlay_plan(self.mode)
        return self._merge_overlay_plans(self.mode, earnings_plan, mm_plan)

    def _empty_overlay_plan(self, overlay: OverlayName, reason: str) -> OverlayPlan:
        return OverlayPlan(overlay=overlay, bid=None, ask=None, aggressive_actions=(), reason=reason)

    def _compute_mm_overlay_plan(self, mode: ModeName) -> OverlayPlan:
        _, mm_budget, _ = self.overlay_budgets(mode)
        if mm_budget <= 0:
            return self._empty_overlay_plan("mm", "MM budget temporarily shifted into earnings handling.")

        mm_inventory = self.mm_position
        mm_buy_exposure, mm_sell_exposure = self.overlay_exposures("mm")
        allowed_buy, allowed_sell = self.overlay_allowed_size(
            "mm",
            mode_cap=(
                self.a_config.opening_max_position if mode == "OPENING_MICRO_MM"
                else self.a_config.discovery_max_position if mode == "MULTIPLIER_DISCOVERY"
                else self.a_config.news_caution_max_position if mode == "NEWS_CAUTIOUS_MM"
                else self.a_config.steady_max_position
            ),
            budget=mm_budget,
        )

        if mode == "OPENING_MICRO_MM":
            plan = self.quote_engine._opening_quotes(
                mode,
                mm_inventory,
                self.book,
                mm_buy_exposure,
                mm_sell_exposure,
                overlay="mm",
                cap_override=min(self.a_config.opening_max_position, mm_budget),
            )
            return self._clamp_overlay_plan("mm", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        if mode == "MULTIPLIER_DISCOVERY":
            plan = self.quote_engine._cautious_quotes(
                mode,
                self.valuation.fair_value,
                mm_inventory,
                self.book,
                mm_buy_exposure,
                mm_sell_exposure,
                overlay="mm",
                max_position_override=min(self.a_config.discovery_max_position, mm_budget),
                quote_size_override=self.a_config.discovery_quote_size,
                half_spread_override=self.a_config.discovery_half_spread_ticks,
            )
            return self._clamp_overlay_plan("mm", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        if mode == "NEWS_CAUTIOUS_MM":
            plan = self.quote_engine._cautious_quotes(
                mode,
                self.valuation.fair_value,
                mm_inventory,
                self.book,
                mm_buy_exposure,
                mm_sell_exposure,
                overlay="mm",
                max_position_override=min(self.a_config.news_caution_max_position, mm_budget),
                quote_size_override=self.a_config.news_caution_quote_size,
                half_spread_override=self.a_config.news_caution_half_spread_ticks,
            )
            return self._clamp_overlay_plan("mm", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        if mode in {"STEADY_MM", "UNWIND"}:
            if self.valuation.fair_value is None:
                return self._empty_overlay_plan("mm", "Waiting for a clean A fair value before steady MM.")
            plan = self.quote_engine._steady_quotes(
                mode,
                self.valuation.fair_value,
                mm_inventory,
                self.book,
                mm_buy_exposure,
                mm_sell_exposure,
                overlay="mm",
                cap_override=min(self.a_config.steady_max_position, mm_budget),
                quote_size_override=self.a_config.steady_quote_size,
            )
            return self._clamp_overlay_plan("mm", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        return self._empty_overlay_plan("mm", "MM overlay idle in earnings-handling mode.")

    def _compute_earnings_overlay_plan(self, mode: ModeName) -> OverlayPlan:
        earnings_budget, _, _ = self.overlay_budgets(mode)
        if earnings_budget <= 0:
            return self._empty_overlay_plan("earnings", "Earnings overlay budget disabled.")

        earnings_inventory = self.earnings_position
        earnings_buy_exposure, earnings_sell_exposure = self.overlay_exposures("earnings")
        mode_cap = (
            0 if mode == "PRE_NEWS_PULLBACK"
            else self.a_config.shock_max_position if mode == "POST_EARNINGS_SHOCK"
            else earnings_budget if mode == "UNWIND"
            else 0
        )
        allowed_buy, allowed_sell = self.overlay_allowed_size("earnings", mode_cap=mode_cap, budget=earnings_budget)

        if mode == "PRE_NEWS_PULLBACK":
            return self._empty_overlay_plan("earnings", "flat before scheduled A earnings")

        if mode == "POST_EARNINGS_SHOCK":
            if self.valuation.fair_value is None:
                return self._empty_overlay_plan("earnings", "Waiting for a trusted fair value before shock trading.")
            plan = self.quote_engine._shock_quotes(
                mode,
                self.valuation.fair_value,
                earnings_inventory,
                self.book,
                earnings_buy_exposure,
                earnings_sell_exposure,
                shock_direction=self.shock_direction,
                shock_threshold=self.shock_threshold or self.a_config.shock_take_min_edge,
                overlay="earnings",
                cap_override=min(self.a_config.shock_max_position, earnings_budget),
                quote_size_override=self.a_config.shock_quote_size,
            )
            return self._clamp_overlay_plan("earnings", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        if mode == "UNWIND":
            if self.valuation.fair_value is None:
                return self._empty_overlay_plan("earnings", "Waiting for a trusted fair value before unwind.")
            plan = self.quote_engine._unwind_quotes(
                mode,
                self.valuation.fair_value,
                earnings_inventory,
                self.book,
                earnings_buy_exposure,
                earnings_sell_exposure,
                aggressive_allowed=self.unwind_aggressive_active,
                overlay="earnings",
                cap_override=earnings_budget,
                quote_size_override=self.a_config.steady_quote_size,
                aggressive_intent="unwind",
                passive_intent="unwind",
            )
            return self._clamp_overlay_plan("earnings", plan, allowed_buy=allowed_buy, allowed_sell=allowed_sell)

        return self._empty_overlay_plan("earnings", "earnings overlay idle outside earnings-active modes")

    def _clamp_overlay_plan(
        self,
        overlay: OverlayName,
        plan: QuotePlan,
        *,
        allowed_buy: int,
        allowed_sell: int,
    ) -> OverlayPlan:
        def clamp_order(order: DesiredOrder | None) -> DesiredOrder | None:
            if order is None:
                return None
            allowed = allowed_buy if order.side == "BUY" else allowed_sell
            qty = min(order.qty, allowed)
            if qty <= 0:
                return None
            return order if qty == order.qty else replace(order, qty=qty)

        return OverlayPlan(
            overlay=overlay,
            bid=clamp_order(plan.bid),
            ask=clamp_order(plan.ask),
            aggressive_actions=tuple(
                clamped
                for action in plan.aggressive_actions
                if (clamped := clamp_order(action)) is not None
            ),
            reason=plan.reason,
        )

    def _merge_overlay_plans(self, mode: ModeName, earnings_plan: OverlayPlan, mm_plan: OverlayPlan) -> QuotePlan:
        aggressive_by_side: dict[SideName, DesiredOrder] = {}
        for action in earnings_plan.aggressive_actions:
            aggressive_by_side[action.side] = action
        for action in mm_plan.aggressive_actions:
            aggressive_by_side.setdefault(action.side, action)

        def choose_passive(side: SideName) -> DesiredOrder | None:
            if side in aggressive_by_side:
                return None
            earnings_order = earnings_plan.bid if side == "BUY" else earnings_plan.ask
            mm_order = mm_plan.bid if side == "BUY" else mm_plan.ask
            if earnings_order is not None and self._order_reduces_position(side, self.earnings_position):
                return earnings_order
            if mm_order is not None:
                return mm_order
            return earnings_order

        bid = choose_passive("BUY")
        ask = choose_passive("SELL")
        aggressive_actions = tuple(
            action
            for side in ("BUY", "SELL")
            if (action := aggressive_by_side.get(side)) is not None
        )
        reason_parts = [part for part in (earnings_plan.reason, mm_plan.reason) if part]
        return QuotePlan(
            mode=mode,
            bid=bid,
            ask=ask,
            aggressive_actions=aggressive_actions,
            observe_only=False,
            reason=" | ".join(reason_parts) if reason_parts else mode.lower(),
        )

    def trace_state(self, now_ms: int) -> dict[str, object]:
        mode = self.mode
        buy_exposure = self.order_manager.buy_exposure()
        sell_exposure = self.order_manager.sell_exposure()
        allowed_buy, allowed_sell = self.global_allowed_size()
        discovery = None
        if self.discovery_window is not None:
            discovery = {
                "started_ms": self.discovery_window.started_ms,
                "min_lock_ms": self.discovery_window.min_lock_ms,
                "max_lock_ms": self.discovery_window.max_lock_ms,
                "next_sample_ms": self.discovery_window.next_sample_ms,
                "earnings_value": self.discovery_window.earnings_value,
                "sample_count": len(self.discovery_window.samples),
                "latest_sample": self.discovery_window.samples[-1] if self.discovery_window.samples else None,
                "invalidated_reason": self.discovery_window.invalidated_reason,
            }
        earnings_budget, mm_budget, budget_shift_active = self.overlay_budgets(mode)
        earnings_buy_exposure, earnings_sell_exposure = self.overlay_exposures("earnings")
        mm_buy_exposure, mm_sell_exposure = self.overlay_exposures("mm")
        earnings_allowed_buy, earnings_allowed_sell = self.overlay_allowed_size(
            "earnings",
            mode_cap=max(self.a_config.shock_max_position, earnings_budget),
            budget=earnings_budget,
        )
        mm_mode_cap = self.a_config.steady_max_position
        if mode == "OPENING_MICRO_MM":
            mm_mode_cap = self.a_config.opening_max_position
        elif mode == "MULTIPLIER_DISCOVERY":
            mm_mode_cap = self.a_config.discovery_max_position
        elif mode == "NEWS_CAUTIOUS_MM":
            mm_mode_cap = self.a_config.news_caution_max_position
        mm_allowed_buy, mm_allowed_sell = self.overlay_allowed_size(
            "mm",
            mode_cap=mm_mode_cap,
            budget=mm_budget,
        )
        position_cap = min(
            self.a_config.total_position_limit,
            max(
                0,
                earnings_budget if mode in {"PRE_NEWS_PULLBACK", "POST_EARNINGS_SHOCK", "UNWIND"} else 0,
                mm_budget,
            ),
        )
        live_orders = self.order_manager.live_orders_snapshot()
        return {
            "mode": mode,
            "fair_value": self.valuation.fair_value,
            "trusted_multiplier": self.valuation.trusted_multiplier,
            "multiplier_confidence": self.valuation.multiplier_confidence,
            "latest_earnings": self.valuation.last_earnings_value,
            "shock_direction": self.shock_direction,
            "shock_threshold": self.shock_threshold,
            "shock_target_fair": self.shock_target_fair,
            "shock_started_ms": self.shock_started_ms,
            "discovery_window": discovery,
            "news_caution_active": self.news_caution_active,
            "inventory": self.inventory,
            "earnings_position": self.earnings_position,
            "mm_position": self.mm_position,
            "earnings_budget": earnings_budget,
            "mm_budget": mm_budget,
            "budget_shift_active": budget_shift_active,
            "unwind_active": self.unwind_active,
            "unwind_aggressive_active": self.unwind_aggressive_active,
            "buy_exposure": buy_exposure,
            "sell_exposure": sell_exposure,
            "allowed_buy_size": allowed_buy,
            "allowed_sell_size": allowed_sell,
            "position_cap": position_cap,
            "overlay_exposures": {
                "earnings": {
                    "buy": earnings_buy_exposure,
                    "sell": earnings_sell_exposure,
                    "allowed_buy": earnings_allowed_buy,
                    "allowed_sell": earnings_allowed_sell,
                },
                "mm": {
                    "buy": mm_buy_exposure,
                    "sell": mm_sell_exposure,
                    "allowed_buy": mm_allowed_buy,
                    "allowed_sell": mm_allowed_sell,
                },
            },
            "live_orders": live_orders,
            "book": {
                "best_bid_px": None if self.book.best_bid is None else self.book.best_bid.px,
                "best_bid_qty": None if self.book.best_bid is None else self.book.best_bid.qty,
                "best_ask_px": None if self.book.best_ask is None else self.book.best_ask.px,
                "best_ask_qty": None if self.book.best_ask is None else self.book.best_ask.qty,
                "spread": self.book.spread,
                "mid": self.book.mid,
                "microprice": self.book.microprice,
                "top_of_book_imbalance": self.book.top_of_book_imbalance,
                "bid_levels": [{"px": level.px, "qty": level.qty} for level in self.book.bid_levels],
                "ask_levels": [{"px": level.px, "qty": level.qty} for level in self.book.ask_levels],
            },
            "last_trade_px": self.last_trade_px,
            "last_trade_qty": self.last_trade_qty,
            "last_trade_ms": self.last_trade_ms,
            "exchange_tick": self.last_news_tick,
            "ms_until_next_earnings": self.ms_until_next_scheduled_earnings(now_ms),
        }

    def ms_until_next_scheduled_earnings(self, now_ms: int | None = None) -> int | None:
        if now_ms is None:
            now_ms = self._now_ms()
        if self.tick_anchor_tick is not None and self.tick_anchor_ms is not None:
            estimated_day_tick = self._estimated_day_tick(now_ms)
            if estimated_day_tick is None:
                return None
            next_tick = self._next_earnings_day_tick(estimated_day_tick)
            delta_ticks = (next_tick - estimated_day_tick) % DAY_TICKS
            return 0 if delta_ticks == 0 else round(delta_ticks * TICK_MS)

        if not self.a_config.startup_assume_fresh_round:
            return None

        day_elapsed_ms = max(0, (now_ms - self.startup_ms) % DAY_MS)
        for earnings_ms in (30_000, 60_000):
            if day_elapsed_ms <= earnings_ms:
                return earnings_ms - day_elapsed_ms
        return DAY_MS - day_elapsed_ms + 30_000

    def _advance_discovery(self, now_ms: int) -> None:
        window = self.discovery_window
        if window is None:
            return

        if window.invalidated_reason is not None:
            self.learning_events.append(
                LearningEvent(
                    status="skipped",
                    estimate=None,
                    trusted_multiplier=self.valuation.trusted_multiplier,
                    confidence=self.valuation.multiplier_confidence,
                    method="skipped",
                    settled_mid=None,
                    reason=window.invalidated_reason,
                )
            )
            self.discovery_window = None
            return

        while window.next_sample_ms <= min(now_ms, window.max_lock_ms):
            if self.book.mid is not None:
                window.samples.append(float(self.book.mid))
            window.next_sample_ms += self.a_config.calibration_sample_period_ms

            if len(window.samples) >= 4:
                recent = window.samples[-4:]
                if max(recent) - min(recent) <= self.a_config.calibration_stability_band_ticks:
                    self._finalize_discovery(
                        settled_mid=float(median(recent)),
                        method="stable_level",
                        reason="Locked the round multiplier once the last four one-second samples fell inside the stability band.",
                    )
                    return

        if now_ms < window.max_lock_ms:
            return

        if not window.samples:
            self.learning_events.append(
                LearningEvent(
                    status="skipped",
                    estimate=None,
                    trusted_multiplier=self.valuation.trusted_multiplier,
                    confidence=self.valuation.multiplier_confidence,
                    method="skipped",
                    settled_mid=None,
                    reason="Timed out the multiplier discovery window without any usable A mid-price samples.",
                )
            )
            self.discovery_window = None
            return

        self._finalize_discovery(
            settled_mid=float(median(window.samples)),
            method="fallback_level",
            reason="Reached the end of the discovery window and accepted the median post-earnings level as the multiplier estimate.",
        )

    def _finalize_discovery(self, *, settled_mid: float, method: str, reason: str) -> None:
        window = self.discovery_window
        if window is None:
            return
        estimate = settled_mid / window.earnings_value if window.earnings_value > 0 else None
        self.discovery_window = None
        if estimate is None or estimate <= 0:
            self.learning_events.append(
                LearningEvent(
                    status="skipped",
                    estimate=None,
                    trusted_multiplier=self.valuation.trusted_multiplier,
                    confidence=self.valuation.multiplier_confidence,
                    method="skipped",
                    settled_mid=settled_mid,
                    reason="Could not extract a positive multiplier estimate from the settled post-earnings level.",
                )
            )
            return

        update = self.valuation.absorb_estimate(estimate)
        self.learning_events.append(
            LearningEvent(
                status=update.status,
                estimate=estimate,
                trusted_multiplier=update.trusted_multiplier,
                confidence=update.confidence,
                method=method,
                settled_mid=settled_mid,
                reason=reason if update.status != "candidate" else update.reason,
                tolerance=update.tolerance,
            )
        )

    def _determine_mode(self, now_ms: int) -> ModeName:
        if self._shock_active(now_ms):
            return "POST_EARNINGS_SHOCK"

        until_next_earnings = self.ms_until_next_scheduled_earnings(now_ms)
        if until_next_earnings is not None and until_next_earnings <= self.a_config.pre_news_pullback_ms:
            return "PRE_NEWS_PULLBACK"

        if self.news_caution_active:
            return "NEWS_CAUTIOUS_MM"

        if self.discovery_window is not None:
            return "MULTIPLIER_DISCOVERY"

        if self.valuation.fair_value is None:
            return "OPENING_MICRO_MM"

        self._refresh_unwind_state()
        if self.unwind_active:
            return "UNWIND"

        return "STEADY_MM"

    def _shock_active(self, now_ms: int) -> bool:
        if self.shock_started_ms is None or self.shock_target_fair is None:
            return False
        return now_ms - self.shock_started_ms < self.a_config.shock_window_ms

    def _refresh_unwind_state(self) -> None:
        abs_inventory = abs(self.earnings_position)
        if self.unwind_active:
            if abs_inventory <= self.a_config.unwind_exit_position:
                self.unwind_active = False
                self.unwind_aggressive_active = False
                return
        elif abs_inventory >= self.a_config.unwind_entry_position:
            self.unwind_active = True

        if not self.unwind_active:
            self.unwind_aggressive_active = False
            return

        if self.unwind_aggressive_active:
            if abs_inventory <= self.a_config.unwind_aggressive_exit:
                self.unwind_aggressive_active = False
            return
        if abs_inventory >= self.a_config.unwind_aggressive_entry:
            self.unwind_aggressive_active = True

    @staticmethod
    def _order_reduces_position(side: SideName, position: int) -> bool:
        if position > 0:
            return side == "SELL"
        if position < 0:
            return side == "BUY"
        return False

    def _preferred_overlay_for_inventory_delta(self, inventory: int, delta: int) -> OverlayName:
        if self.unwind_active or self.mode in {"PRE_NEWS_PULLBACK", "POST_EARNINGS_SHOCK", "UNWIND"}:
            return "earnings"
        if abs(self.earnings_position) > abs(self.mm_position):
            return "earnings"
        if abs(inventory) >= self.a_config.unwind_entry_position or abs(delta) >= self.a_config.shock_quote_size:
            return "earnings"
        return "mm"

    def _ensure_position_invariant(self) -> None:
        split_total = self.earnings_position + self.mm_position
        if split_total == self.inventory:
            return
        correction = self.inventory - split_total
        if self._preferred_overlay_for_inventory_delta(self.inventory, correction) == "earnings":
            self.earnings_position += correction
        else:
            self.mm_position += correction

    def _maybe_finish_recovery(self) -> None:
        if self.recovery_active and not self.recovery_pending:
            self.on_recovery_complete()

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
    def _is_a_unstructured_news(news_release: dict) -> bool:
        if news_release.get("kind") == "structured":
            return False
        symbol = str(news_release.get("symbol") or "").upper()
        if symbol == "A":
            return True
        raw_content = str(news_release.get("raw_content") or news_release.get("content") or "").upper()
        return " A " in f" {raw_content} "

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
