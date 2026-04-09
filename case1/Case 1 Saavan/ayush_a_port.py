from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace
from typing import Any

from a_bot_strategy import (
    BookSnapshot,
    DesiredOrder,
    LearningEvent,
    ManagedOrder,
    NewsReaction,
    OverlayName,
    QuotePlan,
)
from a_bot_config import RiskConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case1.ayush_work.marketA_v3.config import StrategyConfig as AyushStrategyConfig
from case1.ayush_work.marketA_v3.core.types import (
    BookLevel as AyushBookLevel,
    BookSnapshot as AyushBookSnapshot,
    ManagedOrder as AyushManagedOrder,
    NewsEvent as AyushNewsEvent,
    StrategySnapshot as AyushStrategySnapshot,
)
from case1.ayush_work.marketA_v3.market_A_strategy import AStrategy as AyushAStrategy


@dataclass(frozen=True)
class _DecisionEnvelope:
    trigger: str
    decision: Any


class AyushPortStrategy:
    """Faithful adapter for Ayush's standalone v3 A strategy."""

    symbol = "A"

    def __init__(
        self,
        *,
        risk: RiskConfig,
        restored_orders=(),
        recovered_multiplier: float | None = None,
        recovered_multiplier_confidence: int = 0,
        recovered_fair_value: int | None = None,
        recovered_earnings_value: float | None = None,
        initial_multiplier: float | None = None,
        initial_fair_value: int | None = None,
        book_depth_levels: int = 10,
    ) -> None:
        self.risk = risk
        self._config = AyushStrategyConfig()
        self._strategy = AyushAStrategy(self._config)
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.book = BookSnapshot()
        self.inventory = 0
        self._authoritative_inventory = 0
        self._last_position_update_ms: int | None = None
        self.last_trade_px: int | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_ms: int | None = None
        self.mode = "AYUSH_IDLE"
        self.news_caution_until_ms = 0
        self.recovery_pending: set[str] = set()
        self.recovery_active = False
        self.valuation = SimpleNamespace(last_source="ayush_port_boot", last_earnings_value=None)
        self._pending_decision: _DecisionEnvelope | None = None
        self._learning_events: list[LearningEvent] = []
        self._latest_export_state = self._strategy.export_state()
        self._last_fair_value = self._strategy.fair_value
        self._last_trusted_multiplier = self._strategy.trusted_multiplier
        self._last_multiplier_confidence = max(0, recovered_multiplier_confidence)
        self._earnings_signal_seq = 0
        self._news_signal_seq = 0
        self.current_earnings_signal_id: str | None = None
        self.current_news_signal_id: str | None = None
        self.active_earnings_cycle_id: str | None = None
        self.active_news_signal_id: str | None = None
        self.pending_news_signal_id: str | None = None
        self.orders: dict[str, ManagedOrder] = {}

        for order in restored_orders:
            restored = ManagedOrder(
                order_id=order.order_id,
                side=order.side,
                px=order.px,
                qty=order.qty,
                remaining_qty=order.remaining_qty,
                submitted_ms=order.submitted_ms,
                overlay=getattr(order, "overlay", "earnings"),
                aggressive=bool(getattr(order, "aggressive", False)),
                cancel_pending=bool(getattr(order, "cancel_pending", False)),
                restored=True,
                intent=getattr(order, "intent", ""),
                mode_at_submit=getattr(order, "mode_at_submit", ""),
                evaluation_reason=getattr(order, "evaluation_reason", ""),
                market_key=getattr(order, "market_key", "A"),
                strategy_family=getattr(order, "strategy_family", ""),
                action_class=getattr(order, "action_class", ""),
                pnl_owner=getattr(order, "pnl_owner", ""),
                signal_id=getattr(order, "signal_id", ""),
                trade_group_id=getattr(order, "trade_group_id", ""),
                leg_role=getattr(order, "leg_role", "single"),
            )
            self.orders[restored.order_id] = restored
            self.recovery_pending.add(restored.order_id)
        if self.recovery_pending:
            self.recovery_active = True

        seed_multiplier = recovered_multiplier if recovered_multiplier is not None else initial_multiplier
        seed_fair = recovered_fair_value if recovered_fair_value is not None else initial_fair_value
        seed_earnings = recovered_earnings_value
        if seed_multiplier is not None:
            self._strategy.trusted_multiplier = float(seed_multiplier)
            self._last_multiplier_confidence = max(1, self._last_multiplier_confidence)
        if seed_earnings is not None:
            self._strategy.latest_earnings = float(seed_earnings)
            self.valuation.last_earnings_value = float(seed_earnings)
        if seed_fair is not None:
            self._strategy.fair_value = int(seed_fair)
            self._strategy.base_fair_value = int(seed_fair)
            self._last_fair_value = int(seed_fair)
            self.valuation.last_source = "journal"

        self._refresh_public_state()

    @property
    def fair_value(self) -> int | None:
        return self._strategy.fair_value

    @property
    def trusted_multiplier(self) -> float | None:
        return self._strategy.trusted_multiplier

    @property
    def multiplier_confidence(self) -> int:
        return max(
            self._last_multiplier_confidence,
            0 if self._strategy.trusted_multiplier is None else max(1, len(self._strategy.clean_multiplier_samples)),
        )

    @property
    def last_earnings_value(self) -> float | None:
        return self._strategy.latest_earnings

    def sync_inventory_from_exchange(self, inventory: int, *, now_ms: int | None = None) -> None:
        self.inventory = int(inventory)
        self._authoritative_inventory = int(inventory)
        self._last_position_update_ms = self._now_ms() if now_ms is None else int(now_ms)

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol != self.symbol:
            return False
        self.book = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        self._store_decision("book", self._strategy.on_book(self._snapshot(now_ms=now_ms, trigger="book")))
        return True

    def on_market_trade(self, price: int, qty: int, now_ms: int | None = None) -> None:
        event_ms = self._now_ms() if now_ms is None else int(now_ms)
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = event_ms
        self._store_decision("trade", self._strategy.on_trade(self._snapshot(now_ms=event_ms, trigger="trade")))

    def on_fill(self, order_id: str, qty: int, price: int, *, now_ms: int | None = None) -> ManagedOrder | None:
        event_ms = self._now_ms() if now_ms is None else int(now_ms)
        order = self.orders.get(order_id)
        if order is not None:
            fill_qty = min(abs(int(qty)), order.remaining_qty)
            if fill_qty > 0:
                order.remaining_qty = max(0, order.remaining_qty - fill_qty)
                self._apply_fill_inventory_hint(order.side, fill_qty, event_ms)
            if order.remaining_qty <= 0:
                self._drop_order(order_id)
        self.last_trade_px = int(price)
        self.last_trade_qty = int(qty)
        self.last_trade_ms = event_ms
        self._store_decision("fill", self._strategy.on_fill(self._snapshot(now_ms=event_ms, trigger="fill")))
        return order

    def on_cancel_response(self, order_id: str, success: bool) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is not None:
            if success:
                self._drop_order(order_id)
            else:
                order.cancel_pending = False
        self._update_recovery(order_id)
        return order

    def on_rejection(self, order_id: str) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is not None:
            self._drop_order(order_id)
        self._update_recovery(order_id)
        return order

    def on_recovery_complete(self) -> None:
        self.recovery_active = False
        self.recovery_pending.clear()

    def recovery_orders_to_cancel(self) -> list[ManagedOrder]:
        return [
            order
            for order in self.orders.values()
            if order.order_id in self.recovery_pending and order.remaining_qty > 0 and not order.cancel_pending
        ]

    def drain_learning_events(self) -> list[LearningEvent]:
        events = list(self._learning_events)
        self._learning_events.clear()
        return events

    def on_news(self, news_release: dict, now_ms: int) -> NewsReaction:
        news_event, resolved_text, resolved_source = self._to_news_event(now_ms, news_release)
        if news_event.is_structured_a_earnings:
            self._earnings_signal_seq += 1
            self.current_earnings_signal_id = f"ayush_eps_{self._earnings_signal_seq}"
            self.active_earnings_cycle_id = self.current_earnings_signal_id
            self.valuation.last_earnings_value = news_event.value
        elif news_event.is_a_specific_unstructured:
            self._news_signal_seq += 1
            self.current_news_signal_id = f"ayush_news_{self._news_signal_seq}"
            self.pending_news_signal_id = self.current_news_signal_id

        old_fair = self.fair_value
        decision = self._strategy.on_news(self._snapshot(now_ms=now_ms, trigger="news"), news_event)
        self._store_decision("news", decision)

        state = self._latest_export_state
        new_fair = self.fair_value
        active_kind = str(state.get("active_signal_kind") or "")
        if active_kind == "structured":
            self.active_earnings_cycle_id = self.current_earnings_signal_id
        elif active_kind == "unstructured":
            self.active_news_signal_id = self.current_news_signal_id or self.pending_news_signal_id
        fair_value_updated = new_fair is not None and new_fair != old_fair
        if news_event.is_structured_a_earnings:
            self.valuation.last_source = "ayush_structured_news"
        elif news_event.is_a_specific_unstructured:
            self.valuation.last_source = "ayush_unstructured_news"

        relevant = bool(news_event.is_structured_a_earnings or news_event.is_a_specific_unstructured)
        return NewsReaction(
            relevant=relevant,
            fair_value_updated=fair_value_updated,
            note=decision.reason,
            earnings_value=news_event.value,
            news_sentiment_score=self._float_or_none(state.get("news_sentiment_score")),
            news_sentiment_bucket=self._str_or_none(state.get("news_sentiment_bucket")),
            old_fair_value=old_fair,
            new_fair_value=new_fair,
            base_fair_value=self._int_or_none(state.get("base_fair_value")),
            news_fair_value=self._int_or_none(state.get("news_fair_value")),
            pending_news_target_inventory=self._int_or_none(state.get("pending_news_target_inventory")),
            news_confirmation_state=self._str_or_none(state.get("news_confirmation_state")),
            active_news_signal_id=self.active_news_signal_id or self.pending_news_signal_id,
            news_matched_phrases=tuple(state.get("news_matched_phrases") or ()),
            news_matched_unigrams=tuple(state.get("news_matched_unigrams") or ()),
            news_matched_bigrams=tuple(state.get("news_matched_bigrams") or ()),
            unknown_candidate_phrases=tuple(state.get("unknown_candidate_phrases") or ()),
            unknown_candidate_unigrams=tuple(state.get("unknown_candidate_unigrams") or ()),
            unknown_candidate_bigrams=tuple(state.get("unknown_candidate_bigrams") or ()),
            resolved_news_text=resolved_text,
            resolved_news_text_source=resolved_source,
            shock_direction=int(state.get("shock_direction") or 0),
            shock_threshold=None,
            tick=news_event.tick,
        )

    def compute_quotes(self, now_ms: int | None = None) -> QuotePlan:
        _, plan = self.evaluate_runtime(now_ms=now_ms)
        return plan

    def evaluate_runtime(self, *, now_ms: int | None = None) -> tuple[Any | None, QuotePlan]:
        event_ms = self._now_ms() if now_ms is None else int(now_ms)
        if self.recovery_active:
            return None, QuotePlan(
                mode=self.mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for recovered A orders to be cancelled.",
            )

        envelope = self._pending_decision
        if envelope is None:
            decision = self._strategy.on_timer(self._snapshot(now_ms=event_ms, trigger="timer"))
            envelope = _DecisionEnvelope(trigger="timer", decision=decision)
            self._store_decision(envelope.trigger, envelope.decision, replace_pending=False)
        self._pending_decision = None
        return envelope.decision, self._translate_decision(envelope.decision)

    def desired_order_for_decision(self, decision: Any) -> DesiredOrder | None:
        desired = getattr(decision, "desired_order", None)
        if desired is None:
            return None
        normalized_qty = int(desired.qty)
        if normalized_qty <= 0:
            return None
        strategy_family, action_class, pnl_owner, overlay = self._order_identity(str(desired.intent or ""))
        signal_id = self._signal_id_for_order(strategy_family)
        return DesiredOrder(
            side=str(desired.side),
            px=int(desired.px),
            qty=normalized_qty,
            overlay=overlay,
            aggressive=bool(desired.aggressive),
            reason=str(desired.reason or getattr(decision, "reason", "") or ""),
            intent=str(desired.intent or ""),
            mode_at_submit=self.mode,
            evaluation_reason=str(getattr(decision, "reason", "") or ""),
            market_key="A",
            strategy_family=strategy_family,
            action_class=action_class,
            pnl_owner=pnl_owner,
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role="single",
        )

    def current_orders(self) -> list[ManagedOrder]:
        return [order for order in self.orders.values() if order.remaining_qty > 0]

    def live_orders_snapshot(self) -> list[dict[str, Any]]:
        return [
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
                "market_key": order.market_key,
                "strategy_family": order.strategy_family,
                "action_class": order.action_class,
                "pnl_owner": order.pnl_owner,
                "signal_id": order.signal_id,
                "trade_group_id": order.trade_group_id,
                "leg_role": order.leg_role,
            }
            for order in self.current_orders()
        ]

    def buy_exposure(self) -> int:
        return sum(order.remaining_qty for order in self.current_orders() if order.side == "BUY")

    def sell_exposure(self) -> int:
        return sum(order.remaining_qty for order in self.current_orders() if order.side == "SELL")

    def mark_cancel_requested(self, order_id: str, now_ms: int) -> ManagedOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        order.cancel_pending = True
        return order

    def note_submitted(self, *, order_id: str, desired: DesiredOrder, now_ms: int) -> ManagedOrder:
        managed = ManagedOrder(
            order_id=order_id,
            side=desired.side,
            px=desired.px,
            qty=desired.qty,
            remaining_qty=desired.qty,
            submitted_ms=int(now_ms),
            overlay=desired.overlay,
            aggressive=desired.aggressive,
            cancel_pending=False,
            restored=False,
            intent=desired.intent,
            mode_at_submit=desired.mode_at_submit,
            evaluation_reason=desired.evaluation_reason,
            market_key=desired.market_key,
            strategy_family=desired.strategy_family,
            action_class=desired.action_class,
            pnl_owner=desired.pnl_owner,
            signal_id=desired.signal_id,
            trade_group_id=desired.trade_group_id,
            leg_role=desired.leg_role,
        )
        self.orders[order_id] = managed
        return managed

    def any_cancel_pending(self) -> bool:
        return any(order.cancel_pending for order in self.current_orders())

    def normalize_desired_order(self, decision: Any) -> DesiredOrder | None:
        return self.desired_order_for_decision(decision)

    def can_keep_orders(self, active_orders: list[ManagedOrder], desired: DesiredOrder, now_ms: int) -> bool:
        if not active_orders:
            return True
        if any(not self.order_matches(active, desired, now_ms) for active in active_orders):
            return False
        total_live_qty = sum(active.remaining_qty for active in active_orders if active.remaining_qty > 0)
        return total_live_qty <= (int(desired.qty) + self._config.replace_qty_tolerance)

    def order_matches(self, active: ManagedOrder, desired: DesiredOrder, now_ms: int) -> bool:
        if active.cancel_pending:
            return False
        if active.side != desired.side:
            return False
        if abs(active.px - desired.px) > self._config.replace_price_tolerance_ticks:
            age_ms = int(now_ms) - active.submitted_ms
            return age_ms < self._config.min_order_live_ms
        return True

    def next_slice_qty(self, remaining_target_qty: int) -> int:
        max_legal_qty = min(self._config.max_exchange_order_qty, self._config.order_slice_max_qty)
        if remaining_target_qty <= max_legal_qty:
            return int(remaining_target_qty)
        slice_qty = min(max_legal_qty, self._config.order_slice_target_qty)
        remainder = int(remaining_target_qty) - slice_qty
        min_slice = max(1, self._config.order_slice_min_qty)
        if 0 < remainder < min_slice:
            slice_qty = min(max_legal_qty, slice_qty + (min_slice - remainder))
        return max(1, slice_qty)

    def risk_adjusted_order(self, desired: DesiredOrder) -> DesiredOrder | None:
        live_orders = self.current_orders()
        if len(live_orders) >= self._config.max_open_orders:
            return None

        outstanding_volume = sum(order.remaining_qty for order in live_orders)
        available_volume = max(0, self._config.max_outstanding_volume - outstanding_volume)
        if available_volume <= 0:
            return None

        max_qty_from_position = self._max_qty_from_position_limit(desired.side)
        if max_qty_from_position <= 0:
            return None

        allowed_qty = min(
            int(desired.qty),
            available_volume,
            max_qty_from_position,
            self._config.max_exchange_order_qty,
        )
        if allowed_qty <= 0:
            return None
        if allowed_qty == desired.qty:
            return desired
        return replace(desired, qty=allowed_qty)

    def trace_state(self, now_ms: int) -> dict[str, object]:
        state = self._latest_export_state
        active_kind = str(state.get("active_signal_kind") or "")
        earnings_position = self.inventory if active_kind == "structured" else 0
        news_position = self.inventory if active_kind == "unstructured" else 0
        live_orders = self.live_orders_snapshot()
        buy_exposure = self.buy_exposure()
        sell_exposure = self.sell_exposure()
        return {
            "symbol": self.symbol,
            "market_key": "A",
            "mode": self.mode,
            "fair_value": self.fair_value,
            "trusted_multiplier": self.trusted_multiplier,
            "multiplier_confidence": self.multiplier_confidence,
            "latest_earnings": self.last_earnings_value,
            "inventory": self.inventory,
            "earnings_position": earnings_position,
            "news_position": news_position,
            "mm_position": 0,
            "earnings_budget": self._config.position_cap,
            "mm_budget": 0,
            "budget_shift_active": False,
            "unwind_active": self.mode == "UNWIND",
            "unwind_aggressive_active": self.mode == "UNWIND",
            "buy_exposure": buy_exposure,
            "sell_exposure": sell_exposure,
            "allowed_buy_size": max(0, self._config.max_absolute_position - self.inventory),
            "allowed_sell_size": max(0, self._config.max_absolute_position + self.inventory),
            "position_cap": self._config.max_absolute_position,
            "overlay_exposures": {
                "earnings": {
                    "inventory": earnings_position,
                    "buy_exposure": buy_exposure if active_kind == "structured" else 0,
                    "sell_exposure": sell_exposure if active_kind == "structured" else 0,
                },
                "news": {
                    "inventory": news_position,
                    "buy_exposure": buy_exposure if active_kind == "unstructured" else 0,
                    "sell_exposure": sell_exposure if active_kind == "unstructured" else 0,
                },
                "mm": {"inventory": 0, "buy_exposure": 0, "sell_exposure": 0},
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
            "exchange_tick": None,
            "news_caution_active": False,
            "news_caution_until_ms": 0,
            "news_caution_remaining_ms": 0,
            "current_earnings_signal_id": self.current_earnings_signal_id,
            "current_news_signal_id": self.current_news_signal_id,
            "active_earnings_cycle_id": self.active_earnings_cycle_id,
            "active_news_signal_id": self.active_news_signal_id,
            "pending_news_signal_id": self.pending_news_signal_id,
            "active_signal_kind": active_kind,
            "base_fair_value": state.get("base_fair_value"),
            "news_fair_value": state.get("news_fair_value"),
            "news_sentiment_score": state.get("news_sentiment_score"),
            "news_sentiment_bucket": state.get("news_sentiment_bucket"),
            "news_confirmation_state": state.get("news_confirmation_state"),
            "pending_news_target_inventory": state.get("pending_news_target_inventory"),
            "pe_frozen": state.get("pe_frozen"),
            "pending_news": state.get("pending_news"),
            "news_confirmation_deadline_ms": state.get("news_confirmation_deadline_ms"),
            "news_takeover_started_ms": state.get("news_takeover_started_ms"),
            "news_matched_phrases": state.get("news_matched_phrases"),
            "news_matched_unigrams": state.get("news_matched_unigrams"),
            "news_matched_bigrams": state.get("news_matched_bigrams"),
            "unknown_candidate_phrases": state.get("unknown_candidate_phrases"),
            "unknown_candidate_unigrams": state.get("unknown_candidate_unigrams"),
            "unknown_candidate_bigrams": state.get("unknown_candidate_bigrams"),
            "shock_target_inventory": state.get("shock_target_inventory"),
            "original_shock_target_inventory": state.get("original_shock_target_inventory"),
            "shock_peak_inventory_abs": state.get("shock_peak_inventory_abs"),
            "shock_direction": state.get("shock_direction"),
            "shock_reference_mid": state.get("shock_reference_mid"),
            "overshoot_active": state.get("overshoot_active"),
            "overshoot_stage_index": state.get("overshoot_stage_index"),
            "overshoot_trimmed_qty_total": state.get("overshoot_trimmed_qty_total"),
            "shock_decay_steps_applied": state.get("shock_decay_steps_applied"),
            "shock_decay_trimmed_qty_total": state.get("shock_decay_trimmed_qty_total"),
        }

    def _translate_decision(self, decision: Any | None) -> QuotePlan:
        if decision is None:
            return QuotePlan(
                mode=self.mode,
                bid=None,
                ask=None,
                aggressive_actions=(),
                observe_only=True,
                reason="Waiting for recovered A orders to be cancelled.",
            )
        desired = self.desired_order_for_decision(decision)
        aggressive_actions = ()
        bid = None
        ask = None
        if desired is not None:
            if desired.aggressive:
                aggressive_actions = (desired,)
            elif desired.side == "BUY":
                bid = desired
            else:
                ask = desired
        plan_mode = self.mode
        decision_mode = str(getattr(decision, "mode", "") or "")
        if decision_mode == "UNWIND":
            plan_mode = "UNWIND"
        elif decision_mode == "SHOCK":
            plan_mode = "POST_NEWS_SHOCK" if desired is not None and desired.strategy_family == "a_news" else "POST_EARNINGS_SHOCK"
        elif decision_mode == "IDLE":
            confirmation_state = str(self._latest_export_state.get("news_confirmation_state") or "inactive")
            if confirmation_state not in {"inactive", "active"}:
                plan_mode = "NEWS_CONFIRMATION"
        return QuotePlan(
            mode=plan_mode,
            bid=bid,
            ask=ask,
            aggressive_actions=aggressive_actions,
            observe_only=bool(getattr(decision, "observe_only", False) and desired is None),
            reason=str(getattr(decision, "reason", "") or self.mode),
        )

    def _order_identity(self, intent: str) -> tuple[str, str, str, OverlayName]:
        if intent == "post_earnings_shock_take":
            return ("a_earnings", "shock_take", "a_earnings", "earnings")
        if intent == "post_news_shock_take":
            return ("a_news", "news_take", "a_news", "news")
        if intent == "news_takeover_flatten":
            return ("a_news", "news_takeover_flatten", "a_news", "news")
        if intent == "unwind":
            active_kind = str(self._latest_export_state.get("active_signal_kind") or "")
            if active_kind == "unstructured":
                return ("a_news", "news_unwind", "a_news", "news")
            return ("a_earnings", "shock_unwind", "a_earnings", "earnings")
        return ("a_earnings", "shock_take", "a_earnings", "earnings")

    def _signal_id_for_order(self, strategy_family: str) -> str:
        if strategy_family == "a_news":
            return (
                self.active_news_signal_id
                or self.pending_news_signal_id
                or self.current_news_signal_id
                or f"ayush_news_{self._news_signal_seq}"
            )
        return self.active_earnings_cycle_id or self.current_earnings_signal_id or f"ayush_eps_{self._earnings_signal_seq}"

    def _snapshot(self, *, now_ms: int, trigger: str) -> AyushStrategySnapshot:
        open_orders = tuple(self._convert_order(order) for order in self.current_orders())
        books_by_symbol = {self.symbol: self._convert_book(self.book)}
        inventories_by_symbol = {self.symbol: int(self.inventory)}
        open_orders_by_symbol = {self.symbol: open_orders}
        last_trade_px_by_symbol = {self.symbol: self.last_trade_px}
        return AyushStrategySnapshot(
            now_ms=now_ms,
            exchange_tick=None,
            book=self._convert_book(self.book),
            inventory=int(self.inventory),
            cash=0,
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.last_earnings_value,
            mode=self._strategy.mode,
            open_orders=open_orders,
            last_trade_px=self.last_trade_px,
            message_index=None,
            books_by_symbol=books_by_symbol,
            inventories_by_symbol=inventories_by_symbol,
            open_orders_by_symbol=open_orders_by_symbol,
            last_trade_px_by_symbol=last_trade_px_by_symbol,
            event_symbol=self.symbol if trigger != "timer" else None,
        )

    def _store_decision(self, trigger: str, decision: Any, *, replace_pending: bool = True) -> None:
        if replace_pending:
            self._pending_decision = _DecisionEnvelope(trigger=trigger, decision=decision)
        self._refresh_public_state()

    def _refresh_public_state(self) -> None:
        state = self._strategy.export_state()
        self._latest_export_state = state
        self.mode = self._mapped_mode(state)
        active_kind = str(state.get("active_signal_kind") or "")
        confirmation_state = str(state.get("news_confirmation_state") or "inactive")
        if active_kind == "structured":
            self.active_earnings_cycle_id = self.current_earnings_signal_id
            self.active_news_signal_id = None
        elif active_kind == "unstructured":
            self.active_news_signal_id = self.current_news_signal_id or self.pending_news_signal_id
        else:
            self.active_news_signal_id = None

        if state.get("pending_news"):
            self.pending_news_signal_id = self.current_news_signal_id or self.pending_news_signal_id
        elif confirmation_state in {"inactive", ""}:
            self.pending_news_signal_id = None

        new_multiplier = self._strategy.trusted_multiplier
        if new_multiplier is not None:
            if self._last_trusted_multiplier is None or abs(float(new_multiplier) - float(self._last_trusted_multiplier)) > 1e-9:
                self._last_multiplier_confidence = max(1, self._last_multiplier_confidence + 1)
                self.valuation.last_source = "ayush_multiplier_update"
            self._last_trusted_multiplier = float(new_multiplier)

        if self._strategy.latest_earnings is not None:
            self.valuation.last_earnings_value = float(self._strategy.latest_earnings)
        if self._strategy.fair_value is not None and self._strategy.fair_value != self._last_fair_value:
            if active_kind == "unstructured":
                self.valuation.last_source = "ayush_unstructured_news"
            elif active_kind == "structured":
                self.valuation.last_source = "ayush_structured_news"
            self._last_fair_value = int(self._strategy.fair_value)

    def _apply_fill_inventory_hint(self, side: str, qty: int, now_ms: int) -> None:
        authoritative_inventory = int(self._authoritative_inventory)
        if self._last_position_update_ms is not None and (int(now_ms) - self._last_position_update_ms) <= 50:
            self.inventory = authoritative_inventory
            return
        if authoritative_inventory != int(self.inventory):
            self.inventory = authoritative_inventory
            return
        signed_qty = int(qty) if side == "BUY" else -int(qty)
        self.inventory += signed_qty

    def _max_qty_from_position_limit(self, side: str) -> int:
        if side == "BUY":
            return max(0, self._config.max_absolute_position - int(self.inventory))
        return max(0, self._config.max_absolute_position + int(self.inventory))

    def _update_recovery(self, order_id: str) -> None:
        if order_id in self.recovery_pending:
            self.recovery_pending.discard(order_id)
            if not self.recovery_pending:
                self.on_recovery_complete()

    def _drop_order(self, order_id: str) -> None:
        self.orders.pop(order_id, None)

    @staticmethod
    def _convert_book(book: BookSnapshot) -> AyushBookSnapshot:
        best_bid = None if book.best_bid is None else AyushBookLevel(px=book.best_bid.px, qty=book.best_bid.qty)
        best_ask = None if book.best_ask is None else AyushBookLevel(px=book.best_ask.px, qty=book.best_ask.qty)
        return AyushBookSnapshot(best_bid=best_bid, best_ask=best_ask)

    @staticmethod
    def _convert_order(order: ManagedOrder) -> AyushManagedOrder:
        return AyushManagedOrder(
            order_id=order.order_id,
            side=order.side,
            px=order.px,
            qty=order.qty,
            remaining_qty=order.remaining_qty,
            submitted_ms=order.submitted_ms,
            aggressive=order.aggressive,
            intent=order.intent,
            reason=order.evaluation_reason or order.intent or "",
            symbol="A",
            context_id=order.signal_id or None,
        )

    @staticmethod
    def _mapped_mode(state: dict[str, Any]) -> str:
        news_confirmation_state = str(state.get("news_confirmation_state") or "inactive")
        if news_confirmation_state not in {"inactive", "active"}:
            return "NEWS_CONFIRMATION"
        active_kind = str(state.get("active_signal_kind") or "")
        mode = str(state.get("mode") or "IDLE")
        if mode == "SHOCK":
            return "POST_NEWS_SHOCK" if active_kind == "unstructured" else "POST_EARNINGS_SHOCK"
        if mode == "UNWIND":
            return "UNWIND"
        if news_confirmation_state == "active":
            return "POST_NEWS_SHOCK"
        return "AYUSH_IDLE"

    @staticmethod
    def _to_news_event(now_ms: int, news_release: dict) -> tuple[AyushNewsEvent, str | None, str | None]:
        raw = dict(news_release)
        new_data = raw.get("new_data") or {}
        content = None
        content_source = None
        if new_data.get("content") is not None:
            content = str(new_data.get("content"))
            content_source = "new_data.content"
        elif raw.get("raw_content") is not None:
            content = str(raw.get("raw_content"))
            content_source = "raw_content"
        elif raw.get("content") is not None:
            content = str(raw.get("content"))
            content_source = "content"
        return (
            AyushNewsEvent(
                now_ms=now_ms,
                tick=raw.get("tick"),
                kind=str(raw.get("kind") or "unknown"),
                symbol=None if raw.get("symbol") is None else str(raw.get("symbol")),
                structured_subtype=None if new_data.get("structured_subtype") is None else str(new_data.get("structured_subtype")),
                asset=None if new_data.get("asset") is None else str(new_data.get("asset")),
                value=None if new_data.get("value") is None else float(new_data.get("value")),
                content=content,
                news_type=None if new_data.get("type") is None else str(new_data.get("type")),
                forecast=None if new_data.get("forecast") is None else float(new_data.get("forecast")),
                actual=None if new_data.get("actual") is None else float(new_data.get("actual")),
                raw_payload=raw,
            ),
            content,
            content_source,
        )

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _str_or_none(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _now_ms() -> int:
        return int(monotonic() * 1000)
