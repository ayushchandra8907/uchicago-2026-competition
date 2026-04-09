from __future__ import annotations

from collections import deque
from statistics import median

from ..config import StrategyConfig
from ..core.types import Decision, DesiredOrder, ModeState, NewsEvent, StrategySnapshot
from .a_news_sentiment import SentimentResult, score_a_unstructured_headline


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class AStrategy:
    """A shock strategy for structured earnings and A-specific unstructured news."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mode: ModeState = "IDLE"
        self.trusted_multiplier: float | None = None
        self.latest_earnings: float | None = None
        self.fair_value: int | None = None
        self.base_fair_value: int | None = None
        self.news_fair_value: int | None = None
        self.fair_change_ticks: float | None = None
        self.pre_earnings_mids: deque[tuple[int, float]] = deque()
        self.post_event_mids: deque[tuple[int, float]] = deque()
        self.shock_started_ms: int | None = None
        self.shock_direction: int = 0
        self.shock_target_inventory: int = 0
        self.original_shock_target_inventory: int = 0
        self.shock_peak_inventory_abs: int = 0
        self.shock_reference_mid: float | None = None
        self.initial_shock_edge: float = 0.0
        self.equilibrium_reached_ms: int | None = None
        self.unwind_started_ms: int | None = None
        self.unwind_start_inventory: int = 0
        self.pe_frozen: bool = False
        self.clean_multiplier_samples: list[float] = []
        self.current_event_contaminated: bool = False
        self.overshoot_stage_index: int = 0
        self.overshoot_trimmed_qty_total: int = 0
        self.overshoot_active: bool = False
        self.overshoot_trigger_ticks: int | None = None
        self.overshoot_crossed_fair_ms: int | None = None
        self.shock_decay_steps_applied: int = 0
        self.shock_decay_trimmed_qty_total: int = 0
        self.structured_event_count = 0

        self.active_signal_kind: str | None = None
        self.news_sentiment_score: float | None = None
        self.news_sentiment_bucket: str | None = None
        self.news_matched_phrases: tuple[str, ...] = ()
        self.news_matched_unigrams: tuple[str, ...] = ()
        self.news_matched_bigrams: tuple[str, ...] = ()
        self.unknown_candidate_phrases: tuple[str, ...] = ()
        self.unknown_candidate_unigrams: tuple[str, ...] = ()
        self.unknown_candidate_bigrams: tuple[str, ...] = ()
        self.pending_takeover_news: NewsEvent | None = None
        self.pending_unstructured_news: NewsEvent | None = None
        self.pending_news_score: float | None = None
        self.pending_news_fair: int | None = None
        self.pending_news_target_inventory: int | None = None
        self.pending_news_bucket: str | None = None
        self.pending_news_direction: int = 0
        self.pending_news_base_fair: int | None = None
        self.pending_news_matched_phrases: tuple[str, ...] = ()
        self.pending_news_matched_unigrams: tuple[str, ...] = ()
        self.pending_news_matched_bigrams: tuple[str, ...] = ()
        self.pending_news_unknown_phrases: tuple[str, ...] = ()
        self.pending_news_unknown_unigrams: tuple[str, ...] = ()
        self.pending_news_unknown_bigrams: tuple[str, ...] = ()
        self.pending_news_reference_mid: float | None = None
        self.pending_news_reference_bid: int | None = None
        self.pending_news_reference_ask: int | None = None
        self.pending_news_reference_spread: int | None = None
        self.pending_news: str | None = None
        self.news_takeover_started_ms: int | None = None
        self.news_confirmation_deadline_ms: int | None = None
        self.news_confirmation_direction: int = 0
        self.news_confirmed: bool = False
        self.news_confirmation_state: str = "inactive"

    def on_book(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="book")

    def on_trade(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.last_trade_px)
        return self._evaluate(snapshot, trigger="trade")

    def on_fill(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="fill")

    def on_timer(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="timer")

    def on_news(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)

        if news_event.is_structured_a_earnings:
            return self._start_structured_news(snapshot, news_event)

        if news_event.is_a_specific_unstructured:
            return self._handle_unstructured_a_news(snapshot, news_event)

        return self._observe_current_state(snapshot, reason="ignoring non-A or non-tradable news")

    def export_state(self) -> dict[str, int | float | str | bool | list[str] | None]:
        return {
            "mode": self.mode,
            "active_signal_kind": self.active_signal_kind,
            "fair_value": self.fair_value,
            "base_fair_value": self.base_fair_value,
            "news_fair_value": self.news_fair_value,
            "trusted_multiplier": self.trusted_multiplier,
            "latest_earnings": self.latest_earnings,
            "fair_change_ticks": self.fair_change_ticks,
            "shock_target_inventory": self.shock_target_inventory,
            "original_shock_target_inventory": self.original_shock_target_inventory,
            "shock_peak_inventory_abs": self.shock_peak_inventory_abs,
            "shock_direction": self.shock_direction,
            "shock_reference_mid": self.shock_reference_mid,
            "initial_shock_edge": self.initial_shock_edge,
            "pe_frozen": self.pe_frozen,
            "clean_multiplier_sample_count": len(self.clean_multiplier_samples),
            "overshoot_active": self.overshoot_active,
            "overshoot_stage_index": self.overshoot_stage_index,
            "overshoot_trimmed_qty_total": self.overshoot_trimmed_qty_total,
            "overshoot_trigger_ticks": self.overshoot_trigger_ticks,
            "shock_decay_steps_applied": self.shock_decay_steps_applied,
            "shock_decay_trimmed_qty_total": self.shock_decay_trimmed_qty_total,
            "equilibrium_reached": self.equilibrium_reached_ms is not None,
            "structured_event_count": self.structured_event_count,
            "news_sentiment_score": self.news_sentiment_score,
            "news_sentiment_bucket": self.news_sentiment_bucket,
            "news_matched_phrases": list(self.news_matched_phrases),
            "news_matched_unigrams": list(self.news_matched_unigrams),
            "news_matched_bigrams": list(self.news_matched_bigrams),
            "unknown_candidate_phrases": list(self.unknown_candidate_phrases),
            "unknown_candidate_unigrams": list(self.unknown_candidate_unigrams),
            "unknown_candidate_bigrams": list(self.unknown_candidate_bigrams),
            "pending_takeover_news": None if self.pending_takeover_news is None else self.pending_takeover_news.content or self.pending_takeover_news.kind,
            "pending_news": self.pending_news,
            "pending_news_target_inventory": self.pending_news_target_inventory,
            "news_confirmation_state": self.news_confirmation_state,
            "news_confirmation_deadline_ms": self.news_confirmation_deadline_ms,
            "news_takeover_started_ms": self.news_takeover_started_ms,
        }

    def _record_mid(self, now_ms: int, mid: float | int | None) -> None:
        if mid is None:
            return
        numeric_mid = float(mid)
        self.pre_earnings_mids.append((now_ms, numeric_mid))
        self._trim_window(self.pre_earnings_mids, now_ms, self.config.first_earnings_baseline_window_ms)
        if self.mode in {"SHOCK", "UNWIND"} or self.pending_unstructured_news is not None:
            self.post_event_mids.append((now_ms, numeric_mid))
            hold_window = max(
                self.config.flatten_deadline_ms * 2,
                self.config.equilibrium_hold_ms * 3,
            )
            self._trim_window(self.post_event_mids, now_ms, hold_window)

    @staticmethod
    def _trim_window(samples: deque[tuple[int, float]], now_ms: int, window_ms: int) -> None:
        threshold = now_ms - window_ms
        while samples and samples[0][0] < threshold:
            samples.popleft()

    def _baseline_mid(self, snapshot: StrategySnapshot) -> float | None:
        self._trim_window(self.pre_earnings_mids, snapshot.now_ms, self.config.first_earnings_baseline_window_ms)
        if len(self.pre_earnings_mids) >= self.config.first_earnings_min_mid_samples:
            return float(median(value for _, value in self.pre_earnings_mids))
        return snapshot.book.mid

    def _base_fair_for_unstructured(self, snapshot: StrategySnapshot) -> int | None:
        if self.base_fair_value is not None:
            return int(self.base_fair_value)
        if self.fair_value is not None:
            return int(self.fair_value)
        if self.trusted_multiplier is not None and self.latest_earnings is not None:
            return round(self.trusted_multiplier * self.latest_earnings)
        baseline_mid = self._baseline_mid(snapshot)
        if baseline_mid is not None:
            return round(baseline_mid)
        return None

    def _scaled_target(self, edge: float, *, fair_change_ticks: float | None = None) -> int:
        edge_abs = abs(edge)
        min_edge = max(1, self.config.shock_min_edge_ticks)
        if edge_abs < min_edge:
            return 0
        confidence_span = max(1, self.config.shock_full_confidence_edge_ticks - min_edge)
        confidence = min(1.0, max(0.0, (edge_abs - min_edge) / confidence_span))
        scaled_cap = max(0, min(self.config.position_cap, self.config.max_absolute_position))
        base_target = max(self.config.shock_min_position, round(scaled_cap * confidence))
        scaled = max(base_target, round(edge_abs * self.config.shock_position_scale))
        target_abs = min(scaled_cap, scaled)

        if fair_change_ticks is not None:
            fair_change_abs = abs(fair_change_ticks)
            change_confidence_span = max(1, self.config.shock_full_confidence_change_ticks - min_edge)
            change_confidence = min(
                1.0,
                max(0.0, fair_change_abs - min_edge) / change_confidence_span,
            )
            change_base_target = max(self.config.shock_min_position, round(scaled_cap * change_confidence))
            change_scaled_target = max(
                change_base_target,
                round(fair_change_abs * self.config.shock_change_position_scale),
            )
            # Treat fair-value change as an additional confidence signal, not a hard cap.
            target_abs = max(target_abs, min(scaled_cap, change_scaled_target))

        return _sign(edge) * target_abs

    def _news_offset_ticks(self, score: float, bucket: str) -> int:
        absolute = abs(score)
        if absolute >= 5.0:
            return self.config.news_very_extreme_offset_ticks
        if bucket == "extreme":
            return self.config.news_extreme_offset_ticks
        if bucket == "strong":
            return self.config.news_strong_offset_ticks
        if bucket == "medium":
            return self.config.news_medium_offset_ticks
        if bucket == "light":
            return self.config.news_light_offset_ticks
        return 0

    def _news_position_cap(self, score: float, bucket: str) -> int:
        absolute = abs(score)
        if absolute >= 5.0:
            return self.config.news_very_extreme_position
        if bucket == "extreme":
            return self.config.news_extreme_position
        if bucket == "strong":
            return self.config.news_strong_position
        if bucket == "medium":
            return self.config.news_medium_position
        if bucket == "light":
            return self.config.news_light_position
        return 0

    def _news_target_inventory(
        self,
        *,
        news_fair: int,
        score: float,
        bucket: str,
        snapshot: StrategySnapshot,
    ) -> int:
        direction = 1 if score > 0 else -1 if score < 0 else 0
        if snapshot.book.mid is None or direction == 0:
            return 0
        offset_ticks = float(self._news_offset_ticks(score, bucket))
        news_edge_abs = abs(float(news_fair) - float(snapshot.book.mid))
        effective_edge = max(news_edge_abs, offset_ticks)
        raw_target = abs(
            self._scaled_target(
                direction * effective_edge,
                fair_change_ticks=offset_ticks,
            )
        )
        bucket_cap = self._news_position_cap(score, bucket)
        capped_abs = min(raw_target, bucket_cap, self.config.position_cap, self.config.max_absolute_position)
        if capped_abs <= self.config.news_zero_position_threshold:
            return 0
        return direction * capped_abs

    def _reset_overshoot_state(self) -> None:
        self.overshoot_stage_index = 0
        self.overshoot_trimmed_qty_total = 0
        self.overshoot_active = False
        self.overshoot_trigger_ticks = None
        self.overshoot_crossed_fair_ms = None

    def _reset_decay_state(self) -> None:
        self.shock_decay_steps_applied = 0
        self.shock_decay_trimmed_qty_total = 0

    def _clear_pending_news(self) -> None:
        self.pending_unstructured_news = None
        self.pending_news_score = None
        self.pending_news_fair = None
        self.pending_news_target_inventory = None
        self.pending_news_bucket = None
        self.pending_news_direction = 0
        self.pending_news_base_fair = None
        self.pending_news_matched_phrases = ()
        self.pending_news_matched_unigrams = ()
        self.pending_news_matched_bigrams = ()
        self.pending_news_unknown_phrases = ()
        self.pending_news_unknown_unigrams = ()
        self.pending_news_unknown_bigrams = ()
        self.pending_news_reference_mid = None
        self.pending_news_reference_bid = None
        self.pending_news_reference_ask = None
        self.pending_news_reference_spread = None
        self.pending_news = None
        self.news_takeover_started_ms = None
        self.news_confirmation_deadline_ms = None
        self.news_confirmation_direction = 0
        self.news_confirmed = False
        self.news_confirmation_state = "inactive"

    def _queue_takeover_news(
        self,
        snapshot: StrategySnapshot,
        news_event: NewsEvent,
        *,
        trigger: str,
        reason: str,
        preserve_pending_signal: bool = False,
    ) -> Decision:
        if not preserve_pending_signal:
            self._clear_pending_news()
        self.pending_takeover_news = news_event
        self.mode = "UNWIND"
        if self.unwind_started_ms is None:
            self.unwind_started_ms = snapshot.now_ms
            self.unwind_start_inventory = snapshot.inventory
        self.news_takeover_started_ms = snapshot.now_ms
        return self._build_takeover_flatten_decision(snapshot, trigger=trigger, reason=reason)

    def _clear_after_flatten(self) -> None:
        if self.active_signal_kind == "unstructured" and self.base_fair_value is not None:
            self.fair_value = self.base_fair_value
        self.mode = "IDLE"
        self.active_signal_kind = None
        self.news_fair_value = None
        self.unwind_started_ms = None
        self.unwind_start_inventory = 0
        self.equilibrium_reached_ms = None
        self.initial_shock_edge = 0.0
        self.original_shock_target_inventory = 0
        self.shock_target_inventory = 0
        self.shock_peak_inventory_abs = 0
        self.shock_direction = 0
        self.shock_started_ms = None
        self.shock_reference_mid = None
        self._reset_overshoot_state()
        self._reset_decay_state()
        self.post_event_mids.clear()

    def _update_multiplier_from_equilibrium(self) -> None:
        if (
            self.pe_frozen
            or self.current_event_contaminated
            or self.active_signal_kind == "unstructured"
            or len(self.clean_multiplier_samples) >= self.config.multiplier_clean_sample_limit
            or self.latest_earnings is None
            or self.latest_earnings <= 0
            or not self.post_event_mids
        ):
            return
        settled_mid = float(median(value for _, value in self.post_event_mids))
        estimate = settled_mid / self.latest_earnings
        if estimate <= 0:
            return
        bounded = estimate
        if self.trusted_multiplier is not None:
            clamp = self.config.multiplier_sample_clamp_fraction
            low = self.trusted_multiplier * (1.0 - clamp)
            high = self.trusted_multiplier * (1.0 + clamp)
            bounded = min(max(estimate, low), high)
        self.clean_multiplier_samples.append(bounded)
        self.clean_multiplier_samples = self.clean_multiplier_samples[: self.config.multiplier_clean_sample_limit]
        self.trusted_multiplier = float(median(self.clean_multiplier_samples))
        self.base_fair_value = round(self.trusted_multiplier * self.latest_earnings)
        self.fair_value = self.base_fair_value

    def _start_structured_news(self, snapshot: StrategySnapshot, news_event: NewsEvent, *, allow_takeover: bool = True) -> Decision:
        if allow_takeover and self._news_takeover_required(snapshot):
            return self._queue_takeover_news(
                snapshot,
                news_event,
                trigger="structured_news",
                reason="flattening current inventory before the latest structured A earnings signal takes over",
            )
        self._clear_pending_news()
        self.pending_takeover_news = None
        self.active_signal_kind = "structured"
        self.news_sentiment_score = None
        self.news_sentiment_bucket = None
        self.news_matched_phrases = ()
        self.news_matched_unigrams = ()
        self.news_matched_bigrams = ()
        self.unknown_candidate_phrases = ()
        self.unknown_candidate_unigrams = ()
        self.unknown_candidate_bigrams = ()
        self.news_fair_value = None

        self.structured_event_count += 1
        earnings_value = float(news_event.value or 0.0)
        baseline_mid = self._baseline_mid(snapshot)
        prior_base_fair = self.base_fair_value if self.base_fair_value is not None else self.fair_value
        if baseline_mid is None:
            return self._idle_decision(snapshot, reason="waiting for a pre-report A mid baseline")

        if self.trusted_multiplier is None and self.config.first_earnings_anchor > 0:
            self.trusted_multiplier = baseline_mid / self.config.first_earnings_anchor
        if self.trusted_multiplier is None:
            return self._idle_decision(snapshot, reason="unable to derive an initial multiplier")

        self.latest_earnings = earnings_value
        self.base_fair_value = round(self.trusted_multiplier * earnings_value)
        self.fair_value = self.base_fair_value
        self.shock_reference_mid = snapshot.book.mid if snapshot.book.mid is not None else baseline_mid
        edge = float(self.fair_value) - float(self.shock_reference_mid)
        self.fair_change_ticks = None if prior_base_fair is None else float(self.fair_value - prior_base_fair)
        if abs(edge) < self.config.shock_min_edge_ticks:
            self.current_event_contaminated = False
            self._clear_after_flatten()
            return self._idle_decision(snapshot, reason="structured A earnings moved fair too little for shock mode")

        self.mode = "SHOCK"
        self.shock_started_ms = snapshot.now_ms
        self.shock_direction = _sign(edge)
        self.shock_target_inventory = self._scaled_target(edge, fair_change_ticks=self.fair_change_ticks)
        self.original_shock_target_inventory = self.shock_target_inventory
        self.shock_peak_inventory_abs = 0
        self.initial_shock_edge = edge
        self.equilibrium_reached_ms = None
        self.unwind_started_ms = None
        self.unwind_start_inventory = 0
        self.current_event_contaminated = False
        self._reset_overshoot_state()
        self._reset_decay_state()
        self.post_event_mids.clear()
        if snapshot.book.mid is not None:
            self.post_event_mids.append((snapshot.now_ms, float(snapshot.book.mid)))
        return self._evaluate(snapshot, trigger="structured_news")

    def _handle_unstructured_a_news(self, snapshot: StrategySnapshot, news_event: NewsEvent, *, allow_takeover: bool = True) -> Decision:
        self.pe_frozen = True
        self.current_event_contaminated = True

        sentiment = score_a_unstructured_headline(news_event.content)
        self.news_sentiment_score = sentiment.score
        self.news_sentiment_bucket = sentiment.bucket
        self.news_matched_phrases = sentiment.matched_phrases
        self.news_matched_unigrams = sentiment.matched_unigrams
        self.news_matched_bigrams = sentiment.matched_bigrams
        self.unknown_candidate_phrases = sentiment.unknown_candidate_phrases
        self.unknown_candidate_unigrams = sentiment.unknown_candidate_unigrams
        self.unknown_candidate_bigrams = sentiment.unknown_candidate_bigrams

        base_fair = self._base_fair_for_unstructured(snapshot)
        self.base_fair_value = base_fair if base_fair is not None else self.base_fair_value
        self.news_fair_value = None if base_fair is None else base_fair + (self._news_offset_ticks(sentiment.score, sentiment.bucket) * sentiment.direction)

        if base_fair is None:
            return self._observe_current_state(snapshot, reason="A-news logged without a base fair; not trading it")
        if not sentiment.matched_phrases or sentiment.direction == 0:
            return self._observe_current_state(snapshot, reason="A-news logged for future dictionary work but not tradable yet")

        news_fair = int(self.news_fair_value or base_fair)
        target_inventory = self._news_target_inventory(news_fair=news_fair, score=sentiment.score, bucket=sentiment.bucket, snapshot=snapshot)
        if target_inventory == 0:
            return self._observe_current_state(snapshot, reason="A-news edge is too small to trade after sizing limits")

        self.pending_unstructured_news = news_event
        self.pending_news_score = sentiment.score
        self.pending_news_fair = news_fair
        self.pending_news_target_inventory = target_inventory
        self.pending_news_bucket = sentiment.bucket
        self.pending_news_direction = _sign(target_inventory)
        self.pending_news_base_fair = base_fair
        self.pending_news_matched_phrases = sentiment.matched_phrases
        self.pending_news_matched_unigrams = sentiment.matched_unigrams
        self.pending_news_matched_bigrams = sentiment.matched_bigrams
        self.pending_news_unknown_phrases = sentiment.unknown_candidate_phrases
        self.pending_news_unknown_unigrams = sentiment.unknown_candidate_unigrams
        self.pending_news_unknown_bigrams = sentiment.unknown_candidate_bigrams
        self.pending_news_reference_mid = snapshot.book.mid
        self.pending_news_reference_bid = None if snapshot.book.best_bid is None else snapshot.book.best_bid.px
        self.pending_news_reference_ask = None if snapshot.book.best_ask is None else snapshot.book.best_ask.px
        self.pending_news_reference_spread = snapshot.book.spread
        self.pending_news = news_event.content
        self.news_confirmation_direction = sentiment.direction
        self.news_confirmed = sentiment.bucket in {"strong", "extreme"} or abs(sentiment.score) >= 5.0
        self.news_confirmation_deadline_ms = None if self.news_confirmed else snapshot.now_ms + self.config.news_confirmation_timeout_ms
        self.news_confirmation_state = "immediate" if self.news_confirmed else "pending"
        takeover_required = allow_takeover and self._news_takeover_required(snapshot)
        self.news_takeover_started_ms = snapshot.now_ms if takeover_required else None

        if takeover_required:
            self.news_confirmation_state = "flattening"
            return self._queue_takeover_news(
                snapshot,
                news_event,
                trigger="unstructured_news",
                reason="flattening current inventory before the latest A-news signal takes over",
                preserve_pending_signal=True,
            )

        if self.news_confirmed:
            return self._activate_pending_unstructured_shock(snapshot, trigger="unstructured_news")
        return self._observe_current_state(snapshot, reason="waiting for medium A-news confirmation before taking inventory")

    def _handle_pending_takeover_news(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision | None:
        if self.pending_takeover_news is None:
            return None
        if self._news_takeover_required(snapshot):
            self.mode = "UNWIND"
            if self.unwind_started_ms is None:
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
            self.news_takeover_started_ms = self.news_takeover_started_ms or snapshot.now_ms
            return self._build_takeover_flatten_decision(
                snapshot,
                trigger=trigger,
                reason="flattening current inventory before the latest A signal takes over",
            )

        takeover_news = self.pending_takeover_news
        self.pending_takeover_news = None
        if takeover_news.is_structured_a_earnings:
            return self._start_structured_news(snapshot, takeover_news, allow_takeover=False)
        if takeover_news.is_a_specific_unstructured:
            return self._handle_unstructured_a_news(snapshot, takeover_news, allow_takeover=False)
        return self._idle_decision(snapshot, reason="pending takeover news became non-tradable")

    def _handle_pending_unstructured_news(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision | None:
        if self.pending_unstructured_news is None:
            return None

        if self._news_takeover_required(snapshot):
            self.news_confirmation_state = "flattening"
            self.mode = "UNWIND"
            return self._build_takeover_flatten_decision(snapshot, trigger=trigger, reason="flattening current inventory before the latest A-news signal takes over")

        if not self.news_confirmed:
            if self._news_confirmation_satisfied(snapshot):
                self.news_confirmed = True
                self.news_confirmation_state = "confirmed"
            elif self.news_confirmation_deadline_ms is not None and snapshot.now_ms >= self.news_confirmation_deadline_ms:
                self.news_confirmation_state = "failed"
                self._clear_pending_news()
                if snapshot.inventory != 0:
                    self.mode = "UNWIND"
                    if self.unwind_started_ms is None:
                        self.unwind_started_ms = snapshot.now_ms
                        self.unwind_start_inventory = snapshot.inventory
                    return self._build_unwind_decision(snapshot, trigger=trigger, reason="medium A-news confirmation failed; finishing flatten to zero")
                return self._idle_decision(snapshot, reason="medium A-news confirmation failed")
            else:
                self.news_confirmation_state = "pending"
                return self._observe_current_state(snapshot, reason="waiting for medium A-news confirmation")

        return self._activate_pending_unstructured_shock(snapshot, trigger=trigger)

    def _news_confirmation_satisfied(self, snapshot: StrategySnapshot) -> bool:
        if self.pending_news_direction == 0 or snapshot.book.mid is None or self.pending_news_reference_mid is None:
            return False
        move = self.pending_news_direction * (float(snapshot.book.mid) - float(self.pending_news_reference_mid))
        if move >= self.config.news_confirmation_move_ticks:
            return True

        spread = max(1, int(self.pending_news_reference_spread or snapshot.book.spread or 1))
        if self.pending_news_direction > 0:
            bid_confirm = (
                self.pending_news_reference_bid is not None
                and snapshot.book.best_bid is not None
                and snapshot.book.best_bid.px >= self.pending_news_reference_bid + spread
            )
            ask_confirm = (
                self.pending_news_reference_ask is not None
                and snapshot.book.best_ask is not None
                and snapshot.book.best_ask.px >= self.pending_news_reference_ask + spread
            )
            return bid_confirm or ask_confirm

        ask_confirm = (
            self.pending_news_reference_ask is not None
            and snapshot.book.best_ask is not None
            and snapshot.book.best_ask.px <= self.pending_news_reference_ask - spread
        )
        bid_confirm = (
            self.pending_news_reference_bid is not None
            and snapshot.book.best_bid is not None
            and snapshot.book.best_bid.px <= self.pending_news_reference_bid - spread
        )
        return ask_confirm or bid_confirm

    def _activate_pending_unstructured_shock(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        if self.pending_news_target_inventory is None or self.pending_news_fair is None:
            return self._idle_decision(snapshot, reason="pending A-news signal became incomplete")
        if self._news_takeover_required(snapshot):
            self.news_confirmation_state = "flattening"
            return self._build_takeover_flatten_decision(snapshot, trigger=trigger, reason="flattening current inventory before the latest A-news signal takes over")

        self.mode = "SHOCK"
        self.active_signal_kind = "unstructured"
        self.shock_started_ms = snapshot.now_ms
        self.shock_direction = _sign(self.pending_news_target_inventory)
        self.shock_target_inventory = self.pending_news_target_inventory
        self.original_shock_target_inventory = self.pending_news_target_inventory
        self.shock_peak_inventory_abs = 0
        self.base_fair_value = self.pending_news_base_fair
        self.news_fair_value = self.pending_news_fair
        self.fair_value = self.pending_news_fair
        self.fair_change_ticks = None if self.base_fair_value is None else float(self.pending_news_fair - self.base_fair_value)
        self.shock_reference_mid = snapshot.book.mid if snapshot.book.mid is not None else float(self.pending_news_fair)
        self.initial_shock_edge = float(self.pending_news_fair) - float(self.shock_reference_mid)
        self.equilibrium_reached_ms = None
        self.unwind_started_ms = None
        self.unwind_start_inventory = 0
        self.current_event_contaminated = True
        self._reset_overshoot_state()
        self._reset_decay_state()
        self.post_event_mids.clear()
        if snapshot.book.mid is not None:
            self.post_event_mids.append((snapshot.now_ms, float(snapshot.book.mid)))
        self.news_confirmed = True
        self.news_confirmation_state = "active"
        self._clear_pending_news()
        self.news_confirmation_state = "active"
        return self._build_shock_decision(snapshot, trigger=trigger)

    def _evaluate(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        takeover_decision = self._handle_pending_takeover_news(snapshot, trigger=trigger)
        if takeover_decision is not None:
            return takeover_decision

        pending_decision = self._handle_pending_unstructured_news(snapshot, trigger=trigger)
        if pending_decision is not None:
            return pending_decision

        if self.mode == "SHOCK":
            self._record_mid(snapshot.now_ms, snapshot.book.mid)
            self._update_shock_peak_inventory(snapshot)
            self._ratchet_shock_target_toward_zero(snapshot)
            if self._should_emergency_dump(snapshot):
                self.mode = "UNWIND"
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
                self.equilibrium_reached_ms = None
                self.overshoot_active = False
                self.overshoot_trigger_ticks = None
                self.shock_target_inventory = 0
                return self._build_unwind_decision(
                    snapshot,
                    trigger=trigger,
                    reason="emergency dump after a sharp wrong-way move against the current shock inventory",
                )
            if self._should_force_time_flatten(snapshot):
                self.mode = "UNWIND"
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
                self.equilibrium_reached_ms = None
                self.overshoot_active = False
                self.overshoot_trigger_ticks = None
                self.shock_target_inventory = 0
                return self._build_unwind_decision(
                    snapshot,
                    trigger=trigger,
                    reason="maximum post-news hold time reached; flattening all remaining A shock inventory",
                )
            overshoot_decision = self._maybe_build_overshoot_decision(snapshot, trigger=trigger)
            if overshoot_decision is not None:
                return overshoot_decision
            if self._equilibrium_reached(snapshot.now_ms):
                self._update_multiplier_from_equilibrium()
                self.mode = "UNWIND"
                self.equilibrium_reached_ms = snapshot.now_ms
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
                self.overshoot_active = False
                self.overshoot_trigger_ticks = None
                self.shock_target_inventory = 0
                return self._build_unwind_decision(snapshot, trigger=trigger, reason="A price stabilized after shock; switching to unwind")
            self._maybe_apply_shock_decay(snapshot.now_ms)
            return self._build_shock_decision(snapshot, trigger=trigger)

        if self.mode == "UNWIND":
            if abs(snapshot.inventory) <= self.config.flatten_near_zero_threshold:
                self._clear_after_flatten()
                return self._idle_decision(snapshot, reason="shock inventory flattened back to zero")
            return self._build_unwind_decision(snapshot, trigger=trigger, reason="flattening residual shock inventory back to zero")

        if snapshot.inventory != 0 and self.fair_value is not None:
            self.mode = "UNWIND"
            if self.unwind_started_ms is None:
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
            return self._build_unwind_decision(snapshot, trigger=trigger, reason="non-zero inventory must revert back to zero")

        self.mode = "IDLE"
        self.active_signal_kind = None
        if self.fair_value is None:
            return self._idle_decision(snapshot, reason="waiting for the first structured A earnings report")
        return self._idle_decision(snapshot, reason="waiting for the next structured A signal")

    def _stage_thresholds(self) -> list[int]:
        edge_abs = abs(self.initial_shock_edge)
        if self.active_signal_kind == "unstructured":
            return [
                max(8, round(0.14 * edge_abs)),
                max(14, round(0.26 * edge_abs)),
                max(20, round(0.38 * edge_abs)),
            ]
        return [
            max(12, round(0.20 * edge_abs)),
            max(20, round(0.35 * edge_abs)),
            max(28, round(0.50 * edge_abs)),
        ]

    def _ratchet_shock_target_toward_zero(self, snapshot: StrategySnapshot) -> None:
        if self.shock_target_inventory == 0 or snapshot.inventory == 0:
            return
        target_sign = _sign(self.shock_target_inventory)
        inventory_sign = _sign(snapshot.inventory)
        if (
            inventory_sign == target_sign
            and self.shock_peak_inventory_abs > 0
            and abs(snapshot.inventory) < self.shock_peak_inventory_abs
            and abs(snapshot.inventory) < abs(self.shock_target_inventory)
        ):
            self.shock_target_inventory = target_sign * abs(snapshot.inventory)
            return
        if inventory_sign != target_sign:
            self.shock_target_inventory = 0

    def _update_shock_peak_inventory(self, snapshot: StrategySnapshot) -> None:
        if _sign(snapshot.inventory) != self.shock_direction:
            return
        self.shock_peak_inventory_abs = max(self.shock_peak_inventory_abs, abs(snapshot.inventory))

    def _should_emergency_dump(self, snapshot: StrategySnapshot) -> bool:
        if (
            self.shock_started_ms is None
            or self.shock_reference_mid is None
            or snapshot.book.mid is None
            or snapshot.inventory == 0
        ):
            return False
        if abs(snapshot.inventory) < self.config.shock_emergency_dump_min_inventory:
            return False
        if (snapshot.now_ms - self.shock_started_ms) < self.config.shock_emergency_dump_min_elapsed_ms:
            return False

        adverse_direction = -1 if snapshot.inventory > 0 else 1
        move_from_reference = float(snapshot.book.mid) - float(self.shock_reference_mid)
        adverse_move_ticks = adverse_direction * move_from_reference
        initial_edge_abs = abs(self.initial_shock_edge)
        threshold = max(
            self.config.shock_emergency_dump_ticks,
            round(initial_edge_abs * self.config.shock_emergency_dump_fraction),
        )
        return adverse_move_ticks >= float(threshold)

    def _should_force_time_flatten(self, snapshot: StrategySnapshot) -> bool:
        if self.shock_started_ms is None or snapshot.inventory == 0:
            return False
        return (snapshot.now_ms - self.shock_started_ms) >= self.config.shock_max_hold_ms

    def _maybe_apply_shock_decay(self, now_ms: int) -> None:
        if self.shock_started_ms is None or self.shock_direction == 0 or self.shock_target_inventory == 0:
            return
        original_target_abs = abs(self.original_shock_target_inventory)
        if original_target_abs < self.config.shock_decay_min_inventory:
            return
        if now_ms < self.shock_started_ms + self.config.shock_decay_start_ms:
            return
        if not self._shock_decay_stall_confirmed(now_ms):
            return

        interval_ms = max(1, self.config.shock_decay_interval_ms)
        steps_due = 1 + ((now_ms - self.shock_started_ms - self.config.shock_decay_start_ms) // interval_ms)
        if steps_due <= self.shock_decay_steps_applied:
            return

        step_qty = round(original_target_abs * self.config.shock_decay_fraction)
        step_qty = max(self.config.shock_decay_min_qty, step_qty)
        step_qty = min(self.config.shock_decay_max_qty, step_qty)
        if step_qty <= 0:
            self.shock_decay_steps_applied = steps_due
            return

        additional_steps = steps_due - self.shock_decay_steps_applied
        residual_floor = max(
            self.config.shock_min_position,
            round(original_target_abs * self.config.shock_decay_min_residual_fraction),
        )
        current_target_abs = abs(self.shock_target_inventory)
        allowed_trim = max(0, current_target_abs - residual_floor)
        if allowed_trim <= 0:
            self.shock_decay_steps_applied = steps_due
            return

        trim_qty = min(allowed_trim, additional_steps * step_qty)
        if trim_qty <= 0:
            self.shock_decay_steps_applied = steps_due
            return

        self.shock_target_inventory = self.shock_direction * (current_target_abs - trim_qty)
        self.shock_decay_steps_applied = steps_due
        self.shock_decay_trimmed_qty_total += trim_qty

    def _shock_decay_stall_confirmed(self, now_ms: int) -> bool:
        window_ms = max(1, self.config.shock_decay_stall_window_ms)
        threshold = max(0, self.config.shock_decay_stall_threshold_ticks)
        recent_mids = [mid for ts, mid in self.post_event_mids if ts >= (now_ms - window_ms)]
        if len(recent_mids) < 3:
            return True
        return (max(recent_mids) - min(recent_mids)) <= float(threshold)

    def _maybe_build_overshoot_decision(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision | None:
        self.overshoot_active = False
        self.overshoot_trigger_ticks = None
        if (
            self.fair_value is None
            or snapshot.book.mid is None
            or snapshot.inventory == 0
            or self.shock_target_inventory == 0
            or self.overshoot_stage_index >= 3
        ):
            return None
        if snapshot.inventory * self.shock_direction <= 0:
            return None

        stage_thresholds = self._stage_thresholds()
        threshold = stage_thresholds[self.overshoot_stage_index]
        current_mid = float(snapshot.book.mid)
        crossed_fair = current_mid >= float(self.fair_value) if self.shock_direction > 0 else current_mid <= float(self.fair_value)
        if crossed_fair and self.overshoot_crossed_fair_ms is None:
            self.overshoot_crossed_fair_ms = snapshot.now_ms
        beyond_fair = (
            current_mid >= float(self.fair_value) + threshold
            if self.shock_direction > 0
            else current_mid <= float(self.fair_value) - threshold
        )
        if not beyond_fair:
            return None

        hold_ms = self.config.overshoot_hold_ms
        band_ticks = self.config.overshoot_band_ticks
        reversal_ticks_required = self.config.overshoot_reversal_ticks
        if self.active_signal_kind == "unstructured":
            hold_ms = self.config.news_overshoot_hold_ms
            band_ticks = self.config.news_overshoot_band_ticks
            reversal_ticks_required = self.config.news_overshoot_reversal_ticks

        window_threshold = snapshot.now_ms - hold_ms
        window = [(ts, mid) for ts, mid in self.post_event_mids if ts >= window_threshold]
        if len(window) < 3:
            return None
        mids = [mid for _, mid in window]
        if max(mids) - min(mids) > band_ticks:
            return None

        reversal_ticks = max(mids) - current_mid if self.shock_direction > 0 else current_mid - min(mids)
        if reversal_ticks < reversal_ticks_required:
            return None

        self.overshoot_active = True
        self.overshoot_trigger_ticks = threshold

        original_target_abs = abs(self.original_shock_target_inventory)
        stage_fractions = [
            self.config.overshoot_stage1_fraction,
            self.config.overshoot_stage2_fraction,
            self.config.overshoot_stage3_fraction,
        ]
        residual_fraction = self.config.overshoot_min_residual_fraction
        stage_max_qty = self.config.overshoot_stage_max_qty
        if original_target_abs >= self.config.overshoot_large_position_threshold:
            residual_fraction = max(residual_fraction, self.config.overshoot_large_position_residual_fraction)
            if self.overshoot_stage_index == 0:
                stage_fractions[0] = max(stage_fractions[0], self.config.overshoot_large_position_stage1_fraction)
                stage_max_qty = max(
                    stage_max_qty,
                    round(original_target_abs * self.config.overshoot_large_position_stage1_fraction),
                )

        residual_floor = max(
            self.config.shock_min_position,
            round(original_target_abs * residual_fraction),
        )
        remaining_target_abs = abs(self.shock_target_inventory)
        allowed_trim = max(0, remaining_target_abs - residual_floor)
        if allowed_trim <= 0:
            return None

        trim_qty = round(original_target_abs * stage_fractions[self.overshoot_stage_index])
        trim_qty = max(self.config.overshoot_stage_min_qty, trim_qty)
        trim_qty = min(stage_max_qty, trim_qty, allowed_trim)
        if trim_qty <= 0:
            return None

        new_target_abs = remaining_target_abs - trim_qty
        new_target_inventory = self.shock_direction * new_target_abs
        delta = new_target_inventory - snapshot.inventory
        if snapshot.inventory > 0 and delta >= 0:
            return None
        if snapshot.inventory < 0 and delta <= 0:
            return None

        if delta > 0:
            if snapshot.book.best_ask is None:
                return None
            order = DesiredOrder(
                side="BUY",
                px=snapshot.book.best_ask.px,
                qty=abs(delta),
                aggressive=True,
                intent="overshoot_trim",
                reason="buying back part of a short shock basket during a perceived overshoot",
            )
        else:
            if snapshot.book.best_bid is None:
                return None
            order = DesiredOrder(
                side="SELL",
                px=snapshot.book.best_bid.px,
                qty=abs(delta),
                aggressive=True,
                intent="overshoot_trim",
                reason="selling part of a long shock basket during a perceived overshoot",
            )

        self.shock_target_inventory = new_target_inventory
        self.overshoot_trimmed_qty_total += abs(delta)
        self.overshoot_stage_index += 1

        return Decision(
            mode="SHOCK",
            target_inventory=new_target_inventory,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: trimming staged shock inventory during an overshoot beyond fair {self.fair_value}",
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=False,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _equilibrium_reached(self, now_ms: int) -> bool:
        hold_ms = self.config.equilibrium_hold_ms
        min_elapsed_ms = self.config.equilibrium_min_elapsed_ms
        overshoot_max_wait_ms = self.config.overshoot_max_wait_ms
        residual_edge_ticks = self.config.equilibrium_residual_edge_ticks
        min_capture_fraction = self.config.equilibrium_min_capture_fraction
        if self.active_signal_kind == "unstructured":
            residual_edge_ticks = self.config.news_equilibrium_residual_edge_ticks
            min_capture_fraction = self.config.news_equilibrium_min_capture_fraction

        if self.shock_started_ms is not None and (now_ms - self.shock_started_ms) < min_elapsed_ms:
            return False
        self._trim_window(self.post_event_mids, now_ms, hold_ms)
        if len(self.post_event_mids) < self.config.equilibrium_min_samples:
            return False
        if (self.post_event_mids[-1][0] - self.post_event_mids[0][0]) < hold_ms:
            return False
        mids = [value for _, value in self.post_event_mids]
        if max(mids) - min(mids) > self.config.equilibrium_band_ticks:
            return False
        if self.fair_value is None:
            return True

        settled_mid = float(median(mids))
        current_edge = float(self.fair_value) - settled_mid
        current_direction = _sign(current_edge)
        residual_edge = abs(current_edge)
        if current_direction != self.shock_direction or residual_edge <= residual_edge_ticks:
            return True
        if self.shock_reference_mid is None:
            return False

        initial_edge_abs = abs(self.initial_shock_edge)
        if initial_edge_abs <= 0:
            initial_edge_abs = abs(float(self.fair_value) - float(self.shock_reference_mid))
        if initial_edge_abs <= 0:
            return False

        if self.overshoot_crossed_fair_ms is not None:
            overshoot_wait_elapsed = now_ms - self.overshoot_crossed_fair_ms
            if overshoot_wait_elapsed >= overshoot_max_wait_ms and self.overshoot_stage_index == 0:
                return True

        captured_ticks = abs(settled_mid - float(self.shock_reference_mid))
        captured_fraction = min(1.0, captured_ticks / initial_edge_abs)
        return captured_fraction >= min_capture_fraction

    def _build_shock_decision(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        if self.fair_value is None or snapshot.book.mid is None:
            return self._idle_decision(snapshot, reason="waiting for A top-of-book to trade shock inventory")
        target_inventory = self.shock_target_inventory
        delta = target_inventory - snapshot.inventory
        if delta == 0:
            return Decision(
                mode="SHOCK",
                target_inventory=target_inventory,
                desired_order=None,
                cancel_all=False,
                observe_only=True,
                reason=f"{trigger}: already at the A shock target",
                fair_value=self.fair_value,
                trusted_multiplier=self.trusted_multiplier,
                latest_earnings=self.latest_earnings,
                equilibrium_reached=False,
                shock_reference_mid=self.shock_reference_mid,
            )

        intent = "post_news_shock_take" if self.active_signal_kind == "unstructured" else "post_earnings_shock_take"
        signal_label = "unstructured A-news" if self.active_signal_kind == "unstructured" else "structured A earnings"
        is_initial = trigger in {"structured_news", "unstructured_news", "book", "trade", "timer", "fill"}
        clip = self.config.shock_initial_clip if is_initial else self.config.shock_reinforce_clip

        if delta > 0:
            if snapshot.book.best_ask is None:
                return self._idle_decision(snapshot, reason=f"cannot buy {signal_label} shock inventory without an ask")
            order = DesiredOrder(
                side="BUY",
                px=snapshot.book.best_ask.px,
                qty=min(abs(delta), clip),
                aggressive=True,
                intent=intent,
                reason=f"buying aggressively into {signal_label} edge",
            )
        else:
            if snapshot.book.best_bid is None:
                return self._idle_decision(snapshot, reason=f"cannot sell {signal_label} shock inventory without a bid")
            order = DesiredOrder(
                side="SELL",
                px=snapshot.book.best_bid.px,
                qty=min(abs(delta), clip),
                aggressive=True,
                intent=intent,
                reason=f"selling aggressively into {signal_label} edge",
            )
        return Decision(
            mode="SHOCK",
            target_inventory=target_inventory,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: taking {signal_label} shock inventory toward fair {self.fair_value}",
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=False,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _build_takeover_flatten_decision(self, snapshot: StrategySnapshot, *, trigger: str, reason: str) -> Decision:
        if snapshot.inventory == 0 and snapshot.open_orders:
            return Decision(
                mode="UNWIND",
                target_inventory=0,
                desired_order=None,
                cancel_all=True,
                observe_only=True,
                reason=f"{trigger}: clearing live orders before the latest A-news signal takes over",
                fair_value=self.fair_value,
                trusted_multiplier=self.trusted_multiplier,
                latest_earnings=self.latest_earnings,
                equilibrium_reached=False,
                shock_reference_mid=self.shock_reference_mid,
            )
        if snapshot.inventory == 0:
            return self._observe_current_state(snapshot, reason=reason)
        if snapshot.inventory > 0:
            if snapshot.book.best_bid is None:
                return self._observe_current_state(snapshot, reason="waiting for a bid while flattening before A-news takeover")
            order = DesiredOrder(
                side="SELL",
                px=snapshot.book.best_bid.px,
                qty=abs(snapshot.inventory),
                aggressive=True,
                intent="news_takeover_flatten",
                reason="flattening current inventory before the latest A-news signal takes over",
            )
        else:
            if snapshot.book.best_ask is None:
                return self._observe_current_state(snapshot, reason="waiting for an ask while flattening before A-news takeover")
            order = DesiredOrder(
                side="BUY",
                px=snapshot.book.best_ask.px,
                qty=abs(snapshot.inventory),
                aggressive=True,
                intent="news_takeover_flatten",
                reason="flattening current inventory before the latest A-news signal takes over",
            )
        return Decision(
            mode="UNWIND",
            target_inventory=0,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: {reason}",
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=False,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _build_unwind_decision(self, snapshot: StrategySnapshot, *, trigger: str, reason: str) -> Decision:
        if snapshot.inventory == 0:
            return self._idle_decision(snapshot, reason="already flat after A shock")
        target_inventory = 0
        side = "SELL" if snapshot.inventory > 0 else "BUY"
        qty = abs(snapshot.inventory)
        if side == "BUY":
            if snapshot.book.best_ask is None:
                return self._idle_decision(snapshot, reason="cannot unwind short A inventory without an ask")
            px = snapshot.book.best_ask.px
        else:
            if snapshot.book.best_bid is None:
                return self._idle_decision(snapshot, reason="cannot unwind long A inventory without a bid")
            px = snapshot.book.best_bid.px

        return Decision(
            mode="UNWIND",
            target_inventory=0,
            desired_order=DesiredOrder(
                side=side,
                px=px,
                qty=qty,
                aggressive=True,
                intent="unwind",
                reason="flattening A shock inventory back to zero after equilibrium",
            ),
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: {reason}",
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=True,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _news_takeover_required(self, snapshot: StrategySnapshot) -> bool:
        if snapshot.inventory != 0:
            return True
        if snapshot.open_orders:
            return True
        return False

    def _idle_decision(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        return Decision(
            mode=self.mode,
            target_inventory=0 if self.mode != "SHOCK" else self.shock_target_inventory,
            desired_order=None,
            cancel_all=bool(snapshot.open_orders),
            observe_only=True,
            reason=reason,
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=self.equilibrium_reached_ms is not None,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _observe_current_state(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        return Decision(
            mode=self.mode,
            target_inventory=0 if self.mode != "SHOCK" else self.shock_target_inventory,
            desired_order=None,
            cancel_all=False,
            observe_only=True,
            reason=reason,
            fair_value=self.fair_value,
            trusted_multiplier=self.trusted_multiplier,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=self.equilibrium_reached_ms is not None,
            shock_reference_mid=self.shock_reference_mid,
        )
