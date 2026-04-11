from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..config import MarketCStrategyConfig
from ..core.types import BookSnapshot, Decision, DesiredOrder, ModeState, NewsEvent, StrategySnapshot
from .c_news_sentiment import score_fed_speak_headline


@dataclass(frozen=True)
class MacroPairSignal:
    macro_event_id: str
    source: str
    headline: str | None
    prior_probs: dict[str, float]
    posterior_probs: dict[str, float]
    fair_values: dict[str, int]
    expected_rate_delta_bp: float
    positive_symbol: str | None
    negative_symbol: str | None
    pair_size: int
    bucket: str
    relevance_score: float
    delta_hike: float
    delta_hold: float
    delta_cut: float
    matched_phrases: tuple[str, ...]
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    matched_hike_terms: tuple[str, ...]
    matched_hold_terms: tuple[str, ...]
    matched_cut_terms: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]
    unknown_candidate_unigrams: tuple[str, ...]
    unknown_candidate_bigrams: tuple[str, ...]
    no_trade_reason: str | None
    contract_edges: dict[str, dict[str, float | None]]


@dataclass
class MacroLegState:
    symbol: str
    direction: int
    target_inventory: int
    original_target_inventory: int
    reference_mid: float
    fair_value: float
    bucket: str
    started_ms: int
    best_mid: float
    overshoot_trim_applied: bool = False
    post_event_mids: list[tuple[int, float]] = field(default_factory=list)


class CStrategy:
    """Prediction-market strategy for stock C's Fed contracts."""

    def __init__(self, config: MarketCStrategyConfig):
        self.config = config
        self.mode: ModeState = "IDLE"
        self.session_started_ms: int | None = None
        self.last_signal_ms: int | None = None
        self.last_tradeable_macro_ms: int | None = None

        self.prior_probs = self._equal_probs()
        self.posterior_probs = self._equal_probs()
        self.fair_values = self._fair_values_from_probs(self.posterior_probs)
        self.expected_rate_delta_bp = 0.0

        self.baseline_targets_by_symbol = self._zero_targets()
        self.macro_pair_targets_by_symbol = self._zero_targets()
        self.trading_phase_targets_by_symbol = self._zero_targets()
        self.probe_targets_by_symbol = self._zero_targets()
        self.endgame_targets_by_symbol = self._zero_targets()
        self.final_phase_targets_by_symbol = self._zero_targets()
        self.combined_targets_by_symbol = self._zero_targets()
        self.active_target_inventories = self._zero_targets()
        self.active_target_symbol: str | None = None
        self.active_target_inventory: int = 0

        self.active_macro_signal: MacroPairSignal | None = None
        self.pending_macro_signal: MacroPairSignal | None = None
        self.macro_pair_takeover_symbols: set[str] = set()
        self.macro_leg_states_by_symbol: dict[str, MacroLegState | None] = {symbol: None for symbol in self.tracked_symbols}

        self.last_signal_source: str | None = None
        self.rate_macro_event_id: str | None = None
        self.rate_no_trade_reason: str | None = None
        self.rate_relevance_score: float = 0.0
        self.rate_bucket: str = "none"
        self.rate_hawk_score: float = 0.0
        self.rate_hold_score: float = 0.0
        self.rate_cut_score: float = 0.0
        self.rate_matched_phrases: tuple[str, ...] = ()
        self.rate_matched_unigrams: tuple[str, ...] = ()
        self.rate_matched_bigrams: tuple[str, ...] = ()
        self.rate_matched_hike_terms: tuple[str, ...] = ()
        self.rate_matched_hold_terms: tuple[str, ...] = ()
        self.rate_matched_cut_terms: tuple[str, ...] = ()
        self.rate_unknown_candidate_phrases: tuple[str, ...] = ()
        self.rate_unknown_candidate_unigrams: tuple[str, ...] = ()
        self.rate_unknown_candidate_bigrams: tuple[str, ...] = ()
        self.rate_chosen_edge_ticks: float = 0.0
        self.rate_no_arb_gap_ticks: float = 0.0
        self.rate_edge_map: dict[str, dict[str, float | None]] = self._empty_edge_map()
        self.rate_macro_pair_symbols: list[str] | None = None
        self.rate_macro_leg_reference_mids: dict[str, float | None] = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_fairs: dict[str, float | None] = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_bucket: str | None = None
        self.latest_signal_positive_symbol: str | None = None
        self.latest_signal_negative_symbol: str | None = None
        self.latest_signal_pair_size: int = 0
        self._macro_event_counter = 0

    @property
    def tracked_symbols(self) -> tuple[str, str, str]:
        return self.config.tracked_symbols

    def on_book(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="book")

    def on_trade(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="trade")

    def on_fill(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="fill")

    def on_timer(self, snapshot: StrategySnapshot) -> Decision:
        return self._evaluate(snapshot, trigger="timer")

    def on_news(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> Decision:
        self._ensure_session_started(snapshot)
        if self._trading_phase_active(snapshot):
            signal = self._signal_from_news(snapshot, news_event)
            if signal is None:
                if news_event.is_structured_cpi_print:
                    self.rate_macro_event_id = self._next_macro_event_id("cpi_print", news_event)
                    self.rate_no_trade_reason = "small_cpi_move"
                elif news_event.is_fed_speak_unstructured:
                    self.rate_macro_event_id = self._next_macro_event_id("FedSpeak", news_event)
                    self.rate_no_trade_reason = "headline_below_relevance_threshold"
                else:
                    self.rate_no_trade_reason = "irrelevant_macro_news"
                return self._evaluate(snapshot, trigger="news")
            self._apply_signal_state(signal)
            self.last_tradeable_macro_ms = int(snapshot.now_ms)
            self._queue_or_activate_macro_signal(snapshot, signal)
            self.last_signal_ms = snapshot.now_ms
            return self._evaluate(snapshot, trigger=signal.source)

        if news_event.is_structured_cpi_print or news_event.is_fed_speak_unstructured:
            self.rate_no_trade_reason = "final_phase_market_only"
        else:
            self.rate_no_trade_reason = "irrelevant_macro_news"
        return self._evaluate(snapshot, trigger="news")

    def export_state(self) -> dict[str, int | float | str | list[str] | dict[str, int | float | str | None] | None]:
        return {
            "active_signal_kind": "prediction_market",
            "posterior_hike": round(self.posterior_probs[self.config.fed_hike], 6),
            "posterior_hold": round(self.posterior_probs[self.config.fed_hold], 6),
            "posterior_cut": round(self.posterior_probs[self.config.fed_cut], 6),
            "prior_hike": round(self.prior_probs[self.config.fed_hike], 6),
            "prior_hold": round(self.prior_probs[self.config.fed_hold], 6),
            "prior_cut": round(self.prior_probs[self.config.fed_cut], 6),
            "fair_value_hike": self.fair_values[self.config.fed_hike],
            "fair_value_hold": self.fair_values[self.config.fed_hold],
            "fair_value_cut": self.fair_values[self.config.fed_cut],
            "expected_rate_delta_bp": round(self.expected_rate_delta_bp, 3),
            "rate_macro_event_id": self.rate_macro_event_id,
            "rate_signal_source": self.last_signal_source,
            "rate_no_trade_reason": self.rate_no_trade_reason,
            "rate_relevance_score": round(self.rate_relevance_score, 3),
            "rate_bucket": self.rate_bucket,
            "rate_target_symbol": self.latest_signal_positive_symbol,
            "rate_target_inventory": self.latest_signal_pair_size,
            "rate_chosen_edge_ticks": round(self.rate_chosen_edge_ticks, 3),
            "rate_no_arb_gap_ticks": round(self.rate_no_arb_gap_ticks, 3),
            "rate_long_edge_hike": self._rounded_edge(self.config.fed_hike, "long_edge"),
            "rate_long_edge_hold": self._rounded_edge(self.config.fed_hold, "long_edge"),
            "rate_long_edge_cut": self._rounded_edge(self.config.fed_cut, "long_edge"),
            "rate_short_edge_hike": self._rounded_edge(self.config.fed_hike, "short_edge"),
            "rate_short_edge_hold": self._rounded_edge(self.config.fed_hold, "short_edge"),
            "rate_short_edge_cut": self._rounded_edge(self.config.fed_cut, "short_edge"),
            "rate_hawk_score": round(self.rate_hawk_score, 3),
            "rate_hold_score": round(self.rate_hold_score, 3),
            "rate_cut_score": round(self.rate_cut_score, 3),
            "rate_matched_phrases": list(self.rate_matched_phrases),
            "rate_matched_unigrams": list(self.rate_matched_unigrams),
            "rate_matched_bigrams": list(self.rate_matched_bigrams),
            "rate_matched_hike_terms": list(self.rate_matched_hike_terms),
            "rate_matched_hold_terms": list(self.rate_matched_hold_terms),
            "rate_matched_cut_terms": list(self.rate_matched_cut_terms),
            "rate_unknown_candidate_phrases": list(self.rate_unknown_candidate_phrases),
            "rate_unknown_candidate_unigrams": list(self.rate_unknown_candidate_unigrams),
            "rate_unknown_candidate_bigrams": list(self.rate_unknown_candidate_bigrams),
            "rate_baseline_targets_by_symbol": dict(self.baseline_targets_by_symbol),
            "rate_macro_targets_by_symbol": dict(self.macro_pair_targets_by_symbol),
            "rate_macro_pair_targets_by_symbol": dict(self.macro_pair_targets_by_symbol),
            "rate_trading_phase_targets_by_symbol": dict(self.trading_phase_targets_by_symbol),
            "rate_probe_targets_by_symbol": dict(self.probe_targets_by_symbol),
            "rate_endgame_targets_by_symbol": dict(self.endgame_targets_by_symbol),
            "rate_final_phase_targets_by_symbol": dict(self.final_phase_targets_by_symbol),
            "rate_combined_targets_by_symbol": dict(self.combined_targets_by_symbol),
            "rate_macro_pair_symbols": None if self.rate_macro_pair_symbols is None else list(self.rate_macro_pair_symbols),
            "rate_macro_leg_reference_mids": {
                symbol: None if px is None else round(float(px), 3) for symbol, px in self.rate_macro_leg_reference_mids.items()
            },
            "rate_macro_leg_fairs": {
                symbol: None if px is None else round(float(px), 3) for symbol, px in self.rate_macro_leg_fairs.items()
            },
            "rate_macro_leg_bucket": self.rate_macro_leg_bucket,
            "rate_reversion_targets_by_symbol": self._zero_targets(),
            "rate_pair_targets_by_symbol": self._zero_targets(),
            "rate_reversion_active_symbols": [],
            "rate_reversion_entry_px_by_symbol": {symbol: None for symbol in self.tracked_symbols},
            "rate_reversion_reason_by_symbol": {symbol: None for symbol in self.tracked_symbols},
            "rate_pair_active_pair": None,
            "rate_pair_entry_px_by_symbol": {symbol: None for symbol in self.tracked_symbols},
            "rate_pair_reason_by_symbol": {symbol: None for symbol in self.tracked_symbols},
            "rate_pair_move_by_symbol": {symbol: None for symbol in self.tracked_symbols},
            "rate_pair_last_event_id": None,
            "rate_pair_last_event_kind": None,
            "rate_pair_last_event_pair": None,
            "rate_pair_last_event_reason": None,
            "rate_reversion_last_event_id": None,
            "rate_reversion_last_event_kind": None,
            "rate_reversion_last_event_symbol": None,
            "rate_reversion_last_event_reason": None,
            "rate_reversion_last_entry_px": None,
            "rate_reversion_last_exit_px": None,
        }

    def _evaluate(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        self._ensure_session_started(snapshot)

        if self._is_endgame(snapshot):
            self._clear_trading_phase_state()
            self._clear_probe_targets()
            self._refresh_endgame_targets(snapshot)
            self.final_phase_targets_by_symbol = dict(self.endgame_targets_by_symbol)
            self.trading_phase_targets_by_symbol = self._zero_targets()
        elif self._is_decision_probe_phase(snapshot):
            self._clear_trading_phase_state()
            self._clear_endgame_targets()
            self._refresh_probe_targets(snapshot)
            self.final_phase_targets_by_symbol = dict(self.probe_targets_by_symbol)
            self.trading_phase_targets_by_symbol = self._zero_targets()
        else:
            self._clear_probe_targets()
            self._clear_endgame_targets()
            self.final_phase_targets_by_symbol = self._zero_targets()
            if self._macro_signal_window_active(snapshot):
                self._refresh_baseline_targets(snapshot)
                self._refresh_macro_pair_state(snapshot)
                self._refresh_trading_phase_targets(snapshot)
            else:
                had_live_trading_state = (
                    self.active_macro_signal is not None
                    or self.pending_macro_signal is not None
                    or self._has_rate_exposure(snapshot)
                    or any(target != 0 for target in self.trading_phase_targets_by_symbol.values())
                )
                self._clear_trading_phase_state()
                if had_live_trading_state:
                    self.rate_no_trade_reason = "macro_signal_stale_flatten"
                    if self.last_signal_source in {"FedSpeak", "cpi_print"}:
                        self.last_signal_source = "macro_signal_timeout"

        self._refresh_combined_targets(snapshot)
        if self._has_any_combined_target():
            self.mode = "SHOCK"
            return self._build_entry_decision(snapshot, trigger=trigger)
        if self._has_rate_exposure(snapshot):
            self.mode = "UNWIND"
            return self._build_flatten_decision(snapshot, reason="flattening prediction inventory back to the combined zero target")
        self.mode = "IDLE"
        return self._observe("waiting for CPI, FedSpeak, or contract dislocations")

    def _signal_from_news(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> MacroPairSignal | None:
        if news_event.is_structured_cpi_print:
            return self._signal_from_cpi(snapshot, news_event)
        if news_event.is_fed_speak_unstructured:
            return self._signal_from_fedspeak(snapshot, news_event)
        return None

    def _apply_signal_state(self, signal: MacroPairSignal) -> None:
        self.posterior_probs = dict(signal.posterior_probs)
        self.prior_probs = dict(signal.prior_probs)
        self.fair_values = dict(signal.fair_values)
        self.expected_rate_delta_bp = signal.expected_rate_delta_bp
        self.last_signal_source = signal.source
        self.rate_macro_event_id = signal.macro_event_id
        self.rate_no_trade_reason = signal.no_trade_reason
        self.rate_relevance_score = signal.relevance_score
        self.rate_bucket = signal.bucket
        self.rate_hawk_score = signal.delta_hike
        self.rate_hold_score = signal.delta_hold
        self.rate_cut_score = signal.delta_cut
        self.rate_matched_phrases = signal.matched_phrases
        self.rate_matched_unigrams = signal.matched_unigrams
        self.rate_matched_bigrams = signal.matched_bigrams
        self.rate_matched_hike_terms = signal.matched_hike_terms
        self.rate_matched_hold_terms = signal.matched_hold_terms
        self.rate_matched_cut_terms = signal.matched_cut_terms
        self.rate_unknown_candidate_phrases = signal.unknown_candidate_phrases
        self.rate_unknown_candidate_unigrams = signal.unknown_candidate_unigrams
        self.rate_unknown_candidate_bigrams = signal.unknown_candidate_bigrams
        self.rate_edge_map = {symbol: dict(values) for symbol, values in signal.contract_edges.items()}
        self.rate_macro_pair_symbols = None if signal.positive_symbol is None or signal.negative_symbol is None else [
            signal.positive_symbol,
            signal.negative_symbol,
        ]
        self.latest_signal_positive_symbol = signal.positive_symbol
        self.latest_signal_negative_symbol = signal.negative_symbol
        self.latest_signal_pair_size = signal.pair_size
        self.rate_macro_leg_bucket = signal.bucket
        self.rate_chosen_edge_ticks = 0.0

    def _queue_or_activate_macro_signal(self, snapshot: StrategySnapshot, signal: MacroPairSignal) -> None:
        if signal.positive_symbol is None or signal.negative_symbol is None or signal.pair_size <= 0:
            self.pending_macro_signal = None
            return

        if self._active_macro_pair_symbols():
            self.pending_macro_signal = signal
            self.macro_pair_takeover_symbols = set(self._active_macro_pair_symbols())
            self._clear_active_macro_pair_targets()
            return

        self.pending_macro_signal = None
        self.macro_pair_takeover_symbols.clear()
        self._activate_macro_signal(snapshot, signal)

    def _activate_macro_signal(self, snapshot: StrategySnapshot, signal: MacroPairSignal) -> None:
        self.active_macro_signal = signal
        self.macro_pair_targets_by_symbol = self._zero_targets()
        self.macro_leg_states_by_symbol = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_reference_mids = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_fairs = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_bucket = signal.bucket

        for symbol, direction in (
            (signal.positive_symbol, 1),
            (signal.negative_symbol, -1),
        ):
            if symbol is None:
                continue
            mark = self._mark_price(snapshot.book_for(symbol))
            if mark is None:
                continue
            original_target = signal.pair_size if direction > 0 else -signal.pair_size
            fair = self._clip_price(mark + (direction * self._move_ticks_for_bucket(signal.bucket)))
            state = MacroLegState(
                symbol=symbol,
                direction=direction,
                target_inventory=original_target,
                original_target_inventory=original_target,
                reference_mid=float(mark),
                fair_value=float(fair),
                bucket=signal.bucket,
                started_ms=int(snapshot.now_ms),
                best_mid=float(mark),
                post_event_mids=[(int(snapshot.now_ms), float(mark))],
            )
            self.macro_leg_states_by_symbol[symbol] = state
            self.macro_pair_targets_by_symbol[symbol] = original_target
            self.rate_macro_leg_reference_mids[symbol] = float(mark)
            self.rate_macro_leg_fairs[symbol] = float(fair)

    def _refresh_baseline_targets(self, snapshot: StrategySnapshot) -> None:
        if not self._trading_phase_active(snapshot):
            self.baseline_targets_by_symbol = self._zero_targets()
            return

        lower = float(self.config.baseline_neutral_low_price)
        upper = float(self.config.baseline_neutral_high_price)
        cap = int(self.config.baseline_target_cap)
        full_distance = max(1.0, float(self.config.baseline_full_size_distance_ticks))
        marks = {symbol: self._mark_price(snapshot.book_for(symbol)) for symbol in self.tracked_symbols}
        centerish = [
            symbol
            for symbol, mark in marks.items()
            if mark is not None and lower <= float(mark) <= upper
        ]

        targets = self._zero_targets()
        for symbol in self.tracked_symbols:
            mark = marks[symbol]
            if mark is None:
                continue
            if lower <= float(mark) <= upper:
                targets[symbol] = 0
                continue
            if float(mark) > upper:
                distance = float(mark) - upper
                sign = -1
            else:
                distance = lower - float(mark)
                sign = 1
            magnitude = min(float(cap), float(cap) * min(1.0, distance / full_distance))
            targets[symbol] = self._clip_contract_target(sign * int(round(magnitude)))

        if len(centerish) >= 2:
            for symbol in centerish:
                targets[symbol] = 0

        self.baseline_targets_by_symbol = targets

    def _refresh_macro_pair_state(self, snapshot: StrategySnapshot) -> None:
        if not self._trading_phase_active(snapshot):
            self._clear_trading_phase_state()
            return

        if self.pending_macro_signal is not None and self._macro_takeover_complete(snapshot):
            signal = self.pending_macro_signal
            self.pending_macro_signal = None
            self.macro_pair_takeover_symbols.clear()
            self._activate_macro_signal(snapshot, signal)

        if self.active_macro_signal is None:
            self.macro_pair_targets_by_symbol = self._zero_targets()
            return

        updated_targets = self._zero_targets()
        any_live_leg = False
        for symbol, state in list(self.macro_leg_states_by_symbol.items()):
            if state is None:
                continue
            current = self._updated_leg_state(snapshot, state)
            self.macro_leg_states_by_symbol[symbol] = current
            if current.target_inventory != 0:
                updated_targets[symbol] = int(current.target_inventory)
                any_live_leg = True

        self.macro_pair_targets_by_symbol = updated_targets
        if not any_live_leg:
            self.active_macro_signal = None
            self.rate_macro_leg_reference_mids = {symbol: None for symbol in self.tracked_symbols}
            self.rate_macro_leg_fairs = {symbol: None for symbol in self.tracked_symbols}

    def _updated_leg_state(self, snapshot: StrategySnapshot, state: MacroLegState) -> MacroLegState:
        mark = self._mark_price(snapshot.book_for(state.symbol))
        if mark is None or state.target_inventory == 0:
            return state

        post_event_mids = list(state.post_event_mids)
        post_event_mids.append((int(snapshot.now_ms), float(mark)))
        cutoff = int(snapshot.now_ms) - max(
            int(self.config.macro_equilibrium_hold_ms) * 3,
            20_000,
        )
        post_event_mids = [(ms, mid) for ms, mid in post_event_mids if ms >= cutoff]

        best_mid = max(state.best_mid, float(mark)) if state.direction > 0 else min(state.best_mid, float(mark))
        updated = MacroLegState(
            symbol=state.symbol,
            direction=state.direction,
            target_inventory=state.target_inventory,
            original_target_inventory=state.original_target_inventory,
            reference_mid=state.reference_mid,
            fair_value=state.fair_value,
            bucket=state.bucket,
            started_ms=state.started_ms,
            best_mid=float(best_mid),
            overshoot_trim_applied=state.overshoot_trim_applied,
            post_event_mids=post_event_mids,
        )

        if self._macro_leg_reversal_detected(updated, float(mark)):
            return self._flatten_leg(updated)

        if not updated.overshoot_trim_applied and self._macro_leg_overshoot_detected(updated, float(mark)):
            updated = self._trim_leg_on_overshoot(updated)

        if self._macro_leg_equilibrium_reached(updated, snapshot.now_ms):
            return self._flatten_leg(updated)
        return updated

    def _refresh_trading_phase_targets(self, snapshot: StrategySnapshot) -> None:
        combined = {}
        for symbol in self.tracked_symbols:
            raw = int(self.baseline_targets_by_symbol.get(symbol, 0)) + int(self.macro_pair_targets_by_symbol.get(symbol, 0))
            combined[symbol] = self._clip_contract_target(raw)
        self.trading_phase_targets_by_symbol = combined

    def _refresh_combined_targets(self, snapshot: StrategySnapshot) -> None:
        if self._is_endgame(snapshot):
            combined = dict(self.endgame_targets_by_symbol)
        elif self._is_decision_probe_phase(snapshot):
            combined = dict(self.probe_targets_by_symbol)
        else:
            combined = dict(self.trading_phase_targets_by_symbol)

        self.combined_targets_by_symbol = combined
        self.active_target_inventories = dict(combined)

        best_symbol = None
        best_target = 0
        best_delta_abs = 0
        for symbol in self.tracked_symbols:
            target = int(combined.get(symbol, 0))
            delta_abs = abs(target - snapshot.inventory_for(symbol))
            if delta_abs > best_delta_abs:
                best_delta_abs = delta_abs
                best_symbol = symbol
                best_target = target
        self.active_target_symbol = best_symbol
        self.active_target_inventory = best_target

    def _build_entry_decision(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        target_symbol, target_inventory = self._next_target_delta(snapshot)
        if target_symbol is None or target_inventory == snapshot.inventory_for(target_symbol):
            return self._observe(f"{trigger}: already at the active prediction-market target")

        current_inventory = snapshot.inventory_for(target_symbol)
        delta = target_inventory - current_inventory
        book = snapshot.book_for(target_symbol)
        intent, reason, context_id = self._entry_metadata_for_symbol(target_symbol, delta)
        if delta > 0:
            if book.best_ask is None:
                return self._observe(f"{trigger}: waiting for an ask in {target_symbol}")
            order = DesiredOrder(
                side="BUY",
                px=book.best_ask.px,
                qty=abs(delta),
                aggressive=True,
                intent=intent,
                reason=reason,
                symbol=target_symbol,
                context_id=context_id,
            )
        else:
            if book.best_bid is None:
                return self._observe(f"{trigger}: waiting for a bid in {target_symbol}")
            order = DesiredOrder(
                side="SELL",
                px=book.best_bid.px,
                qty=abs(delta),
                aggressive=True,
                intent=intent,
                reason=reason,
                symbol=target_symbol,
                context_id=context_id,
            )
        return Decision(
            mode="SHOCK",
            target_inventory=target_inventory,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: adjusting prediction-market inventory in {target_symbol}",
            fair_value=self.fair_values.get(target_symbol),
        )

    def _build_flatten_decision(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        positions = {symbol: snapshot.inventory_for(symbol) for symbol in self.tracked_symbols}
        nonzero_positions = {symbol: qty for symbol, qty in positions.items() if qty != 0}
        if not nonzero_positions:
            if any(snapshot.open_orders_for(symbol) for symbol in self.tracked_symbols):
                return Decision(
                    mode="UNWIND",
                    target_inventory=0,
                    desired_order=None,
                    cancel_all=True,
                    observe_only=True,
                    reason=reason,
                    fair_value=self.fair_values.get(self.active_target_symbol or self.config.fed_hold),
                )
            return self._observe("prediction inventory already flat")

        symbol, qty = max(nonzero_positions.items(), key=lambda item: abs(item[1]))
        book = snapshot.book_for(symbol)
        intent, context_id = self._flatten_metadata_for_symbol(symbol)
        if qty > 0:
            if book.best_bid is None:
                return self._observe(f"waiting for a bid while flattening {symbol}")
            order = DesiredOrder(
                side="SELL",
                px=book.best_bid.px,
                qty=abs(qty),
                aggressive=True,
                intent=intent,
                reason=reason,
                symbol=symbol,
                context_id=context_id,
            )
        else:
            if book.best_ask is None:
                return self._observe(f"waiting for an ask while flattening {symbol}")
            order = DesiredOrder(
                side="BUY",
                px=book.best_ask.px,
                qty=abs(qty),
                aggressive=True,
                intent=intent,
                reason=reason,
                symbol=symbol,
                context_id=context_id,
            )
        return Decision(
            mode="UNWIND",
            target_inventory=0,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=reason,
            fair_value=self.fair_values.get(symbol),
        )

    def _entry_metadata_for_symbol(self, symbol: str, delta: int) -> tuple[str, str, str | None]:
        if self.endgame_targets_by_symbol.get(symbol, 0) != 0:
            return "prediction_market_take_endgame", "building final winner/loser resolution inventory", self.rate_macro_event_id
        if self.probe_targets_by_symbol.get(symbol, 0) != 0:
            return "prediction_market_take_probe", "building small decision-phase winner/loser inventory", self.rate_macro_event_id
        if self.macro_pair_targets_by_symbol.get(symbol, 0) != 0:
            return "prediction_market_take_macro", "building macro pair shock inventory", self.rate_macro_event_id
        return "prediction_market_take_baseline", "adjusting baseline contrarian inventory", self.rate_macro_event_id

    def _flatten_metadata_for_symbol(self, symbol: str) -> tuple[str, str | None]:
        if self.endgame_targets_by_symbol.get(symbol, 0) != 0:
            return "prediction_market_unwind_endgame", self.rate_macro_event_id
        if self.probe_targets_by_symbol.get(symbol, 0) != 0:
            return "prediction_market_unwind_probe", self.rate_macro_event_id
        if self.macro_pair_targets_by_symbol.get(symbol, 0) == 0 and self.active_macro_signal is not None:
            return "prediction_market_unwind_macro", self.rate_macro_event_id
        return "prediction_market_unwind_trading", self.rate_macro_event_id

    def _signal_from_cpi(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> MacroPairSignal | None:
        prior_probs, no_arb_gap = self._prior_from_market(snapshot)
        surprise = float(news_event.actual or 0.0) - float(news_event.forecast or 0.0)
        abs_surprise = abs(surprise)
        if abs_surprise < self.config.cpi_small_surprise:
            return None
        if abs_surprise < self.config.cpi_medium_surprise:
            shift = self.config.cpi_small_logit_shift
            bucket = "light"
        elif abs_surprise < self.config.cpi_large_surprise:
            shift = self.config.cpi_medium_logit_shift
            bucket = "medium"
        elif abs_surprise < (self.config.cpi_large_surprise * 1.75):
            shift = self.config.cpi_large_logit_shift
            bucket = "strong"
        else:
            shift = self.config.cpi_large_logit_shift * 1.35
            bucket = "extreme"

        if surprise > 0:
            deltas = {
                self.config.fed_hike: shift,
                self.config.fed_hold: -0.45 * shift,
                self.config.fed_cut: -shift,
            }
        else:
            deltas = {
                self.config.fed_hike: -shift,
                self.config.fed_hold: -0.45 * shift,
                self.config.fed_cut: shift,
            }
        return self._build_macro_pair_signal(
            snapshot,
            macro_event_id=self._next_macro_event_id("cpi_print", news_event),
            source="cpi_print",
            headline=f"CPI actual {news_event.actual} vs forecast {news_event.forecast}",
            deltas=deltas,
            bucket=bucket,
            relevance_score=abs_surprise,
            matched_phrases=(),
            matched_unigrams=(),
            matched_bigrams=(),
            matched_hike_terms=(),
            matched_hold_terms=(),
            matched_cut_terms=(),
            unknown_candidate_phrases=(),
            unknown_candidate_unigrams=(),
            unknown_candidate_bigrams=(),
            no_arb_gap_ticks=no_arb_gap,
            prior_probs=prior_probs,
        )

    def _signal_from_fedspeak(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> MacroPairSignal | None:
        sentiment = score_fed_speak_headline(str(news_event.content or ""))
        if sentiment.relevance_score < self.config.news_relevance_threshold or sentiment.bucket == "none":
            return None
        prior_probs, no_arb_gap = self._prior_from_market(snapshot)
        component_scale = self._bucket_logit_shift(sentiment.bucket) / max(self.config.news_score_delta_divisor, 1e-6)
        deltas = {
            self.config.fed_hike: component_scale * sentiment.delta_hike,
            self.config.fed_hold: component_scale * sentiment.delta_hold,
            self.config.fed_cut: component_scale * sentiment.delta_cut,
        }
        return self._build_macro_pair_signal(
            snapshot,
            macro_event_id=self._next_macro_event_id("FedSpeak", news_event),
            source="FedSpeak",
            headline=str(news_event.content or ""),
            deltas=deltas,
            bucket=sentiment.bucket,
            relevance_score=sentiment.relevance_score,
            matched_phrases=sentiment.matched_phrases,
            matched_unigrams=sentiment.matched_unigrams,
            matched_bigrams=sentiment.matched_bigrams,
            matched_hike_terms=sentiment.matched_hike_terms,
            matched_hold_terms=sentiment.matched_hold_terms,
            matched_cut_terms=sentiment.matched_cut_terms,
            unknown_candidate_phrases=sentiment.unknown_candidate_phrases,
            unknown_candidate_unigrams=sentiment.unknown_candidate_unigrams,
            unknown_candidate_bigrams=sentiment.unknown_candidate_bigrams,
            no_arb_gap_ticks=no_arb_gap,
            prior_probs=prior_probs,
        )

    def _build_macro_pair_signal(
        self,
        snapshot: StrategySnapshot,
        *,
        macro_event_id: str,
        source: str,
        headline: str | None,
        deltas: dict[str, float],
        bucket: str,
        relevance_score: float,
        matched_phrases: tuple[str, ...],
        matched_unigrams: tuple[str, ...],
        matched_bigrams: tuple[str, ...],
        matched_hike_terms: tuple[str, ...],
        matched_hold_terms: tuple[str, ...],
        matched_cut_terms: tuple[str, ...],
        unknown_candidate_phrases: tuple[str, ...],
        unknown_candidate_unigrams: tuple[str, ...],
        unknown_candidate_bigrams: tuple[str, ...],
        no_arb_gap_ticks: float,
        prior_probs: dict[str, float],
    ) -> MacroPairSignal | None:
        posterior_probs = self._posterior_from_deltas(
            prior_probs,
            (
                deltas[self.config.fed_hike],
                deltas[self.config.fed_hold],
                deltas[self.config.fed_cut],
            ),
        )
        fair_values = self._fair_values_from_probs(posterior_probs)
        positive_symbol, negative_symbol, no_trade_reason = self._select_macro_pair(snapshot, deltas)
        pair_size = 0 if positive_symbol is None or negative_symbol is None else min(
            self.config.trading_macro_target_cap,
            self._position_for_bucket(bucket),
        )
        contract_edges = self._contract_edges(snapshot, fair_values)
        return MacroPairSignal(
            macro_event_id=macro_event_id,
            source=source,
            headline=headline,
            prior_probs=prior_probs,
            posterior_probs=posterior_probs,
            fair_values=fair_values,
            expected_rate_delta_bp=self._expected_rate_delta_bp(posterior_probs),
            positive_symbol=positive_symbol,
            negative_symbol=negative_symbol,
            pair_size=pair_size,
            bucket=bucket,
            relevance_score=relevance_score,
            delta_hike=deltas[self.config.fed_hike],
            delta_hold=deltas[self.config.fed_hold],
            delta_cut=deltas[self.config.fed_cut],
            matched_phrases=matched_phrases,
            matched_unigrams=matched_unigrams,
            matched_bigrams=matched_bigrams,
            matched_hike_terms=matched_hike_terms,
            matched_hold_terms=matched_hold_terms,
            matched_cut_terms=matched_cut_terms,
            unknown_candidate_phrases=unknown_candidate_phrases,
            unknown_candidate_unigrams=unknown_candidate_unigrams,
            unknown_candidate_bigrams=unknown_candidate_bigrams,
            no_trade_reason=no_trade_reason,
            contract_edges=contract_edges,
        )

    def _select_macro_pair(
        self,
        snapshot: StrategySnapshot,
        deltas: dict[str, float],
    ) -> tuple[str | None, str | None, str | None]:
        positive_symbol = max(self.tracked_symbols, key=lambda symbol: deltas[symbol])
        negative_symbol = min(self.tracked_symbols, key=lambda symbol: deltas[symbol])
        positive_value = float(deltas[positive_symbol])
        negative_value = float(deltas[negative_symbol])
        threshold = float(self.config.macro_pair_min_delta)
        if positive_value < threshold:
            return None, None, "no_meaningful_positive_leg"
        if negative_value > -threshold:
            if positive_symbol == self.config.fed_hold and positive_value >= float(self.config.macro_pair_hold_tail_fallback_delta):
                tail_symbols = [self.config.fed_hike, self.config.fed_cut]
                tail_marks = {
                    symbol: self._mark_price(snapshot.book_for(symbol))
                    for symbol in tail_symbols
                }
                available = [symbol for symbol in tail_symbols if tail_marks[symbol] is not None]
                if not available:
                    return None, None, "macro_pair_tail_unavailable"
                negative_symbol = max(available, key=lambda symbol: float(tail_marks[symbol] or 0.0))
            else:
                return None, None, "no_meaningful_negative_leg"
        if positive_symbol == negative_symbol:
            return None, None, "pair_selection_failed"
        return positive_symbol, negative_symbol, None

    def _contract_edges(self, snapshot: StrategySnapshot, fair_values: dict[str, int]) -> dict[str, dict[str, float | None]]:
        edges = self._empty_edge_map()
        for symbol in self.tracked_symbols:
            book = snapshot.book_for(symbol)
            fair_value = fair_values[symbol]
            if book.best_ask is not None:
                edges[symbol]["long_edge"] = round(float(fair_value) - float(book.best_ask.px), 3)
            if book.best_bid is not None:
                edges[symbol]["short_edge"] = round(float(book.best_bid.px) - float(fair_value), 3)
        return edges

    def _macro_takeover_complete(self, snapshot: StrategySnapshot) -> bool:
        if not self.macro_pair_takeover_symbols:
            return True
        for symbol in self.macro_pair_takeover_symbols:
            if snapshot.inventory_for(symbol) != int(self.baseline_targets_by_symbol.get(symbol, 0)):
                return False
            if snapshot.open_orders_for(symbol):
                return False
        return True

    def _clear_active_macro_pair_targets(self) -> None:
        self.active_macro_signal = None
        self.macro_pair_targets_by_symbol = self._zero_targets()
        self.macro_leg_states_by_symbol = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_reference_mids = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_fairs = {symbol: None for symbol in self.tracked_symbols}

    def _clear_trading_phase_state(self) -> None:
        self.baseline_targets_by_symbol = self._zero_targets()
        self.macro_pair_targets_by_symbol = self._zero_targets()
        self.trading_phase_targets_by_symbol = self._zero_targets()
        self.active_macro_signal = None
        self.pending_macro_signal = None
        self.macro_pair_takeover_symbols.clear()
        self.macro_leg_states_by_symbol = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_pair_symbols = None
        self.rate_macro_leg_reference_mids = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_fairs = {symbol: None for symbol in self.tracked_symbols}
        self.rate_macro_leg_bucket = None

    def _macro_leg_overshoot_detected(self, state: MacroLegState, mark: float) -> bool:
        trigger = max(
            float(self.config.macro_overshoot_min_trigger_ticks),
            float(self._move_ticks_for_bucket(state.bucket)) * float(self.config.macro_overshoot_trigger_fraction),
        )
        return (state.direction * (mark - state.fair_value)) >= trigger

    def _trim_leg_on_overshoot(self, state: MacroLegState) -> MacroLegState:
        trimmed_abs = max(
            int(self.config.macro_overshoot_min_residual_qty),
            int(round(abs(state.original_target_inventory) * (1.0 - float(self.config.macro_overshoot_trim_fraction)))),
        )
        trimmed_abs = min(trimmed_abs, abs(state.target_inventory))
        return MacroLegState(
            symbol=state.symbol,
            direction=state.direction,
            target_inventory=(trimmed_abs * state.direction),
            original_target_inventory=state.original_target_inventory,
            reference_mid=state.reference_mid,
            fair_value=state.fair_value,
            bucket=state.bucket,
            started_ms=state.started_ms,
            best_mid=state.best_mid,
            overshoot_trim_applied=True,
            post_event_mids=list(state.post_event_mids),
        )

    def _macro_leg_equilibrium_reached(self, state: MacroLegState, now_ms: int) -> bool:
        if (now_ms - state.started_ms) < int(self.config.macro_equilibrium_min_elapsed_ms):
            return False
        recent = [mid for ts, mid in state.post_event_mids if ts >= (now_ms - int(self.config.macro_equilibrium_hold_ms))]
        if len(recent) < int(self.config.macro_equilibrium_min_samples):
            return False
        if (max(recent) - min(recent)) > float(self.config.macro_equilibrium_band_ticks):
            return False
        return abs(float(recent[-1]) - float(state.fair_value)) <= float(self.config.macro_equilibrium_residual_edge_ticks)

    def _macro_leg_reversal_detected(self, state: MacroLegState, mark: float) -> bool:
        best_progress = state.direction * (float(state.best_mid) - float(state.reference_mid))
        current_progress = state.direction * (float(mark) - float(state.reference_mid))
        return (
            best_progress >= float(self.config.macro_reversal_min_progress_ticks)
            and (best_progress - current_progress) >= float(self.config.macro_reversal_exit_ticks)
        )

    def _flatten_leg(self, state: MacroLegState) -> MacroLegState:
        return MacroLegState(
            symbol=state.symbol,
            direction=state.direction,
            target_inventory=0,
            original_target_inventory=state.original_target_inventory,
            reference_mid=state.reference_mid,
            fair_value=state.fair_value,
            bucket=state.bucket,
            started_ms=state.started_ms,
            best_mid=state.best_mid,
            overshoot_trim_applied=state.overshoot_trim_applied,
            post_event_mids=list(state.post_event_mids),
        )

    def _move_ticks_for_bucket(self, bucket: str) -> int:
        if bucket == "light":
            return int(self.config.macro_move_light_ticks)
        if bucket == "medium":
            return int(self.config.macro_move_medium_ticks)
        if bucket == "strong":
            return int(self.config.macro_move_strong_ticks)
        if bucket == "extreme":
            return int(self.config.macro_move_extreme_ticks)
        return int(self.config.macro_move_light_ticks)

    def _refresh_endgame_targets(self, snapshot: StrategySnapshot) -> None:
        if not self._books_complete_for_final(snapshot):
            fallback_targets = self._fallback_final_targets(snapshot, self.endgame_targets_by_symbol)
            if fallback_targets is not None:
                self.endgame_targets_by_symbol = fallback_targets
                self.rate_macro_event_id = f"endgame_hold:{snapshot.now_ms}"
                self.last_signal_source = "endgame_hold_previous"
                self.rate_no_trade_reason = "endgame_prior_unavailable_holding_previous"
                return

        prior_probs, no_arb_gap = self._prior_from_market(snapshot)
        self.posterior_probs = dict(prior_probs)
        self.prior_probs = dict(prior_probs)
        self.fair_values = self._fair_values_from_probs(prior_probs)
        self.expected_rate_delta_bp = self._expected_rate_delta_bp(prior_probs)
        self.rate_macro_event_id = f"endgame:{snapshot.now_ms}"
        self.last_signal_source = "endgame_market_prior"
        self.rate_no_trade_reason = None
        self.rate_relevance_score = 1.0
        self.rate_bucket = "extreme"
        self.rate_hawk_score = 0.0
        self.rate_hold_score = 0.0
        self.rate_cut_score = 0.0
        self.rate_no_arb_gap_ticks = no_arb_gap
        self.rate_edge_map = self._empty_edge_map()

        marks = {symbol: float(self._mark_price(snapshot.book_for(symbol)) or 0.0) for symbol in self.tracked_symbols}
        winner_symbol = max(self.tracked_symbols, key=lambda symbol: marks[symbol])
        targets = {symbol: int(self.config.endgame_short_target) for symbol in self.tracked_symbols}
        targets[winner_symbol] = int(self.config.endgame_long_target)
        for symbol in self.tracked_symbols:
            if marks[symbol] < float(self.config.endgame_almost_dead_price):
                targets[symbol] = int(self.config.endgame_short_target)
        self.endgame_targets_by_symbol = targets

    def _refresh_probe_targets(self, snapshot: StrategySnapshot) -> None:
        if not self._books_complete_for_final(snapshot):
            fallback_targets = self._fallback_final_targets(snapshot, self.probe_targets_by_symbol)
            if fallback_targets is not None:
                self.probe_targets_by_symbol = fallback_targets
                self.rate_macro_event_id = f"decision_probe_hold:{snapshot.now_ms}"
                self.last_signal_source = "decision_probe_hold_previous"
                self.rate_no_trade_reason = "decision_probe_prior_unavailable_holding_previous"
                return

        prior_probs, no_arb_gap = self._prior_from_market(snapshot)
        self.posterior_probs = dict(prior_probs)
        self.prior_probs = dict(prior_probs)
        self.fair_values = self._fair_values_from_probs(prior_probs)
        self.expected_rate_delta_bp = self._expected_rate_delta_bp(prior_probs)
        self.rate_macro_event_id = f"decision_probe:{snapshot.now_ms}"
        self.last_signal_source = "decision_probe_market_prior"
        self.rate_no_trade_reason = None
        self.rate_relevance_score = 1.0
        self.rate_bucket = "medium"
        self.rate_hawk_score = 0.0
        self.rate_hold_score = 0.0
        self.rate_cut_score = 0.0
        self.rate_no_arb_gap_ticks = no_arb_gap
        self.rate_edge_map = self._empty_edge_map()

        marks = {symbol: float(self._mark_price(snapshot.book_for(symbol)) or 0.0) for symbol in self.tracked_symbols}
        ranked = sorted(self.tracked_symbols, key=lambda symbol: marks[symbol], reverse=True)
        leader_symbol = ranked[0]
        leader_mark = marks[leader_symbol]
        second_mark = marks[ranked[1]]
        confident = (
            leader_mark >= float(self.config.decision_probe_confident_price)
            or (leader_mark - second_mark) >= float(self.config.decision_probe_confidence_gap_ticks)
        )
        size = int(self.config.decision_probe_confident_target if confident else self.config.decision_probe_base_target)
        targets = {symbol: -size for symbol in self.tracked_symbols}
        targets[leader_symbol] = size
        self.probe_targets_by_symbol = targets

    def _books_complete_for_final(self, snapshot: StrategySnapshot) -> bool:
        return all(self._mark_price(snapshot.book_for(symbol)) is not None for symbol in self.tracked_symbols)

    def _fallback_final_targets(self, snapshot: StrategySnapshot, current_targets: dict[str, int]) -> dict[str, int] | None:
        inventory_targets = {symbol: self._clip_contract_target(snapshot.inventory_for(symbol)) for symbol in self.tracked_symbols}
        if any(inventory_targets.values()):
            return inventory_targets
        existing_targets = {symbol: self._clip_contract_target(current_targets.get(symbol, 0)) for symbol in self.tracked_symbols}
        if any(existing_targets.values()):
            return existing_targets
        previous_targets = {symbol: self._clip_contract_target(self.combined_targets_by_symbol.get(symbol, 0)) for symbol in self.tracked_symbols}
        if any(previous_targets.values()):
            return previous_targets
        return None

    def _next_macro_event_id(self, source: str, news_event: NewsEvent) -> str:
        self._macro_event_counter += 1
        tick = "na" if news_event.tick is None else str(news_event.tick)
        return f"{source}:{tick}:{news_event.now_ms}:{self._macro_event_counter}"

    def _empty_edge_map(self) -> dict[str, dict[str, float | None]]:
        return {symbol: {"long_edge": None, "short_edge": None} for symbol in self.tracked_symbols}

    def _rounded_edge(self, symbol: str, side: str) -> float | None:
        value = self.rate_edge_map.get(symbol, {}).get(side)
        return None if value is None else round(float(value), 3)

    def _next_target_delta(self, snapshot: StrategySnapshot) -> tuple[str | None, int]:
        best_symbol: str | None = None
        best_target = 0
        best_delta_abs = 0
        for symbol in self.tracked_symbols:
            target = int(self.combined_targets_by_symbol.get(symbol, 0))
            delta_abs = abs(target - snapshot.inventory_for(symbol))
            if delta_abs > best_delta_abs:
                best_symbol = symbol
                best_target = target
                best_delta_abs = delta_abs
        return best_symbol, best_target

    @staticmethod
    def _mark_price(book: BookSnapshot) -> float | None:
        if book.mid is not None:
            return float(book.mid)
        if book.best_bid is not None:
            return float(book.best_bid.px)
        if book.best_ask is not None:
            return float(book.best_ask.px)
        return None

    def _position_for_bucket(self, bucket: str) -> int:
        if bucket == "light":
            return int(self.config.signal_light_position)
        if bucket == "medium":
            return int(self.config.signal_medium_position)
        if bucket == "strong":
            return int(self.config.signal_strong_position)
        if bucket == "extreme":
            return int(self.config.signal_extreme_position)
        return 0

    def _bucket_logit_shift(self, bucket: str) -> float:
        if bucket == "light":
            return float(self.config.news_light_logit_shift)
        if bucket == "medium":
            return float(self.config.news_medium_logit_shift)
        if bucket == "strong":
            return float(self.config.news_strong_logit_shift)
        if bucket == "extreme":
            return float(self.config.news_extreme_logit_shift)
        return 0.0

    def _prior_from_market(self, snapshot: StrategySnapshot) -> tuple[dict[str, float], float]:
        raw_values: dict[str, float] = {}
        for symbol in self.tracked_symbols:
            mark = self._mark_price(snapshot.book_for(symbol))
            if mark is None:
                return self._equal_probs(), 0.0
            raw_values[symbol] = max(0.0, min(float(self.config.prediction_scale), float(mark)))
        total = sum(raw_values.values())
        scale = float(self.config.prediction_scale)
        floor = max(float(self.config.posterior_floor_probability), 1e-6)
        logits: dict[str, float] = {}
        for symbol, value in raw_values.items():
            probability_like = max(floor, min(1.0 - floor, value / scale))
            logits[symbol] = math.log(probability_like / (1.0 - probability_like))
        max_logit = max(logits.values())
        exp_values = {symbol: math.exp(logit - max_logit) for symbol, logit in logits.items()}
        exp_total = sum(exp_values.values())
        if exp_total <= 0:
            return self._equal_probs(), total - scale
        return {symbol: exp_values[symbol] / exp_total for symbol in self.tracked_symbols}, total - scale

    def _posterior_from_deltas(self, prior_probs: dict[str, float], deltas: tuple[float, float, float]) -> dict[str, float]:
        floor = max(float(self.config.posterior_floor_probability), 1e-6)
        logits = []
        for index, symbol in enumerate(self.tracked_symbols):
            prior = max(floor, min(1.0 - floor, float(prior_probs.get(symbol, 1.0 / 3.0))))
            logits.append(math.log(prior) + deltas[index])
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        total = sum(exp_values)
        if total <= 0:
            return self._equal_probs()
        return {symbol: exp_values[index] / total for index, symbol in enumerate(self.tracked_symbols)}

    def _fair_values_from_probs(self, probs: dict[str, float]) -> dict[str, int]:
        return {
            symbol: int(round(float(self.config.prediction_scale) * float(probs.get(symbol, 0.0))))
            for symbol in self.tracked_symbols
        }

    def _expected_rate_delta_bp(self, probs: dict[str, float]) -> float:
        return (
            float(self.config.expected_rate_step_bp) * float(probs.get(self.config.fed_hike, 0.0))
            - float(self.config.expected_rate_step_bp) * float(probs.get(self.config.fed_cut, 0.0))
        )

    def _has_rate_exposure(self, snapshot: StrategySnapshot) -> bool:
        if any(snapshot.inventory_for(symbol) != 0 for symbol in self.tracked_symbols):
            return True
        if any(snapshot.open_orders_for(symbol) for symbol in self.tracked_symbols):
            return True
        return False

    def _has_any_combined_target(self) -> bool:
        return any(target != 0 for target in self.combined_targets_by_symbol.values())

    def _active_macro_pair_symbols(self) -> list[str]:
        return [symbol for symbol, target in self.macro_pair_targets_by_symbol.items() if target != 0]

    def _clear_endgame_targets(self) -> None:
        self.endgame_targets_by_symbol = self._zero_targets()

    def _clear_probe_targets(self) -> None:
        self.probe_targets_by_symbol = self._zero_targets()

    def _zero_targets(self) -> dict[str, int]:
        return {symbol: 0 for symbol in self.tracked_symbols}

    def _equal_probs(self) -> dict[str, float]:
        return {symbol: 1.0 / 3.0 for symbol in self.tracked_symbols}

    def _ensure_session_started(self, snapshot: StrategySnapshot) -> None:
        if self.session_started_ms is None:
            self.session_started_ms = int(snapshot.now_ms)

    def _time_remaining_ms(self, snapshot: StrategySnapshot) -> int:
        if self.session_started_ms is None:
            return int(self.config.round_duration_ms)
        return max(0, int(self.config.round_duration_ms) - (int(snapshot.now_ms) - int(self.session_started_ms)))

    def _is_endgame(self, snapshot: StrategySnapshot) -> bool:
        return self._time_remaining_ms(snapshot) <= int(self.config.endgame_countdown_ms)

    def _is_decision_probe_phase(self, snapshot: StrategySnapshot) -> bool:
        remaining = self._time_remaining_ms(snapshot)
        return int(self.config.endgame_countdown_ms) < remaining <= int(self.config.decision_probe_countdown_ms)

    def _trading_phase_active(self, snapshot: StrategySnapshot) -> bool:
        return self._time_remaining_ms(snapshot) > int(self.config.decision_probe_countdown_ms)

    def _macro_signal_window_active(self, snapshot: StrategySnapshot) -> bool:
        if self.last_tradeable_macro_ms is None:
            return False
        return (int(snapshot.now_ms) - int(self.last_tradeable_macro_ms)) <= int(self.config.macro_signal_timeout_ms)

    def _clip_contract_target(self, value: int | float) -> int:
        limit = int(self.config.max_absolute_position_per_contract)
        return max(-limit, min(limit, int(round(float(value)))))

    def _clip_price(self, value: float) -> int:
        return max(0, min(int(self.config.prediction_scale), int(round(float(value)))))

    def _observe(self, reason: str) -> Decision:
        return Decision(
            mode=self.mode,
            target_inventory=self.active_target_inventory if self.mode == "SHOCK" else 0,
            desired_order=None,
            cancel_all=False,
            observe_only=True,
            reason=reason,
            fair_value=None if self.active_target_symbol is None else self.fair_values.get(self.active_target_symbol),
        )
