from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
from statistics import median

from ..config import StrategyConfig
from ..core.types import Decision, DesiredOrder, ModeState, NewsEvent, StrategySnapshot


FLOAT_RE = r"[-+]?\d*\.?\d+"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(frozen=True)
class _CRateContext:
    q_hike: float
    q_hold: float
    q_cut: float
    market_rate_bp: float


class CMarketMakerStrategy:
    """Stock-C market maker with A-style structured earnings shock handling."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mode: ModeState = "IDLE"
        self.trusted_multiplier: float | None = None
        self.fair_value: int | None = None
        self.base_fair_value: int | None = None
        self.news_fair_value: int | None = None
        self.fair_change_ticks: float | None = None
        self.latest_earnings: float | None = None
        self.active_signal_kind: str | None = None

        self.microprice: float | None = None
        self.recent_trade_signals: deque[tuple[int, int]] = deque()
        self.recent_trade_count: int = 0
        self.flow_score: int = 0
        self.burst_active: bool = False
        self.quote_side: str | None = None
        self.quote_px: int | None = None
        self.quote_reason: str | None = None

        self.current_eps_c: float = float(config.c_default_eps)
        self.baseline_eps_c: float | None = None
        self.have_real_eps_c: bool = False
        self.last_c_earnings_delta: float = 0.0
        self.recent_c_earnings_deltas: deque[float] = deque(maxlen=4)
        self.anchor_price: float | None = None
        self.anchor_eps: float | None = None
        self.anchor_rate_bp: float | None = None
        self.last_rate_context: _CRateContext | None = None

        self.pending_earnings_value: float | None = None
        self.pending_earnings_tick: int | None = None
        self.pending_news_target_inventory: int | None = None
        self.pending_news: str | None = None
        self.pending_news_reference_mid: float | None = None

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
        self.overshoot_stage_index: int = 0
        self.overshoot_trimmed_qty_total: int = 0
        self.overshoot_active: bool = False
        self.overshoot_trigger_ticks: int | None = None
        self.overshoot_crossed_fair_ms: int | None = None
        self.shock_decay_steps_applied: int = 0
        self.shock_decay_trimmed_qty_total: int = 0

    def on_book(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="book")

    def on_trade(self, snapshot: StrategySnapshot) -> Decision:
        self._record_trade(snapshot)
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="trade")

    def on_fill(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="fill")

    def on_timer(self, snapshot: StrategySnapshot) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        return self._evaluate(snapshot, trigger="timer")

    def on_news(self, snapshot: StrategySnapshot, news_event: NewsEvent) -> Decision:
        self._record_mid(snapshot.now_ms, snapshot.book.mid)
        earnings_value = self._parse_c_earnings_value(news_event)
        if earnings_value is None:
            return self._observe_current_state(snapshot, reason="ignoring non-C earnings news in stock C mode")
        return self._handle_structured_c_earnings(snapshot, news_event, earnings_value)

    def export_state(self) -> dict[str, int | float | str | bool | None]:
        return {
            "mode": self.mode,
            "active_signal_kind": self.active_signal_kind,
            "fair_value": self.fair_value,
            "base_fair_value": self.base_fair_value,
            "news_fair_value": self.news_fair_value,
            "latest_earnings": self.latest_earnings,
            "fair_change_ticks": self.fair_change_ticks,
            "shock_target_inventory": self.shock_target_inventory,
            "original_shock_target_inventory": self.original_shock_target_inventory,
            "shock_peak_inventory_abs": self.shock_peak_inventory_abs,
            "shock_direction": self.shock_direction,
            "shock_reference_mid": self.shock_reference_mid,
            "initial_shock_edge": self.initial_shock_edge,
            "overshoot_active": self.overshoot_active,
            "overshoot_stage_index": self.overshoot_stage_index,
            "overshoot_trimmed_qty_total": self.overshoot_trimmed_qty_total,
            "overshoot_trigger_ticks": self.overshoot_trigger_ticks,
            "shock_decay_steps_applied": self.shock_decay_steps_applied,
            "shock_decay_trimmed_qty_total": self.shock_decay_trimmed_qty_total,
            "pending_news": self.pending_news,
            "pending_news_target_inventory": self.pending_news_target_inventory,
            "mm_microprice": None if self.microprice is None else round(self.microprice, 3),
            "mm_recent_trade_count": self.recent_trade_count,
            "mm_flow_score": self.flow_score,
            "mm_burst_active": self.burst_active,
            "mm_quote_side": self.quote_side,
            "mm_quote_px": self.quote_px,
            "mm_quote_reason": self.quote_reason,
        }

    def _evaluate(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        self._refresh_export_fair_value(snapshot)

        pending_decision = self._handle_pending_earnings(snapshot, trigger=trigger)
        if pending_decision is not None:
            return pending_decision

        if self.mode == "SHOCK":
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
                    reason="emergency dump after a sharp wrong-way move against the current stock C earnings shock",
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
                    reason="maximum stock C earnings hold time reached; flattening the remaining shock inventory",
                )
            overshoot_decision = self._maybe_build_overshoot_decision(snapshot, trigger=trigger)
            if overshoot_decision is not None:
                return overshoot_decision
            if self._equilibrium_reached(snapshot.now_ms):
                self.mode = "UNWIND"
                self.equilibrium_reached_ms = snapshot.now_ms
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
                self.overshoot_active = False
                self.overshoot_trigger_ticks = None
                self.shock_target_inventory = 0
                return self._build_unwind_decision(
                    snapshot,
                    trigger=trigger,
                    reason="stock C stabilized after the earnings shock; switching to unwind",
                )
            self._maybe_apply_shock_decay(snapshot.now_ms)
            return self._build_shock_decision(snapshot, trigger=trigger)

        if self.mode == "UNWIND":
            if abs(snapshot.inventory) <= self.config.flatten_near_zero_threshold:
                self._clear_after_flatten(snapshot)
                return self._idle_decision(snapshot, reason="stock C inventory flattened back to neutral")
            return self._build_unwind_decision(snapshot, trigger=trigger, reason="flattening residual stock C shock inventory")

        return self._evaluate_market_maker(snapshot, trigger=trigger)

    def _evaluate_market_maker(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        book = snapshot.book
        if book.best_bid is None or book.best_ask is None or book.mid is None:
            self.mode = "IDLE"
            self._update_state(snapshot, quote_side=None, quote_px=None, quote_reason="waiting for a two-sided C book")
            return self._cancel_or_observe(snapshot, reason="waiting for a two-sided C book")

        self._trim_trade_signals(snapshot.now_ms)
        self.microprice = self._microprice(snapshot)
        passive_fair = float(self.microprice if self.microprice is not None else book.mid) + float(self.config.maker_fair_value_offset_ticks)
        model_fair = self._model_fair_value(snapshot)
        self.fair_value = int(round(model_fair if model_fair is not None else passive_fair))
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
                return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=quote_qty, reason="buy burst in C; leaning offer into flow")
            if self.flow_score <= -int(self.config.maker_flow_trigger) and inventory < hard_limit:
                self.mode = "MARKET_MAKING_STUB"
                return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=quote_qty, reason="sell burst in C; leaning bid into flow")

        if inventory > 0:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="SELL", px=book.best_ask.px, qty=min(inventory, quote_qty), reason="no burst; recycling long C inventory")
        if inventory < 0:
            self.mode = "MARKET_MAKING_STUB"
            return self._quote(snapshot, side="BUY", px=book.best_bid.px, qty=min(abs(inventory), quote_qty), reason="no burst; recycling short C inventory")

        self.mode = "IDLE"
        self._update_state(snapshot, quote_side=None, quote_px=None, quote_reason="waiting for a bursty spread in C")
        return self._cancel_or_observe(snapshot, reason="waiting for a bursty spread in C")

    def _handle_structured_c_earnings(self, snapshot: StrategySnapshot, news_event: NewsEvent, earnings_value: float) -> Decision:
        tick = news_event.tick if news_event.tick is not None else snapshot.exchange_tick
        if not self.have_real_eps_c:
            self.have_real_eps_c = True
            self.current_eps_c = float(earnings_value)
            self.latest_earnings = float(earnings_value)
            self.baseline_eps_c = float(earnings_value)
            self.last_c_earnings_delta = 0.0
            self.active_signal_kind = None
            self.pending_news = None
            self.pending_news_target_inventory = None
            self.pending_news_reference_mid = None
            self._maybe_initialize_anchor(snapshot, force=True)
            self._refresh_export_fair_value(snapshot)
            return self._observe_current_state(
                snapshot,
                reason="first structured C earnings report adopted as the stock C baseline; not trading the baseline print",
            )

        if self._news_takeover_required(snapshot):
            self.pending_earnings_value = float(earnings_value)
            self.pending_earnings_tick = tick
            self.pending_news = f"C earnings value={earnings_value:.4f}"
            self.pending_news_reference_mid = snapshot.book.mid
            self.pending_news_target_inventory = None
            self.mode = "UNWIND"
            if self.unwind_started_ms is None:
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
            return self._build_takeover_flatten_decision(
                snapshot,
                trigger="structured_news",
                reason="flattening current stock C inventory before the latest structured C earnings signal takes over",
            )

        return self._activate_c_earnings_shock(
            snapshot,
            tick=tick,
            earnings_value=float(earnings_value),
            trigger="structured_news",
        )

    def _handle_pending_earnings(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision | None:
        if self.pending_earnings_value is None:
            return None
        if self._news_takeover_required(snapshot):
            self.mode = "UNWIND"
            if self.unwind_started_ms is None:
                self.unwind_started_ms = snapshot.now_ms
                self.unwind_start_inventory = snapshot.inventory
            return self._build_takeover_flatten_decision(
                snapshot,
                trigger=trigger,
                reason="flattening current stock C inventory before the latest structured C earnings signal takes over",
            )

        value = float(self.pending_earnings_value)
        tick = self.pending_earnings_tick
        self.pending_earnings_value = None
        self.pending_earnings_tick = None
        self.pending_news = None
        self.pending_news_reference_mid = None
        self.pending_news_target_inventory = None
        return self._activate_c_earnings_shock(snapshot, tick=tick, earnings_value=value, trigger=trigger)

    def _activate_c_earnings_shock(
        self,
        snapshot: StrategySnapshot,
        *,
        tick: int | None,
        earnings_value: float,
        trigger: str,
    ) -> Decision:
        self._maybe_initialize_anchor(snapshot)
        old_eps = self.current_eps_c
        fair_before = self._model_fair_value(snapshot)
        reference_mid = snapshot.book.mid

        self.current_eps_c = float(earnings_value)
        self.latest_earnings = float(earnings_value)
        delta = float(earnings_value) - float(old_eps)
        self.last_c_earnings_delta = delta
        self.recent_c_earnings_deltas.append(delta)

        fair_after = self._model_fair_value(snapshot)
        self.base_fair_value = None if fair_before is None else int(round(fair_before))
        self.news_fair_value = None if fair_after is None else int(round(fair_after))
        self.fair_value = self.news_fair_value if self.news_fair_value is not None else self.fair_value
        self.fair_change_ticks = None if fair_before is None or fair_after is None else float(fair_after - fair_before)

        if fair_after is None or reference_mid is None:
            self.mode = "IDLE"
            self.active_signal_kind = None
            return self._observe_current_state(snapshot, reason="C earnings logged without enough rate/book context to derive a tradable fair")

        if abs(delta) < float(self.config.c_earnings_ignore_delta):
            self.mode = "IDLE"
            self.active_signal_kind = None
            return self._observe_current_state(snapshot, reason="structured C earnings changed EPS too little for shock mode")

        edge = float(fair_after) - float(reference_mid)
        target_inventory = self._scaled_target(edge, fair_change_ticks=self.fair_change_ticks)
        if target_inventory == 0:
            self.mode = "IDLE"
            self.active_signal_kind = None
            return self._observe_current_state(snapshot, reason="structured C earnings moved fair too little for shock mode")

        self.mode = "SHOCK"
        self.active_signal_kind = "structured_c_earnings"
        self.shock_started_ms = snapshot.now_ms
        self.shock_direction = _sign(edge)
        self.shock_target_inventory = target_inventory
        self.original_shock_target_inventory = target_inventory
        self.shock_peak_inventory_abs = 0
        self.shock_reference_mid = float(reference_mid)
        self.initial_shock_edge = edge
        self.equilibrium_reached_ms = None
        self.unwind_started_ms = None
        self.unwind_start_inventory = 0
        self._reset_overshoot_state()
        self._reset_decay_state()
        self.post_event_mids.clear()
        self.post_event_mids.append((snapshot.now_ms, float(reference_mid)))
        return self._build_shock_decision(snapshot, trigger=trigger)

    def _parse_c_earnings_value(self, news_event: NewsEvent) -> float | None:
        if news_event.is_structured_c_earnings:
            return float(news_event.value)

        raw = dict(news_event.raw_payload or {})
        new_data_obj = raw.get("new_data") or {}
        new_data = new_data_obj if isinstance(new_data_obj, dict) else {}
        subtype = str(new_data.get("structured_subtype") or news_event.structured_subtype or "").lower()
        asset = str(new_data.get("asset") or news_event.asset or news_event.symbol or "").upper()
        raw_value = new_data.get("value", news_event.value)
        if subtype == "earnings" and asset == self.config.symbol and raw_value is not None:
            return float(raw_value)

        text = str(new_data.get("content") or news_event.content or (new_data_obj if not isinstance(new_data_obj, dict) else ""))
        if not text:
            return None
        if "earnings" not in text.lower():
            return None
        match = re.search(r"\bC\s+earnings\s+released\s*:?\s*(%s)" % FLOAT_RE, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _refresh_export_fair_value(self, snapshot: StrategySnapshot) -> None:
        model_fair = self._model_fair_value(snapshot)
        if model_fair is not None:
            self.fair_value = int(round(model_fair))
            if self.base_fair_value is None and self.have_real_eps_c:
                self.base_fair_value = self.fair_value
            return
        if snapshot.book.mid is not None:
            self.fair_value = int(round(float(snapshot.book.mid) + float(self.config.maker_fair_value_offset_ticks)))

    def _rate_context(self, snapshot: StrategySnapshot) -> _CRateContext | None:
        hike_mid = snapshot.book_for(self.config.rate_hike_symbol).mid
        hold_mid = snapshot.book_for(self.config.rate_hold_symbol).mid
        cut_mid = snapshot.book_for(self.config.rate_cut_symbol).mid
        if hike_mid is None or hold_mid is None or cut_mid is None:
            return self.last_rate_context

        total = float(hike_mid) + float(hold_mid) + float(cut_mid)
        if total <= 0:
            return self.last_rate_context

        context = _CRateContext(
            q_hike=float(hike_mid) / total,
            q_hold=float(hold_mid) / total,
            q_cut=float(cut_mid) / total,
            market_rate_bp=float(self.config.c_rate_step_bp) * ((float(hike_mid) / total) - (float(cut_mid) / total)),
        )
        self.last_rate_context = context
        return context

    def _maybe_initialize_anchor(self, snapshot: StrategySnapshot, *, force: bool = False) -> bool:
        rate = self._rate_context(snapshot)
        mid_c = snapshot.book.mid
        if rate is None or mid_c is None:
            return False
        if force or self.anchor_price is None or self.anchor_eps is None or self.anchor_rate_bp is None:
            self.anchor_price = float(mid_c)
            self.anchor_eps = float(self.current_eps_c)
            self.anchor_rate_bp = float(rate.market_rate_bp)
            return True
        return False

    def _model_fair_value(self, snapshot: StrategySnapshot) -> float | None:
        rate = self._rate_context(snapshot)
        if (
            rate is None
            or self.anchor_price is None
            or self.anchor_eps is None
            or self.anchor_rate_bp is None
            or self.anchor_eps == 0
        ):
            return None

        dy = (float(rate.market_rate_bp) - float(self.anchor_rate_bp)) / 10_000.0
        ops_anchor = float(self.config.c_ops_weight) * float(self.anchor_price)
        bond_anchor = float(self.config.c_bond_weight) * float(self.anchor_price)
        ops_fair = ops_anchor * (float(self.current_eps_c) / float(self.anchor_eps)) * math.exp(-float(self.config.c_pe_yield_gamma) * dy)
        bond_fair = bond_anchor * (
            1.0
            - (float(self.config.c_bond_duration) * dy)
            + (0.5 * float(self.config.c_bond_convexity) * dy * dy)
        )
        return ops_fair + bond_fair

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
            target_abs = max(target_abs, min(scaled_cap, change_scaled_target))

        return _sign(edge) * target_abs

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

    def _record_mid(self, now_ms: int, mid: float | int | None) -> None:
        if mid is None:
            return
        if self.mode in {"SHOCK", "UNWIND"} or self.pending_earnings_value is not None:
            self.post_event_mids.append((now_ms, float(mid)))
            hold_window = max(self.config.flatten_deadline_ms * 2, self.config.equilibrium_hold_ms * 3)
            self._trim_window(self.post_event_mids, now_ms, hold_window)

    @staticmethod
    def _trim_window(samples: deque[tuple[int, float]], now_ms: int, window_ms: int) -> None:
        threshold = now_ms - window_ms
        while samples and samples[0][0] < threshold:
            samples.popleft()

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
            latest_earnings=self.latest_earnings,
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
                latest_earnings=self.latest_earnings,
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
                latest_earnings=self.latest_earnings,
            )
        return self._observe_current_state(snapshot, reason=reason)

    def _stage_thresholds(self) -> list[int]:
        edge_abs = abs(self.initial_shock_edge)
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

        window_threshold = snapshot.now_ms - self.config.overshoot_hold_ms
        window = [(ts, mid) for ts, mid in self.post_event_mids if ts >= window_threshold]
        if len(window) < 3:
            return None
        mids = [mid for _, mid in window]
        if max(mids) - min(mids) > self.config.overshoot_band_ticks:
            return None

        reversal_ticks = max(mids) - current_mid if self.shock_direction > 0 else current_mid - min(mids)
        if reversal_ticks < self.config.overshoot_reversal_ticks:
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
                reason="buying back part of the short stock C earnings shock during an overshoot",
                symbol=self.config.symbol,
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
                reason="selling part of the long stock C earnings shock during an overshoot",
                symbol=self.config.symbol,
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
            reason=f"{trigger}: trimming staged stock C shock inventory during an overshoot beyond fair {self.fair_value}",
            fair_value=self.fair_value,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=False,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _equilibrium_reached(self, now_ms: int) -> bool:
        if self.shock_started_ms is not None and (now_ms - self.shock_started_ms) < self.config.equilibrium_min_elapsed_ms:
            return False
        self._trim_window(self.post_event_mids, now_ms, self.config.equilibrium_hold_ms)
        if len(self.post_event_mids) < self.config.equilibrium_min_samples:
            return False
        if (self.post_event_mids[-1][0] - self.post_event_mids[0][0]) < self.config.equilibrium_hold_ms:
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
        if current_direction != self.shock_direction or residual_edge <= self.config.equilibrium_residual_edge_ticks:
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
            if overshoot_wait_elapsed >= self.config.overshoot_max_wait_ms and self.overshoot_stage_index == 0:
                return True

        captured_ticks = abs(settled_mid - float(self.shock_reference_mid))
        captured_fraction = min(1.0, captured_ticks / initial_edge_abs)
        return captured_fraction >= self.config.equilibrium_min_capture_fraction

    def _build_shock_decision(self, snapshot: StrategySnapshot, *, trigger: str) -> Decision:
        if self.fair_value is None or snapshot.book.mid is None:
            return self._idle_decision(snapshot, reason="waiting for a C top-of-book to trade the earnings shock")
        target_inventory = self.shock_target_inventory
        delta = target_inventory - snapshot.inventory
        if delta == 0:
            return Decision(
                mode="SHOCK",
                target_inventory=target_inventory,
                desired_order=None,
                cancel_all=False,
                observe_only=True,
                reason=f"{trigger}: already at the stock C earnings shock target",
                fair_value=self.fair_value,
                latest_earnings=self.latest_earnings,
                equilibrium_reached=False,
                shock_reference_mid=self.shock_reference_mid,
            )

        clip = self.config.shock_initial_clip if trigger in {"structured_news", "book", "trade", "timer", "fill"} else self.config.shock_reinforce_clip
        if delta > 0:
            if snapshot.book.best_ask is None:
                return self._idle_decision(snapshot, reason="cannot buy stock C shock inventory without an ask")
            order = DesiredOrder(
                side="BUY",
                px=snapshot.book.best_ask.px,
                qty=min(abs(delta), clip),
                aggressive=True,
                intent="post_c_earnings_shock_take",
                reason="buying aggressively into the structured C earnings edge",
                symbol=self.config.symbol,
            )
        else:
            if snapshot.book.best_bid is None:
                return self._idle_decision(snapshot, reason="cannot sell stock C shock inventory without a bid")
            order = DesiredOrder(
                side="SELL",
                px=snapshot.book.best_bid.px,
                qty=min(abs(delta), clip),
                aggressive=True,
                intent="post_c_earnings_shock_take",
                reason="selling aggressively into the structured C earnings edge",
                symbol=self.config.symbol,
            )
        return Decision(
            mode="SHOCK",
            target_inventory=target_inventory,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: taking stock C earnings shock inventory toward fair {self.fair_value}",
            fair_value=self.fair_value,
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
                reason=f"{trigger}: clearing live stock C orders before the latest earnings signal takes over",
                fair_value=self.fair_value,
                latest_earnings=self.latest_earnings,
                equilibrium_reached=False,
                shock_reference_mid=self.shock_reference_mid,
            )
        if snapshot.inventory == 0:
            return self._observe_current_state(snapshot, reason=reason)
        if snapshot.inventory > 0:
            if snapshot.book.best_bid is None:
                return self._observe_current_state(snapshot, reason="waiting for a bid while flattening before the latest C earnings signal")
            order = DesiredOrder(
                side="SELL",
                px=snapshot.book.best_bid.px,
                qty=abs(snapshot.inventory),
                aggressive=True,
                intent="news_takeover_flatten",
                reason="flattening current stock C inventory before the latest C earnings signal takes over",
                symbol=self.config.symbol,
            )
        else:
            if snapshot.book.best_ask is None:
                return self._observe_current_state(snapshot, reason="waiting for an ask while flattening before the latest C earnings signal")
            order = DesiredOrder(
                side="BUY",
                px=snapshot.book.best_ask.px,
                qty=abs(snapshot.inventory),
                aggressive=True,
                intent="news_takeover_flatten",
                reason="flattening current stock C inventory before the latest C earnings signal takes over",
                symbol=self.config.symbol,
            )
        return Decision(
            mode="UNWIND",
            target_inventory=0,
            desired_order=order,
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: {reason}",
            fair_value=self.fair_value,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=False,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _build_unwind_decision(self, snapshot: StrategySnapshot, *, trigger: str, reason: str) -> Decision:
        if snapshot.inventory == 0:
            return self._idle_decision(snapshot, reason="already flat after the stock C earnings shock")
        side = "SELL" if snapshot.inventory > 0 else "BUY"
        qty = abs(snapshot.inventory)
        if side == "BUY":
            if snapshot.book.best_ask is None:
                return self._idle_decision(snapshot, reason="cannot unwind short stock C inventory without an ask")
            px = snapshot.book.best_ask.px
        else:
            if snapshot.book.best_bid is None:
                return self._idle_decision(snapshot, reason="cannot unwind long stock C inventory without a bid")
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
                reason="flattening stock C earnings shock inventory back to neutral",
                symbol=self.config.symbol,
            ),
            cancel_all=True,
            observe_only=False,
            reason=f"{trigger}: {reason}",
            fair_value=self.fair_value,
            latest_earnings=self.latest_earnings,
            equilibrium_reached=True,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _news_takeover_required(self, snapshot: StrategySnapshot) -> bool:
        return snapshot.inventory != 0 or bool(snapshot.open_orders)

    def _clear_after_flatten(self, snapshot: StrategySnapshot) -> None:
        self.mode = "IDLE"
        self.active_signal_kind = None
        self.news_fair_value = None
        self.fair_change_ticks = None
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
        self._refresh_export_fair_value(snapshot)
        if self.fair_value is not None:
            self.base_fair_value = self.fair_value

    def _reset_overshoot_state(self) -> None:
        self.overshoot_stage_index = 0
        self.overshoot_trimmed_qty_total = 0
        self.overshoot_active = False
        self.overshoot_trigger_ticks = None
        self.overshoot_crossed_fair_ms = None

    def _reset_decay_state(self) -> None:
        self.shock_decay_steps_applied = 0
        self.shock_decay_trimmed_qty_total = 0

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
            latest_earnings=self.latest_earnings,
            equilibrium_reached=self.equilibrium_reached_ms is not None,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _idle_decision(self, snapshot: StrategySnapshot, *, reason: str) -> Decision:
        return Decision(
            mode=self.mode,
            target_inventory=0 if self.mode != "SHOCK" else self.shock_target_inventory,
            desired_order=None,
            cancel_all=bool(snapshot.open_orders),
            observe_only=True,
            reason=reason,
            fair_value=self.fair_value,
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
            latest_earnings=self.latest_earnings,
            equilibrium_reached=self.equilibrium_reached_ms is not None,
            shock_reference_mid=self.shock_reference_mid,
        )

    def _update_state(self, snapshot: StrategySnapshot, *, quote_side: str | None, quote_px: int | None, quote_reason: str | None) -> None:
        self.quote_side = quote_side
        self.quote_px = quote_px
        self.quote_reason = quote_reason
        if self.fair_value is None and snapshot.book.mid is not None:
            self.fair_value = int(round(float(snapshot.book.mid) + float(self.config.maker_fair_value_offset_ticks)))
