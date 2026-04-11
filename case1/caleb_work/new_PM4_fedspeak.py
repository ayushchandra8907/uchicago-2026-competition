from __future__ import annotations

import asyncio
from dataclasses import asdict
import subprocess
import time
from typing import Optional

try:
    from new_PM3_heuristic import (
        MarketLabels,
        NewPM3HeuristicClient,
        PendingSignal,
        env_bool,
        env_float,
        env_int,
        env_str,
        parse_cpi,
    )
    from PM_fedspeak_sentiment import PMFedSpeakSentiment, score_pm_fedspeak
    from PM_fedspeak_tracker import PMFedSpeakTracker
except ModuleNotFoundError:
    from case1.caleb_work.new_PM3_heuristic import (
        MarketLabels,
        NewPM3HeuristicClient,
        PendingSignal,
        env_bool,
        env_float,
        env_int,
        env_str,
        parse_cpi,
    )
    from case1.caleb_work.PM_fedspeak_sentiment import PMFedSpeakSentiment, score_pm_fedspeak
    from case1.caleb_work.PM_fedspeak_tracker import PMFedSpeakTracker


FEDSPEAK_KEYWORDS = (
    "fed",
    "fomc",
    "inflation",
    "policy",
    "rates",
    "cuts",
    "hikes",
    "restrictive",
    "neutral rate",
    "labor market",
    "wages",
)


class NewPM4FedSpeakClient(NewPM3HeuristicClient):
    """
    Copy of PM3 strategy behavior with one extension:
    - CPI handling (unchanged from PM3)
    - FedSpeak handling (sentiment-driven signal mapped to winner/loser pair)
    """

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.fedspeak_enabled = env_bool("PM4_FEDSPEAK_ENABLED", True)
        self.fedspeak_min_abs_bias_bp = env_float("PM4_FEDSPEAK_MIN_ABS_BIAS_BP", 0.75)
        self.fedspeak_event_cooldown_sec = env_float("PM4_FEDSPEAK_EVENT_COOLDOWN_SEC", 0.75)
        self.fedspeak_target_light = env_int("PM4_FEDSPEAK_TARGET_LIGHT", 20)
        self.fedspeak_target_medium = env_int("PM4_FEDSPEAK_TARGET_MEDIUM", 30)
        self.fedspeak_target_strong = env_int("PM4_FEDSPEAK_TARGET_STRONG", 40)
        self.fedspeak_cut_min_q = env_float("PM4_FEDSPEAK_CUT_MIN_Q", 0.10)
        self.fedspeak_strong_dovish_bp = env_float("PM4_FEDSPEAK_STRONG_DOVISH_BP", 2.25)
        self.fedspeak_tracker = PMFedSpeakTracker(
            resolve_after_sec=env_float("PM4_FEDSPEAK_TRACK_RESOLVE_SEC", 1.50),
            max_history=env_int("PM4_FEDSPEAK_TRACK_HISTORY", 256),
        )
        self.last_fedspeak_ts = 0.0

        # Handoff guard mode: on trigger, flatten to 0/0/0 fast.
        # In hybrid mode (default), the guard re-arms after flatten so click trading can continue.
        self.hybrid_guard_enabled = env_bool("PM4_HYBRID_GUARD_ENABLED", True)
        self.hybrid_pause_after_flatten = env_bool("PM4_HYBRID_PAUSE_AFTER_FLATTEN", False)
        self.hybrid_guard_rearm_sec = env_float("PM4_HYBRID_GUARD_REARM_SEC", 0.75)
        self.adverse_only_flatten_enabled = env_bool("PM4_ADVERSE_ONLY_FLATTEN_ENABLED", True)
        self.adverse_flatten_constant = env_bool("PM4_ADVERSE_FLATTEN_CONSTANT", False)
        self.handoff_ding_enabled = env_bool("PM4_HANDOFF_DING_ENABLED", True)
        self.handoff_ding_sound = env_str("PM4_HANDOFF_DING_SOUND", "/System/Library/Sounds/Glass.aiff")
        self.handoff_skip_low_price_enabled = env_bool("PM4_HANDOFF_SKIP_LOW_PRICE_ENABLED", True)
        self.handoff_skip_low_price_px = env_int("PM4_HANDOFF_SKIP_LOW_PRICE_PX", 100)
        self.handoff_skip_high_price_enabled = env_bool("PM4_HANDOFF_SKIP_HIGH_PRICE_ENABLED", True)
        self.handoff_skip_high_price_px = env_int("PM4_HANDOFF_SKIP_HIGH_PRICE_PX", 900)
        self.last_guard_trigger_ts = 0.0
        self.guard_trigger_count = 0
        self.handoff_initial_pos: dict[str, int] = {s: 0 for s in self.cfg.rate_symbols}
        self.handoff_plan_abs: dict[str, int] = {s: 0 for s in self.cfg.rate_symbols}
        self.handoff_pred_winner: Optional[str] = None
        self.handoff_pred_loser: Optional[str] = None
        self.handoff_pred_constant: Optional[str] = None
        self.handoff_pred_source: Optional[str] = None

        self.handoff_cpi_abs = env_float("PM4_HANDOFF_CPI_ABS", 0.0003)
        self.handoff_fedspeak_min_abs_bias_bp = env_float(
            "PM4_HANDOFF_FEDSPEAK_MIN_ABS_BIAS_BP", self.fedspeak_min_abs_bias_bp
        )
        self.handoff_active = False
        self.handoff_done = False
        self.auto_paused = False
        self.handoff_started_ts = 0.0
        self.handoff_trigger_reason: Optional[str] = None

    # -----------------------
    # Unwind disabled (for experiment)
    # -----------------------
    # Keep underlying code intact in PM3, but force unwind triggers off here.
    def should_emergency_unwind(self) -> bool:
        return False

    def should_retrace_unwind(self) -> bool:
        return False

    def should_time_unwind(self) -> bool:
        return False

    @staticmethod
    def _is_fedspeak_candidate(content: str, message_type: str) -> bool:
        mt = message_type.lower().strip()
        if "fedspeak" in mt or mt in {"macro", "rates"}:
            return True
        lowered = content.lower()
        if "fed" not in lowered and "fomc" not in lowered:
            return False
        return any(keyword in lowered for keyword in FEDSPEAK_KEYWORDS)

    def _fedspeak_target_abs(self, sentiment: PMFedSpeakSentiment) -> int:
        if sentiment.bucket in {"strong", "extreme"}:
            return min(self.fedspeak_target_strong, self.cfg.max_order_size)
        if sentiment.bucket == "medium":
            return min(self.fedspeak_target_medium, self.cfg.max_order_size)
        return min(self.fedspeak_target_light, self.cfg.max_order_size)

    def _build_signal_from_fedspeak(
        self,
        *,
        content: str,
        message_type: str,
    ) -> Optional[tuple[PendingSignal, PMFedSpeakSentiment, str]]:
        sentiment = score_pm_fedspeak(content)
        if abs(sentiment.implied_bias_bp) < self.fedspeak_min_abs_bias_bp:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="fedspeak_skip_low_bias",
                score=sentiment.score,
                bucket=sentiment.bucket,
                implied_bias_bp=sentiment.implied_bias_bp,
                message_type=message_type,
                content=content,
            )
            return None

        probs = self.current_probabilities()
        if probs is None:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="fedspeak_skip_missing_probabilities",
                message_type=message_type,
                content=content,
            )
            return None

        q_hike = float(probs.get(self.cfg.fed_hike, 0.0))
        q_hold = float(probs.get(self.cfg.fed_hold, 0.0))
        q_cut = float(probs.get(self.cfg.fed_cut, 0.0))

        if sentiment.direction > 0:
            winner = self.cfg.fed_hike
            loser = self.choose_counterpart(winner)
        elif sentiment.direction < 0:
            cut_pref = (q_cut >= self.fedspeak_cut_min_q) or (abs(sentiment.implied_bias_bp) >= self.fedspeak_strong_dovish_bp)
            winner = self.cfg.fed_cut if cut_pref else self.cfg.fed_hold
            loser = self.choose_counterpart(winner)
        else:
            # Mixed/neutral but still above bias threshold: default to HOLD.
            winner = self.cfg.fed_hold
            loser = self.choose_counterpart(winner)

        constant = next(sym for sym in self.cfg.rate_symbols if sym not in {winner, loser})
        target_abs = max(self.cfg.entry_initial_clip, self._fedspeak_target_abs(sentiment))

        self.labels = MarketLabels(
            important_1=winner,
            important_2=loser,
            irrelevant=constant,
            pair_corr=self.labels.pair_corr,
            updated_ts=time.time(),
        )

        signal = PendingSignal(
            winner=winner,
            loser=loser,
            # Keep CPI learning isolated to CPI events.
            surprise=0.0,
            target_abs=target_abs,
            reason=f"fedspeak_{sentiment.bucket}_{winner}_over_{loser}",
            source="fedspeak",
            score_winner=abs(sentiment.score),
            score_loser=-abs(sentiment.score),
            score_constant=0.0,
            q_hike=q_hike,
            q_hold=q_hold,
            q_cut=q_cut,
        )
        return signal, sentiment, constant

    async def on_fedspeak_signal(self, content: str, message_type: str) -> None:
        if not self.fedspeak_enabled:
            return
        now = time.time()
        if now - self.last_fedspeak_ts < self.fedspeak_event_cooldown_sec:
            self._trace("decision", tick=self.current_tick, mode=self.mode, reason="fedspeak_event_cooldown_skip")
            return

        built = self._build_signal_from_fedspeak(content=content, message_type=message_type)
        if built is None:
            return
        signal, sentiment, constant = built

        tracker_event_id = self.fedspeak_tracker.register(
            tick=self.current_tick,
            headline=content,
            message_type=message_type,
            sentiment=sentiment,
            winner_predicted=signal.winner,
            loser_predicted=signal.loser,
            constant_predicted=constant,
            target_abs=signal.target_abs,
            q_hike_before=signal.q_hike,
            q_hold_before=signal.q_hold,
            q_cut_before=signal.q_cut,
            now_ts=now,
        )

        self.pending_signal = signal
        self.last_fedspeak_ts = now

        if self.mode == "IDLE" and self.is_flat() and self.total_open_orders() == 0 and self.startup_flatten_complete:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="queued_fedspeak_signal_for_confirm",
                tracker_event_id=tracker_event_id,
                pending=vars(signal),
                sentiment=asdict(sentiment),
            )
            return

        # Unwind disabled for this experimental variant:
        # keep running position and queue signal instead of forced flatten-takeover.
        self._trace(
            "decision",
            tick=self.current_tick,
            mode=self.mode,
            reason="queued_fedspeak_signal_while_active_no_unwind",
            tracker_event_id=tracker_event_id,
            pending=vars(signal),
            sentiment=asdict(sentiment),
        )

    async def _cancel_all_open_orders_fast(self) -> None:
        for order_id in list(self.open_orders.keys()):
            try:
                await self.cancel_order(order_id)
            except Exception:
                pass

    async def _flatten_all_fast(self, reason: str) -> bool:
        acted = False
        for symbol in self.cfg.rate_symbols:
            planned_abs = int(self.handoff_plan_abs.get(symbol, 0))
            if planned_abs <= self.cfg.near_flat_threshold:
                continue
            initial_pos = int(self.handoff_initial_pos.get(symbol, 0))
            current_pos = int(self.get_position(symbol))
            if initial_pos > 0:
                flattened_abs = max(0, min(planned_abs, initial_pos - current_pos))
                side = self.side_from_name("SELL")
            elif initial_pos < 0:
                flattened_abs = max(0, min(planned_abs, current_pos - initial_pos))
                side = self.side_from_name("BUY")
            else:
                continue
            remaining_abs = max(0, planned_abs - flattened_abs)
            if remaining_abs <= self.cfg.near_flat_threshold:
                continue
            liq_px = self._liquidation_px(symbol, side)
            price_skip = self._handoff_price_skip(liq_px)
            if price_skip is not None:
                skip_label, threshold = price_skip
                self._trace(
                    "decision",
                    tick=self.current_tick,
                    mode=self.mode,
                    reason=f"handoff_skip_{skip_label}",
                    symbol=symbol,
                    liquidation_px=liq_px,
                    threshold=threshold,
                    initial_pos=initial_pos,
                    current_pos=current_pos,
                    planned_abs=planned_abs,
                    remaining_abs=remaining_abs,
                )
                continue
            qty = self.clip_qty(symbol, side, min(remaining_abs, self.cfg.max_order_size))
            if qty <= 0:
                continue
            placed = await self.submit_order(symbol, side, qty, aggressive=True, reason=reason)
            acted = acted or placed
        if acted:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason=reason,
                positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
            )
        return acted

    def _set_handoff_prediction(
        self,
        *,
        winner: Optional[str],
        loser: Optional[str],
        constant: Optional[str],
        source: str,
    ) -> None:
        self.handoff_pred_winner = winner
        self.handoff_pred_loser = loser
        self.handoff_pred_constant = constant
        self.handoff_pred_source = source
        self._trace(
            "decision",
            tick=self.current_tick,
            mode=self.mode,
            reason="handoff_prediction_set",
            source=source,
            winner=winner,
            loser=loser,
            constant=constant,
        )

    def _clear_handoff_prediction(self, source: str) -> None:
        self.handoff_pred_winner = None
        self.handoff_pred_loser = None
        self.handoff_pred_constant = None
        self.handoff_pred_source = source

    def _planned_abs_for_symbol(self, symbol: str, pos: int) -> int:
        base_abs = abs(pos)
        if base_abs <= self.cfg.near_flat_threshold:
            return 0
        if not self.adverse_only_flatten_enabled:
            return base_abs
        winner = self.handoff_pred_winner
        loser = self.handoff_pred_loser
        constant = self.handoff_pred_constant
        # If no prediction available, fall back to full flatten for safety.
        if winner is None or loser is None:
            return base_abs
        if symbol == winner:
            # Winner expected up: adverse side is short winner.
            return max(0, -pos)
        if symbol == loser:
            # Loser expected down: adverse side is long loser.
            return max(0, pos)
        if symbol == constant:
            return base_abs if self.adverse_flatten_constant else 0
        return base_abs

    def _liquidation_px(self, symbol: str, side) -> Optional[int]:
        top = self.top(symbol)
        if side == self.side_from_name("BUY"):
            return None if top.ask is None else int(top.ask)
        return None if top.bid is None else int(top.bid)

    def _handoff_price_skip(self, liq_px: Optional[int]) -> Optional[tuple[str, int]]:
        if liq_px is None:
            return None
        if self.handoff_skip_low_price_enabled and liq_px < self.handoff_skip_low_price_px:
            return ("low_price", int(self.handoff_skip_low_price_px))
        if self.handoff_skip_high_price_enabled and liq_px > self.handoff_skip_high_price_px:
            return ("high_price", int(self.handoff_skip_high_price_px))
        return None

    def _handoff_plan_complete(self) -> bool:
        for symbol in self.cfg.rate_symbols:
            planned_abs = int(self.handoff_plan_abs.get(symbol, 0))
            if planned_abs <= self.cfg.near_flat_threshold:
                continue
            initial_pos = int(self.handoff_initial_pos.get(symbol, 0))
            current_pos = int(self.get_position(symbol))
            if initial_pos > 0:
                flattened_abs = max(0, min(planned_abs, initial_pos - current_pos))
                side = self.side_from_name("SELL")
            elif initial_pos < 0:
                flattened_abs = max(0, min(planned_abs, current_pos - initial_pos))
                side = self.side_from_name("BUY")
            else:
                continue
            remaining_abs = max(0, planned_abs - flattened_abs)
            if remaining_abs <= self.cfg.near_flat_threshold:
                continue
            liq_px = self._liquidation_px(symbol, side)
            if self._handoff_price_skip(liq_px) is not None:
                continue
            return False
        return True

    def _emit_handoff_ding(self, trigger_reason: str, planned_total_abs: int) -> None:
        if not self.handoff_ding_enabled:
            return
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass
        sound = str(self.handoff_ding_sound or "").strip()
        if not sound:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="handoff_ding_terminal_bell_only",
                trigger_reason=trigger_reason,
                planned_total_abs=planned_total_abs,
            )
            return
        try:
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="handoff_ding",
                trigger_reason=trigger_reason,
                planned_total_abs=planned_total_abs,
                sound=sound,
            )
        except Exception as exc:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="handoff_ding_failed",
                trigger_reason=trigger_reason,
                planned_total_abs=planned_total_abs,
                error=repr(exc),
            )

    async def _trigger_handoff(self, reason: str, **payload) -> None:
        if self.auto_paused:
            return
        now = time.time()
        if (now - self.last_guard_trigger_ts) < self.hybrid_guard_rearm_sec:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="handoff_guard_cooldown_skip",
                trigger_reason=reason,
                cooldown_sec=self.hybrid_guard_rearm_sec,
            )
            return
        self.last_guard_trigger_ts = now

        if not self.handoff_active:
            self.handoff_active = True
            self.handoff_started_ts = now
            self.handoff_trigger_reason = reason
            self.guard_trigger_count += 1
            self.handoff_done = False
            self.pending_signal = None
            self.mode = "UNWIND"

            for symbol in self.cfg.rate_symbols:
                pos = int(self.get_position(symbol))
                self.handoff_initial_pos[symbol] = pos
                plan_abs = self._planned_abs_for_symbol(symbol, pos)
                if plan_abs <= self.cfg.near_flat_threshold:
                    self.handoff_plan_abs[symbol] = 0
                    if abs(pos) > self.cfg.near_flat_threshold:
                        self._trace(
                            "decision",
                            tick=self.current_tick,
                            mode=self.mode,
                            reason="handoff_plan_keep_position",
                            symbol=symbol,
                            initial_pos=pos,
                            winner=self.handoff_pred_winner,
                            loser=self.handoff_pred_loser,
                            constant=self.handoff_pred_constant,
                            adverse_only=self.adverse_only_flatten_enabled,
                        )
                    continue
                side = self.side_from_name("SELL" if pos > 0 else "BUY")
                liq_px = self._liquidation_px(symbol, side)
                price_skip = self._handoff_price_skip(liq_px)
                if price_skip is not None:
                    skip_label, threshold = price_skip
                    self.handoff_plan_abs[symbol] = 0
                    self._trace(
                        "decision",
                        tick=self.current_tick,
                        mode=self.mode,
                        reason=f"handoff_plan_skip_{skip_label}",
                        symbol=symbol,
                        liquidation_px=liq_px,
                        threshold=threshold,
                        initial_pos=pos,
                    )
                    continue
                self.handoff_plan_abs[symbol] = plan_abs

            planned_total_abs = int(sum(max(0, int(v)) for v in self.handoff_plan_abs.values()))
            if planned_total_abs > self.cfg.near_flat_threshold:
                self._emit_handoff_ding(reason, planned_total_abs)

            self._trace(
                "transition",
                tick=self.current_tick,
                mode=self.mode,
                reason="handoff_triggered",
                hybrid_guard_enabled=self.hybrid_guard_enabled,
                guard_trigger_count=self.guard_trigger_count,
                prediction_source=self.handoff_pred_source,
                predicted_winner=self.handoff_pred_winner,
                predicted_loser=self.handoff_pred_loser,
                predicted_constant=self.handoff_pred_constant,
                trigger_reason=reason,
                **payload,
            )
        await self._cancel_all_open_orders_fast()
        if self.total_open_orders() == 0:
            await self._flatten_all_fast("handoff_flatten")

    async def on_cpi_signal(self, surprise: float, source: str) -> None:
        if abs(surprise) <= self.cfg.cpi_trigger_abs:
            return
        now = time.time()
        if now - self.last_event_ts < self.cfg.cpi_event_cooldown_sec:
            self._trace("decision", tick=self.current_tick, mode=self.mode, reason="event_cooldown_skip", surprise=surprise)
            return

        signal = self.build_signal_from_cpi(surprise, source)
        if signal is None:
            return

        self.pending_signal = signal
        if self.mode == "IDLE" and self.is_flat() and self.total_open_orders() == 0 and self.startup_flatten_complete:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode=self.mode,
                reason="queued_signal_for_confirm",
                pending=vars(signal),
            )
            return

        # Unwind disabled for this experimental variant:
        # keep running position and queue CPI signal instead of forced flatten-takeover.
        self._trace(
            "decision",
            tick=self.current_tick,
            mode=self.mode,
            reason="queued_cpi_signal_while_active_no_unwind",
            surprise=surprise,
            pending=vars(signal),
        )

    async def _evaluate_and_sync(self, reason: str) -> None:
        if not self.connected or not self._position_snapshot_seen.is_set():
            return

        if self.auto_paused:
            self._trace(
                "cycle",
                tick=self.current_tick,
                reason=reason,
                mode="PAUSED_MANUAL",
                handoff_done=self.handoff_done,
                positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
                open_orders=len(self.open_orders),
            )
            return

        if self.handoff_active:
            async with self._quote_lock:
                for symbol in self.cfg.rate_symbols:
                    self.refresh_book(symbol)
                await self._cancel_all_open_orders_fast()
                if self.total_open_orders() == 0:
                    await self._flatten_all_fast("handoff_flatten")

                if self._handoff_plan_complete() and self.total_open_orders() == 0:
                    self.handoff_done = True
                    self.handoff_active = False
                    self.auto_paused = self.hybrid_pause_after_flatten
                    self.mode = "IDLE"
                    self.pending_signal = None
                    self._clear_handoff_prediction("handoff_complete")
                    self._trace(
                        "transition",
                        tick=self.current_tick,
                        mode=self.mode,
                        reason=(
                            "handoff_flat_complete_manual_mode"
                            if self.auto_paused
                            else "handoff_flat_complete_hybrid_rearm"
                        ),
                        handoff_trigger_reason=self.handoff_trigger_reason,
                        handoff_elapsed_sec=(time.time() - self.handoff_started_ts),
                        guard_trigger_count=self.guard_trigger_count,
                    )
                self._trace(
                    "cycle",
                    tick=self.current_tick,
                    reason=reason,
                    mode="HANDOFF_FLATTEN",
                    handoff_trigger_reason=self.handoff_trigger_reason,
                    handoff_elapsed_sec=(time.time() - self.handoff_started_ts),
                    positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
                    open_orders=len(self.open_orders),
                )
            return

        if self.hybrid_guard_enabled:
            async with self._quote_lock:
                for symbol in self.cfg.rate_symbols:
                    self.refresh_book(symbol)
                self._trace(
                    "cycle",
                    tick=self.current_tick,
                    reason=reason,
                    mode="HYBRID_GUARD_STANDBY",
                    handoff_done=self.handoff_done,
                    guard_trigger_count=self.guard_trigger_count,
                    positions={s: self.get_position(s) for s in self.cfg.rate_symbols},
                    open_orders=len(self.open_orders),
                )
        else:
            await super()._evaluate_and_sync(reason)

        probs = self.current_probabilities()
        if probs is None:
            return
        resolved = self.fedspeak_tracker.resolve_ready(
            q_hike_now=float(probs.get(self.cfg.fed_hike, 0.0)),
            q_hold_now=float(probs.get(self.cfg.fed_hold, 0.0)),
            q_cut_now=float(probs.get(self.cfg.fed_cut, 0.0)),
            now_ts=time.time(),
        )
        for row in resolved:
            self._trace("fedspeak_tracker_resolved", tick=self.current_tick, row=asdict(row))
        if resolved:
            self._trace("fedspeak_tracker_summary", tick=self.current_tick, summary=self.fedspeak_tracker.summary())

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")
        self._clear_handoff_prediction("news_start")

        if self.auto_paused:
            self._trace(
                "decision",
                tick=self.current_tick,
                mode="PAUSED_MANUAL",
                reason="news_ignored_manual_mode",
                kind=kind,
            )
            return

        # CPI trigger for handoff flatten.
        actual, forecast, source = parse_cpi(kind, new_data)
        if actual is not None and forecast is not None:
            surprise = actual - forecast
            self._trace(
                "news",
                tick=self.current_tick,
                kind=kind,
                parsed_type="cpi",
                source=source,
                actual=actual,
                forecast=forecast,
                surprise=surprise,
                mode=self.mode,
                labels=vars(self.labels),
            )
            if abs(surprise) > self.handoff_cpi_abs:
                signal = self.build_signal_from_cpi(surprise, source)
                if signal is not None:
                    constant = next(
                        sym for sym in self.cfg.rate_symbols if sym not in {signal.winner, signal.loser}
                    )
                    self._set_handoff_prediction(
                        winner=signal.winner,
                        loser=signal.loser,
                        constant=constant,
                        source=f"cpi:{source}",
                    )
                await self._trigger_handoff("cpi_trigger", source=source, surprise=surprise)
            else:
                self._trace(
                    "decision",
                    tick=self.current_tick,
                    mode=self.mode,
                    reason="cpi_guard_not_triggered",
                    surprise=surprise,
                    threshold=self.handoff_cpi_abs,
                )
            await self._evaluate_and_sync("news")
            return

        # FedSpeak trigger for handoff flatten.
        content = str(new_data.get("content", "") or "")
        message_type = str(new_data.get("type", "") or "")
        if kind == "unstructured" and content and self._is_fedspeak_candidate(content, message_type):
            sentiment = score_pm_fedspeak(content)
            self._trace(
                "news",
                tick=self.current_tick,
                kind=kind,
                parsed_type="fedspeak",
                message_type=message_type,
                content=content,
                sentiment=asdict(sentiment),
                mode=self.mode,
                labels=vars(self.labels),
            )
            if abs(sentiment.implied_bias_bp) >= self.handoff_fedspeak_min_abs_bias_bp:
                built = self._build_signal_from_fedspeak(content=content, message_type=message_type)
                if built is not None:
                    signal, _, constant = built
                    self._set_handoff_prediction(
                        winner=signal.winner,
                        loser=signal.loser,
                        constant=constant,
                        source=f"fedspeak:{message_type}",
                    )
                await self._trigger_handoff(
                    "fedspeak_trigger",
                    message_type=message_type,
                    score=sentiment.score,
                    bucket=sentiment.bucket,
                    implied_bias_bp=sentiment.implied_bias_bp,
                    matched_phrases=sentiment.matched_phrases,
                )
            else:
                self._trace(
                    "decision",
                    tick=self.current_tick,
                    mode=self.mode,
                    reason="fedspeak_guard_not_triggered",
                    implied_bias_bp=sentiment.implied_bias_bp,
                    threshold=self.handoff_fedspeak_min_abs_bias_bp,
                )
            await self._evaluate_and_sync("news")
            return


async def main():
    client = NewPM4FedSpeakClient(
        env_str("UTC_HOST", "34.197.188.76:3333"),
        env_str("UTC_USERNAME", "uiuc"),
        env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
