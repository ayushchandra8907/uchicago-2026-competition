from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

try:
    from PM_fedspeak_sentiment import PMFedSpeakSentiment
except ModuleNotFoundError:
    from case1.caleb_work.PM_fedspeak_sentiment import PMFedSpeakSentiment


@dataclass(frozen=True)
class PMFedSpeakHeadlineAnalysis:
    event_id: int
    tick: int | None
    headline: str
    message_type: str
    score: float
    bucket: str
    implied_bias_bp: float
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]
    q_hike_before: float
    q_hold_before: float
    q_cut_before: float
    q_hike_after: float
    q_hold_after: float
    q_cut_after: float
    delta_hike: float
    delta_hold: float
    delta_cut: float
    winner_observed: str
    loser_observed: str
    winner_predicted: str
    loser_predicted: str
    target_abs: int
    verdict: str


@dataclass
class _PendingFedSpeakEvent:
    event_id: int
    tick: int | None
    headline: str
    message_type: str
    sentiment: PMFedSpeakSentiment
    winner_predicted: str
    loser_predicted: str
    constant_predicted: str
    target_abs: int
    q_hike_before: float
    q_hold_before: float
    q_cut_before: float
    started_ts: float


class PMFedSpeakTracker:
    """Lightweight online tracker for PM FedSpeak events."""

    def __init__(self, resolve_after_sec: float = 1.50, max_history: int = 256):
        self.resolve_after_sec = float(resolve_after_sec)
        self.max_history = int(max_history)
        self._next_event_id = 1
        self._pending: dict[int, _PendingFedSpeakEvent] = {}
        self._history: list[PMFedSpeakHeadlineAnalysis] = []

    def register(
        self,
        *,
        tick: int | None,
        headline: str,
        message_type: str,
        sentiment: PMFedSpeakSentiment,
        winner_predicted: str,
        loser_predicted: str,
        constant_predicted: str,
        target_abs: int,
        q_hike_before: float,
        q_hold_before: float,
        q_cut_before: float,
        now_ts: float | None = None,
    ) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1
        self._pending[event_id] = _PendingFedSpeakEvent(
            event_id=event_id,
            tick=tick,
            headline=headline,
            message_type=message_type,
            sentiment=sentiment,
            winner_predicted=winner_predicted,
            loser_predicted=loser_predicted,
            constant_predicted=constant_predicted,
            target_abs=int(target_abs),
            q_hike_before=float(q_hike_before),
            q_hold_before=float(q_hold_before),
            q_cut_before=float(q_cut_before),
            started_ts=time.time() if now_ts is None else float(now_ts),
        )
        return event_id

    def resolve_ready(
        self,
        *,
        q_hike_now: float,
        q_hold_now: float,
        q_cut_now: float,
        now_ts: float | None = None,
    ) -> list[PMFedSpeakHeadlineAnalysis]:
        now = time.time() if now_ts is None else float(now_ts)
        ready: list[PMFedSpeakHeadlineAnalysis] = []
        labels = ("R_HIKE", "R_HOLD", "R_CUT")

        for event_id, event in list(self._pending.items()):
            if (now - event.started_ts) < self.resolve_after_sec:
                continue

            delta_hike = float(q_hike_now - event.q_hike_before)
            delta_hold = float(q_hold_now - event.q_hold_before)
            delta_cut = float(q_cut_now - event.q_cut_before)
            deltas = (delta_hike, delta_hold, delta_cut)
            winner_observed = labels[max(range(3), key=lambda idx: deltas[idx])]
            loser_observed = labels[min(range(3), key=lambda idx: deltas[idx])]
            verdict = "matched" if (winner_observed == event.winner_predicted and loser_observed == event.loser_predicted) else "mismatch"

            resolved = PMFedSpeakHeadlineAnalysis(
                event_id=event.event_id,
                tick=event.tick,
                headline=event.headline,
                message_type=event.message_type,
                score=event.sentiment.score,
                bucket=event.sentiment.bucket,
                implied_bias_bp=event.sentiment.implied_bias_bp,
                matched_unigrams=event.sentiment.matched_unigrams,
                matched_bigrams=event.sentiment.matched_bigrams,
                unknown_candidate_phrases=event.sentiment.unknown_candidate_phrases,
                q_hike_before=event.q_hike_before,
                q_hold_before=event.q_hold_before,
                q_cut_before=event.q_cut_before,
                q_hike_after=float(q_hike_now),
                q_hold_after=float(q_hold_now),
                q_cut_after=float(q_cut_now),
                delta_hike=delta_hike,
                delta_hold=delta_hold,
                delta_cut=delta_cut,
                winner_observed=winner_observed,
                loser_observed=loser_observed,
                winner_predicted=event.winner_predicted,
                loser_predicted=event.loser_predicted,
                target_abs=event.target_abs,
                verdict=verdict,
            )
            ready.append(resolved)
            self._history.append(resolved)
            self._pending.pop(event_id, None)

        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]
        return ready

    def summary(self) -> dict[str, Any]:
        total = len(self._history)
        matched = sum(1 for row in self._history if row.verdict == "matched")
        bucket_counts: dict[str, int] = {}
        for row in self._history:
            bucket_counts[row.bucket] = bucket_counts.get(row.bucket, 0) + 1
        return {
            "resolved_events": total,
            "matched_events": matched,
            "match_rate": 0.0 if total == 0 else float(matched) / float(total),
            "bucket_counts": bucket_counts,
            "pending_events": len(self._pending),
        }

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(row) for row in self._history]
