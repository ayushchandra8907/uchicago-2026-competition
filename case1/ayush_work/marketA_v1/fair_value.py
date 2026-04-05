from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import SessionData


@dataclass(frozen=True)
class PEFitResult:
    pe_ratio: float
    observations: pd.DataFrame
    summary: pd.DataFrame


class FairValueModel:
    def __init__(self, pe_ratio: float, price_scale: int = 100) -> None:
        self.pe_ratio = float(pe_ratio)
        self.price_scale = int(price_scale)
        self.latest_earnings: float | None = None
        self.current_fair_px: int | None = None

    def update_earnings(self, earnings_value: float) -> int:
        self.latest_earnings = float(earnings_value)
        self.current_fair_px = int(round(self.latest_earnings * self.pe_ratio * self.price_scale))
        return self.current_fair_px

    def reference_fair_px(self, fallback_mid_px: float | None) -> int | None:
        if self.current_fair_px is not None:
            return self.current_fair_px
        if fallback_mid_px is None:
            return None
        return int(round(fallback_mid_px))


def fit_pe_ratio(
    sessions: tuple[SessionData, ...],
    *,
    price_scale: int = 100,
    trim_fraction: float = 0.1,
    default_pe_ratio: float = 10.0,
) -> PEFitResult:
    observations = collect_pe_observations(sessions, price_scale=price_scale)
    if observations.empty:
        summary = pd.DataFrame(
            [
                {
                    "global_pe_ratio": default_pe_ratio,
                    "observation_count": 0,
                    "trim_fraction": trim_fraction,
                }
            ]
        )
        return PEFitResult(pe_ratio=default_pe_ratio, observations=observations, summary=summary)

    ratios = sorted(value for value in observations["observed_pe_ratio"].tolist() if pd.notna(value))
    trim_count = int(len(ratios) * trim_fraction)
    if trim_count * 2 >= len(ratios):
        trimmed = ratios
    else:
        trimmed = ratios[trim_count : len(ratios) - trim_count]
    pe_ratio = float(pd.Series(trimmed).median()) if trimmed else default_pe_ratio

    per_session = (
        observations.groupby("session_id", as_index=False)["observed_pe_ratio"]
        .median()
        .rename(columns={"observed_pe_ratio": "session_pe_ratio"})
    )
    summary = pd.DataFrame(
        [
            {
                "global_pe_ratio": pe_ratio,
                "observation_count": int(len(observations)),
                "trim_fraction": trim_fraction,
                "median_observed_pe_ratio": float(observations["observed_pe_ratio"].median()),
                "mean_observed_pe_ratio": float(observations["observed_pe_ratio"].mean()),
            }
        ]
    )
    summary = summary.merge(
        pd.DataFrame(
            [
                {
                    "session_pe_ratio_min": float(per_session["session_pe_ratio"].min()) if not per_session.empty else pe_ratio,
                    "session_pe_ratio_max": float(per_session["session_pe_ratio"].max()) if not per_session.empty else pe_ratio,
                    "session_count": int(len(per_session)),
                }
            ]
        ),
        how="cross",
    )
    return PEFitResult(pe_ratio=pe_ratio, observations=observations, summary=summary)


def collect_pe_observations(sessions: tuple[SessionData, ...], *, price_scale: int = 100) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for session in sessions:
        book_rows = session.book_rows[["time_ms", "mid_px"]].dropna(subset=["time_ms", "mid_px"]).copy()
        if book_rows.empty:
            continue
        news_rows = session.news_rows[
            (session.news_rows["kind"] == "structured")
            & (session.news_rows["structured_subtype"] == "earnings")
            & (session.news_rows["earnings_asset"] == "A")
        ].copy()
        if news_rows.empty:
            continue
        news_rows["earnings_value"] = pd.to_numeric(news_rows["earnings_value"], errors="coerce")
        news_rows = news_rows.dropna(subset=["time_ms", "earnings_value"])
        for _, event in news_rows.iterrows():
            start_ms = float(event["time_ms"]) + 2_000.0
            end_ms = float(event["time_ms"]) + 5_000.0
            reaction_window = book_rows[(book_rows["time_ms"] >= start_ms) & (book_rows["time_ms"] <= end_ms)]
            if reaction_window.empty:
                continue
            observed_mid_px = float(reaction_window["mid_px"].median())
            earnings_value = float(event["earnings_value"])
            if earnings_value == 0:
                continue
            observed_pe_ratio = (observed_mid_px / price_scale) / earnings_value
            rows.append(
                {
                    "session_id": session.session_id,
                    "event_time_ms": float(event["time_ms"]),
                    "earnings_value": earnings_value,
                    "observed_mid_px": observed_mid_px,
                    "observed_pe_ratio": observed_pe_ratio,
                    "source_layout": session.source_layout,
                }
            )
    return pd.DataFrame(rows)


def write_pe_outputs(result: PEFitResult, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result.observations.to_csv(output_root / "earnings_event_analysis.csv", index=False)
    result.summary.to_csv(output_root / "pe_fit_summary.csv", index=False)
