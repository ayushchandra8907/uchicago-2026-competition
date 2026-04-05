from __future__ import annotations

import argparse
import math

import pandas as pd

from ..config import load_app_config
from ..data_loader import build_run_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline transition table for A earnings.")
    parser.add_argument("--config", help="Optional JSON config override path.")
    parser.add_argument("--bucket-size", type=float, default=0.1, help="Earnings bucket size.")
    return parser.parse_args()


def _bucket(value: float, bucket_size: float) -> float:
    return round(math.floor(value / bucket_size) * bucket_size, 3)


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    catalog = build_run_catalog(config.paths.data_root)

    rows: list[dict[str, float | int | str]] = []
    for session in catalog.sessions:
        earnings = session.news_rows[
            (session.news_rows["kind"] == "structured")
            & (session.news_rows["structured_subtype"] == "earnings")
            & (session.news_rows["earnings_asset"] == "A")
        ][["time_ms", "earnings_value"]].dropna().sort_values("time_ms")
        if len(earnings) < 2:
            continue
        values = earnings["earnings_value"].astype(float).tolist()
        for current_value, next_value in zip(values, values[1:]):
            rows.append(
                {
                    "session_id": session.session_id,
                    "current_earnings": current_value,
                    "next_earnings": next_value,
                    "current_bucket": _bucket(current_value, args.bucket_size),
                    "next_bucket": _bucket(next_value, args.bucket_size),
                }
            )

    transitions = pd.DataFrame(rows)
    if transitions.empty:
        transitions.to_csv(config.paths.output_root / "earnings_transition_summary.csv", index=False)
        return

    summary = (
        transitions.groupby(["current_bucket", "next_bucket"], as_index=False)
        .agg(
            transition_count=("session_id", "count"),
            mean_next_earnings=("next_earnings", "mean"),
        )
        .sort_values(["current_bucket", "transition_count"], ascending=[True, False])
    )
    total_counts = summary.groupby("current_bucket")["transition_count"].transform("sum")
    summary["transition_probability"] = summary["transition_count"] / total_counts
    summary.to_csv(config.paths.output_root / "earnings_transition_summary.csv", index=False)


if __name__ == "__main__":
    main()
