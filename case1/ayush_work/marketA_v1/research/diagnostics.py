from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_app_config
from ..data_loader import build_run_catalog
from ..fair_value import fit_pe_ratio, write_pe_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write loader and PE-fit diagnostics for the A-only strategy.")
    parser.add_argument("--config", help="Optional JSON config override path.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap on sessions processed.")
    return parser.parse_args()


def _session_diagnostics_frame(catalog) -> pd.DataFrame:
    rows: list[dict[str, str | float | int]] = []
    for session in catalog.sessions:
        rows.append(
            {
                "session_id": session.session_id,
                "layout": session.source_layout,
                "book_event_count": session.diagnostics["book_event_count"],
                "trade_event_count": session.diagnostics["trade_event_count"],
                "news_event_count": session.diagnostics["news_event_count"],
                "earnings_event_count": session.diagnostics["earnings_event_count"],
                "duration_ms": float(session.events[-1].time_ms - session.events[0].time_ms) if session.events else 0.0,
            }
        )
    for skipped in catalog.skipped_runs:
        rows.append(
            {
                "session_id": skipped.path.name,
                "layout": "skipped",
                "book_event_count": 0,
                "trade_event_count": 0,
                "news_event_count": 0,
                "earnings_event_count": 0,
                "duration_ms": 0.0,
                "skip_reason": skipped.reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    catalog = build_run_catalog(config.paths.data_root)
    sessions = catalog.sessions[: args.max_sessions] if args.max_sessions else catalog.sessions
    diagnostics_df = _session_diagnostics_frame(catalog)
    diagnostics_df.to_csv(config.paths.output_root / "session_diagnostics.csv", index=False)

    pe_result = fit_pe_ratio(
        sessions,
        price_scale=config.strategy.price_scale,
        default_pe_ratio=config.strategy.initial_pe_ratio,
    )
    write_pe_outputs(pe_result, config.paths.output_root)


if __name__ == "__main__":
    main()
