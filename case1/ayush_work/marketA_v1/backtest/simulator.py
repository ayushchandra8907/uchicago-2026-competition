from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import AppConfig, build_app_config, load_app_config, write_best_params
from ..data_loader import build_run_catalog
from ..fair_value import fit_pe_ratio, write_pe_outputs
from ..models import BacktestResult, SessionData
from .metrics import write_metrics_outputs
from .replay_engine import run_session_backtest


def run_backtests(sessions: tuple[SessionData, ...], config: AppConfig) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    for session in sessions:
        artifacts = run_session_backtest(session, config)
        results.append(artifacts.result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-only replay/backtest suite.")
    parser.add_argument("--config", help="Optional JSON config override path.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap on sessions processed.")
    parser.add_argument("--pe-ratio", type=float, default=None, help="Override fitted PE ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    catalog = build_run_catalog(config.paths.data_root)
    sessions = catalog.sessions[: args.max_sessions] if args.max_sessions else catalog.sessions

    pe_result = fit_pe_ratio(sessions, price_scale=config.strategy.price_scale, default_pe_ratio=config.strategy.initial_pe_ratio)
    pe_ratio = args.pe_ratio if args.pe_ratio is not None else pe_result.pe_ratio
    tuned_config = build_app_config(
        repo_root=config.paths.repo_root,
        risk=config.risk,
        replay=config.replay,
        strategy=replace(config.strategy, initial_pe_ratio=pe_ratio),
    )
    results = run_backtests(sessions, tuned_config)
    write_metrics_outputs(results, tuned_config.paths.output_root)
    write_pe_outputs(pe_result, tuned_config.paths.output_root)
    write_best_params(tuned_config)


if __name__ == "__main__":
    main()
