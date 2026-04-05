from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json

import pandas as pd

from ..backtest.metrics import summarize_results, write_metrics_outputs
from ..backtest.simulator import run_backtests
from ..config import AppConfig, build_app_config, load_app_config, write_best_params
from ..data_loader import build_run_catalog
from ..fair_value import fit_pe_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a discrete parameter sweep for the A-only market maker.")
    parser.add_argument("--config", help="Optional JSON config override path.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap on sessions processed.")
    parser.add_argument(
        "--sweep-sessions",
        type=int,
        default=4,
        help="Number of representative sessions to use during the sweep before validating the winner on all selected sessions.",
    )
    return parser.parse_args()


def _candidate_configs(base: AppConfig) -> list[AppConfig]:
    grid = itertools.product(
        (2, 3),
        (0.35, 0.55),
        (12.0, 20.0),
        (5, 7),
        (150, 250),
    )
    configs: list[AppConfig] = []
    for base_half_spread_px, inventory_penalty, vol_widening, aggressive_edge_px, requote_cooldown_ms in grid:
        strategy = replace(
            base.strategy,
            base_half_spread_px=base_half_spread_px,
            inventory_penalty=inventory_penalty,
            vol_widening=vol_widening,
            aggressive_edge_px=aggressive_edge_px,
            shock_aggressive_edge_px=max(2, aggressive_edge_px - 2),
            requote_cooldown_ms=requote_cooldown_ms,
        )
        configs.append(
            build_app_config(
                repo_root=base.paths.repo_root,
                risk=base.risk,
                replay=base.replay,
                strategy=strategy,
            )
        )
    return configs


def _sample_sessions(sessions, limit: int):
    if limit <= 0 or len(sessions) <= limit:
        return sessions
    indices = sorted({round(index * (len(sessions) - 1) / (limit - 1)) for index in range(limit)})
    return tuple(sessions[index] for index in indices)


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    catalog = build_run_catalog(config.paths.data_root)
    sessions = catalog.sessions[: args.max_sessions] if args.max_sessions else catalog.sessions
    sweep_sessions = _sample_sessions(sessions, args.sweep_sessions)

    pe_fit = fit_pe_ratio(
        sessions,
        price_scale=config.strategy.price_scale,
        default_pe_ratio=config.strategy.initial_pe_ratio,
    )
    base = build_app_config(
        repo_root=config.paths.repo_root,
        risk=config.risk,
        replay=config.replay,
        strategy=replace(config.strategy, initial_pe_ratio=pe_fit.pe_ratio),
    )

    sweep_rows: list[dict[str, float | int]] = []
    best_config = base
    best_results = run_backtests(sweep_sessions, base)
    best_summary, _, _, _ = summarize_results(best_results)
    best_key = (
        float(best_summary.loc[0, "avg_pnl"]),
        -float(best_summary.loc[0, "stdev_pnl"]),
        -float(best_summary.loc[0, "avg_max_drawdown"]),
    )

    for candidate in _candidate_configs(base):
        results = run_backtests(sweep_sessions, candidate)
        summary, _, _, _ = summarize_results(results)
        avg_pnl = float(summary.loc[0, "avg_pnl"])
        stdev_pnl = float(summary.loc[0, "stdev_pnl"])
        avg_drawdown = float(summary.loc[0, "avg_max_drawdown"])
        sweep_rows.append(
            {
                "base_half_spread_px": candidate.strategy.base_half_spread_px,
                "inventory_penalty": candidate.strategy.inventory_penalty,
                "vol_widening": candidate.strategy.vol_widening,
                "aggressive_edge_px": candidate.strategy.aggressive_edge_px,
                "requote_cooldown_ms": candidate.strategy.requote_cooldown_ms,
                "avg_pnl": avg_pnl,
                "median_pnl": float(summary.loc[0, "median_pnl"]),
                "stdev_pnl": stdev_pnl,
                "avg_max_drawdown": avg_drawdown,
                "passive_fill_count": int(summary.loc[0, "passive_fill_count"]),
                "aggressive_fill_count": int(summary.loc[0, "aggressive_fill_count"]),
            }
        )
        key = (avg_pnl, -stdev_pnl, -avg_drawdown)
        if key > best_key:
            best_key = key
            best_config = candidate
            best_results = results

    results_df = pd.DataFrame(sweep_rows).sort_values(
        ["avg_pnl", "stdev_pnl", "avg_max_drawdown"],
        ascending=[False, True, True],
    )
    results_df.to_csv(config.paths.output_root / "parameter_sweep_results.csv", index=False)
    best_results = run_backtests(sessions, best_config)
    write_metrics_outputs(best_results, config.paths.output_root)
    write_best_params(best_config)
    (config.paths.output_root / "best_params_summary.json").write_text(
        json.dumps(
            {
                "pe_ratio": best_config.strategy.initial_pe_ratio,
                "strategy": {
                    "base_half_spread_px": best_config.strategy.base_half_spread_px,
                    "inventory_penalty": best_config.strategy.inventory_penalty,
                    "vol_widening": best_config.strategy.vol_widening,
                    "aggressive_edge_px": best_config.strategy.aggressive_edge_px,
                    "requote_cooldown_ms": best_config.strategy.requote_cooldown_ms,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
