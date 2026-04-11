from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import research_backtester as rb
from research_utils import load_module_from_path, rank_corr


def contingent_sharpe(daily_returns: np.ndarray, remove_top: int = 0, remove_bottom: int = 0) -> float:
    x = np.asarray(daily_returns, dtype=float).copy()
    if x.size == 0:
        return float("nan")
    keep = np.ones(len(x), dtype=bool)
    if remove_top > 0:
        top_idx = np.argsort(x)[-remove_top:]
        keep[top_idx] = False
    if remove_bottom > 0:
        bottom_idx = np.argsort(x)[:remove_bottom]
        keep[bottom_idx] = False
    return rb.annualized_sharpe(x[keep])


def quartile_bucket_table(values: np.ndarray, returns: np.ndarray, label: str) -> pd.DataFrame:
    x = np.asarray(values, dtype=float)
    y = np.asarray(returns, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return pd.DataFrame(columns=[label, "count", "mean_return", "sharpe"])

    quantiles = np.quantile(x, [0.25, 0.5, 0.75])
    bucket_ids = np.digitize(x, quantiles, right=True)
    rows = []
    for bucket in range(4):
        bucket_ret = y[bucket_ids == bucket]
        rows.append({
            label: f"Q{bucket + 1}",
            "count": int(len(bucket_ret)),
            "mean_return": float(np.mean(bucket_ret)) if len(bucket_ret) else float("nan"),
            "sharpe": rb.annualized_sharpe(bucket_ret) if len(bucket_ret) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def signal_ic_table(aggregate: dict, rolling_window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_debug = aggregate["signal_debug"]
    asset_daily_returns = aggregate["asset_daily_returns"]
    asset_net_contrib = aggregate["asset_net_contrib"]
    turnover = aggregate["daily_turnover"]

    component_names = [
        "alpha",
        "model_signal",
        "raw_family_signal",
        "resid_family_signal",
        "hgb_signal",
        "fast_signal",
        "intraday_signal",
    ]
    rows = []
    rolling_rows = []

    for name in component_names:
        ics = []
        cost_adjusted = []
        turnover_adjusted = []
        for day, debug in enumerate(signal_debug):
            if day >= len(asset_daily_returns):
                break
            if not isinstance(debug, dict):
                continue
            signal = debug.get(name)
            if signal is None:
                continue
            signal = np.asarray(signal, dtype=float)
            ic = rank_corr(signal, asset_daily_returns[day])
            cost_ic = rank_corr(signal, asset_net_contrib[day])
            turn_adj = ic / (1.0 + float(turnover[day])) if day < len(turnover) else ic
            ics.append(ic)
            cost_adjusted.append(cost_ic)
            turnover_adjusted.append(turn_adj)

        if not ics:
            continue

        rows.append({
            "signal": name,
            "mean_ic": float(np.mean(ics)),
            "std_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
            "mean_cost_adjusted_ic": float(np.mean(cost_adjusted)),
            "mean_turnover_adjusted_ic": float(np.mean(turnover_adjusted)),
            "positive_ic_share": float(np.mean(np.array(ics) > 0.0)),
        })

        series = pd.Series(ics, dtype=float)
        rolling = series.rolling(rolling_window, min_periods=max(5, rolling_window // 2)).mean()
        for idx, value in enumerate(rolling):
            rolling_rows.append({
                "signal": name,
                "day_index": idx,
                "rolling_ic_20": float(value) if np.isfinite(value) else np.nan,
            })

    summary = pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)
    rolling = pd.DataFrame(rolling_rows)
    return summary, rolling


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfitting diagnostics for Case 2 strategies")
    parser.add_argument("--module-path", default="submission_converted.py")
    parser.add_argument("--prices-path", default="prices.csv")
    parser.add_argument("--meta-path", default="meta.csv")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    strategy_module = load_module_from_path("submission_converted", args.module_path)
    prices = strategy_module.load_prices(args.prices_path)
    meta = strategy_module.load_meta(args.meta_path)
    strategy_factory = rb.create_strategy_factory(strategy_module)
    total_days = prices.shape[0] // rb.TICKS_PER_DAY

    evaluations = {
        "parity": rb.evaluate_windows(prices, meta, strategy_factory, rb.make_official_parity_windows(total_days)),
        "yearly": rb.evaluate_windows(prices, meta, strategy_factory, rb.make_yearly_windows(total_days)),
        "quarterly": rb.evaluate_windows(
            prices,
            meta,
            strategy_factory,
            rb.make_expanding_windows(total_days, min_train_days=2 * rb.TRADING_DAYS_PER_YEAR, test_days=63, step_days=63, prefix="quarterly"),
        ),
        "semiannual": rb.evaluate_windows(
            prices,
            meta,
            strategy_factory,
            rb.make_expanding_windows(total_days, min_train_days=2 * rb.TRADING_DAYS_PER_YEAR, test_days=126, step_days=63, prefix="semiannual"),
        ),
    }

    sharpe_rows = []
    for name, evaluation in evaluations.items():
        summary = evaluation["summary"]
        sharpe_rows.append({
            "mode": name,
            "mean_sharpe": summary["mean_sharpe"],
            "std_sharpe": summary["std_sharpe"],
            "min_sharpe": summary["min_sharpe"],
            "recent_weighted_mean_sharpe": summary["recent_weighted_mean_sharpe"],
            "robust_score": summary["robust_score"],
        })
    sharpe_stability = pd.DataFrame(sharpe_rows)

    quarterly_agg = rb.aggregate_daily_metrics(evaluations["quarterly"])
    parity_agg = rb.aggregate_daily_metrics(evaluations["parity"])

    return_contingent = pd.DataFrame([
        {"sample": "quarterly_full", "sharpe": rb.annualized_sharpe(quarterly_agg["daily_returns_net"])},
        {"sample": "quarterly_drop_top5", "sharpe": contingent_sharpe(quarterly_agg["daily_returns_net"], remove_top=5)},
        {"sample": "quarterly_drop_top10", "sharpe": contingent_sharpe(quarterly_agg["daily_returns_net"], remove_top=10)},
        {"sample": "quarterly_drop_bottom5", "sharpe": contingent_sharpe(quarterly_agg["daily_returns_net"], remove_bottom=5)},
        {"sample": "quarterly_drop_top5_bottom5", "sharpe": contingent_sharpe(quarterly_agg["daily_returns_net"], remove_top=5, remove_bottom=5)},
        {"sample": "parity_full", "sharpe": rb.annualized_sharpe(parity_agg["daily_returns_net"])},
    ])

    cost_contingent = pd.DataFrame([
        {"path": "before_costs", "sharpe": rb.annualized_sharpe(quarterly_agg["daily_returns_gross"])},
        {"path": "after_txn_only", "sharpe": rb.annualized_sharpe(quarterly_agg["daily_returns_txn"])},
        {"path": "after_txn_and_borrow", "sharpe": rb.annualized_sharpe(quarterly_agg["daily_returns_net"])},
    ])

    exposure_gross = quartile_bucket_table(quarterly_agg["daily_gross"], quarterly_agg["daily_returns_net"], "gross_quartile")
    exposure_turnover = quartile_bucket_table(quarterly_agg["daily_turnover"], quarterly_agg["daily_returns_net"], "turnover_quartile")
    exposure_short = quartile_bucket_table(quarterly_agg["daily_short"], quarterly_agg["daily_returns_net"], "short_quartile")

    ic_summary, ic_rolling = signal_ic_table(quarterly_agg)

    print("\nSharpe Stability")
    print(sharpe_stability.to_string(index=False))
    print("\nReturn-Contingent Sharpe")
    print(return_contingent.to_string(index=False))
    print("\nCost-Contingent Sharpe")
    print(cost_contingent.to_string(index=False))
    print("\nExposure-Contingent Sharpe: Gross")
    print(exposure_gross.to_string(index=False))
    print("\nExposure-Contingent Sharpe: Turnover")
    print(exposure_turnover.to_string(index=False))
    print("\nExposure-Contingent Sharpe: Short Borrow Exposure")
    print(exposure_short.to_string(index=False))
    print("\nSignal IC Summary")
    print(ic_summary.to_string(index=False))

    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        sharpe_stability.to_csv(outdir / "sharpe_stability.csv", index=False)
        return_contingent.to_csv(outdir / "return_contingent_sharpe.csv", index=False)
        cost_contingent.to_csv(outdir / "cost_contingent_sharpe.csv", index=False)
        exposure_gross.to_csv(outdir / "exposure_gross.csv", index=False)
        exposure_turnover.to_csv(outdir / "exposure_turnover.csv", index=False)
        exposure_short.to_csv(outdir / "exposure_short.csv", index=False)
        ic_summary.to_csv(outdir / "signal_ic_summary.csv", index=False)
        ic_rolling.to_csv(outdir / "signal_ic_rolling.csv", index=False)


if __name__ == "__main__":
    main()
