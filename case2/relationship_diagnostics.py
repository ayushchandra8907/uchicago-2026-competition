from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_utils import daily_intraday_arrays, daily_returns_from_prices, load_module_from_path, rank_corr, rank_vector, rolling_ols_beta

ROLL_WINDOW = 63
SIGNAL_ROLL_WINDOW = 20


def mean_offdiag(corr: np.ndarray) -> float:
    if corr.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = corr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def residual_block(window: np.ndarray, sector_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_days, n_assets = window.shape
    market = np.mean(window, axis=1)
    resid = np.zeros_like(window)
    beta_market = np.zeros(n_assets, dtype=float)
    beta_sector = np.zeros(n_assets, dtype=float)

    for i in range(n_assets):
        y = window[:, i]
        beta_market[i] = rolling_ols_beta(y, market)
        sec = sector_id[i]
        peer_idx = np.where((sector_id == sec) & (np.arange(n_assets) != i))[0]
        sector_peer = np.mean(window[:, peer_idx], axis=1) if len(peer_idx) > 0 else np.zeros(n_days, dtype=float)
        beta_sector[i] = rolling_ols_beta(y, sector_peer)
        resid[:, i] = y - beta_market[i] * market - beta_sector[i] * sector_peer

    return resid, beta_market, beta_sector


def candidate_signals(R: np.ndarray, intraday: dict[str, np.ndarray], sector_id: np.ndarray, t: int, resid_win: np.ndarray) -> dict[str, np.ndarray]:
    ret_1 = R[t]
    ret_5 = np.mean(R[t - 4 : t + 1], axis=0)
    ret_10 = np.mean(R[t - 9 : t + 1], axis=0)
    rel_5 = np.zeros(R.shape[1], dtype=float)
    for sec in np.unique(sector_id):
        idx = np.where(sector_id == sec)[0]
        rel_5[idx] = ret_5[idx] - np.mean(ret_5[idx])

    resid_5 = np.mean(resid_win[-5:], axis=0)
    intraday_sector_rel = intraday["sector_rel"][t + 1]
    intraday_last5 = intraday["last5"][t + 1]
    intraday_oc = intraday["open_close"][t + 1]

    combo_fast = (
        rank_vector(ret_1)
        + rank_vector(ret_5)
        + rank_vector(ret_10)
        + rank_vector(resid_5)
        + rank_vector(rel_5)
    ) / 5.0

    return {
        "mom_1": ret_1,
        "mom_5": ret_5,
        "mom_10": ret_10,
        "rel_5": rel_5,
        "resid_5": resid_5,
        "intraday_sector_rel": intraday_sector_rel,
        "intraday_last5": intraday_last5,
        "intraday_oc": intraday_oc,
        "combo_fast": combo_fast,
    }


def compute_relationship_tables(prices: np.ndarray, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_id = meta_df["sector_id"].to_numpy(dtype=int)
    returns = daily_returns_from_prices(prices)
    intraday = daily_intraday_arrays(prices, sector_id)

    relationship_rows = []
    signal_rows = []

    for t in range(max(ROLL_WINDOW - 1, 10), len(returns) - 1):
        window = returns[t - ROLL_WINDOW + 1 : t + 1]
        asset_corr = np.corrcoef(window, rowvar=False)

        within_vals = []
        between_vals = []
        for i in range(asset_corr.shape[0]):
            for j in range(i + 1, asset_corr.shape[1]):
                if sector_id[i] == sector_id[j]:
                    within_vals.append(asset_corr[i, j])
                else:
                    between_vals.append(asset_corr[i, j])

        sector_baskets = []
        for sec in np.unique(sector_id):
            idx = np.where(sector_id == sec)[0]
            sector_baskets.append(np.mean(window[:, idx], axis=1))
        sector_baskets = np.column_stack(sector_baskets)
        sector_corr = np.corrcoef(sector_baskets, rowvar=False)

        cov = np.cov(window, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        top_eigen_share = float(eigvals[-1] / np.sum(eigvals)) if eigvals.size else 0.0

        resid, beta_market, beta_sector = residual_block(window, sector_id)
        resid_corrs = []
        for sec in np.unique(sector_id):
            idx = np.where(sector_id == sec)[0]
            if len(idx) <= 1:
                continue
            corr = np.corrcoef(resid[:, idx], rowvar=False)
            resid_corrs.append(mean_offdiag(corr))

        autocorr_lag1 = []
        autocorr_lag2 = []
        autocorr_lag5 = []
        for i in range(window.shape[1]):
            s = pd.Series(window[:, i])
            autocorr_lag1.append(float(s.autocorr(1)))
            autocorr_lag2.append(float(s.autocorr(2)))
            autocorr_lag5.append(float(s.autocorr(5)))

        sector_lead_lag = []
        for i in range(window.shape[1]):
            sec = sector_id[i]
            idx = np.where((sector_id == sec) & (np.arange(window.shape[1]) != i))[0]
            if len(idx) == 0:
                continue
            sector_series = np.mean(window[:, idx], axis=1)
            residual_series = resid[:, i]
            sector_lead_lag.append(np.corrcoef(sector_series[:-1], residual_series[1:])[0, 1])

        relationship_rows.append({
            "day_index": t,
            "within_sector_corr_mean": float(np.nanmean(within_vals)),
            "between_sector_corr_mean": float(np.nanmean(between_vals)),
            "sector_basket_corr_mean": mean_offdiag(sector_corr),
            "top_eigen_share": top_eigen_share,
            "same_sector_resid_corr_mean": float(np.nanmean(resid_corrs)) if resid_corrs else 0.0,
            "mean_abs_beta_market": float(np.mean(np.abs(beta_market))),
            "mean_abs_beta_sector": float(np.mean(np.abs(beta_sector))),
            "autocorr_lag1_mean": float(np.nanmean(autocorr_lag1)),
            "autocorr_lag2_mean": float(np.nanmean(autocorr_lag2)),
            "autocorr_lag5_mean": float(np.nanmean(autocorr_lag5)),
            "lead_lag_sector_resid_mean": float(np.nanmean(sector_lead_lag)) if sector_lead_lag else 0.0,
            "cross_sectional_dispersion": float(np.std(returns[t])),
        })

        signals = candidate_signals(returns, intraday, sector_id, t, resid)
        next_ret = returns[t + 1]
        for name, signal in signals.items():
            ranked = rank_vector(signal)
            top = next_ret[ranked >= np.quantile(ranked, 0.8)]
            bottom = next_ret[ranked <= np.quantile(ranked, 0.2)]
            signal_rows.append({
                "day_index": t,
                "signal": name,
                "rank_ic": rank_corr(signal, next_ret),
                "top_bottom_spread": float(np.mean(top) - np.mean(bottom)),
            })

    relationship_df = pd.DataFrame(relationship_rows)
    signal_df = pd.DataFrame(signal_rows)
    signal_df["rolling_ic_20"] = signal_df.groupby("signal")["rank_ic"].transform(
        lambda s: s.rolling(SIGNAL_ROLL_WINDOW, min_periods=max(5, SIGNAL_ROLL_WINDOW // 2)).mean()
    )
    signal_df["rolling_spread_20"] = signal_df.groupby("signal")["top_bottom_spread"].transform(
        lambda s: s.rolling(SIGNAL_ROLL_WINDOW, min_periods=max(5, SIGNAL_ROLL_WINDOW // 2)).mean()
    )
    return relationship_df, signal_df


def add_regime_labels(relationship_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    regime = relationship_df.copy()

    top_eig_q25, top_eig_q75 = regime["top_eigen_share"].quantile([0.25, 0.75]).tolist()
    disp_q25, disp_q75 = regime["cross_sectional_dispersion"].quantile([0.25, 0.75]).tolist()

    benchmark = signal_df[signal_df["signal"] == "combo_fast"][["day_index", "rolling_ic_20"]].copy()
    benchmark = benchmark.rename(columns={"rolling_ic_20": "signal_benchmark_rolling_ic"})
    regime = regime.merge(benchmark, on="day_index", how="left")

    regime["high_common_factor"] = regime["top_eigen_share"] >= top_eig_q75
    regime["weak_common_factor"] = regime["top_eigen_share"] <= top_eig_q25
    regime["high_dispersion"] = regime["cross_sectional_dispersion"] >= disp_q75
    regime["low_dispersion"] = regime["cross_sectional_dispersion"] <= disp_q25
    regime["signal_on"] = regime["signal_benchmark_rolling_ic"] > 0.01
    regime["signal_off"] = regime["signal_benchmark_rolling_ic"] <= 0.0
    return regime


def regime_summary(regime_df: pd.DataFrame, signal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    regime_flags = [
        "high_common_factor",
        "weak_common_factor",
        "high_dispersion",
        "low_dispersion",
        "signal_on",
        "signal_off",
    ]
    relationship_rows = []
    signal_rows = []

    for flag in regime_flags:
        subset = regime_df[regime_df[flag]]
        if subset.empty:
            continue
        relationship_rows.append({
            "regime": flag,
            "count_days": int(len(subset)),
            "within_sector_corr_mean": float(subset["within_sector_corr_mean"].mean()),
            "between_sector_corr_mean": float(subset["between_sector_corr_mean"].mean()),
            "top_eigen_share_mean": float(subset["top_eigen_share"].mean()),
            "same_sector_resid_corr_mean": float(subset["same_sector_resid_corr_mean"].mean()),
            "signal_benchmark_rolling_ic_mean": float(subset["signal_benchmark_rolling_ic"].mean()),
        })

        joined = signal_df.merge(subset[["day_index"]], on="day_index", how="inner")
        for signal_name, grp in joined.groupby("signal"):
            signal_rows.append({
                "regime": flag,
                "signal": signal_name,
                "mean_rank_ic": float(grp["rank_ic"].mean()),
                "positive_ic_share": float((grp["rank_ic"] > 0.0).mean()),
                "mean_top_bottom_spread": float(grp["top_bottom_spread"].mean()),
            })

    return pd.DataFrame(relationship_rows), pd.DataFrame(signal_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relationship and regime diagnostics for Case 2")
    parser.add_argument("--module-path", default="submission_converted.py")
    parser.add_argument("--prices-path", default="prices.csv")
    parser.add_argument("--meta-path", default="meta.csv")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    strategy_module = load_module_from_path("submission_converted", args.module_path)
    prices = strategy_module.load_prices(args.prices_path)
    meta_df = pd.read_csv(args.meta_path)

    relationship_df, signal_df = compute_relationship_tables(prices, meta_df)
    regime_df = add_regime_labels(relationship_df, signal_df)
    relationship_summary, signal_summary = regime_summary(regime_df, signal_df)

    overall_signal_summary = (
        signal_df.groupby("signal")
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            std_rank_ic=("rank_ic", "std"),
            positive_ic_share=("rank_ic", lambda s: float((s > 0.0).mean())),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
        )
        .reset_index()
        .sort_values("mean_rank_ic", ascending=False)
    )

    print("\nRelationship Summary (full sample)")
    print(relationship_df.mean(numeric_only=True).to_frame("mean").T.to_string(index=False))
    print("\nOverall Signal Summary")
    print(overall_signal_summary.to_string(index=False))
    print("\nRegime Relationship Summary")
    print(relationship_summary.to_string(index=False))
    print("\nRegime Signal Summary")
    print(signal_summary.to_string(index=False))

    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        relationship_df.to_csv(outdir / "rolling_relationships.csv", index=False)
        signal_df.to_csv(outdir / "rolling_signal_metrics.csv", index=False)
        regime_df.to_csv(outdir / "regime_labels.csv", index=False)
        relationship_summary.to_csv(outdir / "regime_relationship_summary.csv", index=False)
        signal_summary.to_csv(outdir / "regime_signal_summary.csv", index=False)
        overall_signal_summary.to_csv(outdir / "overall_signal_summary.csv", index=False)


if __name__ == "__main__":
    main()
