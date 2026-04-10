from __future__ import annotations

"""Participant submission scaffold for the portfolio optimization case.

Implement your strategy by modifying the MyStrategy class below.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import HuberRegressor, Ridge

N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


@dataclass(frozen=True)
class PublicMeta:
    """Per-asset metadata visible to participants."""

    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    """Load the price matrix from CSV. Returns shape (n_ticks, 25)."""
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
    """Load asset metadata from CSV."""
    df = pd.read_csv(path)
    return PublicMeta(
        sector_id=df["sector_id"].to_numpy(dtype=int),
        spread_bps=df["spread_bps"].to_numpy(dtype=float),
        borrow_bps_annual=df["borrow_bps_annual"].to_numpy(dtype=float),
    )


class StrategyBase:
    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        pass

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        raise NotImplementedError


def project_to_gross_limit(w: np.ndarray) -> np.ndarray:
    """Project weights back onto the L1 gross-exposure constraint (<=1)."""
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross):
        return np.zeros_like(w)
    if gross > 1.0:
        w /= gross
    return w


def fit_covariances(train_ret: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = train_ret.values
    cov_sample = np.cov(x, rowvar=False)
    lw = LedoitWolf().fit(x)
    cov_lw = lw.covariance_
    return cov_sample, cov_lw


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n, dtype=float) / n
    eps = 1e-12

    for _ in range(max_iter):
        marginal = cov @ w
        rc = w * marginal
        port_var = float(w @ marginal)

        if port_var <= eps:
            return np.ones(n, dtype=float) / n

        target = port_var / n
        denom = np.maximum(rc, eps)
        w_new = w * (target / denom)
        w_new = np.maximum(w_new, eps)
        w_new /= w_new.sum()

        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new

    return w


class MyStrategy(StrategyBase):
    """Ridge raw-return + Huber residual-return alpha with regime-scaled execution."""

    def __init__(self):
        self.ticks_per_day = TICKS_PER_DAY
        self.sector_id = None
        self.spread_bps = None
        self.borrow_bps_annual = None
        self.sector_to_indices = None
        self.sector_peer_indices = None

        self.lookback_cov = 84
        self.min_days = 120

        self.beta = 1.12
        self.rebalance_rate = 0.14
        self.k_frac = 0.08
        self.dataset_min_days = 60
        self.dataset_lookback = 60
        self.fit_min_samples = 100

        self.momentum_windows = (1, 5, 10, 20, 60)
        self.vol_windows = (5, 20, 60)
        self.market_windows = (1, 5, 20)
        self.sector_windows = (5, 20, 60)

        self.alpha_vol_floor = 1e-6
        self.alpha_shrink = 0.7
        self.alpha_tanh_scale = 1.5
        self.turnover_band = 0.011

        self.resid_feature_lookback = 60
        self.resid_feature_ridge = 1e-5
        self.resid_momentum_windows = (1, 5, 20, 60)
        self.use_residual_features = True
        self.use_residual_ensemble = True
        self.resid_feature_lookbacks = (40, 60, 80)
        self.resid_ensemble_weights = (0.55, 0.30, 0.15)
        self.agreement_scale_floor = 0.50
        self.use_confidence_gross_scaling = True
        self.gross_scale_floor = 0.80
        self.use_recent_ic_scaling = True
        self.recent_ic_lookback_days = 20
        self.recent_ic_scale_floor = 0.80
        self.recent_ic_bad = -0.02
        self.recent_ic_good = 0.03

        self.cov_alpha_mix = 0.02
        self.cov_alpha_ridge = 0.05
        self.cov_alpha_floor = 1e-6

        self.cs_percentiles = (10.0, 90.0)
        self.cs_scale_floor = 1e-6
        self.cs_clip = 3.0

        self.trend_window = 20
        self.trend_threshold = 0.001
        self.trend_beta_scale = 0.8
        self.history_scale_days = 756.0

        self.ridge_alpha = 1.0
        self.target_mode = "raw"

        self.raw_family_weight = 0.4
        self.resid_family_weight = 0.6
        self.raw_model_kind = "ridge"
        self.resid_model_kind = "huber"
        self.raw_family_lookbacks = (40, 60, 80)
        self.raw_family_weights = (0.55, 0.30, 0.15)
        self.resid_family_lookbacks = (40, 60, 80)
        self.resid_family_weights = (0.55, 0.30, 0.15)
        self.huber_alpha = 1e-4
        self.huber_epsilon = 1.35
        self.huber_max_iter = 1000

        self.global_top_k = 2
        self.global_bottom_k = 2
        self.borrow_short_damp = 0.0
        self.spread_turnover_scale = 0.5
        self.spread_turnover_min_mult = 0.75
        self.spread_turnover_max_mult = 1.25

        self.ridge_model = Ridge(alpha=self.ridge_alpha)
        self.model_families = {}

        self.ridge_fitted = False
        self.alpha_agreement_scale = 1.0
        self.last_debug = {}
        self.online_signal_history = {}
        self.realized_return_history = []
        self.pending_signals = None
        self.last_recorded_return_count = 0
        self.online_min_ic_obs = 5
        self.use_regime_quality_model = True
        self.regime_lookback_days = 63
        self.regime_signal_lookback_days = 20
        self.regime_horizon_days = 21
        self.regime_model_alpha = 2.0
        self.regime_floor = 0.70
        self.regime_beta_floor = 0.75
        self.regime_scale_beta = True
        self.regime_scale_gross = True
        self.regime_pred_bad_quantile = 0.25
        self.regime_pred_good_quantile = 0.75
        self.regime_model = Ridge(alpha=self.regime_model_alpha)
        self.regime_model_fitted = False
        self.regime_feature_mean = None
        self.regime_feature_std = None
        self.regime_pred_low = None
        self.regime_pred_high = None

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.ticks_per_day = int(kwargs.get("ticks_per_day", TICKS_PER_DAY))
        self.sector_id = np.asarray(meta.sector_id, dtype=int)
        self.spread_bps = np.asarray(meta.spread_bps, dtype=float)
        self.borrow_bps_annual = np.asarray(meta.borrow_bps_annual, dtype=float)
        unique_sectors = np.unique(self.sector_id)
        self.sector_to_indices = {
            int(sec): np.where(self.sector_id == sec)[0] for sec in unique_sectors
        }
        self.sector_peer_indices = []
        for i, sec in enumerate(self.sector_id):
            sec_idx = self.sector_to_indices[int(sec)]
            self.sector_peer_indices.append(sec_idx[sec_idx != i])
        self.prev_weights = None
        self.last_debug = {}
        self.online_signal_history = {}
        self.realized_return_history = []
        self.pending_signals = None
        self.last_recorded_return_count = 0
        self.regime_model_fitted = False
        self.regime_feature_mean = None
        self.regime_feature_std = None
        self.regime_pred_low = None
        self.regime_pred_high = None
    
        self.train_days_available = len(self._daily_returns_df(train_prices))
    
        daily_ret = self._daily_returns_df(train_prices)
        intraday_arrays = self._daily_intraday_arrays(train_prices)
        self.last_recorded_return_count = len(daily_ret)
        self.model_families = {}

        raw_family = self._train_ridge_family(
            daily_ret,
            intraday_arrays,
            target_mode="raw",
            lookbacks=self.raw_family_lookbacks,
            weights=self.raw_family_weights,
        )
        resid_family = self._train_ridge_family(
            daily_ret,
            intraday_arrays,
            target_mode="residual_vol_adj",
            lookbacks=self.resid_family_lookbacks,
            weights=self.resid_family_weights,
        )

        if raw_family is not None:
            self.model_families["raw"] = raw_family
        if resid_family is not None:
            self.model_families["resid"] = resid_family

        self.ridge_fitted = len(self.model_families) > 0

        self._train_regime_quality_model(daily_ret)

    def _daily_closes(self, price_history) -> np.ndarray:
        prices = np.asarray(price_history, dtype=float)
        n_ticks = prices.shape[0]
        n_days = n_ticks // self.ticks_per_day
        if n_days == 0:
            return np.empty((0, prices.shape[1]), dtype=float)

        close_idx = np.arange(
            self.ticks_per_day - 1,
            n_days * self.ticks_per_day,
            self.ticks_per_day,
        )
        return prices[close_idx]

    def _daily_returns_df(self, price_history) -> pd.DataFrame:
        closes = self._daily_closes(price_history)
        if closes.shape[0] <= 1:
            return pd.DataFrame(np.empty((0, closes.shape[1])), columns=ASSET_COLUMNS)

        ret = closes[1:] / closes[:-1] - 1.0
        return pd.DataFrame(ret, columns=ASSET_COLUMNS)

    def _rolling_mean(self, arr: np.ndarray, start: int, end: int) -> float:
        x = arr[start:end]
        if len(x) == 0:
            return 0.0
        return float(np.mean(x))

    def _rolling_std(self, arr: np.ndarray, start: int, end: int) -> float:
        x = arr[start:end]
        if len(x) <= 1:
            return 1e-6
        v = float(np.std(x, ddof=1))
        return max(v, 1e-6)

    def _family_specs(self, lookbacks: tuple[int, ...], weights: tuple[float, ...]) -> list[tuple[int, float]]:
        lbs = [int(x) for x in lookbacks]
        w = np.asarray(weights, dtype=float)
        if len(lbs) == 0:
            return [(self.resid_feature_lookback, 1.0)]
        if len(w) != len(lbs):
            w = np.ones(len(lbs), dtype=float)
        w = np.maximum(w, 0.0)
        wsum = float(np.sum(w))
        if wsum < 1e-12:
            w = np.ones(len(lbs), dtype=float) / len(lbs)
        else:
            w = w / wsum
        return list(zip(lbs, w.tolist()))

    def _train_ridge_family(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
        *,
        target_mode: str,
        lookbacks: tuple[int, ...],
        weights: tuple[float, ...],
    ) -> dict | None:
        family_models = []
        family_kind = self.raw_model_kind if target_mode == "raw" else self.resid_model_kind
        for resid_lookback, model_weight in self._family_specs(lookbacks, weights):
            X_ridge, y_ridge = self._build_dataset(
                daily_ret,
                intraday_arrays=intraday_arrays,
                target_mode=target_mode,
                resid_lookback=resid_lookback,
            )
            if X_ridge is None or len(X_ridge) <= self.fit_min_samples:
                continue
            if family_kind == "huber":
                model = HuberRegressor(
                    alpha=self.huber_alpha,
                    epsilon=self.huber_epsilon,
                    max_iter=self.huber_max_iter,
                )
            else:
                model = Ridge(alpha=self.ridge_alpha)
            model.fit(X_ridge, y_ridge)
            family_models.append({
                "model": model,
                "resid_lookback": resid_lookback,
                "weight": model_weight,
            })

        if not family_models:
            return None

        return {"target_mode": target_mode, "models": family_models}

    def _daily_intraday_arrays(self, price_history: np.ndarray) -> dict[str, np.ndarray]:
        prices = np.asarray(price_history, dtype=float)
        n_ticks = prices.shape[0]
        n_days = n_ticks // self.ticks_per_day
        if n_days == 0:
            zeros = np.empty((0, N_ASSETS), dtype=float)
            return {
                "open_close": zeros,
                "first5": zeros,
                "last5": zeros,
                "range": zeros,
                "intraday_vol": zeros,
                "close_loc": zeros,
                "sector_rel": zeros,
            }

        day_prices = prices[: n_days * self.ticks_per_day].reshape(n_days, self.ticks_per_day, N_ASSETS)
        opens = day_prices[:, 0, :]
        closes = day_prices[:, -1, :]
        highs = np.max(day_prices, axis=1)
        lows = np.min(day_prices, axis=1)
        first5 = day_prices[:, min(4, self.ticks_per_day - 1), :] / opens - 1.0
        last5 = closes / day_prices[:, max(self.ticks_per_day - 5, 0), :] - 1.0
        tick_rets = day_prices[:, 1:, :] / day_prices[:, :-1, :] - 1.0
        intraday_vol = np.std(tick_rets, axis=1, ddof=1)
        open_close = closes / opens - 1.0
        close_loc = (closes - lows) / (highs - lows + 1e-12)

        sector_rel = np.zeros_like(open_close)
        for _, idx in self.sector_to_indices.items():
            sector_rel[:, idx] = open_close[:, idx] - np.mean(open_close[:, idx], axis=1, keepdims=True)

        return {
            "open_close": open_close,
            "first5": first5,
            "last5": last5,
            "range": highs / lows - 1.0,
            "intraday_vol": intraday_vol,
            "close_loc": close_loc,
            "sector_rel": sector_rel,
        }

    def _target_value(
        self,
        R: np.ndarray,
        t: int,
        i: int,
        beta_mkt: float,
        beta_sec: float,
        vol_scale: float,
        target_mode: str | None = None,
    ) -> float:
        mode = self.target_mode if target_mode is None else str(target_mode)
        next_ret = float(R[t + 1, i])
        if mode == "raw":
            return next_ret

        sec_idx = self.sector_to_indices[int(self.sector_id[i])]
        next_sector = float(np.mean(R[t + 1, sec_idx]))
        if mode == "sector_relative":
            return next_ret - next_sector

        peer_idx = self.sector_peer_indices[i]
        next_peer = float(np.mean(R[t + 1, peer_idx])) if len(peer_idx) > 0 else 0.0
        next_market = float(np.mean(R[t + 1]))
        residual = next_ret - beta_mkt * next_market - beta_sec * next_peer

        if mode == "residual":
            return residual
        if mode == "residual_vol_adj":
            return residual / max(vol_scale, self.alpha_vol_floor)

        return next_ret

    def _build_dataset(
        self,
        daily_ret: pd.DataFrame,
        *,
        intraday_arrays: dict[str, np.ndarray] | None = None,
        target_mode: str | None = None,
        resid_lookback: int | None = None,
    ):
        R = daily_ret.values
        n_days, n_assets = R.shape

        if n_days < self.dataset_min_days:
            return None, None

        rows = []
        y = []

        spread_frac = self.spread_bps / 1e4
        borrow_frac = self.borrow_bps_annual / 1e4
        sectors = self.sector_id
        resid_lb = self.resid_feature_lookback if resid_lookback is None else int(resid_lookback)
        tgt_mode = self.target_mode if target_mode is None else str(target_mode)
        if intraday_arrays is None:
            raise ValueError("intraday_arrays are required for dataset construction")

        warmup = max(
            self.dataset_lookback,
            resid_lb,
            max(self.momentum_windows),
            max(self.vol_windows),
            max(self.market_windows),
            max(self.sector_windows),
        )
        for t in range(warmup, n_days - 1):
            market_1 = np.mean(R[t - self.market_windows[0]])
            market_5 = np.mean(R[t - self.market_windows[1]:t])
            market_20 = np.mean(R[t - self.market_windows[2]:t])
            residual_block = self._market_sector_residual_features(R, t, lookback=resid_lb)

            for i in range(n_assets):
                r_i = R[:, i]

                ret_1 = r_i[t - self.momentum_windows[0]]
                ret_5 = self._rolling_mean(r_i, t - self.momentum_windows[1], t)
                ret_10 = self._rolling_mean(r_i, t - self.momentum_windows[2], t)
                ret_20 = self._rolling_mean(r_i, t - self.momentum_windows[3], t)
                ret_60 = self._rolling_mean(r_i, t - self.momentum_windows[4], t)

                vol_5 = self._rolling_std(r_i, t - self.vol_windows[0], t)
                vol_20 = self._rolling_std(r_i, t - self.vol_windows[1], t)
                vol_60 = self._rolling_std(r_i, t - self.vol_windows[2], t)

                sec = sectors[i]
                sec_idx = self.sector_to_indices[int(sec)]

                sec_ret_5 = float(np.mean(R[t - self.sector_windows[0]:t, sec_idx]))
                sec_ret_20 = float(np.mean(R[t - self.sector_windows[1]:t, sec_idx]))
                sec_ret_60 = float(np.mean(R[t - self.sector_windows[2]:t, sec_idx]))

                rel_5 = ret_5 - sec_ret_5
                rel_20 = ret_20 - sec_ret_20
                rel_60 = ret_60 - sec_ret_60

                resid_1, resid_5, resid_20, resid_60, beta_mkt, beta_sec = residual_block[i]
                summary_idx = t
                intraday_oc = float(intraday_arrays["open_close"][summary_idx, i])
                intraday_first5 = float(intraday_arrays["first5"][summary_idx, i])
                intraday_last5 = float(intraday_arrays["last5"][summary_idx, i])
                intraday_range = float(intraday_arrays["range"][summary_idx, i])
                intraday_vol = float(intraday_arrays["intraday_vol"][summary_idx, i])
                close_loc = float(intraday_arrays["close_loc"][summary_idx, i])
                intraday_sector_rel = float(intraday_arrays["sector_rel"][summary_idx, i])
                vol_ratio_5_20 = vol_5 / max(vol_20, self.alpha_vol_floor) - 1.0
                signal_to_cost_20 = abs(ret_20) / (spread_frac[i] + 5.0 * borrow_frac[i] / 252.0 + 1e-8)

                rows.append([
                    ret_1, ret_5, ret_10, ret_20, ret_60,
                    vol_5, vol_20, vol_60,
                    vol_ratio_5_20,
                    rel_5, rel_20, rel_60,
                    resid_1, resid_5, resid_20, resid_60,
                    beta_mkt, beta_sec,
                    intraday_oc, intraday_first5, intraday_last5,
                    intraday_range, intraday_vol, close_loc, intraday_sector_rel,
                    market_1, market_5, market_20,
                    spread_frac[i], borrow_frac[i], signal_to_cost_20, float(sec),
                ])
                y.append(self._target_value(R, t, i, beta_mkt, beta_sec, vol_20, target_mode=tgt_mode))

        return np.asarray(rows, dtype=float), np.asarray(y, dtype=float)

    def _build_current_feature_matrix(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
        resid_lookback: int | None = None,
    ) -> np.ndarray:
        R = daily_ret.values
        n_days, n_assets = R.shape

        spread_frac = self.spread_bps / 1e4
        borrow_frac = self.borrow_bps_annual / 1e4
        sectors = self.sector_id
        resid_lb = self.resid_feature_lookback if resid_lookback is None else int(resid_lookback)

        t = n_days
        market_1 = np.mean(R[t - 1])
        market_5 = np.mean(R[t - 5:t])
        market_20 = np.mean(R[t - 20:t])
        residual_block = self._market_sector_residual_features(R, t, lookback=resid_lb)

        rows = []

        for i in range(n_assets):
            r_i = R[:, i]

            ret_1 = r_i[t - 1]
            ret_5 = self._rolling_mean(r_i, t - 5, t)
            ret_10 = self._rolling_mean(r_i, t - 10, t)
            ret_20 = self._rolling_mean(r_i, t - 20, t)
            ret_60 = self._rolling_mean(r_i, t - 60, t)

            vol_5 = self._rolling_std(r_i, t - 5, t)
            vol_20 = self._rolling_std(r_i, t - 20, t)
            vol_60 = self._rolling_std(r_i, t - 60, t)

            sec = sectors[i]
            sec_idx = self.sector_to_indices[int(sec)]

            sec_ret_5 = float(np.mean(R[t - 5:t, sec_idx]))
            sec_ret_20 = float(np.mean(R[t - 20:t, sec_idx]))
            sec_ret_60 = float(np.mean(R[t - 60:t, sec_idx]))

            rel_5 = ret_5 - sec_ret_5
            rel_20 = ret_20 - sec_ret_20
            rel_60 = ret_60 - sec_ret_60

            resid_1, resid_5, resid_20, resid_60, beta_mkt, beta_sec = residual_block[i]
            summary_idx = t
            intraday_oc = float(intraday_arrays["open_close"][summary_idx, i])
            intraday_first5 = float(intraday_arrays["first5"][summary_idx, i])
            intraday_last5 = float(intraday_arrays["last5"][summary_idx, i])
            intraday_range = float(intraday_arrays["range"][summary_idx, i])
            intraday_vol = float(intraday_arrays["intraday_vol"][summary_idx, i])
            close_loc = float(intraday_arrays["close_loc"][summary_idx, i])
            intraday_sector_rel = float(intraday_arrays["sector_rel"][summary_idx, i])
            vol_ratio_5_20 = vol_5 / max(vol_20, self.alpha_vol_floor) - 1.0
            signal_to_cost_20 = abs(ret_20) / (spread_frac[i] + 5.0 * borrow_frac[i] / 252.0 + 1e-8)

            rows.append([
                ret_1, ret_5, ret_10, ret_20, ret_60,
                vol_5, vol_20, vol_60,
                vol_ratio_5_20,
                rel_5, rel_20, rel_60,
                resid_1, resid_5, resid_20, resid_60,
                beta_mkt, beta_sec,
                intraday_oc, intraday_first5, intraday_last5,
                intraday_range, intraday_vol, close_loc, intraday_sector_rel,
                market_1, market_5, market_20,
                spread_frac[i], borrow_frac[i], signal_to_cost_20, float(sec),
            ])

        return np.asarray(rows, dtype=float)

    def _market_sector_residual_features(
        self,
        R: np.ndarray,
        t: int,
        lookback: int | None = None,
    ) -> np.ndarray:
        n_assets = R.shape[1]
        feat = np.zeros((n_assets, 6), dtype=float)

        if not self.use_residual_features:
            return feat

        resid_lb = self.resid_feature_lookback if lookback is None else int(lookback)
        resid_lb = min(resid_lb, t)
        if resid_lb < max(self.resid_momentum_windows):
            return feat

        hist = R[t - resid_lb:t]
        market = hist.mean(axis=1)
        market_c = market - np.mean(market)

        for i in range(n_assets):
            y = hist[:, i]
            y_c = y - np.mean(y)

            peer_idx = self.sector_peer_indices[i]
            if len(peer_idx) > 0:
                sector_peer = hist[:, peer_idx].mean(axis=1)
                sector_peer_c = sector_peer - np.mean(sector_peer)
            else:
                sector_peer_c = np.zeros(resid_lb, dtype=float)

            X = np.column_stack([market_c, sector_peer_c])
            xtx = X.T @ X
            xtx[np.diag_indices_from(xtx)] += self.resid_feature_ridge

            try:
                beta = np.linalg.solve(xtx, X.T @ y_c)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X, y_c, rcond=None)[0]

            resid = y_c - X @ beta
            feat[i] = np.array([
                resid[-1],
                np.mean(resid[-5:]),
                np.mean(resid[-20:]),
                np.mean(resid[-60:]),
                beta[0],
                beta[1],
            ])

        return feat

    def _rank_vector(self, x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(np.asarray(x, dtype=float)))
        if len(order) <= 1:
            return np.zeros_like(order, dtype=float)
        return order.astype(float) / (len(order) - 1.0) - 0.5

    def _pairwise_rank_agreement(self, signals: list[np.ndarray]) -> float:
        if len(signals) <= 1:
            return 1.0

        corrs = []
        ranked = [self._rank_vector(sig) for sig in signals]

        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                corr = np.corrcoef(ranked[i], ranked[j])[0, 1]
                if np.isfinite(corr):
                    corrs.append(float(corr))

        if not corrs:
            return 1.0

        return float(np.mean(corrs))

    def _rank_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        xr = self._rank_vector(x)
        yr = self._rank_vector(y)
        corr = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(corr)

    def _update_online_histories(self, daily_ret: pd.DataFrame) -> None:
        n_days = len(daily_ret)
        if (
            self.pending_signals is None
            or n_days <= self.last_recorded_return_count
            or n_days == 0
        ):
            self.last_recorded_return_count = n_days
            return

        realized = np.asarray(daily_ret.iloc[-1].values, dtype=float).copy()
        self.realized_return_history.append(realized)
        for key, signal in self.pending_signals.items():
            if signal is None:
                continue
            self.online_signal_history.setdefault(key, []).append(np.asarray(signal, dtype=float).copy())
        self.last_recorded_return_count = n_days

    def _recent_signal_ic_from_history(
        self,
        signal_name: str,
        lookback_days: int,
    ) -> float | None:
        signal_hist = self.online_signal_history.get(signal_name, [])
        n = min(len(signal_hist), len(self.realized_return_history), int(lookback_days))
        if n < self.online_min_ic_obs:
            return None

        ics = []
        for signal, realized in zip(signal_hist[-n:], self.realized_return_history[-n:]):
            ic = self._rank_corr(signal, realized)
            if np.isfinite(ic):
                ics.append(float(ic))
        if len(ics) < self.online_min_ic_obs:
            return None
        return float(np.mean(ics))

    def _recent_ic_scale(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
    ) -> float:
        if not self.use_recent_ic_scaling:
            return 1.0

        del daily_ret, intraday_arrays
        mean_ic = self._recent_signal_ic_from_history("alpha", self.recent_ic_lookback_days)
        if mean_ic is None:
            return 1.0

        denom = self.recent_ic_good - self.recent_ic_bad
        if denom <= 1e-12:
            return 1.0

        z = (mean_ic - self.recent_ic_bad) / denom
        z = float(np.clip(z, 0.0, 1.0))
        return self.recent_ic_scale_floor + (1.0 - self.recent_ic_scale_floor) * z

    def _mean_offdiag(self, corr: np.ndarray) -> float:
        if corr.ndim != 2 or corr.shape[0] <= 1:
            return 0.0
        mask = ~np.eye(corr.shape[0], dtype=bool)
        vals = corr[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else 0.0

    def _ols_beta_scalar(self, y: np.ndarray, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_c = x - np.mean(x)
        denom = float(np.dot(x_c, x_c))
        if denom <= 1e-12:
            return 0.0
        y_c = y - np.mean(y)
        return float(np.dot(y_c, x_c) / denom)

    def _same_sector_residual_corr_mean(self, window: np.ndarray) -> float:
        n_days, n_assets = window.shape
        if n_days <= 2:
            return 0.0

        market = np.mean(window, axis=1)
        residuals = np.zeros_like(window)
        for i in range(n_assets):
            y = window[:, i]
            sec = self.sector_id[i]
            peer_idx = np.where((self.sector_id == sec) & (np.arange(n_assets) != i))[0]
            sector_peer = np.mean(window[:, peer_idx], axis=1) if len(peer_idx) > 0 else np.zeros(n_days, dtype=float)
            beta_market = self._ols_beta_scalar(y, market)
            beta_sector = self._ols_beta_scalar(y, sector_peer)
            residuals[:, i] = y - beta_market * market - beta_sector * sector_peer

        resid_corrs = []
        for _, idx in self.sector_to_indices.items():
            if len(idx) <= 1:
                continue
            corr = np.corrcoef(residuals[:, idx], rowvar=False)
            resid_corrs.append(self._mean_offdiag(corr))
        return float(np.mean(resid_corrs)) if resid_corrs else 0.0

    def _simple_signal_components(self, R: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ret_1 = R[d]
        ret_5 = np.mean(R[d - 4:d + 1], axis=0)
        ret_10 = np.mean(R[d - 9:d + 1], axis=0)
        rel_5 = np.zeros(R.shape[1], dtype=float)
        for _, idx in self.sector_to_indices.items():
            rel_5[idx] = ret_5[idx] - np.mean(ret_5[idx])

        residual_block = self._market_sector_residual_features(R, d + 1, lookback=self.resid_feature_lookback)
        resid_5 = residual_block[:, 1]
        combo_fast = (
            self._rank_vector(ret_1)
            + self._rank_vector(ret_5)
            + self._rank_vector(ret_10)
            + self._rank_vector(rel_5)
            + self._rank_vector(resid_5)
        ) / 5.0
        return rel_5, resid_5, combo_fast

    def _signal_metric_tuple(self, signal: np.ndarray, next_ret: np.ndarray) -> tuple[float, float]:
        ic = self._rank_corr(signal, next_ret)
        ranked = self._rank_vector(signal)
        top_cut = np.quantile(ranked, 0.8)
        bot_cut = np.quantile(ranked, 0.2)
        top = next_ret[ranked >= top_cut]
        bottom = next_ret[ranked <= bot_cut]
        spread = float(np.mean(top) - np.mean(bottom)) if len(top) and len(bottom) else 0.0
        return ic, spread

    def _compute_signal_metric_arrays(self, daily_ret: pd.DataFrame) -> dict[str, np.ndarray]:
        R = daily_ret.values
        n_days = len(R)
        metrics = {
            "rel5_ic": np.full(n_days, np.nan, dtype=float),
            "resid5_ic": np.full(n_days, np.nan, dtype=float),
            "combo_ic": np.full(n_days, np.nan, dtype=float),
            "rel5_spread": np.full(n_days, np.nan, dtype=float),
            "resid5_spread": np.full(n_days, np.nan, dtype=float),
            "combo_spread": np.full(n_days, np.nan, dtype=float),
        }
        warmup = max(self.resid_feature_lookback, 10)
        for d in range(warmup, n_days - 1):
            rel_5, resid_5, combo_fast = self._simple_signal_components(R, d)
            next_ret = R[d + 1]
            rel_ic, rel_spread = self._signal_metric_tuple(rel_5, next_ret)
            resid_ic, resid_spread = self._signal_metric_tuple(resid_5, next_ret)
            combo_ic, combo_spread = self._signal_metric_tuple(combo_fast, next_ret)
            metrics["rel5_ic"][d] = rel_ic
            metrics["resid5_ic"][d] = resid_ic
            metrics["combo_ic"][d] = combo_ic
            metrics["rel5_spread"][d] = rel_spread
            metrics["resid5_spread"][d] = resid_spread
            metrics["combo_spread"][d] = combo_spread
        return metrics

    def _signal_strength_summary_from_arrays(
        self,
        metric_arrays: dict[str, np.ndarray],
        t: int,
    ) -> tuple[float, float, float, float, float, float]:
        end_idx = t - 1
        start_idx = max(0, end_idx - self.regime_signal_lookback_days)
        if end_idx <= start_idx:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        def mean_slice(key: str) -> float:
            vals = metric_arrays[key][start_idx:end_idx]
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else 0.0

        return (
            mean_slice("combo_ic"),
            mean_slice("rel5_ic"),
            mean_slice("resid5_ic"),
            mean_slice("combo_spread"),
            mean_slice("rel5_spread"),
            mean_slice("resid5_spread"),
        )

    def _signal_strength_summary_live(self, daily_ret: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
        R = daily_ret.values
        n_days = len(R)
        warmup = max(self.resid_feature_lookback, 10)
        d_end = n_days - 1
        d_start = max(warmup, d_end - self.regime_signal_lookback_days)
        if d_end <= d_start:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        rel_ics = []
        resid_ics = []
        combo_ics = []
        rel_spreads = []
        resid_spreads = []
        combo_spreads = []

        for d in range(d_start, d_end):
            rel_5, resid_5, combo_fast = self._simple_signal_components(R, d)
            next_ret = R[d + 1]
            rel_ic, rel_spread = self._signal_metric_tuple(rel_5, next_ret)
            resid_ic, resid_spread = self._signal_metric_tuple(resid_5, next_ret)
            combo_ic, combo_spread = self._signal_metric_tuple(combo_fast, next_ret)
            rel_ics.append(rel_ic)
            resid_ics.append(resid_ic)
            combo_ics.append(combo_ic)
            rel_spreads.append(rel_spread)
            resid_spreads.append(resid_spread)
            combo_spreads.append(combo_spread)

        def _safe_mean(vals: list[float]) -> float:
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else 0.0

        return (
            _safe_mean(combo_ics),
            _safe_mean(rel_ics),
            _safe_mean(resid_ics),
            _safe_mean(combo_spreads),
            _safe_mean(rel_spreads),
            _safe_mean(resid_spreads),
        )

    def _regime_feature_vector_live(self, daily_ret: pd.DataFrame) -> np.ndarray | None:
        t = len(daily_ret)
        if t < max(self.regime_lookback_days, self.resid_feature_lookback, 10) + 2:
            return None
        R = daily_ret.values
        eig_share, within_corr, resid_corr, dispersion = self._regime_relationship_features(R, t)
        combo_ic, rel5_ic, resid5_ic, combo_spread, rel5_spread, resid5_spread = (
            self._signal_strength_summary_live(daily_ret)
        )
        feats = np.array(
            [
                eig_share,
                within_corr,
                resid_corr,
                dispersion,
                combo_ic,
                rel5_ic,
                resid5_ic,
                combo_spread,
                rel5_spread,
                resid5_spread,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(feats)):
            return None
        return feats

    def _regime_relationship_features(self, R: np.ndarray, t: int) -> tuple[float, float, float, float]:
        window = R[t - self.regime_lookback_days:t]
        corr = np.corrcoef(window, rowvar=False)
        within_vals = []
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                if self.sector_id[i] == self.sector_id[j] and np.isfinite(corr[i, j]):
                    within_vals.append(corr[i, j])

        cov = np.cov(window, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        eig_share = float(eigvals[-1] / np.sum(eigvals)) if eigvals.size else 0.0
        within_corr = float(np.mean(within_vals)) if within_vals else 0.0
        resid_corr = self._same_sector_residual_corr_mean(window)
        dispersion = float(np.mean(np.std(window, axis=1)))
        return eig_share, within_corr, resid_corr, dispersion

    def _regime_feature_vector_from_arrays(
        self,
        daily_ret: pd.DataFrame,
        metric_arrays: dict[str, np.ndarray],
        t: int,
    ) -> np.ndarray | None:
        if t < max(self.regime_lookback_days, self.resid_feature_lookback, 10) + 2:
            return None
        R = daily_ret.values
        eig_share, within_corr, resid_corr, dispersion = self._regime_relationship_features(R, t)
        combo_ic, rel5_ic, resid5_ic, combo_spread, rel5_spread, resid5_spread = (
            self._signal_strength_summary_from_arrays(metric_arrays, t)
        )
        feats = np.array(
            [
                eig_share,
                within_corr,
                resid_corr,
                dispersion,
                combo_ic,
                rel5_ic,
                resid5_ic,
                combo_spread,
                rel5_spread,
                resid5_spread,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(feats)):
            return None
        return feats

    def _future_regime_quality_from_arrays(
        self,
        metric_arrays: dict[str, np.ndarray],
        t: int,
    ) -> float | None:
        start_idx = t - 1
        end_idx = min(len(metric_arrays["combo_ic"]), t + self.regime_horizon_days - 1)
        if end_idx <= start_idx:
            return None

        vals = []
        for key in ("combo_ic", "rel5_ic", "resid5_ic"):
            arr = metric_arrays[key][start_idx:end_idx]
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(float(np.mean(arr)))
        for key in ("combo_spread", "rel5_spread", "resid5_spread"):
            arr = metric_arrays[key][start_idx:end_idx]
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(float(np.mean(arr)))
        if not vals:
            return None
        return float(np.mean(vals))

    def _train_regime_quality_model(self, daily_ret: pd.DataFrame) -> None:
        self.regime_model_fitted = False
        if not self.use_regime_quality_model or len(daily_ret) < self.min_days:
            return

        metric_arrays = self._compute_signal_metric_arrays(daily_ret)
        X_rows = []
        y_vals = []
        for t in range(
            max(self.min_days, self.regime_lookback_days, self.resid_feature_lookback, 10) + 2,
            len(daily_ret) - self.regime_horizon_days + 1,
        ):
            feats = self._regime_feature_vector_from_arrays(daily_ret, metric_arrays, t)
            label = self._future_regime_quality_from_arrays(metric_arrays, t)
            if feats is None or label is None or not np.isfinite(label):
                continue
            X_rows.append(feats)
            y_vals.append(label)

        if len(X_rows) < 50:
            return

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        self.regime_feature_mean = np.mean(X, axis=0)
        self.regime_feature_std = np.maximum(np.std(X, axis=0), 1e-6)
        Xz = (X - self.regime_feature_mean) / self.regime_feature_std

        self.regime_model = Ridge(alpha=self.regime_model_alpha)
        self.regime_model.fit(Xz, y)
        pred = self.regime_model.predict(Xz)
        self.regime_pred_low, self.regime_pred_high = np.quantile(
            pred,
            [self.regime_pred_bad_quantile, self.regime_pred_good_quantile],
        ).tolist()
        self.regime_model_fitted = True

    def _regime_quality_scale(self, daily_ret: pd.DataFrame) -> tuple[float, float, float]:
        if (
            not self.use_regime_quality_model
            or not self.regime_model_fitted
            or self.regime_feature_mean is None
            or self.regime_feature_std is None
        ):
            return 1.0, 1.0, 0.0

        feats = self._regime_feature_vector_live(daily_ret)
        if feats is None:
            return 1.0, 1.0, 0.0

        x = (feats - self.regime_feature_mean) / self.regime_feature_std
        pred = float(self.regime_model.predict(x.reshape(1, -1))[0])
        low = float(self.regime_pred_low) if self.regime_pred_low is not None else pred
        high = float(self.regime_pred_high) if self.regime_pred_high is not None else pred
        if high - low <= 1e-12:
            z = 0.5
        else:
            z = float(np.clip((pred - low) / (high - low), 0.0, 1.0))

        gross_scale = self.regime_floor + (1.0 - self.regime_floor) * z if self.regime_scale_gross else 1.0
        beta_scale = self.regime_beta_floor + (1.0 - self.regime_beta_floor) * z if self.regime_scale_beta else 1.0
        return gross_scale, beta_scale, pred

    def _normalize_l1(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        denom = float(np.sum(np.abs(x)))
        if denom < 1e-12 or not np.isfinite(denom):
            return np.zeros_like(x)
        return x / denom

    def _apply_rebalance(self, target: np.ndarray) -> np.ndarray:
        target = project_to_gross_limit(target)
        if self.prev_weights is None:
            w = target
        else:
            target = target.copy()
            spread_rank = self._rank_vector(self.spread_bps / 1e4) + 0.5
            raw_mult = (
                self.spread_turnover_min_mult
                + (self.spread_turnover_max_mult - self.spread_turnover_min_mult) * spread_rank
            )
            band_mult = 1.0 + self.spread_turnover_scale * (raw_mult - 1.0)
            band = self.turnover_band * band_mult
            small_moves = np.abs(target - self.prev_weights) < band
            target[small_moves] = self.prev_weights[small_moves]
            target = project_to_gross_limit(target)
            w = (1.0 - self.rebalance_rate) * self.prev_weights + self.rebalance_rate * target

        w = project_to_gross_limit(w)
        return w

    def _robust_cs_score(self, x: np.ndarray) -> np.ndarray:
        """
        Cross-sectional robust score:
        - de-mean
        - scale by MAD-like percentile spread
        - clip extremes
        """
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
    
        q10, q90 = np.percentile(x, self.cs_percentiles)
        scale = max(q90 - q10, self.cs_scale_floor)

        z = x / scale
        z = np.clip(z, -self.cs_clip, self.cs_clip)
        return z

    def _covariance_aware_overlay(self, alpha_filtered: np.ndarray, cov_lw: np.ndarray) -> np.ndarray:
        n = len(alpha_filtered)
        if np.sum(np.abs(alpha_filtered)) < 1e-12:
            return np.zeros_like(alpha_filtered)

        diag_scale = float(np.trace(cov_lw)) / max(n, 1)
        reg = cov_lw + (
            self.cov_alpha_ridge * diag_scale + self.cov_alpha_floor
        ) * np.eye(n, dtype=float)

        try:
            raw = np.linalg.solve(reg, alpha_filtered)
        except np.linalg.LinAlgError:
            raw = alpha_filtered / (np.diag(reg) + self.cov_alpha_floor)

        raw = raw - np.mean(raw)
        raw = self._robust_cs_score(raw)
        return self._normalize_l1(raw)

    def _predict_family_signal(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
        family_name: str,
    ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        family = self.model_families.get(family_name)
        if family is None:
            return None, []

        family_pred = np.zeros(N_ASSETS, dtype=float)
        family_signals = []
        total_weight = 0.0
        feature_cache = {}

        for spec in family["models"]:
            resid_lb = int(spec["resid_lookback"])
            if resid_lb not in feature_cache:
                feature_cache[resid_lb] = self._build_current_feature_matrix(
                    daily_ret,
                    intraday_arrays,
                    resid_lookback=resid_lb,
                )
            pred_component = spec["model"].predict(feature_cache[resid_lb])
            family_pred += float(spec["weight"]) * pred_component
            total_weight += float(spec["weight"])
            family_signals.append(pred_component)

        if total_weight <= 0.0:
            return None, []

        return family_pred / total_weight, family_signals

    def _model_signal(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, list[np.ndarray], dict]:
        if not self.ridge_fitted:
            return None, [], {}

        family_preds = {}
        agreement_parts = []
        static_weights = {}

        if self.ridge_fitted:
            raw_pred, raw_parts = self._predict_family_signal(daily_ret, intraday_arrays, "raw")
            resid_pred, resid_parts = self._predict_family_signal(daily_ret, intraday_arrays, "resid")
            if raw_pred is not None and self.raw_family_weight > 0.0:
                family_preds["raw"] = raw_pred
                static_weights["raw"] = self.raw_family_weight
                agreement_parts.extend(raw_parts)
                agreement_parts.append(raw_pred)
            if resid_pred is not None and self.resid_family_weight > 0.0:
                family_preds["resid"] = resid_pred
                static_weights["resid"] = self.resid_family_weight
                agreement_parts.extend(resid_parts)
                agreement_parts.append(resid_pred)

        if not family_preds:
            return None, [], {}

        total = float(sum(static_weights.values()))
        family_weights = {k: v / total for k, v in static_weights.items()} if total > 0.0 else {}

        model_pred = np.zeros(N_ASSETS, dtype=float)
        for family_name, pred in family_preds.items():
            model_pred += float(family_weights.get(family_name, 0.0)) * pred

        ridge_signal = model_pred.copy()

        if np.sum(np.abs(model_pred)) < 1e-12:
            return None, agreement_parts, {}
        model_debug = {
            "raw_family_signal": family_preds.get("raw"),
            "resid_family_signal": family_preds.get("resid"),
            "raw_family_weight": family_weights.get("raw", 0.0),
            "resid_family_weight": family_weights.get("resid", 0.0),
            "ridge_signal": ridge_signal,
            "model_signal": model_pred.copy(),
        }
        return model_pred, agreement_parts, model_debug

    def _build_alpha(
        self,
        daily_ret: pd.DataFrame,
        intraday_arrays: dict[str, np.ndarray],
    ) -> np.ndarray:
        pred, agreement_parts, debug = self._model_signal(daily_ret, intraday_arrays)
        if pred is None:
            self.alpha_agreement_scale = 1.0
            self.last_debug = {"alpha": np.zeros(N_ASSETS, dtype=float)}
            return np.zeros(N_ASSETS, dtype=float)

        if len(agreement_parts) > 1:
            agreement = max(0.0, self._pairwise_rank_agreement(agreement_parts))
            self.alpha_agreement_scale = (
                self.agreement_scale_floor
                + (1.0 - self.agreement_scale_floor) * agreement
            )
        else:
            self.alpha_agreement_scale = 1.0

        alpha = pred - np.mean(pred)
        vol = daily_ret.std(axis=0).values + self.alpha_vol_floor
        alpha = alpha / vol

        alpha = self.alpha_shrink * alpha
        alpha = np.tanh(self.alpha_tanh_scale * alpha)
        debug["alpha"] = alpha.copy()
        self.last_debug = debug
        return alpha

    def _filtered_overlay_signal(self, alpha: np.ndarray) -> np.ndarray:
        global_filtered = np.zeros_like(alpha)

        global_top_k = max(1, int(self.global_top_k))
        global_bottom_k = max(1, int(self.global_bottom_k))
        sorted_idx = np.argsort(alpha)
        global_filtered[sorted_idx[-global_top_k:]] = alpha[sorted_idx[-global_top_k:]]
        global_filtered[sorted_idx[:global_bottom_k]] = alpha[sorted_idx[:global_bottom_k]]
        combined = global_filtered

        if self.borrow_short_damp > 0.0:
            borrow_rank = self._rank_vector(self.borrow_bps_annual / 1e4) + 0.5
            damp = 1.0 / (1.0 + self.borrow_short_damp * borrow_rank)
            short_idx = combined < 0.0
            combined[short_idx] = combined[short_idx] * damp[short_idx]

        return combined

    def get_weights(self, price_history, meta: PublicMeta, day: int) -> np.ndarray:
        daily_ret = self._daily_returns_df(price_history)
        intraday_arrays = self._daily_intraday_arrays(price_history)
        self._update_online_histories(daily_ret)
        n_days = len(daily_ret)
        beta_eff = 0.0
        gross_scale = 1.0
        regime_gross_scale = 1.0
        regime_beta_scale = 1.0
        regime_pred = 0.0
        alpha_overlay = np.zeros(N_ASSETS, dtype=float)
        base_target = np.zeros(N_ASSETS, dtype=float)

        if n_days < self.min_days:
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS
            self.prev_weights = w.copy()
            self.last_debug = {
                "alpha": np.zeros(N_ASSETS, dtype=float),
                "alpha_overlay": np.zeros(N_ASSETS, dtype=float),
                "target": w.copy(),
                "weights": w.copy(),
                "beta_eff": 0.0,
                "gross_scale": 1.0,
                "day": day,
            }
            return w

        cov_lb = min(self.lookback_cov, n_days)
        train_ret = daily_ret.iloc[-cov_lb:]
        _, cov_lw = fit_covariances(train_ret)
        w_rp = risk_parity_weights(cov_lw)

        alpha = self._build_alpha(daily_ret, intraday_arrays)

        if np.sum(np.abs(alpha)) < 1e-12:
            base_target = w_rp.copy()
            w = self._apply_rebalance(w_rp)
        else:
            alpha_filtered = self._filtered_overlay_signal(alpha)
            alpha_overlay = self._normalize_l1(alpha_filtered)

            if self.cov_alpha_mix > 0.0:
                cov_overlay = self._covariance_aware_overlay(alpha_filtered, cov_lw)
                alpha_overlay = self._normalize_l1(
                    (1.0 - self.cov_alpha_mix) * alpha_overlay
                    + self.cov_alpha_mix * cov_overlay
                )

            market_ret = daily_ret.mean(axis=1)
            trend_strength = np.mean(market_ret[-self.trend_window:])

            if abs(trend_strength) < self.trend_threshold:
                beta_eff = self.beta * self.trend_beta_scale
            else:
                beta_eff = self.beta

            history_scale = min(1.0, self.train_days_available / self.history_scale_days)
            beta_eff = beta_eff * history_scale * self.alpha_agreement_scale

            regime_gross_scale, regime_beta_scale, regime_pred = self._regime_quality_scale(daily_ret)
            beta_eff = beta_eff * regime_beta_scale
            target = (1.0 - beta_eff) * w_rp + beta_eff * alpha_overlay

            gross_scale = 1.0
            if self.use_confidence_gross_scaling:
                gross_scale = self.gross_scale_floor + (
                    1.0 - self.gross_scale_floor
                ) * self.alpha_agreement_scale
            if self.use_recent_ic_scaling:
                gross_scale = gross_scale * self._recent_ic_scale(daily_ret, intraday_arrays)
            gross_scale = gross_scale * regime_gross_scale

            base_target = gross_scale * target
            w = self._apply_rebalance(base_target)

        w = project_to_gross_limit(w)

        if not np.all(np.isfinite(w)):
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS

        debug = dict(self.last_debug) if isinstance(self.last_debug, dict) else {}
        debug.update({
            "alpha": np.asarray(alpha, dtype=float).copy(),
            "alpha_overlay": np.asarray(alpha_overlay, dtype=float).copy(),
            "target": np.asarray(base_target, dtype=float).copy(),
            "weights": np.asarray(w, dtype=float).copy(),
            "beta_eff": float(beta_eff),
            "gross_scale": float(gross_scale),
            "regime_gross_scale": float(regime_gross_scale),
            "regime_beta_scale": float(regime_beta_scale),
            "regime_pred": float(regime_pred),
            "day": int(day),
        })
        self.last_debug = debug
        self.pending_signals = {
            "alpha": debug.get("alpha"),
            "model": debug.get("model_signal"),
            "raw": debug.get("raw_family_signal"),
            "resid": debug.get("resid_family_signal"),
            "ridge": debug.get("ridge_signal"),
        }
        self.prev_weights = w.copy()
        return w


def create_strategy() -> StrategyBase:
    """Entry point called by validate.py."""
    return MyStrategy()
