from __future__ import annotations

"""Participant submission scaffold for the portfolio optimization case.

Converted from the provided notebook into the official submission template.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

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
    """
    Train-once ML alpha:
    - build training dataset from train_prices in fit()
    - fit one Ridge model
    - in get_weights(), only predict current cross-section
    - blend with RP base
    """

    def __init__(self):
        self.ticks_per_day = TICKS_PER_DAY
        self.sector_id = None
        self.spread_bps = None
        self.borrow_bps_annual = None

        self.lookback_cov = 84
        self.min_days = 120

        self.beta = 0.9
        self.rebalance_rate = 0.1
        self.k_frac = 0.08
        self.dataset_min_days = 60
        self.dataset_lookback = 60
        self.fit_min_samples = 100

        self.momentum_windows = (1, 5, 10, 20, 60)
        self.vol_windows = (5, 20, 60)
        self.market_windows = (1, 5, 20)
        self.sector_windows = (5, 20, 60)

        self.pred_ridge_weight = 0.45
        self.pred_hgb_weight = 0.55
        self.alpha_vol_floor = 1e-6
        self.alpha_shrink = 0.7
        self.alpha_tanh_scale = 1.5

        self.cs_percentiles = (10.0, 90.0)
        self.cs_scale_floor = 1e-6
        self.cs_clip = 3.0

        self.trend_window = 20
        self.trend_threshold = 0.001
        self.trend_beta_scale = 0.8
        self.history_scale_days = 756.0

        self.ridge_alpha = 1.0
        self.hgb_max_depth = 3
        self.hgb_learning_rate = 0.08
        self.hgb_max_iter = 100
        self.hgb_l2_regularization = 1.0

        self.ridge_model = Ridge(alpha=self.ridge_alpha)

        self.hgb_model = HistGradientBoostingRegressor(
            max_depth=self.hgb_max_depth,
            learning_rate=self.hgb_learning_rate,
            max_iter=self.hgb_max_iter,
            l2_regularization=self.hgb_l2_regularization,
            random_state=42
        )

        self.ridge_fitted = False
        self.hgb_fitted = False

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.ticks_per_day = int(kwargs.get("ticks_per_day", TICKS_PER_DAY))
        self.sector_id = np.asarray(meta.sector_id, dtype=int)
        self.spread_bps = np.asarray(meta.spread_bps, dtype=float)
        self.borrow_bps_annual = np.asarray(meta.borrow_bps_annual, dtype=float)
        self.prev_weights = None
    
        # ADD THIS: how many daily observations were available in training
        self.train_days_available = len(self._daily_returns_df(train_prices))
    
        daily_ret = self._daily_returns_df(train_prices)
        X, y = self._build_dataset(daily_ret)
        
        if X is not None and len(X) > self.fit_min_samples:
            self.ridge_model.fit(X, y)
            self.ridge_fitted = True
        
            self.hgb_model.fit(X, y)
            self.hgb_fitted = True
        else:
            self.ridge_fitted = False
            self.hgb_fitted = False

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

    def _build_dataset(self, daily_ret: pd.DataFrame):
        R = daily_ret.values
        n_days, n_assets = R.shape

        if n_days < self.dataset_min_days:
            return None, None

        rows = []
        y = []

        spread_frac = self.spread_bps / 1e4
        borrow_frac = self.borrow_bps_annual / 1e4
        sectors = self.sector_id

        warmup = self.dataset_lookback
        for t in range(warmup, n_days - 1):
            market_1 = np.mean(R[t - self.market_windows[0]])
            market_5 = np.mean(R[t - self.market_windows[1]:t])
            market_20 = np.mean(R[t - self.market_windows[2]:t])

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
                sec_idx = np.where(sectors == sec)[0]

                sec_ret_5 = float(np.mean(R[t - self.sector_windows[0]:t, sec_idx]))
                sec_ret_20 = float(np.mean(R[t - self.sector_windows[1]:t, sec_idx]))
                sec_ret_60 = float(np.mean(R[t - self.sector_windows[2]:t, sec_idx]))

                rel_5 = ret_5 - sec_ret_5
                rel_20 = ret_20 - sec_ret_20
                rel_60 = ret_60 - sec_ret_60

                rows.append([
                    ret_1, ret_5, ret_10, ret_20, ret_60,
                    vol_5, vol_20, vol_60,
                    rel_5, rel_20, rel_60,
                    market_1, market_5, market_20,
                    spread_frac[i], borrow_frac[i], float(sec),
                ])
                y.append(R[t + 1, i])

        return np.asarray(rows, dtype=float), np.asarray(y, dtype=float)

    def _build_current_feature_matrix(self, daily_ret: pd.DataFrame) -> np.ndarray:
        R = daily_ret.values
        n_days, n_assets = R.shape

        spread_frac = self.spread_bps / 1e4
        borrow_frac = self.borrow_bps_annual / 1e4
        sectors = self.sector_id

        t = n_days
        market_1 = np.mean(R[t - 1])
        market_5 = np.mean(R[t - 5:t])
        market_20 = np.mean(R[t - 20:t])

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
            sec_idx = np.where(sectors == sec)[0]

            sec_ret_5 = float(np.mean(R[t - 5:t, sec_idx]))
            sec_ret_20 = float(np.mean(R[t - 20:t, sec_idx]))
            sec_ret_60 = float(np.mean(R[t - 60:t, sec_idx]))

            rel_5 = ret_5 - sec_ret_5
            rel_20 = ret_20 - sec_ret_20
            rel_60 = ret_60 - sec_ret_60

            rows.append([
                ret_1, ret_5, ret_10, ret_20, ret_60,
                vol_5, vol_20, vol_60,
                rel_5, rel_20, rel_60,
                market_1, market_5, market_20,
                spread_frac[i], borrow_frac[i], float(sec),
            ])

        return np.asarray(rows, dtype=float)

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
    
    def _build_alpha(self, daily_ret: pd.DataFrame) -> np.ndarray:
        X_now = self._build_current_feature_matrix(daily_ret)
    
        preds = []

        if self.ridge_fitted:
            pred_ridge = self.ridge_model.predict(X_now)
            preds.append(self.pred_ridge_weight * pred_ridge)

        if self.hgb_fitted:
            pred_hgb = self.hgb_model.predict(X_now)
            preds.append(self.pred_hgb_weight * pred_hgb)
    
        if not preds:
            return np.zeros(N_ASSETS, dtype=float)
    
        pred = np.sum(preds, axis=0)
    
        # cross-sectional de-mean
        alpha = pred - np.mean(pred)
    
        # SINGLE clean regularization (not repeated)
        vol = daily_ret.std(axis=0).values + self.alpha_vol_floor
        alpha = alpha / vol

        # LIGHT shrink (keep dispersion)
        alpha = self.alpha_shrink * alpha

        # VERY mild clipping (do NOT kill tails)
        alpha = np.tanh(self.alpha_tanh_scale * alpha)
    
        return alpha
    def get_weights(self, price_history, meta: PublicMeta, day: int) -> np.ndarray:
        daily_ret = self._daily_returns_df(price_history)
        n_days = len(daily_ret)
    
        if n_days < self.min_days:
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS
            self.prev_weights = w.copy()
            return w
    
        cov_lb = min(self.lookback_cov, n_days)
        train_ret = daily_ret.iloc[-cov_lb:]
        _, cov_lw = fit_covariances(train_ret)
        w_rp = risk_parity_weights(cov_lw)
    
        alpha = self._build_alpha(daily_ret)
    
        if np.sum(np.abs(alpha)) < 1e-12:
            target = w_rp
        else:
            n = len(alpha)
            k = max(1, int(self.k_frac * n))
    
            sorted_idx = np.argsort(alpha)
            long_idx = sorted_idx[-k:]
            short_idx = sorted_idx[:k]
    
            alpha_filtered = np.zeros_like(alpha)
            alpha_filtered[long_idx] = alpha[long_idx]
            alpha_filtered[short_idx] = alpha[short_idx]
    
            alpha_overlay = alpha_filtered / (np.sum(np.abs(alpha_filtered)) + 1e-12)
    
            market_ret = daily_ret.mean(axis=1)
            trend_strength = np.mean(market_ret[-self.trend_window:])

            if abs(trend_strength) < self.trend_threshold:
                beta_eff = self.beta * self.trend_beta_scale
            else:
                beta_eff = self.beta

            history_scale = min(1.0, self.train_days_available / self.history_scale_days)
            beta_eff = beta_eff * history_scale
    
            target = (1.0 - beta_eff) * w_rp + beta_eff * alpha_overlay
    
        target = project_to_gross_limit(target)
    
        if self.prev_weights is None:
            w = target
        else:
            w = (1.0 - self.rebalance_rate) * self.prev_weights + self.rebalance_rate * target
    
        w = project_to_gross_limit(w)
    
        if not np.all(np.isfinite(w)):
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS
    
        self.prev_weights = w.copy()
        return w


def create_strategy() -> StrategyBase:
    """Entry point called by validate.py."""
    return MyStrategy()
