from __future__ import annotations

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
    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
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
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross):
        return np.zeros_like(w)
    if gross > 1.0:
        w /= gross
    return w


def fit_covariances(train_ret: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(train_ret.values).covariance_


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
        w_new = np.maximum(w * ((port_var / n) / np.maximum(rc, eps)), eps)
        w_new /= w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < tol:
            return w_new
        w = w_new
    return w


class MyStrategy(StrategyBase):
    def __init__(self):
        self.ticks_per_day = TICKS_PER_DAY
        self.sector_id = None
        self.spread_bps = None
        self.borrow_bps_annual = None
        self.sector_to_indices = None
        self.sector_peer_indices = None

        self.lookback_cov = 84
        self.min_days = 120
        self.dataset_min_days = 60
        self.dataset_lookback = 60
        self.fit_min_samples = 100

        self.beta = 1.12
        self.rebalance_rate = 0.14
        self.turnover_band = 0.011
        self.alpha_vol_floor = 1e-6
        self.alpha_shrink = 0.7
        self.alpha_tanh_scale = 1.5
        self.resid_feature_lookback = 60
        self.resid_feature_ridge = 1e-5
        self.agreement_scale_floor = 0.50
        self.gross_scale_floor = 0.80
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

        self.raw_family_weight = 0.4
        self.resid_family_weight = 0.6
        self.ridge_alpha = 1.0
        self.huber_alpha = 1e-4
        self.huber_epsilon = 1.35
        self.huber_max_iter = 1000
        self.family_specs = {
            "raw": ((40, 0.55), (60, 0.30), (80, 0.15)),
            "resid": ((40, 0.55), (60, 0.30), (80, 0.15)),
        }

        self.spread_turnover_scale = 0.5
        self.spread_turnover_min_mult = 0.75
        self.spread_turnover_max_mult = 1.25

        self.model_families = {}
        self.ridge_fitted = False
        self.alpha_agreement_scale = 1.0
        self.alpha_signal_history = []
        self.realized_return_history = []
        self.pending_alpha = None
        self.last_recorded_return_count = 0
        self.online_min_ic_obs = 5

        self.regime_lookback_days = 63
        self.regime_signal_lookback_days = 20
        self.regime_horizon_days = 21
        self.regime_floor = 0.70
        self.regime_beta_floor = 0.75
        self.regime_model = Ridge(alpha=2.0)
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
        self.sector_to_indices = {int(sec): np.where(self.sector_id == sec)[0] for sec in unique_sectors}
        self.sector_peer_indices = [
            self.sector_to_indices[int(sec)][self.sector_to_indices[int(sec)] != i]
            for i, sec in enumerate(self.sector_id)
        ]

        self.prev_weights = None
        self.alpha_signal_history = []
        self.realized_return_history = []
        self.pending_alpha = None
        self.last_recorded_return_count = 0
        self.regime_model_fitted = False
        self.regime_feature_mean = None
        self.regime_feature_std = None
        self.regime_pred_low = None
        self.regime_pred_high = None

        daily_ret = self._daily_returns_df(train_prices)
        intraday_arrays = self._daily_intraday_arrays(train_prices)
        self.train_days_available = len(daily_ret)
        self.last_recorded_return_count = len(daily_ret)

        self.model_families = {}
        for family_name, target_mode in (("raw", "raw"), ("resid", "residual_vol_adj")):
            family = self._train_family(daily_ret, intraday_arrays, family_name, target_mode)
            if family is not None:
                self.model_families[family_name] = family
        self.ridge_fitted = bool(self.model_families)
        self._train_regime_quality_model(daily_ret)

    def _daily_closes(self, price_history) -> np.ndarray:
        prices = np.asarray(price_history, dtype=float)
        n_days = prices.shape[0] // self.ticks_per_day
        if n_days == 0:
            return np.empty((0, prices.shape[1]), dtype=float)
        close_idx = np.arange(self.ticks_per_day - 1, n_days * self.ticks_per_day, self.ticks_per_day)
        return prices[close_idx]

    def _daily_returns_df(self, price_history) -> pd.DataFrame:
        closes = self._daily_closes(price_history)
        if closes.shape[0] <= 1:
            return pd.DataFrame(np.empty((0, closes.shape[1])), columns=ASSET_COLUMNS)
        return pd.DataFrame(closes[1:] / closes[:-1] - 1.0, columns=ASSET_COLUMNS)

    def _rolling_mean(self, arr, start, end) -> float:
        x = arr[start:end]
        return 0.0 if len(x) == 0 else float(np.mean(x))

    def _rolling_std(self, arr, start, end) -> float:
        x = arr[start:end]
        if len(x) <= 1:
            return 1e-6
        return max(float(np.std(x, ddof=1)), 1e-6)

    def _daily_intraday_arrays(self, price_history) -> dict[str, np.ndarray]:
        prices = np.asarray(price_history, dtype=float)
        n_days = prices.shape[0] // self.ticks_per_day
        if n_days == 0:
            zeros = np.empty((0, N_ASSETS), dtype=float)
            return {k: zeros for k in ("open_close", "first5", "last5", "range", "intraday_vol", "close_loc", "sector_rel")}

        day_prices = prices[: n_days * self.ticks_per_day].reshape(n_days, self.ticks_per_day, N_ASSETS)
        opens = day_prices[:, 0, :]
        closes = day_prices[:, -1, :]
        highs = np.max(day_prices, axis=1)
        lows = np.min(day_prices, axis=1)
        tick_rets = day_prices[:, 1:, :] / day_prices[:, :-1, :] - 1.0
        open_close = closes / opens - 1.0
        sector_rel = np.zeros_like(open_close)
        for idx in self.sector_to_indices.values():
            sector_rel[:, idx] = open_close[:, idx] - np.mean(open_close[:, idx], axis=1, keepdims=True)

        return {
            "open_close": open_close,
            "first5": day_prices[:, min(4, self.ticks_per_day - 1), :] / opens - 1.0,
            "last5": closes / day_prices[:, max(self.ticks_per_day - 5, 0), :] - 1.0,
            "range": highs / lows - 1.0,
            "intraday_vol": np.std(tick_rets, axis=1, ddof=1),
            "close_loc": (closes-lows) / (highs - lows + 1e-12),
            "sector_rel": sector_rel,
        }

    def _market_sector_residual_features(self, R, t, lookback=None) -> np.ndarray:
        resid_lb = min(self.resid_feature_lookback if lookback is None else int(lookback), t)
        feat = np.zeros((R.shape[1], 6), dtype=float)
        if resid_lb < 60:
            return feat

        hist = R[t - resid_lb:t]
        market_c = hist.mean(axis=1) - np.mean(hist.mean(axis=1))
        for i in range(R.shape[1]):
            y_c = hist[:, i] - np.mean(hist[:, i])
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
            feat[i] = [resid[-1], np.mean(resid[-5:]), np.mean(resid[-20:]), np.mean(resid[-60:]), beta[0], beta[1]]
        return feat

    def _feature_matrix(self, daily_ret, intraday_arrays, resid_lookback, target_mode=None):
        R = daily_ret.values
        n_days = R.shape[0]
        n_assets = R.shape[1]
        if target_mode is not None and n_days < self.dataset_min_days:
            return None, None

        spread_frac = self.spread_bps / 1e4
        borrow_frac = self.borrow_bps_annual / 1e4
        rows = []
        y = []
        t_range = range(max(self.dataset_lookback, int(resid_lookback), 60), n_days - 1) if target_mode else [n_days]

        for t in t_range:
            market_1 = float(np.mean(R[t - 1]))
            market_5 = float(np.mean(R[t - 5:t]))
            market_20 = float(np.mean(R[t - 20:t]))
            residual_block = self._market_sector_residual_features(R, t, lookback=resid_lookback)

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

                sec_idx = self.sector_to_indices[int(self.sector_id[i])]
                rel_5 = ret_5 - float(np.mean(R[t - 5:t, sec_idx]))
                rel_20 = ret_20 - float(np.mean(R[t - 20:t, sec_idx]))
                rel_60 = ret_60 - float(np.mean(R[t - 60:t, sec_idx]))

                resid_1 = residual_block[i, 0]
                resid_5 = residual_block[i, 1]
                resid_20 = residual_block[i, 2]
                resid_60 = residual_block[i, 3]
                beta_mkt = residual_block[i, 4]
                beta_sec = residual_block[i, 5]
                row = [
                    ret_1, ret_5, ret_10, ret_20, ret_60,
                    vol_5, vol_20, vol_60, vol_5 / max(vol_20, self.alpha_vol_floor) - 1.0,
                    rel_5, rel_20, rel_60,
                    resid_1, resid_5, resid_20, resid_60,
                    beta_mkt, beta_sec,
                    float(intraday_arrays["open_close"][t, i]),
                    float(intraday_arrays["first5"][t, i]),
                    float(intraday_arrays["last5"][t, i]),
                    float(intraday_arrays["range"][t, i]),
                    float(intraday_arrays["intraday_vol"][t, i]),
                    float(intraday_arrays["close_loc"][t, i]),
                    float(intraday_arrays["sector_rel"][t, i]),
                    market_1, market_5, market_20,
                    spread_frac[i], borrow_frac[i],
                    abs(ret_20) / (spread_frac[i] + 5.0 * borrow_frac[i] / 252.0 + 1e-8),
                    float(self.sector_id[i]),
                ]
                rows.append(row)

                if target_mode is None:
                    continue
                next_ret = float(R[t + 1, i])
                if target_mode == "raw":
                    y.append(next_ret)
                    continue
                peer_idx = self.sector_peer_indices[i]
                next_peer = float(np.mean(R[t + 1, peer_idx])) if len(peer_idx) > 0 else 0.0
                next_market = float(np.mean(R[t + 1]))
                residual = next_ret - beta_mkt * next_market - beta_sec * next_peer
                y.append(residual / max(vol_20, self.alpha_vol_floor))

        X = np.asarray(rows, dtype=float)
        return X if target_mode is None else (X, np.asarray(y, dtype=float))

    def _train_family(self, daily_ret, intraday_arrays, family_name, target_mode):
        family_models = []
        for resid_lookback, weight in self.family_specs[family_name]:
            dataset = self._feature_matrix(
                daily_ret,
                intraday_arrays,
                resid_lookback,
                target_mode=target_mode,
            )
            X = dataset[0]
            y = dataset[1]
            if X is None or len(X) <= self.fit_min_samples:
                continue
            model = Ridge(alpha=self.ridge_alpha) if family_name == "raw" else HuberRegressor(
                alpha=self.huber_alpha,
                epsilon=self.huber_epsilon,
                max_iter=self.huber_max_iter,
            )
            model.fit(X, y)
            family_models.append({"model": model, "resid_lookback": resid_lookback, "weight": weight})
        return None if not family_models else {"models": family_models}

    def _predict_family_signal(self, daily_ret, intraday_arrays, family_name):
        family = self.model_families.get(family_name)
        if family is None:
            return None, []

        family_pred = np.zeros(N_ASSETS, dtype=float)
        family_parts = []
        feature_cache = {}
        total_weight = 0.0
        for spec in family["models"]:
            resid_lb = int(spec["resid_lookback"])
            if resid_lb not in feature_cache:
                feature_cache[resid_lb] = self._feature_matrix(daily_ret, intraday_arrays, resid_lb)
            pred = spec["model"].predict(feature_cache[resid_lb])
            family_pred += float(spec["weight"]) * pred
            total_weight += float(spec["weight"])
            family_parts.append(pred)
        return (None, []) if total_weight <= 0.0 else (family_pred / total_weight, family_parts)

    def _rank_vector(self, x) -> np.ndarray:
        order = np.argsort(np.argsort(np.asarray(x, dtype=float)))
        return np.zeros_like(order, dtype=float) if len(order) <= 1 else order.astype(float) / (len(order) - 1.0) - 0.5

    def _rank_corr(self, x, y) -> float:
        corr = np.corrcoef(self._rank_vector(x), self._rank_vector(y))[0, 1]
        return 0.0 if not np.isfinite(corr) else float(corr)

    def _pairwise_rank_agreement(self, signals) -> float:
        if len(signals) <= 1:
            return 1.0
        ranked = [self._rank_vector(sig) for sig in signals]
        corrs = [
            float(np.corrcoef(ranked[i], ranked[j])[0, 1])
            for i in range(len(ranked))
            for j in range(i + 1, len(ranked))
            if np.isfinite(np.corrcoef(ranked[i], ranked[j])[0, 1])
        ]
        return 1.0 if not corrs else float(np.mean(corrs))

    def _model_signal(self, daily_ret, intraday_arrays):
        if not self.ridge_fitted:
            return None, []

        family_preds = {}
        agreement_parts = []
        raw_result = self._predict_family_signal(daily_ret, intraday_arrays, "raw")
        raw_pred = raw_result[0]
        raw_parts = raw_result[1]
        resid_result = self._predict_family_signal(daily_ret, intraday_arrays, "resid")
        resid_pred = resid_result[0]
        resid_parts = resid_result[1]
        if raw_pred is not None and self.raw_family_weight > 0.0:
            family_preds["raw"] = (raw_pred, self.raw_family_weight)
            agreement_parts.extend(raw_parts + [raw_pred])
        if resid_pred is not None and self.resid_family_weight > 0.0:
            family_preds["resid"] = (resid_pred, self.resid_family_weight)
            agreement_parts.extend(resid_parts + [resid_pred])
        if not family_preds:
            return None, []

        total_weight = float(sum(weight for _, weight in family_preds.values()))
        model_pred = np.zeros(N_ASSETS, dtype=float)
        for pred, weight in family_preds.values():
            model_pred += (weight / total_weight) * pred
        return (None, agreement_parts) if np.sum(np.abs(model_pred)) < 1e-12 else (model_pred, agreement_parts)

    def _build_alpha(self, daily_ret, intraday_arrays) -> np.ndarray:
        model_signal = self._model_signal(daily_ret, intraday_arrays)
        pred = model_signal[0]
        agreement_parts = model_signal[1]
        if pred is None:
            self.alpha_agreement_scale = 1.0
            return np.zeros(N_ASSETS, dtype=float)

        if len(agreement_parts) > 1:
            agreement = max(0.0, self._pairwise_rank_agreement(agreement_parts))
            self.alpha_agreement_scale = self.agreement_scale_floor + (1.0 - self.agreement_scale_floor) * agreement
        else:
            self.alpha_agreement_scale = 1.0

        alpha = pred - np.mean(pred)
        alpha = self.alpha_shrink * (alpha / (daily_ret.std(axis=0).values + self.alpha_vol_floor))
        return np.tanh(self.alpha_tanh_scale * alpha)

    def _update_online_histories(self, daily_ret) -> None:
        n_days = len(daily_ret)
        if self.pending_alpha is None or n_days <= self.last_recorded_return_count or n_days == 0:
            self.last_recorded_return_count = n_days
            return
        self.realized_return_history.append(np.asarray(daily_ret.iloc[-1].values, dtype=float).copy())
        self.alpha_signal_history.append(np.asarray(self.pending_alpha, dtype=float).copy())
        self.last_recorded_return_count = n_days

    def _recent_alpha_ic(self, lookback_days):
        n = min(len(self.alpha_signal_history), len(self.realized_return_history), int(lookback_days))
        if n < self.online_min_ic_obs:
            return None
        ics = [
            self._rank_corr(signal, realized)
            for signal, realized in zip(self.alpha_signal_history[-n:], self.realized_return_history[-n:])
        ]
        ics = [ic for ic in ics if np.isfinite(ic)]
        return None if len(ics) < self.online_min_ic_obs else float(np.mean(ics))

    def _recent_ic_scale(self) -> float:
        mean_ic = self._recent_alpha_ic(self.recent_ic_lookback_days)
        if mean_ic is None or self.recent_ic_good - self.recent_ic_bad <= 1e-12:
            return 1.0
        z = float(np.clip((mean_ic - self.recent_ic_bad) / (self.recent_ic_good - self.recent_ic_bad), 0.0, 1.0))
        return self.recent_ic_scale_floor + (1.0 - self.recent_ic_scale_floor) * z

    def _signal_snapshot(self, R, d) -> np.ndarray:
        ret_1 = R[d]
        ret_5 = np.mean(R[d - 4:d + 1], axis=0)
        ret_10 = np.mean(R[d - 9:d + 1], axis=0)
        rel_5 = np.zeros(R.shape[1], dtype=float)
        for idx in self.sector_to_indices.values():
            rel_5[idx] = ret_5[idx] - np.mean(ret_5[idx])
        resid_5 = self._market_sector_residual_features(R, d + 1, lookback=self.resid_feature_lookback)[:, 1]
        combo = (
            self._rank_vector(ret_1)
            + self._rank_vector(ret_5)
            + self._rank_vector(ret_10)
            + self._rank_vector(rel_5)
            + self._rank_vector(resid_5)
        ) / 5.0

        next_ret = R[d + 1]
        signals = (combo, rel_5, resid_5)
        ics = []
        spreads = []
        for signal in signals:
            ranked = self._rank_vector(signal)
            ics.append(self._rank_corr(signal, next_ret))
            top = next_ret[ranked >= np.quantile(ranked, 0.8)]
            bottom = next_ret[ranked <= np.quantile(ranked, 0.2)]
            spreads.append(float(np.mean(top) - np.mean(bottom)) if len(top) and len(bottom) else 0.0)
        return np.array([ics[0], ics[1], ics[2], spreads[0], spreads[1], spreads[2]], dtype=float)

    def _safe_mean(self, values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else 0.0

    def _signal_metric_matrix(self, daily_ret) -> np.ndarray:
        R = daily_ret.values
        metric_matrix = np.full((len(R), 6), np.nan, dtype=float)
        warmup = max(self.resid_feature_lookback, 10)
        for d in range(warmup, len(R) - 1):
            metric_matrix[d] = self._signal_snapshot(R, d)
        return metric_matrix

    def _signal_strength_summary(self, daily_ret=None, metric_matrix=None, t=None):
        if metric_matrix is not None and t is not None:
            end_idx = t - 1
            start_idx = max(0, end_idx - self.regime_signal_lookback_days)
            if end_idx <= start_idx:
                return (0.0,) * 6
            return tuple(self._safe_mean(metric_matrix[start_idx:end_idx, i]) for i in range(6))

        if daily_ret is None:
            return (0.0,) * 6
        R = daily_ret.values
        d_end = len(R) - 1
        d_start = max(max(self.resid_feature_lookback, 10), d_end - self.regime_signal_lookback_days)
        if d_end <= d_start:
            return (0.0,) * 6
        snaps = np.asarray([self._signal_snapshot(R, d) for d in range(d_start, d_end)], dtype=float)
        return tuple(self._safe_mean(snaps[:, i]) for i in range(6))

    def _mean_offdiag(self, corr) -> float:
        if corr.ndim != 2 or corr.shape[0] <= 1:
            return 0.0
        vals = corr[~np.eye(corr.shape[0], dtype=bool)]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else 0.0

    def _ols_beta_scalar(self, y, x) -> float:
        x_c = np.asarray(x, dtype=float) - np.mean(x)
        denom = float(np.dot(x_c, x_c))
        if denom <= 1e-12:
            return 0.0
        y_c = np.asarray(y, dtype=float) - np.mean(y)
        return float(np.dot(y_c, x_c) / denom)

    def _same_sector_residual_corr_mean(self, window) -> float:
        if window.shape[0] <= 2:
            return 0.0
        market = np.mean(window, axis=1)
        residuals = np.zeros_like(window)
        for i in range(window.shape[1]):
            sec = self.sector_id[i]
            peer_idx = np.where((self.sector_id == sec) & (np.arange(window.shape[1]) != i))[0]
            sector_peer = np.mean(window[:, peer_idx], axis=1) if len(peer_idx) > 0 else np.zeros(window.shape[0], dtype=float)
            residuals[:, i] = (
                window[:, i]
                - self._ols_beta_scalar(window[:, i], market) * market
                - self._ols_beta_scalar(window[:, i], sector_peer) * sector_peer
            )
        resid_corrs = [
            self._mean_offdiag(np.corrcoef(residuals[:, idx], rowvar=False))
            for idx in self.sector_to_indices.values()
            if len(idx) > 1
        ]
        return float(np.mean(resid_corrs)) if resid_corrs else 0.0

    def _regime_relationship_features(self, R, t):
        window = R[t - self.regime_lookback_days:t]
        corr = np.corrcoef(window, rowvar=False)
        within_vals = [
            corr[i, j]
            for i in range(corr.shape[0])
            for j in range(i + 1, corr.shape[1])
            if self.sector_id[i] == self.sector_id[j] and np.isfinite(corr[i, j])
        ]
        eigvals = np.linalg.eigvalsh(np.cov(window, rowvar=False))
        eigvals = eigvals[eigvals > 0]
        return (
            float(eigvals[-1] / np.sum(eigvals)) if eigvals.size else 0.0,
            float(np.mean(within_vals)) if within_vals else 0.0,
            self._same_sector_residual_corr_mean(window),
            float(np.mean(np.std(window, axis=1))),
        )

    def _regime_feature_vector(self, daily_ret, summary, t):
        if t < max(self.regime_lookback_days, self.resid_feature_lookback, 10) + 2:
            return None
        feats = np.array([*self._regime_relationship_features(daily_ret.values, t), *summary], dtype=float)
        return feats if np.all(np.isfinite(feats)) else None

    def _train_regime_quality_model(self, daily_ret) -> None:
        self.regime_model_fitted = False
        if len(daily_ret) < self.min_days:
            return

        metric_matrix = self._signal_metric_matrix(daily_ret)
        X_rows = []
        y_vals = []
        start_t = max(self.min_days, self.regime_lookback_days, self.resid_feature_lookback, 10) + 2
        for t in range(start_t, len(daily_ret) - self.regime_horizon_days + 1):
            feats = self._regime_feature_vector(daily_ret, self._signal_strength_summary(metric_matrix=metric_matrix, t=t), t)
            end_idx = min(len(metric_matrix), t + self.regime_horizon_days - 1)
            label = self._safe_mean(metric_matrix[t - 1:end_idx].ravel())
            if feats is None or not np.isfinite(label):
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
        self.regime_model = Ridge(alpha=2.0)
        self.regime_model.fit(Xz, y)
        pred = self.regime_model.predict(Xz)
        self.regime_pred_low = float(np.quantile(pred, 0.25))
        self.regime_pred_high = float(np.quantile(pred, 0.75))
        self.regime_model_fitted = True

    def _regime_quality_scale(self, daily_ret):
        if not self.regime_model_fitted or self.regime_feature_mean is None or self.regime_feature_std is None:
            return 1.0, 1.0, 0.0
        feats = self._regime_feature_vector(daily_ret, self._signal_strength_summary(daily_ret=daily_ret), len(daily_ret))
        if feats is None:
            return 1.0, 1.0, 0.0

        pred = float(self.regime_model.predict(((feats - self.regime_feature_mean) / self.regime_feature_std).reshape(1, -1))[0])
        low = pred if self.regime_pred_low is None else float(self.regime_pred_low)
        high = pred if self.regime_pred_high is None else float(self.regime_pred_high)
        z = 0.5 if high - low <= 1e-12 else float(np.clip((pred - low) / (high - low), 0.0, 1.0))
        gross_scale = self.regime_floor + (1.0 - self.regime_floor) * z
        beta_scale = self.regime_beta_floor + (1.0 - self.regime_beta_floor) * z
        return gross_scale, beta_scale, pred

    def _normalize_l1(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        denom = float(np.sum(np.abs(x)))
        return np.zeros_like(x) if denom < 1e-12 or not np.isfinite(denom) else x / denom

    def _apply_rebalance(self, target) -> np.ndarray:
        target = project_to_gross_limit(target)
        if self.prev_weights is None:
            return target

        spread_rank = self._rank_vector(self.spread_bps / 1e4) + 0.5
        band_mult = 1.0 + self.spread_turnover_scale * (
            self.spread_turnover_min_mult
            + (self.spread_turnover_max_mult - self.spread_turnover_min_mult) * spread_rank
            - 1.0
        )
        band = self.turnover_band * band_mult
        target = target.copy()
        target[np.abs(target - self.prev_weights) < band] = self.prev_weights[np.abs(target - self.prev_weights) < band]
        target = project_to_gross_limit(target)
        return project_to_gross_limit((1.0 - self.rebalance_rate) * self.prev_weights + self.rebalance_rate * target)

    def _robust_cs_score(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float) - np.mean(x)
        q10 = np.percentile(x, self.cs_percentiles[0])
        q90 = np.percentile(x, self.cs_percentiles[1])
        return np.clip(x / max(q90 - q10, self.cs_scale_floor), -self.cs_clip, self.cs_clip)

    def _covariance_aware_overlay(self, alpha_filtered, cov_lw) -> np.ndarray:
        if np.sum(np.abs(alpha_filtered)) < 1e-12:
            return np.zeros_like(alpha_filtered)
        reg = cov_lw + (self.cov_alpha_ridge * (float(np.trace(cov_lw)) / max(len(alpha_filtered), 1)) + self.cov_alpha_floor) * np.eye(len(alpha_filtered))
        try:
            raw = np.linalg.solve(reg, alpha_filtered)
        except np.linalg.LinAlgError:
            raw = alpha_filtered / (np.diag(reg) + self.cov_alpha_floor)
        return self._normalize_l1(self._robust_cs_score(raw - np.mean(raw)))

    def _filtered_overlay_signal(self, alpha) -> np.ndarray:
        filtered = np.zeros_like(alpha)
        order = np.argsort(alpha)
        filtered[order[-2:]] = alpha[order[-2:]]
        filtered[order[:2]] = alpha[order[:2]]
        return filtered

    def get_weights(self, price_history, meta: PublicMeta, day: int) -> np.ndarray:
        del meta, day
        daily_ret = self._daily_returns_df(price_history)
        intraday_arrays = self._daily_intraday_arrays(price_history)
        self._update_online_histories(daily_ret)

        if len(daily_ret) < self.min_days:
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS
            self.prev_weights = w.copy()
            self.pending_alpha = None
            return w

        cov_lw = fit_covariances(daily_ret.iloc[-min(self.lookback_cov, len(daily_ret)):])
        w_rp = risk_parity_weights(cov_lw)
        alpha = self._build_alpha(daily_ret, intraday_arrays)

        if np.sum(np.abs(alpha)) < 1e-12:
            w = self._apply_rebalance(w_rp)
        else:
            alpha_overlay = self._normalize_l1(self._filtered_overlay_signal(alpha))
            if self.cov_alpha_mix > 0.0:
                cov_overlay = self._covariance_aware_overlay(self._filtered_overlay_signal(alpha), cov_lw)
                alpha_overlay = self._normalize_l1((1.0 - self.cov_alpha_mix) * alpha_overlay + self.cov_alpha_mix * cov_overlay)

            beta_eff = self.beta * (self.trend_beta_scale if abs(np.mean(daily_ret.mean(axis=1).iloc[-self.trend_window:])) < self.trend_threshold else 1.0)
            beta_eff *= min(1.0, self.train_days_available / self.history_scale_days) * self.alpha_agreement_scale

            regime_scale = self._regime_quality_scale(daily_ret)
            regime_gross_scale = regime_scale[0]
            regime_beta_scale = regime_scale[1]
            beta_eff *= regime_beta_scale
            target = (1.0 - beta_eff) * w_rp + beta_eff * alpha_overlay

            gross_scale = self.gross_scale_floor + (1.0 - self.gross_scale_floor) * self.alpha_agreement_scale
            w = self._apply_rebalance(gross_scale * self._recent_ic_scale() * regime_gross_scale * target)

        w = project_to_gross_limit(w)
        if not np.all(np.isfinite(w)):
            w = np.ones(N_ASSETS, dtype=float) / N_ASSETS
        self.pending_alpha = np.asarray(alpha, dtype=float).copy()
        self.prev_weights = w.copy()
        return w


def create_strategy() -> StrategyBase:
    return MyStrategy()
