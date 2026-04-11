from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

N_ASSETS = 25
TICKS_PER_DAY = 30
TRADING_DAYS_PER_YEAR = 252


def load_module_from_path(module_name: str, module_path: str | Path):
    path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def daily_closes(price_history: np.ndarray, ticks_per_day: int = TICKS_PER_DAY) -> np.ndarray:
    prices = np.asarray(price_history, dtype=float)
    n_ticks = prices.shape[0]
    n_days = n_ticks // ticks_per_day
    if n_days == 0:
        return np.empty((0, prices.shape[1]), dtype=float)
    close_idx = np.arange(ticks_per_day - 1, n_days * ticks_per_day, ticks_per_day)
    return prices[close_idx]


def daily_returns_from_prices(price_history: np.ndarray, ticks_per_day: int = TICKS_PER_DAY) -> np.ndarray:
    closes = daily_closes(price_history, ticks_per_day=ticks_per_day)
    if closes.shape[0] <= 1:
        return np.empty((0, closes.shape[1]), dtype=float)
    return closes[1:] / closes[:-1] - 1.0


def daily_intraday_arrays(price_history: np.ndarray, sector_id: np.ndarray, ticks_per_day: int = TICKS_PER_DAY) -> dict[str, np.ndarray]:
    prices = np.asarray(price_history, dtype=float)
    n_ticks = prices.shape[0]
    n_days = n_ticks // ticks_per_day
    zeros = np.empty((0, prices.shape[1]), dtype=float)
    if n_days == 0:
        return {
            "open_close": zeros,
            "first5": zeros,
            "last5": zeros,
            "range": zeros,
            "intraday_vol": zeros,
            "close_loc": zeros,
            "sector_rel": zeros,
        }

    day_prices = prices[: n_days * ticks_per_day].reshape(n_days, ticks_per_day, prices.shape[1])
    opens = day_prices[:, 0, :]
    closes = day_prices[:, -1, :]
    highs = np.max(day_prices, axis=1)
    lows = np.min(day_prices, axis=1)
    first5 = day_prices[:, min(4, ticks_per_day - 1), :] / opens - 1.0
    last5 = closes / day_prices[:, max(ticks_per_day - 5, 0), :] - 1.0
    tick_rets = day_prices[:, 1:, :] / day_prices[:, :-1, :] - 1.0
    intraday_vol = np.std(tick_rets, axis=1, ddof=1)
    open_close = closes / opens - 1.0
    close_loc = (closes - lows) / (highs - lows + 1e-12)

    sector_rel = np.zeros_like(open_close)
    for sec in np.unique(sector_id):
        idx = np.where(sector_id == sec)[0]
        sector_rel[:, idx] = open_close[:, idx] - np.mean(open_close[:, idx], axis=1, keepdims=True)

    return {
        "open_close": open_close,
        "first5": first5,
        "last5": last5,
        "range": highs / np.maximum(lows, 1e-12) - 1.0,
        "intraday_vol": intraday_vol,
        "close_loc": close_loc,
        "sector_rel": sector_rel,
    }


def rank_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    order = np.argsort(np.argsort(arr))
    if len(order) <= 1:
        return np.zeros_like(order, dtype=float)
    return order.astype(float) / (len(order) - 1.0) - 0.5


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rank_vector(x)
    yr = rank_vector(y)
    corr = np.corrcoef(xr, yr)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def max_drawdown(daily_returns: np.ndarray) -> float:
    dr = np.asarray(daily_returns, dtype=float)
    if dr.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + dr)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak - 1.0))


def rolling_ols_beta(y: np.ndarray, x: np.ndarray, ridge: float = 1e-8) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = float(np.dot(xc, xc) + ridge)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def to_frame(arr: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=[f"{prefix}_{i:02d}" for i in range(arr.shape[1])])
