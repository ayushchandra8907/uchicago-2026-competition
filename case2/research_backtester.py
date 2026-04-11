from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from research_utils import daily_returns_from_prices, load_module_from_path, max_drawdown, rank_corr

N_ASSETS = 25
TICKS_PER_DAY = 30
TRADING_DAYS_PER_YEAR = 252
IMPACT_MULT = 2.5
DT_YEAR = 1.0 / (TRADING_DAYS_PER_YEAR * TICKS_PER_DAY)

TRAIN_YEARS = 4
HOLDOUT_YEARS = 1
TRAIN_TICKS = TRAIN_YEARS * TRADING_DAYS_PER_YEAR * TICKS_PER_DAY
HOLDOUT_TICKS = HOLDOUT_YEARS * TRADING_DAYS_PER_YEAR * TICKS_PER_DAY


@dataclass(frozen=True)
class WindowSpec:
    label: str
    train_start_day: int
    train_end_day: int
    test_start_day: int
    test_end_day: int


def project_to_gross_limit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross):
        return w
    if gross > 1.0:
        w /= gross
    return w


def transaction_cost_components(spread: np.ndarray, delta_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = (spread / 2.0) * np.abs(delta_weights)
    quadratic = (IMPACT_MULT * spread) * (delta_weights**2)
    return linear, quadratic


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    x = np.asarray(daily_returns, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return -np.inf if mu <= 0 else np.inf
    return math.sqrt(TRADING_DAYS_PER_YEAR) * mu / sd


def copy_debug(debug: object) -> dict:
    if not isinstance(debug, dict):
        return {}
    out = {}
    for key, value in debug.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def history_through_day(train_prices: np.ndarray, hold_prices: np.ndarray, day: int) -> np.ndarray:
    cutoff = (day + 1) * TICKS_PER_DAY
    return np.vstack([train_prices, hold_prices[:cutoff]])


def hold_fixed_weights_one_day_detailed(
    wealth_gross: float,
    wealth_txn: float,
    wealth_full: float,
    weights: np.ndarray,
    logret: np.ndarray,
    borrow: np.ndarray,
    *,
    day: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    t0 = day * TICKS_PER_DAY
    t_begin = t0 + 1 if day == 0 else t0
    gross_contrib = np.zeros_like(weights)
    borrow_contrib = np.zeros_like(weights)

    for t in range(t_begin, t0 + TICKS_PER_DAY):
        simple_ret = np.exp(logret[t]) - 1.0
        gross_contrib += weights * simple_ret
        borrow_tick = np.maximum(-weights, 0.0) * borrow * DT_YEAR

        wealth_gross *= 1.0 + float(np.sum(weights * simple_ret))
        wealth_txn *= 1.0 + float(np.sum(weights * simple_ret))
        wealth_full *= 1.0 + float(np.sum(weights * simple_ret)) - float(np.sum(borrow_tick))
        borrow_contrib += borrow_tick

    return wealth_gross, wealth_txn, wealth_full, gross_contrib, borrow_contrib


def make_official_parity_windows(total_days: int) -> list[WindowSpec]:
    train_days = TRAIN_YEARS * TRADING_DAYS_PER_YEAR
    hold_days = HOLDOUT_YEARS * TRADING_DAYS_PER_YEAR
    if total_days < train_days + hold_days:
        return []
    return [
        WindowSpec(
            label="official_parity",
            train_start_day=0,
            train_end_day=train_days,
            test_start_day=train_days,
            test_end_day=train_days + hold_days,
        )
    ]


def make_yearly_windows(total_days: int) -> list[WindowSpec]:
    windows = []
    for k in range(2, total_days // TRADING_DAYS_PER_YEAR):
        train_end_day = k * TRADING_DAYS_PER_YEAR
        test_end_day = (k + 1) * TRADING_DAYS_PER_YEAR
        if test_end_day > total_days:
            break
        windows.append(
            WindowSpec(
                label=f"yearly_train0_{k-1}_test_{k}",
                train_start_day=0,
                train_end_day=train_end_day,
                test_start_day=train_end_day,
                test_end_day=test_end_day,
            )
        )
    return windows


def make_expanding_windows(
    total_days: int,
    *,
    min_train_days: int,
    test_days: int,
    step_days: int,
    prefix: str,
) -> list[WindowSpec]:
    windows = []
    for test_start_day in range(min_train_days, total_days - test_days + 1, step_days):
        test_end_day = test_start_day + test_days
        windows.append(
            WindowSpec(
                label=f"{prefix}_{test_start_day}_{test_end_day}",
                train_start_day=0,
                train_end_day=test_start_day,
                test_start_day=test_start_day,
                test_end_day=test_end_day,
            )
        )
    return windows


def summarize_folds(folds: list[dict]) -> dict:
    sharpes = np.array([fold["sharpe_net"] for fold in folds], dtype=float)
    if sharpes.size == 0:
        return {
            "mean_sharpe": float("nan"),
            "std_sharpe": float("nan"),
            "min_sharpe": float("nan"),
            "recent_weighted_mean_sharpe": float("nan"),
            "robust_score": float("nan"),
        }
    weights = np.linspace(1.0, 2.0, len(folds))
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(folds) > 1 else 0.0
    min_sharpe = float(np.min(sharpes))
    recent_weighted_mean = float(np.average(sharpes, weights=weights))
    robust_score = recent_weighted_mean + 0.5 * min_sharpe - 0.25 * std_sharpe
    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "min_sharpe": min_sharpe,
        "recent_weighted_mean_sharpe": recent_weighted_mean,
        "robust_score": robust_score,
    }


def create_strategy_factory(strategy_module, attrs: dict | None = None) -> Callable[[], object]:
    attrs = {} if attrs is None else dict(attrs)

    def factory():
        strat = strategy_module.create_strategy()
        for key, value in attrs.items():
            setattr(strat, key, value)
        return strat

    return factory


def run_backtest_detailed(
    train_prices: np.ndarray,
    hold_prices: np.ndarray,
    strategy,
    meta,
) -> dict:
    spread = np.asarray(meta.spread_bps, dtype=float) / 1e4
    borrow = np.asarray(meta.borrow_bps_annual, dtype=float) / 1e4

    strategy.fit(train_prices, meta, ticks_per_day=TICKS_PER_DAY)
    weights = project_to_gross_limit(strategy.get_weights(train_prices, meta, day=0))
    assert np.all(np.isfinite(weights)), "Non-finite weights at initialization"

    wealth_gross = 1.0
    wealth_txn = 1.0
    wealth_full = 1.0

    init_linear, init_quadratic = transaction_cost_components(spread, weights)
    init_trade_cost = float(np.sum(init_linear + init_quadratic))
    wealth_txn *= 1.0 - init_trade_cost
    wealth_full *= 1.0 - init_trade_cost

    logret = np.zeros_like(hold_prices)
    logret[1:] = np.log(hold_prices[1:] / hold_prices[:-1])
    hold_asset_daily_returns = daily_returns_from_prices(np.vstack([train_prices[-TICKS_PER_DAY:], hold_prices]))[: hold_prices.shape[0] // TICKS_PER_DAY]

    n_days = hold_prices.shape[0] // TICKS_PER_DAY
    daily_returns_gross = np.zeros(n_days)
    daily_returns_txn = np.zeros(n_days)
    daily_returns_net = np.zeros(n_days)
    daily_txn_costs = np.zeros(n_days + 1)
    daily_borrow_costs = np.zeros(n_days)
    daily_turnover = np.zeros(n_days + 1)
    daily_gross = np.zeros(n_days)
    daily_short = np.zeros(n_days)
    asset_gross_contrib = np.zeros((n_days, N_ASSETS))
    asset_borrow_contrib = np.zeros((n_days, N_ASSETS))
    asset_txn_contrib = np.zeros((n_days, N_ASSETS))
    signal_debug = [None] * n_days
    signal_debug[0] = copy_debug(getattr(strategy, "last_debug", {}))

    daily_txn_costs[0] = init_trade_cost
    daily_turnover[0] = float(np.sum(np.abs(weights)))

    for day in range(n_days):
        gross_start = wealth_gross
        txn_start = wealth_txn
        full_start = wealth_full
        daily_gross[day] = float(np.sum(np.abs(weights)))
        daily_short[day] = float(np.sum(np.maximum(-weights, 0.0)))

        wealth_gross, wealth_txn, wealth_full, gross_contrib, borrow_contrib = hold_fixed_weights_one_day_detailed(
            wealth_gross,
            wealth_txn,
            wealth_full,
            weights,
            logret,
            borrow,
            day=day,
        )

        if wealth_full <= 0 or not np.isfinite(wealth_full):
            daily_returns_net[day:] = -1.0
            return {
                "daily_returns_gross": daily_returns_gross,
                "daily_returns_txn": daily_returns_txn,
                "daily_returns_net": daily_returns_net,
                "daily_txn_costs": daily_txn_costs[: day + 1],
                "daily_borrow_costs": daily_borrow_costs[: day + 1],
                "daily_turnover": daily_turnover[: day + 1],
                "daily_gross": daily_gross[: day + 1],
                "daily_short": daily_short[: day + 1],
                "asset_daily_returns": hold_asset_daily_returns[: day + 1],
                "asset_gross_contrib": asset_gross_contrib[: day + 1],
                "asset_borrow_contrib": asset_borrow_contrib[: day + 1],
                "asset_txn_contrib": asset_txn_contrib[: day + 1],
                "signal_debug": signal_debug[: day + 1],
                "blown_up": True,
            }

        history = history_through_day(train_prices, hold_prices, day)
        target = project_to_gross_limit(strategy.get_weights(history, meta, day=day + 1))
        assert np.all(np.isfinite(target)), f"Non-finite weights on day {day}"

        delta = target - weights
        linear, quadratic = transaction_cost_components(spread, delta)
        trade_cost_vec = linear + quadratic
        trade_cost = float(np.sum(trade_cost_vec))

        wealth_txn *= 1.0 - trade_cost
        wealth_full *= 1.0 - trade_cost

        asset_gross_contrib[day] = gross_contrib
        asset_borrow_contrib[day] = borrow_contrib
        asset_txn_contrib[day] = trade_cost_vec
        daily_borrow_costs[day] = float(np.sum(borrow_contrib))
        daily_txn_costs[day + 1] = trade_cost
        daily_turnover[day + 1] = float(np.sum(np.abs(delta)))
        daily_returns_gross[day] = wealth_gross / gross_start - 1.0
        daily_returns_txn[day] = wealth_txn / txn_start - 1.0
        daily_returns_net[day] = wealth_full / full_start - 1.0
        weights = target

        if day + 1 < n_days:
            signal_debug[day + 1] = copy_debug(getattr(strategy, "last_debug", {}))

    return {
        "daily_returns_gross": daily_returns_gross,
        "daily_returns_txn": daily_returns_txn,
        "daily_returns_net": daily_returns_net,
        "daily_txn_costs": daily_txn_costs,
        "daily_borrow_costs": daily_borrow_costs,
        "daily_turnover": daily_turnover,
        "daily_gross": daily_gross,
        "daily_short": daily_short,
        "asset_daily_returns": hold_asset_daily_returns,
        "asset_gross_contrib": asset_gross_contrib,
        "asset_borrow_contrib": asset_borrow_contrib,
        "asset_txn_contrib": asset_txn_contrib,
        "signal_debug": signal_debug,
        "blown_up": False,
    }


def evaluate_windows(
    prices: np.ndarray,
    meta,
    strategy_factory: Callable[[], object],
    windows: list[WindowSpec],
) -> dict:
    folds = []
    for window in windows:
        train_prices = prices[
            window.train_start_day * TICKS_PER_DAY : window.train_end_day * TICKS_PER_DAY
        ]
        hold_prices = prices[
            window.test_start_day * TICKS_PER_DAY : window.test_end_day * TICKS_PER_DAY
        ]
        result = run_backtest_detailed(train_prices, hold_prices, strategy_factory(), meta)
        fold = {
            "label": window.label,
            "train_start_day": window.train_start_day,
            "train_end_day": window.train_end_day,
            "test_start_day": window.test_start_day,
            "test_end_day": window.test_end_day,
            "sharpe_gross": annualized_sharpe(result["daily_returns_gross"]),
            "sharpe_txn_only": annualized_sharpe(result["daily_returns_txn"]),
            "sharpe_net": annualized_sharpe(result["daily_returns_net"]),
            "total_return_net": float(np.prod(1.0 + result["daily_returns_net"]) - 1.0),
            "total_txn_cost": float(np.sum(result["daily_txn_costs"])),
            "total_borrow_cost": float(np.sum(result["daily_borrow_costs"])),
            "max_drawdown": max_drawdown(result["daily_returns_net"]),
            "avg_turnover": float(np.mean(result["daily_turnover"])),
            "avg_gross": float(np.mean(result["daily_gross"])),
            "result": result,
        }
        folds.append(fold)
    return {
        "folds": folds,
        "summary": summarize_folds(folds),
    }


def aggregate_daily_metrics(evaluation: dict) -> dict:
    daily_returns_gross = []
    daily_returns_txn = []
    daily_returns_net = []
    daily_txn_costs = []
    daily_borrow_costs = []
    daily_turnover = []
    daily_gross = []
    daily_short = []
    asset_daily_returns = []
    asset_net_contrib = []
    signal_debug = []

    for fold in evaluation["folds"]:
        result = fold["result"]
        daily_returns_gross.append(result["daily_returns_gross"])
        daily_returns_txn.append(result["daily_returns_txn"])
        daily_returns_net.append(result["daily_returns_net"])
        daily_txn_costs.append(result["daily_txn_costs"][1:])
        daily_borrow_costs.append(result["daily_borrow_costs"])
        daily_turnover.append(result["daily_turnover"][1:])
        daily_gross.append(result["daily_gross"])
        daily_short.append(result["daily_short"])
        asset_daily_returns.append(result["asset_daily_returns"])
        asset_net_contrib.append(
            result["asset_gross_contrib"] - result["asset_borrow_contrib"] - result["asset_txn_contrib"]
        )
        signal_debug.extend(result["signal_debug"])

    return {
        "daily_returns_gross": np.concatenate(daily_returns_gross) if daily_returns_gross else np.array([]),
        "daily_returns_txn": np.concatenate(daily_returns_txn) if daily_returns_txn else np.array([]),
        "daily_returns_net": np.concatenate(daily_returns_net) if daily_returns_net else np.array([]),
        "daily_txn_costs": np.concatenate(daily_txn_costs) if daily_txn_costs else np.array([]),
        "daily_borrow_costs": np.concatenate(daily_borrow_costs) if daily_borrow_costs else np.array([]),
        "daily_turnover": np.concatenate(daily_turnover) if daily_turnover else np.array([]),
        "daily_gross": np.concatenate(daily_gross) if daily_gross else np.array([]),
        "daily_short": np.concatenate(daily_short) if daily_short else np.array([]),
        "asset_daily_returns": np.vstack(asset_daily_returns) if asset_daily_returns else np.empty((0, N_ASSETS)),
        "asset_net_contrib": np.vstack(asset_net_contrib) if asset_net_contrib else np.empty((0, N_ASSETS)),
        "signal_debug": signal_debug,
    }


def print_evaluation(name: str, evaluation: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for fold in evaluation["folds"]:
        print(
            f"{fold['label']:28s} "
            f"Sharpe(net)={fold['sharpe_net']:+.4f} "
            f"ret={fold['total_return_net']:+.2%} "
            f"txn={fold['total_txn_cost']:.4%} "
            f"borrow={fold['total_borrow_cost']:.4%} "
            f"maxdd={fold['max_drawdown']:.2%} "
            f"turn={fold['avg_turnover']:.4f} "
            f"gross={fold['avg_gross']:.4f}"
        )
    summary = evaluation["summary"]
    print(
        f"summary mean={summary['mean_sharpe']:+.4f} "
        f"std={summary['std_sharpe']:.4f} "
        f"min={summary['min_sharpe']:+.4f} "
        f"recent_mean={summary['recent_weighted_mean_sharpe']:+.4f} "
        f"robust={summary['robust_score']:+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research backtester for Case 2")
    parser.add_argument("--mode", choices=["parity", "yearly", "quarterly", "semiannual", "all"], default="all")
    parser.add_argument("--module-path", default="submission_converted.py")
    parser.add_argument("--prices-path", default="prices.csv")
    parser.add_argument("--meta-path", default="meta.csv")
    args = parser.parse_args()

    strategy_module = load_module_from_path("submission_converted", args.module_path)
    prices = strategy_module.load_prices(args.prices_path)
    meta = strategy_module.load_meta(args.meta_path)
    total_days = prices.shape[0] // TICKS_PER_DAY
    strategy_factory = create_strategy_factory(strategy_module)

    mode_builders = {
        "parity": make_official_parity_windows(total_days),
        "yearly": make_yearly_windows(total_days),
        "quarterly": make_expanding_windows(
            total_days,
            min_train_days=2 * TRADING_DAYS_PER_YEAR,
            test_days=63,
            step_days=63,
            prefix="quarterly",
        ),
        "semiannual": make_expanding_windows(
            total_days,
            min_train_days=2 * TRADING_DAYS_PER_YEAR,
            test_days=126,
            step_days=63,
            prefix="semiannual",
        ),
    }

    selected = mode_builders.keys() if args.mode == "all" else [args.mode]
    for mode in selected:
        print_evaluation(mode, evaluate_windows(prices, meta, strategy_factory, mode_builders[mode]))


if __name__ == "__main__":
    main()
