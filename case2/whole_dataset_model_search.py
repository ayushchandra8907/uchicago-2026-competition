from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

CASE_DIR = Path("/Users/parrynall/UChicago Trading Competition/uchicago-2026-competition/case2")
SANDBOX_SUBMISSION = Path("/Users/parrynall/Documents/Playground/case2_sandbox/submission_converted.py")
VALIDATE_PATH = CASE_DIR / "validate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


submission = _load_module("case2_submission_search", SANDBOX_SUBMISSION)
validate = _load_module("case2_validate_search", VALIDATE_PATH)


@dataclass
class SearchMetrics:
    name: str
    holdout_sharpe: float
    cv_sharpes: list[float]
    total_return: float
    total_cost: float
    max_drawdown: float

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_sharpes))

    @property
    def cv_std(self) -> float:
        if len(self.cv_sharpes) <= 1:
            return 0.0
        return float(np.std(self.cv_sharpes, ddof=1))

    @property
    def cv_min(self) -> float:
        return float(np.min(self.cv_sharpes))

    @property
    def robust_score(self) -> float:
        return self.cv_mean + 0.5 * self.cv_min - 0.25 * self.cv_std


class SearchStrategy(submission.MyStrategy):
    def __init__(self, *, raw_model_kind: str = "ridge", resid_model_kind: str = "ridge", **attrs):
        super().__init__()
        self.raw_model_kind = raw_model_kind
        self.resid_model_kind = resid_model_kind
        for key, value in attrs.items():
            setattr(self, key, value)

    def _make_estimator(self, kind: str):
        if kind == "ridge":
            return Ridge(alpha=self.ridge_alpha)
        if kind == "hgb":
            return HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=80,
                l2_regularization=2.0,
                min_samples_leaf=20,
                random_state=42,
            )
        if kind == "rf":
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=1,
            )
        if kind == "extra":
            return ExtraTreesRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=1,
            )
        if kind == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=42,
            )
        raise ValueError(f"Unknown model kind: {kind}")

    def _train_ridge_family(
        self,
        daily_ret,
        intraday_arrays,
        *,
        target_mode: str,
        lookbacks: tuple[int, ...],
        weights: tuple[float, ...],
    ):
        family_models = []
        family_kind = self.raw_model_kind if target_mode == "raw" else self.resid_model_kind
        for resid_lookback, model_weight in self._family_specs(lookbacks, weights):
            X, y = self._build_dataset(
                daily_ret,
                intraday_arrays=intraday_arrays,
                target_mode=target_mode,
                resid_lookback=resid_lookback,
            )
            if X is None or len(X) <= self.fit_min_samples:
                continue
            model = self._make_estimator(family_kind)
            model.fit(X, y)
            family_models.append(
                {
                    "model": model,
                    "resid_lookback": resid_lookback,
                    "weight": model_weight,
                }
            )

        if not family_models:
            return None
        return {"target_mode": target_mode, "models": family_models}


def evaluate_candidate(name: str, raw_model_kind: str, resid_model_kind: str, attrs: dict) -> SearchMetrics:
    prices = submission.load_prices(str(CASE_DIR / "prices.csv"))
    meta = submission.load_meta(str(CASE_DIR / "meta.csv"))
    ticks_per_year = validate.TRADING_DAYS_PER_YEAR * validate.TICKS_PER_DAY

    def make_strategy():
        return SearchStrategy(
            raw_model_kind=raw_model_kind,
            resid_model_kind=resid_model_kind,
            **attrs,
        )

    cv_sharpes = []
    last_result = None
    for k in range(2, prices.shape[0] // ticks_per_year):
        train_end = k * ticks_per_year
        test_end = (k + 1) * ticks_per_year
        if test_end > prices.shape[0]:
            break
        result = validate.run_backtest(
            prices[:train_end],
            prices[train_end:test_end],
            make_strategy(),
            meta,
        )
        cv_sharpes.append(validate.annualized_sharpe(result["daily_returns"]))
        last_result = result

    assert last_result is not None
    dr = last_result["daily_returns"]
    cum = np.cumprod(1.0 + dr)
    maxdd = float(np.min(cum / np.maximum.accumulate(cum) - 1.0))
    return SearchMetrics(
        name=name,
        holdout_sharpe=cv_sharpes[-1],
        cv_sharpes=cv_sharpes,
        total_return=float(np.prod(1.0 + dr) - 1.0),
        total_cost=float(np.sum(last_result["daily_costs"])),
        max_drawdown=maxdd,
    )


def main() -> None:
    profiles = {
        "stable_ridge": {
            "use_fast_sleeve": False,
            "model_sleeve_weight": 1.0,
            "fast_sleeve_weight": 0.0,
            "raw_family_weight": 1.0,
            "resid_family_weight": 0.0,
            "pred_hgb_weight": 0.0,
            "beta": 0.88,
            "gross_scale_floor": 0.80,
            "turnover_band": 0.018,
        },
        "moderate_fast": {
            "use_fast_sleeve": True,
            "model_sleeve_weight": 0.75,
            "fast_sleeve_weight": 0.25,
            "raw_family_weight": 0.7,
            "resid_family_weight": 0.3,
            "pred_hgb_weight": 0.0,
            "beta": 0.92,
            "gross_scale_floor": 0.75,
            "turnover_band": 0.021,
        },
        "aggressive_fast": {
            "use_fast_sleeve": True,
            "model_sleeve_weight": 0.70,
            "fast_sleeve_weight": 0.30,
            "raw_family_weight": 0.7,
            "resid_family_weight": 0.3,
            "pred_hgb_weight": 0.0,
            "beta": 1.08,
            "gross_scale_floor": 0.75,
            "turnover_band": 0.021,
        },
    }

    model_pairs = [
        ("ridge", "ridge"),
        ("extra", "ridge"),
        ("rf", "ridge"),
        ("mlp", "ridge"),
        ("ridge", "mlp"),
        ("hgb", "ridge"),
    ]

    results = []
    for profile_name, attrs in profiles.items():
        for raw_kind, resid_kind in model_pairs:
            name = f"{profile_name}:{raw_kind}/{resid_kind}"
            metrics = evaluate_candidate(name, raw_kind, resid_kind, attrs)
            results.append(metrics)
            fold_str = ",".join(f"{x:+.4f}" for x in metrics.cv_sharpes)
            print(
                f"{name:30s} "
                f"hold={metrics.holdout_sharpe:+.4f} "
                f"cv_mean={metrics.cv_mean:+.4f} "
                f"cv_std={metrics.cv_std:.4f} "
                f"cv_min={metrics.cv_min:+.4f} "
                f"robust={metrics.robust_score:+.4f} "
                f"ret={metrics.total_return:+.2%} "
                f"cost={metrics.total_cost:.4%} "
                f"maxdd={metrics.max_drawdown:.2%} "
                f"folds=[{fold_str}]"
            )

    print("\nTop by robust score")
    for metrics in sorted(results, key=lambda x: x.robust_score, reverse=True)[:10]:
        print(
            f"{metrics.name:30s} "
            f"robust={metrics.robust_score:+.4f} "
            f"hold={metrics.holdout_sharpe:+.4f} "
            f"cv_mean={metrics.cv_mean:+.4f} "
            f"cv_min={metrics.cv_min:+.4f}"
        )


if __name__ == "__main__":
    main()
