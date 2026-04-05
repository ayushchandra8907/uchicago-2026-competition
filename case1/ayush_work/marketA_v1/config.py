from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskLimits:
    position_limit: int = 200
    order_size_limit: int = 40
    open_volume_limit: int = 80
    normal_soft_position_limit: int = 80
    shock_soft_position_limit: int = 140


@dataclass(frozen=True)
class StrategyParameters:
    price_scale: int = 100
    initial_pe_ratio: float = 10.0
    min_half_spread_px: int = 2
    base_half_spread_px: int = 3
    market_spread_weight: float = 0.35
    vol_widening: float = 18.0
    toxicity_widening: float = 4.0
    fill_widening: float = 2.0
    microprice_alpha: float = 0.65
    trade_pressure_alpha: float = 3.0
    inventory_penalty: float = 0.45
    normal_order_size: int = 12
    shock_order_size: int = 20
    unwind_order_size: int = 16
    aggressive_edge_px: int = 7
    shock_aggressive_edge_px: int = 4
    overshoot_trigger_px: int = 10
    overshoot_fade_edge_px: int = 5
    requote_cooldown_ms: int = 250
    shock_window_ms: int = 2_500
    overshoot_window_ms: int = 7_000


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    project_root: Path
    data_root: Path
    output_root: Path


@dataclass(frozen=True)
class ReplayConfig:
    passive_queue_book_move: bool = True
    mark_to_market: str = "mid"


@dataclass(frozen=True)
class AppConfig:
    paths: ProjectPaths
    risk: RiskLimits
    strategy: StrategyParameters
    replay: ReplayConfig


def _merge_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    valid = {field.name for field in instance.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {key: value for key, value in overrides.items() if key in valid}
    return replace(instance, **filtered)


def build_app_config(
    *,
    risk: RiskLimits | None = None,
    strategy: StrategyParameters | None = None,
    replay: ReplayConfig | None = None,
    repo_root: Path | None = None,
) -> AppConfig:
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[3]
    project_root = Path(__file__).resolve().parent
    paths = ProjectPaths(
        repo_root=resolved_repo_root,
        project_root=project_root,
        data_root=resolved_repo_root / "data_scraping" / "data",
        output_root=project_root / "outputs",
    )
    paths.output_root.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        paths=paths,
        risk=risk or RiskLimits(),
        strategy=strategy or StrategyParameters(),
        replay=replay or ReplayConfig(),
    )


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    app_config = build_app_config()
    raw_path = config_path or os.getenv("A_BOT_CONFIG_PATH")
    if not raw_path:
        return app_config

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (app_config.paths.project_root / path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))

    risk = _merge_dataclass(app_config.risk, payload.get("risk", {}))
    strategy = _merge_dataclass(app_config.strategy, payload.get("strategy", {}))
    replay = _merge_dataclass(app_config.replay, payload.get("replay", {}))
    return build_app_config(risk=risk, strategy=strategy, replay=replay, repo_root=app_config.paths.repo_root)


def config_as_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": str(config.paths.repo_root),
            "project_root": str(config.paths.project_root),
            "data_root": str(config.paths.data_root),
            "output_root": str(config.paths.output_root),
        },
        "risk": asdict(config.risk),
        "strategy": asdict(config.strategy),
        "replay": asdict(config.replay),
    }


def write_best_params(config: AppConfig, path: Path | None = None) -> Path:
    target = path or (config.paths.output_root / "best_params.json")
    target.write_text(json.dumps(config_as_dict(config), indent=2, sort_keys=True), encoding="utf-8")
    return target
