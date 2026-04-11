from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the v3 bot configuration is incomplete or invalid."""


@dataclass(frozen=True)
class ExchangeConfig:
    host: str
    username: str
    password: str


@dataclass(frozen=True)
class MarketCStrategyConfig:
    symbol_c: str = "C"
    fed_hike: str = "R_HIKE"
    fed_hold: str = "R_HOLD"
    fed_cut: str = "R_CUT"
    prediction_scale: int = 1_000
    max_exchange_order_qty: int = 40
    max_open_orders: int = 50
    max_outstanding_volume: int = 120
    max_absolute_position_per_contract: int = 200
    shared_rate_position_budget: int = 600
    signal_light_position: int = 40
    signal_medium_position: int = 80
    signal_strong_position: int = 140
    signal_extreme_position: int = 200
    entry_edge_ticks: int = 25
    exit_edge_ticks: int = 10
    no_arb_sum_tolerance_ticks: int = 120
    expected_rate_step_bp: float = 25.0
    cpi_small_surprise: float = 0.0002
    cpi_medium_surprise: float = 0.0005
    cpi_large_surprise: float = 0.0010
    cpi_small_logit_shift: float = 0.35
    cpi_medium_logit_shift: float = 0.75
    cpi_large_logit_shift: float = 1.20
    news_relevance_threshold: float = 0.75
    news_light_logit_shift: float = 0.30
    news_medium_logit_shift: float = 0.65
    news_strong_logit_shift: float = 1.05
    news_extreme_logit_shift: float = 1.40
    news_score_delta_divisor: float = 2.5
    posterior_floor_probability: float = 1e-4
    flatten_near_zero_threshold: int = 0
    macro_signal_timeout_ms: int = 10_000
    round_duration_ms: int = 900_000
    decision_probe_countdown_ms: int = 180_000
    decision_probe_base_target: int = 20
    decision_probe_confident_target: int = 80
    decision_probe_confidence_gap_ticks: int = 175
    decision_probe_confident_price: int = 700
    baseline_center_price: int = 400
    baseline_neutral_low_price: int = 320
    baseline_neutral_high_price: int = 480
    baseline_target_cap: int = 60
    baseline_full_size_distance_ticks: int = 320
    trading_macro_target_cap: int = 120
    macro_pair_min_delta: float = 0.35
    macro_pair_hold_tail_fallback_delta: float = 0.15
    macro_move_light_ticks: int = 25
    macro_move_medium_ticks: int = 50
    macro_move_strong_ticks: int = 80
    macro_move_extreme_ticks: int = 120
    macro_equilibrium_hold_ms: int = 1_000
    macro_equilibrium_min_elapsed_ms: int = 1_000
    macro_equilibrium_min_samples: int = 3
    macro_equilibrium_band_ticks: int = 12
    macro_equilibrium_residual_edge_ticks: int = 15
    macro_overshoot_trigger_fraction: float = 0.25
    macro_overshoot_min_trigger_ticks: int = 8
    macro_overshoot_trim_fraction: float = 0.50
    macro_overshoot_min_residual_qty: int = 20
    macro_reversal_min_progress_ticks: int = 18
    macro_reversal_exit_ticks: int = 8
    reversion_disable_countdown_ms: int = 180_000
    reversion_low_price_threshold: int = 40
    reversion_high_price_threshold: int = 960
    reversion_overlay_target: int = 120
    reversion_take_profit_ticks: int = 60
    reversion_stop_loss_ticks: int = 25
    reversion_reversal_min_progress_ticks: int = 18
    reversion_reversal_exit_ticks: int = 8
    pair_disable_countdown_ms: int = 180_000
    pair_lookback_ms: int = 5_000
    pair_min_move_ticks: int = 18
    pair_overlay_target: int = 120
    pair_reversal_min_progress_ticks: int = 18
    pair_reversal_exit_ticks: int = 8
    endgame_countdown_ms: int = 120_000
    endgame_long_target: int = 200
    endgame_short_target: int = -200
    endgame_almost_dead_price: int = 50
    order_slice_target_qty: int = 12
    order_slice_min_qty: int = 7
    order_slice_max_qty: int = 15
    timer_interval_ms: int = 60
    min_order_live_ms: int = 75
    replace_qty_tolerance: int = 1
    replace_price_tolerance_ticks: int = 0

    @property
    def tracked_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "A"
    position_cap: int = 200
    max_exchange_order_qty: int = 40
    max_open_orders: int = 50
    max_outstanding_volume: int = 120
    max_absolute_position: int = 200
    first_earnings_anchor: float = 1.0
    first_earnings_baseline_window_ms: int = 12_000
    first_earnings_min_mid_samples: int = 8
    shock_min_edge_ticks: int = 10
    shock_position_scale: float = 1.20
    shock_full_confidence_edge_ticks: int = 80
    shock_change_position_scale: float = 0.75
    shock_full_confidence_change_ticks: int = 40
    shock_min_position: int = 4
    shock_initial_clip: int = 200
    shock_reinforce_clip: int = 80
    shock_emergency_dump_min_elapsed_ms: int = 250
    shock_emergency_dump_ticks: int = 40
    shock_emergency_dump_fraction: float = 0.20
    shock_emergency_dump_min_inventory: int = 12
    shock_max_hold_ms: int = 12_500
    shock_decay_start_ms: int = 5_000
    shock_decay_interval_ms: int = 500
    shock_decay_fraction: float = 0.08
    shock_decay_min_qty: int = 6
    shock_decay_max_qty: int = 10
    shock_decay_min_inventory: int = 40
    shock_decay_min_residual_fraction: float = 0.10
    shock_decay_stall_window_ms: int = 1_200
    shock_decay_stall_threshold_ticks: int = 12
    overshoot_hold_ms: int = 225
    overshoot_max_wait_ms: int = 600
    overshoot_band_ticks: int = 10
    overshoot_reversal_ticks: int = 2
    overshoot_stage1_fraction: float = 0.30
    overshoot_stage2_fraction: float = 0.25
    overshoot_stage3_fraction: float = 0.20
    overshoot_stage_min_qty: int = 4
    overshoot_stage_max_qty: int = 16
    overshoot_min_residual_fraction: float = 0.30
    overshoot_large_position_threshold: int = 100
    overshoot_large_position_stage1_fraction: float = 0.50
    overshoot_large_position_residual_fraction: float = 0.50
    news_overshoot_hold_ms: int = 200
    news_overshoot_band_ticks: int = 10
    news_overshoot_reversal_ticks: int = 2
    equilibrium_band_ticks: int = 8
    equilibrium_hold_ms: int = 1_000
    equilibrium_min_samples: int = 6
    equilibrium_min_elapsed_ms: int = 1_000
    equilibrium_residual_edge_ticks: int = 40
    equilibrium_min_capture_fraction: float = 0.55
    news_light_offset_ticks: int = 12
    news_medium_offset_ticks: int = 24
    news_strong_offset_ticks: int = 48
    news_extreme_offset_ticks: int = 80
    news_very_extreme_offset_ticks: int = 120
    news_light_position: int = 8
    news_medium_position: int = 36
    news_strong_position: int = 90
    news_extreme_position: int = 130
    news_very_extreme_position: int = 200
    news_zero_position_threshold: int = 3
    news_confirmation_timeout_ms: int = 900
    news_confirmation_move_ticks: int = 3
    news_takeover_flatten_ms: int = 1_200
    news_takeover_near_flat_threshold: int = 4
    news_equilibrium_hold_ms: int = 1_400
    news_equilibrium_min_elapsed_ms: int = 1_200
    news_equilibrium_residual_edge_ticks: int = 40
    news_equilibrium_min_capture_fraction: float = 0.55
    news_overshoot_max_wait_ms: int = 700
    flatten_deadline_ms: int = 2_400
    flatten_force_cross_ms: int = 700
    flatten_near_zero_threshold: int = 1
    order_slice_target_qty: int = 12
    order_slice_min_qty: int = 7
    order_slice_max_qty: int = 15
    multiplier_update_alpha: float = 0.35
    multiplier_update_clamp_fraction: float = 0.18
    multiplier_clean_sample_limit: int = 5
    multiplier_sample_clamp_fraction: float = 0.15
    timer_interval_ms: int = 60
    min_order_live_ms: int = 75
    replace_qty_tolerance: int = 1
    replace_price_tolerance_ticks: int = 0


@dataclass(frozen=True)
class LoggerConfig:
    enabled: bool = True
    run_root: Path | None = None
    queue_max_events: int = 2_000
    write_decision_snapshots: bool = True
    midrun_checkpoint_enabled: bool = True
    midrun_checkpoint_ms: int = 450_000


@dataclass(frozen=True)
class BotPaths:
    base_dir: Path


@dataclass(frozen=True)
class BotConfig:
    exchange: ExchangeConfig
    strategy: StrategyConfig
    logger: LoggerConfig
    paths: BotPaths
    c_strategy: MarketCStrategyConfig = field(default_factory=MarketCStrategyConfig)


def _optional_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _optional_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float.") from exc


def _optional_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean.")


def _load_local_exchange_defaults(base_path: Path) -> tuple[str | None, str | None, str | None]:
    candidate_paths = [
        base_path / "local_config.json",
        base_path.parent / "marketA_v2" / "local_config.json",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to read exchange config at {candidate}: {exc}") from exc
        return (
            None if raw.get("host") is None else str(raw.get("host")).strip(),
            None if raw.get("username") is None else str(raw.get("username")).strip(),
            None if raw.get("password") is None else str(raw.get("password")).strip(),
        )
    return None, None, None


def _required_value(name: str, fallback: str | None) -> str:
    env_value = os.getenv(name)
    if env_value is not None and env_value.strip():
        return env_value.strip()
    if fallback is not None and fallback.strip():
        return fallback.strip()
    raise ConfigError(f"Missing required value for {name}.")


def load_bot_config(base_dir: str | Path) -> BotConfig:
    base_path = Path(base_dir).resolve()
    default_host, default_username, default_password = _load_local_exchange_defaults(base_path)
    exchange = ExchangeConfig(
        host=_required_value("UTC_HOST", default_host),
        username=_required_value("UTC_USERNAME", default_username),
        password=_required_value("UTC_PASSWORD", default_password),
    )
    strategy = StrategyConfig(
        position_cap=_optional_int("A_V3_POSITION_CAP", 200),
        max_exchange_order_qty=_optional_int("A_V3_MAX_EXCHANGE_ORDER_QTY", 40),
        max_open_orders=_optional_int("A_V3_MAX_OPEN_ORDERS", 50),
        max_outstanding_volume=_optional_int("A_V3_MAX_OUTSTANDING_VOLUME", 120),
        max_absolute_position=_optional_int("A_V3_MAX_ABSOLUTE_POSITION", 200),
        first_earnings_anchor=_optional_float("A_V3_FIRST_EARNINGS_ANCHOR", 1.0),
        first_earnings_baseline_window_ms=_optional_int("A_V3_FIRST_EARNINGS_BASELINE_WINDOW_MS", 12_000),
        first_earnings_min_mid_samples=_optional_int("A_V3_FIRST_EARNINGS_MIN_MID_SAMPLES", 8),
        shock_min_edge_ticks=_optional_int("A_V3_SHOCK_MIN_EDGE_TICKS", 10),
        shock_position_scale=_optional_float("A_V3_SHOCK_POSITION_SCALE", 1.20),
        shock_full_confidence_edge_ticks=_optional_int("A_V3_SHOCK_FULL_CONFIDENCE_EDGE_TICKS", 80),
        shock_change_position_scale=_optional_float("A_V3_SHOCK_CHANGE_POSITION_SCALE", 0.75),
        shock_full_confidence_change_ticks=_optional_int("A_V3_SHOCK_FULL_CONFIDENCE_CHANGE_TICKS", 40),
        shock_min_position=_optional_int("A_V3_SHOCK_MIN_POSITION", 4),
        shock_initial_clip=_optional_int("A_V3_SHOCK_INITIAL_CLIP", 200),
        shock_reinforce_clip=_optional_int("A_V3_SHOCK_REINFORCE_CLIP", 80),
        shock_emergency_dump_min_elapsed_ms=_optional_int("A_V3_SHOCK_EMERGENCY_DUMP_MIN_ELAPSED_MS", 250),
        shock_emergency_dump_ticks=_optional_int("A_V3_SHOCK_EMERGENCY_DUMP_TICKS", 40),
        shock_emergency_dump_fraction=_optional_float("A_V3_SHOCK_EMERGENCY_DUMP_FRACTION", 0.20),
        shock_emergency_dump_min_inventory=_optional_int("A_V3_SHOCK_EMERGENCY_DUMP_MIN_INVENTORY", 12),
        shock_max_hold_ms=_optional_int("A_V3_SHOCK_MAX_HOLD_MS", 12_500),
        shock_decay_start_ms=_optional_int("A_V3_SHOCK_DECAY_START_MS", 5_000),
        shock_decay_interval_ms=_optional_int("A_V3_SHOCK_DECAY_INTERVAL_MS", 500),
        shock_decay_fraction=_optional_float("A_V3_SHOCK_DECAY_FRACTION", 0.08),
        shock_decay_min_qty=_optional_int("A_V3_SHOCK_DECAY_MIN_QTY", 6),
        shock_decay_max_qty=_optional_int("A_V3_SHOCK_DECAY_MAX_QTY", 10),
        shock_decay_min_inventory=_optional_int("A_V3_SHOCK_DECAY_MIN_INVENTORY", 40),
        shock_decay_min_residual_fraction=_optional_float("A_V3_SHOCK_DECAY_MIN_RESIDUAL_FRACTION", 0.10),
        shock_decay_stall_window_ms=_optional_int("A_V3_SHOCK_DECAY_STALL_WINDOW_MS", 1_200),
        shock_decay_stall_threshold_ticks=_optional_int("A_V3_SHOCK_DECAY_STALL_THRESHOLD_TICKS", 12),
        overshoot_hold_ms=_optional_int("A_V3_OVERSHOOT_HOLD_MS", 225),
        overshoot_max_wait_ms=_optional_int("A_V3_OVERSHOOT_MAX_WAIT_MS", 600),
        overshoot_band_ticks=_optional_int("A_V3_OVERSHOOT_BAND_TICKS", 10),
        overshoot_reversal_ticks=_optional_int("A_V3_OVERSHOOT_REVERSAL_TICKS", 2),
        overshoot_stage1_fraction=_optional_float("A_V3_OVERSHOOT_STAGE1_FRACTION", 0.30),
        overshoot_stage2_fraction=_optional_float("A_V3_OVERSHOOT_STAGE2_FRACTION", 0.25),
        overshoot_stage3_fraction=_optional_float("A_V3_OVERSHOOT_STAGE3_FRACTION", 0.20),
        overshoot_stage_min_qty=_optional_int("A_V3_OVERSHOOT_STAGE_MIN_QTY", 4),
        overshoot_stage_max_qty=_optional_int("A_V3_OVERSHOOT_STAGE_MAX_QTY", 16),
        overshoot_min_residual_fraction=_optional_float("A_V3_OVERSHOOT_MIN_RESIDUAL_FRACTION", 0.30),
        overshoot_large_position_threshold=_optional_int("A_V3_OVERSHOOT_LARGE_POSITION_THRESHOLD", 100),
        overshoot_large_position_stage1_fraction=_optional_float("A_V3_OVERSHOOT_LARGE_POSITION_STAGE1_FRACTION", 0.50),
        overshoot_large_position_residual_fraction=_optional_float("A_V3_OVERSHOOT_LARGE_POSITION_RESIDUAL_FRACTION", 0.50),
        news_overshoot_hold_ms=_optional_int("A_V3_NEWS_OVERSHOOT_HOLD_MS", 200),
        news_overshoot_band_ticks=_optional_int("A_V3_NEWS_OVERSHOOT_BAND_TICKS", 10),
        news_overshoot_reversal_ticks=_optional_int("A_V3_NEWS_OVERSHOOT_REVERSAL_TICKS", 2),
        equilibrium_band_ticks=_optional_int("A_V3_EQUILIBRIUM_BAND_TICKS", 8),
        equilibrium_hold_ms=_optional_int("A_V3_EQUILIBRIUM_HOLD_MS", 1_000),
        equilibrium_min_samples=_optional_int("A_V3_EQUILIBRIUM_MIN_SAMPLES", 6),
        equilibrium_min_elapsed_ms=_optional_int("A_V3_EQUILIBRIUM_MIN_ELAPSED_MS", 1_000),
        equilibrium_residual_edge_ticks=_optional_int("A_V3_EQUILIBRIUM_RESIDUAL_EDGE_TICKS", 40),
        equilibrium_min_capture_fraction=_optional_float("A_V3_EQUILIBRIUM_MIN_CAPTURE_FRACTION", 0.55),
        news_light_offset_ticks=_optional_int("A_V3_NEWS_LIGHT_OFFSET_TICKS", 12),
        news_medium_offset_ticks=_optional_int("A_V3_NEWS_MEDIUM_OFFSET_TICKS", 24),
        news_strong_offset_ticks=_optional_int("A_V3_NEWS_STRONG_OFFSET_TICKS", 48),
        news_extreme_offset_ticks=_optional_int("A_V3_NEWS_EXTREME_OFFSET_TICKS", 80),
        news_very_extreme_offset_ticks=_optional_int("A_V3_NEWS_VERY_EXTREME_OFFSET_TICKS", 120),
        news_light_position=_optional_int("A_V3_NEWS_LIGHT_POSITION", 8),
        news_medium_position=_optional_int("A_V3_NEWS_MEDIUM_POSITION", 36),
        news_strong_position=_optional_int("A_V3_NEWS_STRONG_POSITION", 90),
        news_extreme_position=_optional_int("A_V3_NEWS_EXTREME_POSITION", 130),
        news_very_extreme_position=_optional_int("A_V3_NEWS_VERY_EXTREME_POSITION", 200),
        news_zero_position_threshold=_optional_int("A_V3_NEWS_ZERO_POSITION_THRESHOLD", 3),
        news_confirmation_timeout_ms=_optional_int("A_V3_NEWS_CONFIRMATION_TIMEOUT_MS", 900),
        news_confirmation_move_ticks=_optional_int("A_V3_NEWS_CONFIRMATION_MOVE_TICKS", 3),
        news_takeover_flatten_ms=_optional_int("A_V3_NEWS_TAKEOVER_FLATTEN_MS", 1_200),
        news_takeover_near_flat_threshold=_optional_int("A_V3_NEWS_TAKEOVER_NEAR_FLAT_THRESHOLD", 4),
        news_equilibrium_hold_ms=_optional_int("A_V3_NEWS_EQUILIBRIUM_HOLD_MS", 1_400),
        news_equilibrium_min_elapsed_ms=_optional_int("A_V3_NEWS_EQUILIBRIUM_MIN_ELAPSED_MS", 1_200),
        news_equilibrium_residual_edge_ticks=_optional_int("A_V3_NEWS_EQUILIBRIUM_RESIDUAL_EDGE_TICKS", 40),
        news_equilibrium_min_capture_fraction=_optional_float("A_V3_NEWS_EQUILIBRIUM_MIN_CAPTURE_FRACTION", 0.55),
        news_overshoot_max_wait_ms=_optional_int("A_V3_NEWS_OVERSHOOT_MAX_WAIT_MS", 700),
        flatten_deadline_ms=_optional_int("A_V3_FLATTEN_DEADLINE_MS", 2_400),
        flatten_force_cross_ms=_optional_int("A_V3_FLATTEN_FORCE_CROSS_MS", 700),
        flatten_near_zero_threshold=_optional_int("A_V3_FLATTEN_NEAR_ZERO_THRESHOLD", 1),
        order_slice_target_qty=_optional_int("A_V3_ORDER_SLICE_TARGET_QTY", 12),
        order_slice_min_qty=_optional_int("A_V3_ORDER_SLICE_MIN_QTY", 7),
        order_slice_max_qty=_optional_int("A_V3_ORDER_SLICE_MAX_QTY", 15),
        multiplier_update_alpha=_optional_float("A_V3_MULTIPLIER_UPDATE_ALPHA", 0.35),
        multiplier_update_clamp_fraction=_optional_float("A_V3_MULTIPLIER_UPDATE_CLAMP_FRACTION", 0.18),
        multiplier_clean_sample_limit=_optional_int("A_V3_MULTIPLIER_CLEAN_SAMPLE_LIMIT", 5),
        multiplier_sample_clamp_fraction=_optional_float("A_V3_MULTIPLIER_SAMPLE_CLAMP_FRACTION", 0.15),
        timer_interval_ms=_optional_int("A_V3_TIMER_INTERVAL_MS", 60),
        min_order_live_ms=_optional_int("A_V3_MIN_ORDER_LIVE_MS", 75),
        replace_qty_tolerance=_optional_int("A_V3_REPLACE_QTY_TOLERANCE", 1),
        replace_price_tolerance_ticks=_optional_int("A_V3_REPLACE_PRICE_TOLERANCE_TICKS", 0),
    )
    logger = LoggerConfig(
        enabled=_optional_bool("A_V3_LOGGER_ENABLED", True),
        run_root=(base_path / "analysis_runs"),
        queue_max_events=_optional_int("A_V3_LOGGER_QUEUE_MAX_EVENTS", 2_000),
        write_decision_snapshots=_optional_bool("A_V3_WRITE_DECISION_SNAPSHOTS", True),
        midrun_checkpoint_enabled=_optional_bool("A_V3_MIDRUN_CHECKPOINT_ENABLED", True),
        midrun_checkpoint_ms=_optional_int("A_V3_MIDRUN_CHECKPOINT_MS", 450_000),
    )
    return BotConfig(
        exchange=exchange,
        strategy=strategy,
        c_strategy=MarketCStrategyConfig(),
        logger=logger,
        paths=BotPaths(base_dir=base_path),
    )
