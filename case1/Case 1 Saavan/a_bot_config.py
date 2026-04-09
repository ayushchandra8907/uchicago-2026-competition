from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


class ConfigError(ValueError):
    """Raised when the bot configuration is incomplete or invalid."""


@dataclass(frozen=True)
class ExchangeConfig:
    host: str
    username: str
    password: str


@dataclass(frozen=True)
class AConfig:
    initial_multiplier: float | None = None
    initial_fair_value: int | None = None
    recover_pricing_state: bool = False
    total_position_limit: int = 180
    earnings_base_budget: int = 120
    mm_base_budget: int = 60
    earnings_shift_budget: int = 180
    mm_shift_budget: int = 0
    startup_assume_fresh_round: bool = True
    pre_news_pullback_ms: int = 4_000
    calibration_min_delay_ms: int = 5_000
    calibration_max_delay_ms: int = 20_000
    calibration_sample_period_ms: int = 1_000
    calibration_stability_band_ticks: int = 8
    calibration_tolerance_fraction: float = 0.10
    calibration_min_tolerance_fraction: float = 0.03
    candidate_confirmations: int = 2
    discovery_quote_size: int = 2
    discovery_max_position: int = 4
    discovery_half_spread_ticks: int = 8
    news_caution_duration_ms: int = 12_000
    news_caution_quote_size: int = 1
    news_caution_max_position: int = 2
    news_caution_half_spread_ticks: int = 8
    news_light_offset_ticks: int = 12
    news_medium_offset_ticks: int = 24
    news_strong_offset_ticks: int = 48
    news_extreme_offset_ticks: int = 80
    news_very_extreme_offset_ticks: int = 120
    news_light_position: int = 8
    news_medium_position: int = 36
    news_strong_position: int = 90
    news_extreme_position: int = 130
    news_very_extreme_position: int = 180
    news_zero_position_threshold: int = 3
    news_confirmation_timeout_ms: int = 900
    news_confirmation_move_ticks: int = 3
    freeze_multiplier_after_unstructured_news: bool = True
    steady_half_spread_ticks: int = 1
    steady_take_min_edge: int = 2
    steady_take_large_inventory_edge: int = 4
    opening_quote_size: int = 1
    opening_max_position: int = 8
    opening_half_spread_ticks: int = 4
    opening_min_book_spread: int = 10
    steady_quote_size: int = 3
    steady_max_position: int = 32
    steady_inventory_skew: float = 0.75
    steady_take_inventory_guard: int = 8
    steady_passive_reduce_start: int = 8
    steady_passive_reduce_full: int = 20
    unwind_inventory_skew: float = 1.50
    unwind_flatten_threshold: int = 2
    unwind_entry_position: int = 24
    unwind_exit_position: int = 12
    unwind_aggressive_entry: int = 24
    unwind_aggressive_exit: int = 16
    earnings_unwind_aggressive_entry: int = 48
    earnings_unwind_aggressive_exit: int = 24
    earnings_unwind_passive_exit: int = 8
    earnings_unwind_passive_take_edge: int = 8
    post_earnings_mm_cooldown_ms: int = 10_000
    unwind_fast_entry: int = 36
    unwind_fast_exit: int = 12
    unwind_fast_quote_size: int = 12
    shock_quote_size: int = 15
    shock_base_max_position: int = 100
    shock_shift_max_position: int = 180
    shock_window_ms: int = 3_000
    shock_take_fraction: float = 0.20
    shock_take_min_edge: int = 4
    news_emergency_dump_min_elapsed_ms: int = 250
    news_emergency_dump_ticks: int = 40
    news_emergency_dump_fraction: float = 0.20
    news_emergency_dump_min_inventory: int = 12
    news_max_hold_ms: int = 12_500
    news_decay_start_ms: int = 5_000
    news_decay_interval_ms: int = 500
    news_decay_fraction: float = 0.08
    news_decay_min_qty: int = 6
    news_decay_max_qty: int = 10
    news_decay_min_inventory: int = 40
    news_decay_min_residual_fraction: float = 0.10
    news_decay_stall_window_ms: int = 1_200
    news_decay_stall_threshold_ticks: int = 12
    news_overshoot_hold_ms: int = 200
    news_overshoot_max_wait_ms: int = 700
    news_overshoot_band_ticks: int = 10
    news_overshoot_reversal_ticks: int = 2
    news_overshoot_stage1_fraction: float = 0.30
    news_overshoot_stage2_fraction: float = 0.25
    news_overshoot_stage3_fraction: float = 0.20
    news_overshoot_stage_min_qty: int = 4
    news_overshoot_stage_max_qty: int = 16
    news_overshoot_min_residual_fraction: float = 0.30
    news_overshoot_large_position_threshold: int = 100
    news_overshoot_large_position_stage1_fraction: float = 0.50
    news_overshoot_large_position_residual_fraction: float = 0.50
    news_equilibrium_band_ticks: int = 8
    news_equilibrium_hold_ms: int = 1_400
    news_equilibrium_min_samples: int = 6
    news_equilibrium_min_elapsed_ms: int = 1_200
    news_equilibrium_residual_edge_ticks: int = 40
    news_equilibrium_min_capture_fraction: float = 0.55
    prejump_enabled: bool = False
    prejump_window_ms: int = 1_200
    prejump_low_threshold: float = 0.85
    prejump_high_threshold: float = 1.35
    prejump_max_position: int = 24
    prejump_quote_size: int = 6
    prejump_aggressive_edge: int = 2


@dataclass(frozen=True)
class BConfig:
    enabled: bool = True
    trading_enabled: bool = True
    observe_only: bool = False
    signal_snapshot_interval_ms: int = 250
    signal_change_threshold_ticks: int = 1
    quote_size: int = 1
    max_position: int = 8
    base_half_spread_ticks: int = 2
    inventory_skew_ticks_per_unit: float = 0.5
    passive_reduce_start: int = 4
    passive_reduce_full: int = 8
    min_book_spread: int = 6
    max_synthetic_dispersion: int = 4
    underlying_symbol: str = "B"
    option_symbols: tuple[str, ...] = (
        "B_C_950",
        "B_P_950",
        "B_C_1000",
        "B_P_1000",
        "B_C_1050",
        "B_P_1050",
    )


@dataclass(frozen=True)
class RiskConfig:
    reprice_cooldown_ms: int = 250
    passive_reprice_threshold_ticks: int = 2
    passive_quote_ttl_ms: int = 3_000

    @property
    def stale_quote_ms(self) -> int:
        """Refresh stale quotes periodically without thrashing the exchange."""
        return max(self.passive_quote_ttl_ms, self.reprice_cooldown_ms)


@dataclass(frozen=True)
class BotPaths:
    base_dir: Path
    journal_path: Path


@dataclass(frozen=True)
class TraceConfig:
    trace_enabled: bool = False
    trace_root: Path | None = None
    trace_snapshot_interval_ms: int = 500
    trace_book_depth_levels: int = 10
    trace_markout_windows_ms: tuple[int, ...] = (250, 1_000, 5_000)
    trace_write_summary_on_shutdown: bool = True


@dataclass(frozen=True)
class BotConfig:
    exchange: ExchangeConfig
    market_a: AConfig
    market_b: BConfig
    risk: RiskConfig
    paths: BotPaths
    trace: TraceConfig
    trading_enabled: bool = True
    trading_disabled_reason: str | None = None


def _required_value(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if default is None or str(default).strip() == "":
            raise ConfigError(f"Missing required environment variable: {name}")
        return str(default).strip()
    return value.strip()


def _optional_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if default is None:
            return None
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a float.") from exc


def _optional_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer.") from exc


def _optional_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Environment variable {name} must be a boolean.")


def _optional_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parts = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            parts.append(int(stripped))
        except ValueError as exc:
            raise ConfigError(f"Environment variable {name} must be a comma-separated list of integers.") from exc
    if not parts:
        raise ConfigError(f"Environment variable {name} must include at least one integer.")
    return tuple(parts)


def load_bot_config(
    base_dir: str | Path,
    *,
    default_host: str | None = None,
    default_username: str | None = None,
    default_password: str | None = None,
    default_initial_multiplier: float | None = None,
    default_initial_fair_value: int | None = None,
) -> BotConfig:
    """Load the exchange, valuation, and risk parameters from env vars or quick-start defaults."""
    base_path = Path(base_dir).resolve()
    host = _required_value("UTC_HOST", default_host)
    username = _required_value("UTC_USERNAME", default_username)
    password = _required_value("UTC_PASSWORD", default_password)

    journal_env = os.getenv("A_JOURNAL_PATH")
    if journal_env and journal_env.strip():
        journal_path = Path(journal_env).expanduser()
        if not journal_path.is_absolute():
            journal_path = (base_path / journal_path).resolve()
    else:
        journal_path = base_path / "runtime" / "a_bot_journal.jsonl"

    trace_root_env = os.getenv("TRACE_ROOT")
    if trace_root_env and trace_root_env.strip():
        trace_root = Path(trace_root_env).expanduser()
        if not trace_root.is_absolute():
            trace_root = (base_path / trace_root).resolve()
    else:
        trace_root = base_path / "analysis_runs"

    return BotConfig(
        exchange=ExchangeConfig(
            host=host,
            username=username,
            password=password,
        ),
        market_a=AConfig(
            initial_multiplier=_optional_float("A_INITIAL_MULTIPLIER", default=default_initial_multiplier),
            initial_fair_value=_optional_int("A_INITIAL_FAIR_VALUE", default_initial_fair_value),
            recover_pricing_state=_optional_bool("A_RECOVER_PRICING_STATE", False),
            total_position_limit=_optional_int("A_TOTAL_POSITION_LIMIT", 180) or 180,
            earnings_base_budget=_optional_int("A_EARNINGS_BASE_BUDGET", 120) or 120,
            mm_base_budget=_optional_int("A_MM_BASE_BUDGET", 60) or 60,
            earnings_shift_budget=_optional_int("A_EARNINGS_SHIFT_BUDGET", 180) or 180,
            mm_shift_budget=_optional_int("A_MM_SHIFT_BUDGET", 0) or 0,
            startup_assume_fresh_round=_optional_bool("A_STARTUP_ASSUME_FRESH_ROUND", True),
            pre_news_pullback_ms=_optional_int("A_PRE_NEWS_PULLBACK_MS", 4_000) or 4_000,
            calibration_min_delay_ms=_optional_int("A_CALIBRATION_MIN_DELAY_MS", 5_000) or 5_000,
            calibration_max_delay_ms=_optional_int("A_CALIBRATION_MAX_DELAY_MS", 20_000) or 20_000,
            calibration_sample_period_ms=_optional_int("A_CALIBRATION_SAMPLE_PERIOD_MS", 1_000) or 1_000,
            calibration_stability_band_ticks=_optional_int("A_CALIBRATION_STABILITY_BAND_TICKS", 8) or 8,
            calibration_tolerance_fraction=_optional_float("A_CALIBRATION_TOLERANCE_FRACTION", 0.10) or 0.10,
            calibration_min_tolerance_fraction=_optional_float("A_CALIBRATION_MIN_TOLERANCE_FRACTION", 0.03) or 0.03,
            candidate_confirmations=_optional_int("A_CANDIDATE_CONFIRMATIONS", 2) or 2,
            discovery_quote_size=_optional_int("A_DISCOVERY_QUOTE_SIZE", 2) or 2,
            discovery_max_position=_optional_int("A_DISCOVERY_MAX_POSITION", 4) or 4,
            discovery_half_spread_ticks=_optional_int("A_DISCOVERY_HALF_SPREAD_TICKS", 8) or 8,
            news_caution_duration_ms=_optional_int("A_NEWS_CAUTION_DURATION_MS", 12_000) or 12_000,
            news_caution_quote_size=_optional_int("A_NEWS_CAUTION_QUOTE_SIZE", 1) or 1,
            news_caution_max_position=_optional_int("A_NEWS_CAUTION_MAX_POSITION", 2) or 2,
            news_caution_half_spread_ticks=_optional_int("A_NEWS_CAUTION_HALF_SPREAD_TICKS", 8) or 8,
            news_light_offset_ticks=_optional_int("A_NEWS_LIGHT_OFFSET_TICKS", 12) or 12,
            news_medium_offset_ticks=_optional_int("A_NEWS_MEDIUM_OFFSET_TICKS", 24) or 24,
            news_strong_offset_ticks=_optional_int("A_NEWS_STRONG_OFFSET_TICKS", 48) or 48,
            news_extreme_offset_ticks=_optional_int("A_NEWS_EXTREME_OFFSET_TICKS", 80) or 80,
            news_very_extreme_offset_ticks=_optional_int("A_NEWS_VERY_EXTREME_OFFSET_TICKS", 120) or 120,
            news_light_position=_optional_int("A_NEWS_LIGHT_POSITION", 8) or 8,
            news_medium_position=_optional_int("A_NEWS_MEDIUM_POSITION", 36) or 36,
            news_strong_position=_optional_int("A_NEWS_STRONG_POSITION", 90) or 90,
            news_extreme_position=_optional_int("A_NEWS_EXTREME_POSITION", 130) or 130,
            news_very_extreme_position=_optional_int("A_NEWS_VERY_EXTREME_POSITION", 180) or 180,
            news_zero_position_threshold=_optional_int("A_NEWS_ZERO_POSITION_THRESHOLD", 3) or 3,
            news_confirmation_timeout_ms=_optional_int("A_NEWS_CONFIRMATION_TIMEOUT_MS", 900) or 900,
            news_confirmation_move_ticks=_optional_int("A_NEWS_CONFIRMATION_MOVE_TICKS", 3) or 3,
            freeze_multiplier_after_unstructured_news=_optional_bool("A_FREEZE_MULTIPLIER_AFTER_UNSTRUCTURED_NEWS", True),
            steady_half_spread_ticks=_optional_int("A_STEADY_HALF_SPREAD_TICKS", 1) or 1,
            steady_take_min_edge=_optional_int("A_STEADY_TAKE_MIN_EDGE", 2) or 2,
            steady_take_large_inventory_edge=_optional_int("A_STEADY_TAKE_LARGE_INVENTORY_EDGE", 4) or 4,
            opening_quote_size=_optional_int("A_OPENING_QUOTE_SIZE", 1) or 1,
            opening_max_position=_optional_int("A_OPENING_MAX_POSITION", 8) or 8,
            opening_half_spread_ticks=_optional_int("A_OPENING_HALF_SPREAD_TICKS", 4) or 4,
            opening_min_book_spread=_optional_int("A_OPENING_MIN_BOOK_SPREAD", 10) or 10,
            steady_quote_size=_optional_int("A_STEADY_QUOTE_SIZE", 3) or 3,
            steady_max_position=_optional_int("A_STEADY_MAX_POSITION", 32) or 32,
            steady_inventory_skew=_optional_float("A_STEADY_INVENTORY_SKEW", 0.75) or 0.75,
            steady_take_inventory_guard=_optional_int("A_STEADY_TAKE_INVENTORY_GUARD", 8) or 8,
            steady_passive_reduce_start=_optional_int("A_STEADY_PASSIVE_REDUCE_START", 8) or 8,
            steady_passive_reduce_full=_optional_int("A_STEADY_PASSIVE_REDUCE_FULL", 20) or 20,
            unwind_inventory_skew=_optional_float("A_UNWIND_INVENTORY_SKEW", 1.50) or 1.50,
            unwind_flatten_threshold=_optional_int("A_UNWIND_FLATTEN_THRESHOLD", 2) or 2,
            unwind_entry_position=_optional_int("A_UNWIND_ENTRY_POSITION", 24) or 24,
            unwind_exit_position=_optional_int("A_UNWIND_EXIT_POSITION", 12) or 12,
            unwind_aggressive_entry=_optional_int("A_UNWIND_AGGRESSIVE_ENTRY", 24) or 24,
            unwind_aggressive_exit=_optional_int("A_UNWIND_AGGRESSIVE_EXIT", 16) or 16,
            earnings_unwind_aggressive_entry=_optional_int("A_EARNINGS_UNWIND_AGGRESSIVE_ENTRY", 48) or 48,
            earnings_unwind_aggressive_exit=_optional_int("A_EARNINGS_UNWIND_AGGRESSIVE_EXIT", 24) or 24,
            earnings_unwind_passive_exit=_optional_int("A_EARNINGS_UNWIND_PASSIVE_EXIT", 8) or 8,
            earnings_unwind_passive_take_edge=_optional_int("A_EARNINGS_UNWIND_PASSIVE_TAKE_EDGE", 8) or 8,
            post_earnings_mm_cooldown_ms=_optional_int("A_POST_EARNINGS_MM_COOLDOWN_MS", 10_000) or 10_000,
            unwind_fast_entry=_optional_int("A_UNWIND_FAST_ENTRY", 36) or 36,
            unwind_fast_exit=_optional_int("A_UNWIND_FAST_EXIT", 12) or 12,
            unwind_fast_quote_size=_optional_int("A_UNWIND_FAST_QUOTE_SIZE", 12) or 12,
            shock_quote_size=_optional_int("A_SHOCK_QUOTE_SIZE", 15) or 15,
            shock_base_max_position=_optional_int("A_SHOCK_BASE_MAX_POSITION", 100) or 100,
            shock_shift_max_position=_optional_int("A_SHOCK_SHIFT_MAX_POSITION", 180) or 180,
            shock_window_ms=_optional_int("A_SHOCK_WINDOW_MS", 3_000) or 3_000,
            shock_take_fraction=_optional_float("A_SHOCK_TAKE_FRACTION", 0.20) or 0.20,
            shock_take_min_edge=_optional_int("A_SHOCK_TAKE_MIN_EDGE", 4) or 4,
            news_emergency_dump_min_elapsed_ms=_optional_int("A_NEWS_EMERGENCY_DUMP_MIN_ELAPSED_MS", 250) or 250,
            news_emergency_dump_ticks=_optional_int("A_NEWS_EMERGENCY_DUMP_TICKS", 40) or 40,
            news_emergency_dump_fraction=_optional_float("A_NEWS_EMERGENCY_DUMP_FRACTION", 0.20) or 0.20,
            news_emergency_dump_min_inventory=_optional_int("A_NEWS_EMERGENCY_DUMP_MIN_INVENTORY", 12) or 12,
            news_max_hold_ms=_optional_int("A_NEWS_MAX_HOLD_MS", 12_500) or 12_500,
            news_decay_start_ms=_optional_int("A_NEWS_DECAY_START_MS", 5_000) or 5_000,
            news_decay_interval_ms=_optional_int("A_NEWS_DECAY_INTERVAL_MS", 500) or 500,
            news_decay_fraction=_optional_float("A_NEWS_DECAY_FRACTION", 0.08) or 0.08,
            news_decay_min_qty=_optional_int("A_NEWS_DECAY_MIN_QTY", 6) or 6,
            news_decay_max_qty=_optional_int("A_NEWS_DECAY_MAX_QTY", 10) or 10,
            news_decay_min_inventory=_optional_int("A_NEWS_DECAY_MIN_INVENTORY", 40) or 40,
            news_decay_min_residual_fraction=_optional_float("A_NEWS_DECAY_MIN_RESIDUAL_FRACTION", 0.10) or 0.10,
            news_decay_stall_window_ms=_optional_int("A_NEWS_DECAY_STALL_WINDOW_MS", 1_200) or 1_200,
            news_decay_stall_threshold_ticks=_optional_int("A_NEWS_DECAY_STALL_THRESHOLD_TICKS", 12) or 12,
            news_overshoot_hold_ms=_optional_int("A_NEWS_OVERSHOOT_HOLD_MS", 200) or 200,
            news_overshoot_max_wait_ms=_optional_int("A_NEWS_OVERSHOOT_MAX_WAIT_MS", 700) or 700,
            news_overshoot_band_ticks=_optional_int("A_NEWS_OVERSHOOT_BAND_TICKS", 10) or 10,
            news_overshoot_reversal_ticks=_optional_int("A_NEWS_OVERSHOOT_REVERSAL_TICKS", 2) or 2,
            news_overshoot_stage1_fraction=_optional_float("A_NEWS_OVERSHOOT_STAGE1_FRACTION", 0.30) or 0.30,
            news_overshoot_stage2_fraction=_optional_float("A_NEWS_OVERSHOOT_STAGE2_FRACTION", 0.25) or 0.25,
            news_overshoot_stage3_fraction=_optional_float("A_NEWS_OVERSHOOT_STAGE3_FRACTION", 0.20) or 0.20,
            news_overshoot_stage_min_qty=_optional_int("A_NEWS_OVERSHOOT_STAGE_MIN_QTY", 4) or 4,
            news_overshoot_stage_max_qty=_optional_int("A_NEWS_OVERSHOOT_STAGE_MAX_QTY", 16) or 16,
            news_overshoot_min_residual_fraction=_optional_float("A_NEWS_OVERSHOOT_MIN_RESIDUAL_FRACTION", 0.30) or 0.30,
            news_overshoot_large_position_threshold=_optional_int("A_NEWS_OVERSHOOT_LARGE_POSITION_THRESHOLD", 100) or 100,
            news_overshoot_large_position_stage1_fraction=_optional_float("A_NEWS_OVERSHOOT_LARGE_POSITION_STAGE1_FRACTION", 0.50) or 0.50,
            news_overshoot_large_position_residual_fraction=_optional_float("A_NEWS_OVERSHOOT_LARGE_POSITION_RESIDUAL_FRACTION", 0.50) or 0.50,
            news_equilibrium_band_ticks=_optional_int("A_NEWS_EQUILIBRIUM_BAND_TICKS", 8) or 8,
            news_equilibrium_hold_ms=_optional_int("A_NEWS_EQUILIBRIUM_HOLD_MS", 1_400) or 1_400,
            news_equilibrium_min_samples=_optional_int("A_NEWS_EQUILIBRIUM_MIN_SAMPLES", 6) or 6,
            news_equilibrium_min_elapsed_ms=_optional_int("A_NEWS_EQUILIBRIUM_MIN_ELAPSED_MS", 1_200) or 1_200,
            news_equilibrium_residual_edge_ticks=_optional_int("A_NEWS_EQUILIBRIUM_RESIDUAL_EDGE_TICKS", 40) or 40,
            news_equilibrium_min_capture_fraction=_optional_float("A_NEWS_EQUILIBRIUM_MIN_CAPTURE_FRACTION", 0.55) or 0.55,
            prejump_enabled=_optional_bool("A_PREJUMP_ENABLED", False),
            prejump_window_ms=_optional_int("A_PREJUMP_WINDOW_MS", 1_200) or 1_200,
            prejump_low_threshold=_optional_float("A_PREJUMP_LOW_THRESHOLD", 0.85) or 0.85,
            prejump_high_threshold=_optional_float("A_PREJUMP_HIGH_THRESHOLD", 1.35) or 1.35,
            prejump_max_position=_optional_int("A_PREJUMP_MAX_POSITION", 24) or 24,
            prejump_quote_size=_optional_int("A_PREJUMP_QUOTE_SIZE", 6) or 6,
            prejump_aggressive_edge=_optional_int("A_PREJUMP_AGGRESSIVE_EDGE", 2) or 2,
        ),
        market_b=BConfig(
            enabled=_optional_bool("B_ENABLED", True),
            trading_enabled=_optional_bool("B_TRADING_ENABLED", True),
            observe_only=_optional_bool("B_OBSERVE_ONLY", False),
            signal_snapshot_interval_ms=_optional_int("B_SIGNAL_SNAPSHOT_INTERVAL_MS", 250) or 250,
            signal_change_threshold_ticks=_optional_int("B_SIGNAL_CHANGE_THRESHOLD_TICKS", 1) or 1,
            quote_size=_optional_int("B_QUOTE_SIZE", 1) or 1,
            max_position=_optional_int("B_MAX_POSITION", 8) or 8,
            base_half_spread_ticks=_optional_int("B_BASE_HALF_SPREAD_TICKS", 2) or 2,
            inventory_skew_ticks_per_unit=_optional_float("B_INVENTORY_SKEW_TICKS_PER_UNIT", 0.5) or 0.5,
            passive_reduce_start=_optional_int("B_PASSIVE_REDUCE_START", 4) or 4,
            passive_reduce_full=_optional_int("B_PASSIVE_REDUCE_FULL", 8) or 8,
            min_book_spread=_optional_int("B_MIN_BOOK_SPREAD", 6) or 6,
            max_synthetic_dispersion=_optional_int("B_MAX_SYNTHETIC_DISPERSION", 4) or 4,
        ),
        risk=RiskConfig(
            reprice_cooldown_ms=_optional_int("A_REPRICE_COOLDOWN_MS", 250) or 250,
            passive_reprice_threshold_ticks=_optional_int("A_PASSIVE_REPRICE_THRESHOLD_TICKS", 2) or 2,
            passive_quote_ttl_ms=_optional_int("A_PASSIVE_QUOTE_TTL_MS", 3_000) or 3_000,
        ),
        paths=BotPaths(
            base_dir=base_path,
            journal_path=journal_path,
        ),
        trace=TraceConfig(
            trace_enabled=_optional_bool("TRACE_ENABLED", False),
            trace_root=trace_root,
            trace_snapshot_interval_ms=_optional_int("TRACE_SNAPSHOT_INTERVAL_MS", 500) or 500,
            trace_book_depth_levels=_optional_int("TRACE_BOOK_DEPTH_LEVELS", 10) or 10,
            trace_markout_windows_ms=_optional_int_tuple("TRACE_MARKOUT_WINDOWS_MS", (250, 1_000, 5_000)),
            trace_write_summary_on_shutdown=_optional_bool("TRACE_WRITE_SUMMARY_ON_SHUTDOWN", True),
        ),
    )
