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
    earnings_shock_entry_guard_ticks: int = 24
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
    allow_trading_with_ayush_port: bool = True
    mm_v2_enabled: bool = True
    meanrev_enabled: bool = True
    signal_snapshot_interval_ms: int = 250
    signal_change_threshold_ticks: int = 1
    quote_size: int = 1
    max_position: int = 4
    base_half_spread_ticks: int = 2
    inventory_skew_ticks_per_unit: float = 0.5
    passive_reduce_start: int = 3
    passive_reduce_full: int = 4
    min_book_spread: int = 6
    max_synthetic_dispersion: int = 4
    mm_v2_book_weight: float = 0.70
    mm_v2_synth_weight: float = 0.30
    mm_v2_min_half_spread_ticks: int = 1
    mm_v2_inside_improve_ticks: int = 1
    mm_v2_dispersion_widen_factor: float = 0.50
    mm_v2_reduce_size_bonus: int = 1
    mm_min_eval_interval_ms: int = 150
    mm_reprice_threshold_ticks: int = 3
    mm_min_valid_spread_ticks: int = 3
    mm_min_healthy_book_age_ms: int = 500
    mm_cancel_on_bad_book: bool = True
    mm_bad_fill_cooldown_ms: int = 750
    meanrev_max_position: int = 16
    meanrev_quote_size: int = 2
    meanrev_ema_fast_ms: int = 30_000
    meanrev_ema_slow_ms: int = 180_000
    meanrev_vol_ewma_ms: int = 60_000
    meanrev_sigma_floor: float = 4.0
    meanrev_entry_z: float = 1.25
    meanrev_entry_z2: float = 2.25
    meanrev_exit_z: float = 0.35
    meanrev_stop_z: float = 5.0
    meanrev_min_spread_ticks: int = 3
    meanrev_max_hold_ms: int = 120_000
    meanrev_cooldown_ms: int = 1_500
    meanrev_aggressive_entry_z: float = 2.75
    meanrev_aggressive_exit: bool = True
    meanrev_entry_ticks: int = 10
    meanrev_full_entry_ticks: int = 15
    meanrev_exit_ticks: int = 3
    meanrev_base_target: int = 6
    meanrev_full_target: int = 16
    meanrev_extreme_entry_ticks: int = 20
    meanrev_risk_off_deviation_ticks: int = 35
    meanrev_turn_confirm_ms: int = 300
    meanrev_min_healthy_book_age_ms: int = 500
    meanrev_bad_fill_cooldown_ms: int = 1_000
    basis_entry_threshold_ticks: float = 1.25
    basis_strong_threshold_ticks: float = 2.5
    imbalance_confirmation_threshold: float = 0.15
    far_side_widen_ticks: int = 4
    parity_enabled: bool = False
    parity_shadow_enabled: bool = True
    parity_edge_threshold_ticks: int = 8
    parity_trade_size: int = 1
    parity_max_exposure: int = 3
    parity_max_quote_age_ms: int = 1_000
    option_lottery_enabled: bool = False
    option_lottery_max_ask: int = 3
    option_lottery_floor_ask: int = 0
    option_lottery_quote_size: int = 1
    option_lottery_max_position_per_symbol: int = 20
    option_lottery_wing_max_position: int = 200
    option_lottery_atm_max_position: int = 40
    option_lottery_total_premium_budget: int = 1_500
    option_lottery_wing_premium_budget: int = 600
    option_lottery_c1050_premium_budget: int = 0
    option_lottery_p950_premium_budget: int = 0
    option_lottery_atm_total_premium_budget: int = 300
    option_lottery_near_strike_ticks: int = 80
    option_lottery_min_momentum_ticks: float = 1.0
    option_lottery_rebuy_cooldown_ms: int = 1_000
    option_lottery_profit_take_enabled: bool = True
    option_lottery_profit_take_min_edge: int = 6
    option_lottery_profit_take_multiple: float = 2.0
    option_lottery_profit_take_quote_size: int = 20
    option_lottery_stale_hold_ms: int = 45_000
    option_hedge_enabled: bool = False
    option_hedge_max_ask: int = 6
    option_hedge_min_underlying_inventory: int = 4
    option_hedge_target_ratio: float = 0.5
    option_hedge_premium_budget: int = 300
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
class MarketCConfig:
    enabled: bool = True
    trading_enabled: bool = True
    symbol: str = "C"
    live_earnings_enabled: bool = True
    live_cpi_enabled: bool = False
    live_macro_enabled: bool = False
    mm_enabled: bool = False
    min_eval_interval_ms: int = 100
    pm_symbols: tuple[str, ...] = ("R_HIKE", "R_HOLD", "R_CUT")


@dataclass(frozen=True)
class ETFConfig:
    enabled: bool = True
    trading_enabled: bool = True
    symbol: str = "ETF"
    alpha_from_a: float = 0.60
    alpha_from_a_earnings: float | None = None
    alpha_from_a_news: float | None = None
    alpha_max: float = 1.0
    alpha_step: float = 0.05
    max_position: int = 100
    quote_size: int = 24
    target_position_per_etf_tick: float = 1.0
    target_position_per_a_shock_inventory: float = 0.35
    min_a_fair_shift_ticks: int = 20
    min_projected_edge_ticks: int = 3
    exit_band_ticks: int = 2
    min_hold_ms: int = 3_000
    max_hold_ms: int = 12_500
    major_a_shock_fair_shift_ticks: int = 60
    major_a_shock_target_inventory: int = 150
    min_target_position_for_major_a_shock: int = 60
    min_book_spread_ticks: int = 1
    reprice_cooldown_ms: int = 100
    reprice_threshold_ticks: int = 2
    min_eval_interval_ms: int = 100
    unwind_reprice_threshold_ticks: int = 8
    entry_retry_window_ms: int = 2_500
    entry_force_aggressive_ms: int = 250
    entry_retry_reprice_ms: int = 125
    churn_window_ms: int = 250
    churn_max_top_of_book_updates: int = 25
    churn_resume_stable_ms: int = 500
    enable_c_earnings: bool = True
    alpha_from_c_earnings: float = 0.35
    min_c_fair_shift_ticks: int = 18
    ac_conflict_policy: str = "suppress"


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
    trace_snapshot_interval_ms: int = 2_000
    trace_book_depth_levels: int = 1
    trace_markout_windows_ms: tuple[int, ...] = (250, 1_000, 5_000)
    trace_write_summary_on_shutdown: bool = True
    trace_record_book_updates: bool = False
    trace_record_observe_only_decisions: bool = False


@dataclass(frozen=True)
class BotConfig:
    exchange: ExchangeConfig
    market_a: AConfig
    market_b: BConfig
    market_c: MarketCConfig
    etf: ETFConfig
    risk: RiskConfig
    paths: BotPaths
    trace: TraceConfig
    a_strategy_mode: str = "ayush_port"
    auto_stop_after_round_complete: bool = True
    assumed_round_duration_ms: int = 900_000
    round_completion_grace_ms: int = 5_000
    auto_stop_on_followup_position_snapshot: bool = False
    auto_stop_on_market_resolved: bool = False
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
            earnings_shock_entry_guard_ticks=_optional_int("A_EARNINGS_SHOCK_ENTRY_GUARD_TICKS", 24) or 24,
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
            allow_trading_with_ayush_port=_optional_bool("B_ALLOW_TRADING_WITH_AYUSH_PORT", True),
            mm_v2_enabled=_optional_bool("B_MM_V2_ENABLED", True),
            meanrev_enabled=_optional_bool("B_MEANREV_ENABLED", True),
            signal_snapshot_interval_ms=_optional_int("B_SIGNAL_SNAPSHOT_INTERVAL_MS", 250) or 250,
            signal_change_threshold_ticks=_optional_int("B_SIGNAL_CHANGE_THRESHOLD_TICKS", 1) or 1,
            quote_size=_optional_int(
                "B_MM_V2_QUOTE_SIZE",
                _optional_int("B_QUOTE_SIZE", 1) or 1,
            ) or 1,
            max_position=_optional_int(
                "B_MM_V2_MAX_POSITION",
                _optional_int("B_MAX_POSITION", 4) or 4,
            ) or 4,
            base_half_spread_ticks=_optional_int("B_BASE_HALF_SPREAD_TICKS", 2) or 2,
            inventory_skew_ticks_per_unit=_optional_float("B_INVENTORY_SKEW_TICKS_PER_UNIT", 0.5) or 0.5,
            passive_reduce_start=_optional_int("B_PASSIVE_REDUCE_START", 3) or 3,
            passive_reduce_full=_optional_int("B_PASSIVE_REDUCE_FULL", 4) or 4,
            min_book_spread=_optional_int("B_MIN_BOOK_SPREAD", 6) or 6,
            max_synthetic_dispersion=_optional_int("B_MAX_SYNTHETIC_DISPERSION", 4) or 4,
            mm_v2_book_weight=_optional_float("B_MM_V2_BOOK_WEIGHT", 0.70) or 0.70,
            mm_v2_synth_weight=_optional_float("B_MM_V2_SYNTH_WEIGHT", 0.30) or 0.30,
            mm_v2_min_half_spread_ticks=_optional_int("B_MM_V2_MIN_HALF_SPREAD_TICKS", 1) or 1,
            mm_v2_inside_improve_ticks=_optional_int("B_MM_V2_INSIDE_IMPROVE_TICKS", 1) or 1,
            mm_v2_dispersion_widen_factor=_optional_float("B_MM_V2_DISPERSION_WIDEN_FACTOR", 0.50) or 0.50,
            mm_v2_reduce_size_bonus=_optional_int("B_MM_V2_REDUCE_SIZE_BONUS", 1) or 1,
            mm_min_eval_interval_ms=_optional_int("B_MM_MIN_EVAL_INTERVAL_MS", 150) or 150,
            mm_reprice_threshold_ticks=_optional_int("B_MM_REPRICE_THRESHOLD_TICKS", 3) or 3,
            mm_min_valid_spread_ticks=_optional_int("B_MM_MIN_VALID_SPREAD_TICKS", 3) or 3,
            mm_min_healthy_book_age_ms=_optional_int("B_MM_MIN_HEALTHY_BOOK_AGE_MS", 500) or 500,
            mm_cancel_on_bad_book=_optional_bool("B_MM_CANCEL_ON_BAD_BOOK", True),
            mm_bad_fill_cooldown_ms=_optional_int("B_MM_BAD_FILL_COOLDOWN_MS", 750) or 750,
            meanrev_max_position=_optional_int("B_MEANREV_MAX_POSITION", 16) or 16,
            meanrev_quote_size=_optional_int("B_MEANREV_QUOTE_SIZE", 2) or 2,
            meanrev_ema_fast_ms=_optional_int("B_MEANREV_EMA_FAST_MS", 30_000) or 30_000,
            meanrev_ema_slow_ms=_optional_int("B_MEANREV_EMA_SLOW_MS", 180_000) or 180_000,
            meanrev_vol_ewma_ms=_optional_int("B_MEANREV_VOL_EWMA_MS", 60_000) or 60_000,
            meanrev_sigma_floor=_optional_float("B_MEANREV_SIGMA_FLOOR", 4.0) or 4.0,
            meanrev_entry_z=_optional_float("B_MEANREV_ENTRY_Z", 1.25) or 1.25,
            meanrev_entry_z2=_optional_float("B_MEANREV_ENTRY_Z2", 2.25) or 2.25,
            meanrev_exit_z=_optional_float("B_MEANREV_EXIT_Z", 0.35) or 0.35,
            meanrev_stop_z=_optional_float("B_MEANREV_STOP_Z", 5.0) or 5.0,
            meanrev_min_spread_ticks=_optional_int("B_MEANREV_MIN_SPREAD_TICKS", 3) or 3,
            meanrev_max_hold_ms=_optional_int("B_MEANREV_MAX_HOLD_MS", 120_000) or 120_000,
            meanrev_cooldown_ms=_optional_int("B_MEANREV_COOLDOWN_MS", 1_500) or 1_500,
            meanrev_aggressive_entry_z=_optional_float("B_MEANREV_AGGRESSIVE_ENTRY_Z", 2.75) or 2.75,
            meanrev_aggressive_exit=_optional_bool("B_MEANREV_AGGRESSIVE_EXIT", True),
            meanrev_entry_ticks=_optional_int("B_MEANREV_ENTRY_TICKS", 10) or 10,
            meanrev_full_entry_ticks=_optional_int("B_MEANREV_FULL_ENTRY_TICKS", 15) or 15,
            meanrev_exit_ticks=_optional_int("B_MEANREV_EXIT_TICKS", 3) or 3,
            meanrev_base_target=_optional_int("B_MEANREV_BASE_TARGET", 6) or 6,
            meanrev_full_target=_optional_int("B_MEANREV_FULL_TARGET", 16) or 16,
            meanrev_extreme_entry_ticks=_optional_int("B_MEANREV_EXTREME_ENTRY_TICKS", 20) or 20,
            meanrev_risk_off_deviation_ticks=_optional_int("B_MEANREV_RISK_OFF_DEVIATION_TICKS", 35) or 35,
            meanrev_turn_confirm_ms=_optional_int("B_MEANREV_TURN_CONFIRM_MS", 300) or 300,
            meanrev_min_healthy_book_age_ms=_optional_int("B_MEANREV_MIN_HEALTHY_BOOK_AGE_MS", 500) or 500,
            meanrev_bad_fill_cooldown_ms=_optional_int("B_MEANREV_BAD_FILL_COOLDOWN_MS", 1_000) or 1_000,
            basis_entry_threshold_ticks=_optional_float("B_BASIS_ENTRY_THRESHOLD_TICKS", 1.25) or 1.25,
            basis_strong_threshold_ticks=_optional_float("B_BASIS_STRONG_THRESHOLD_TICKS", 2.5) or 2.5,
            imbalance_confirmation_threshold=_optional_float("B_IMBALANCE_CONFIRMATION_THRESHOLD", 0.15) or 0.15,
            far_side_widen_ticks=_optional_int("B_FAR_SIDE_WIDEN_TICKS", 4) or 4,
            parity_enabled=_optional_bool("B_PARITY_ENABLED", False),
            parity_shadow_enabled=_optional_bool("B_PARITY_SHADOW_ENABLED", True),
            parity_edge_threshold_ticks=_optional_int("B_PARITY_EDGE_THRESHOLD_TICKS", 8) or 8,
            parity_trade_size=_optional_int("B_PARITY_TRADE_SIZE", 1) or 1,
            parity_max_exposure=_optional_int("B_PARITY_MAX_EXPOSURE", 3) or 3,
            parity_max_quote_age_ms=_optional_int("B_PARITY_MAX_QUOTE_AGE_MS", 1_000) or 1_000,
            option_lottery_enabled=_optional_bool("B_OPTION_LOTTERY_ENABLED", False),
            option_lottery_max_ask=_optional_int("B_OPTION_LOTTERY_MAX_ASK", 3) or 3,
            option_lottery_floor_ask=_optional_int("B_OPTION_LOTTERY_FLOOR_ASK", 0) or 0,
            option_lottery_quote_size=_optional_int("B_OPTION_LOTTERY_QUOTE_SIZE", 1) or 1,
            option_lottery_max_position_per_symbol=_optional_int("B_OPTION_LOTTERY_MAX_POSITION_PER_SYMBOL", 20) or 20,
            option_lottery_wing_max_position=_optional_int("B_OPTION_LOTTERY_WING_MAX_POSITION", 200) or 200,
            option_lottery_atm_max_position=_optional_int("B_OPTION_LOTTERY_ATM_MAX_POSITION", 40) or 40,
            option_lottery_total_premium_budget=_optional_int("B_OPTION_LOTTERY_TOTAL_PREMIUM_BUDGET", 1_500) or 1_500,
            option_lottery_wing_premium_budget=_optional_int("B_OPTION_LOTTERY_WING_PREMIUM_BUDGET", 600) or 600,
            option_lottery_c1050_premium_budget=_optional_int("B_OPTION_C1050_PREMIUM_BUDGET", 0),
            option_lottery_p950_premium_budget=_optional_int("B_OPTION_P950_PREMIUM_BUDGET", 0),
            option_lottery_atm_total_premium_budget=_optional_int("B_OPTION_LOTTERY_ATM_TOTAL_PREMIUM_BUDGET", 300) or 300,
            option_lottery_near_strike_ticks=_optional_int("B_OPTION_LOTTERY_NEAR_STRIKE_TICKS", 80) or 80,
            option_lottery_min_momentum_ticks=_optional_float("B_OPTION_LOTTERY_MIN_MOMENTUM_TICKS", 1.0) or 1.0,
            option_lottery_rebuy_cooldown_ms=_optional_int("B_OPTION_LOTTERY_REBUY_COOLDOWN_MS", 1_000) or 1_000,
            option_lottery_profit_take_enabled=_optional_bool("B_OPTION_LOTTERY_PROFIT_TAKE_ENABLED", True),
            option_lottery_profit_take_min_edge=_optional_int("B_OPTION_LOTTERY_PROFIT_TAKE_MIN_EDGE", 6) or 6,
            option_lottery_profit_take_multiple=_optional_float("B_OPTION_LOTTERY_PROFIT_TAKE_MULTIPLE", 2.0) or 2.0,
            option_lottery_profit_take_quote_size=_optional_int("B_OPTION_LOTTERY_PROFIT_TAKE_QUOTE_SIZE", 20) or 20,
            option_lottery_stale_hold_ms=_optional_int("B_OPTION_LOTTERY_STALE_HOLD_MS", 45_000) or 45_000,
            option_hedge_enabled=_optional_bool("B_OPTION_HEDGE_ENABLED", False),
            option_hedge_max_ask=_optional_int("B_OPTION_HEDGE_MAX_ASK", 6) or 6,
            option_hedge_min_underlying_inventory=_optional_int("B_OPTION_HEDGE_MIN_UNDERLYING_INVENTORY", 4) or 4,
            option_hedge_target_ratio=_optional_float("B_OPTION_HEDGE_TARGET_RATIO", 0.5) or 0.5,
            option_hedge_premium_budget=_optional_int("B_OPTION_HEDGE_PREMIUM_BUDGET", 300) or 300,
        ),
        market_c=MarketCConfig(
            enabled=_optional_bool("C_ENABLED", True),
            trading_enabled=_optional_bool("C_TRADING_ENABLED", True),
            symbol=str(os.getenv("C_SYMBOL", "C") or "C").strip(),
            live_earnings_enabled=_optional_bool("C_LIVE_EARNINGS_ENABLED", True),
            live_cpi_enabled=_optional_bool("C_LIVE_CPI_ENABLED", False),
            live_macro_enabled=_optional_bool("C_LIVE_MACRO_ENABLED", False),
            mm_enabled=_optional_bool("C_MM_ENABLED", False),
            min_eval_interval_ms=_optional_int("C_MIN_EVAL_INTERVAL_MS", 100) or 100,
        ),
        etf=ETFConfig(
            enabled=_optional_bool("ETF_ENABLED", True),
            trading_enabled=_optional_bool("ETF_TRADING_ENABLED", True),
            symbol=str(os.getenv("ETF_SYMBOL", "ETF") or "ETF").strip(),
            alpha_from_a=min(
                _optional_float("ETF_ALPHA_MAX", 1.0) or 1.0,
                max(0.0, _optional_float("ETF_ALPHA_FROM_A", 0.60) or 0.60),
            ),
            alpha_from_a_earnings=_optional_float("ETF_ALPHA_FROM_A_EARNINGS", None),
            alpha_from_a_news=_optional_float("ETF_ALPHA_FROM_A_NEWS", None),
            alpha_max=_optional_float("ETF_ALPHA_MAX", 1.0) or 1.0,
            alpha_step=_optional_float("ETF_ALPHA_STEP", 0.05) or 0.05,
            max_position=_optional_int("ETF_MAX_POSITION", 100) or 100,
            quote_size=_optional_int("ETF_QUOTE_SIZE", 24) or 24,
            target_position_per_etf_tick=_optional_float("ETF_TARGET_POSITION_PER_TICK", 1.0) or 1.0,
            target_position_per_a_shock_inventory=(
                _optional_float("ETF_TARGET_POSITION_PER_A_SHOCK_INVENTORY", 0.35) or 0.35
            ),
            min_a_fair_shift_ticks=_optional_int("ETF_MIN_A_FAIR_SHIFT_TICKS", 20) or 20,
            min_projected_edge_ticks=_optional_int("ETF_MIN_PROJECTED_EDGE_TICKS", 3) or 3,
            exit_band_ticks=_optional_int("ETF_EXIT_BAND_TICKS", 2) or 2,
            min_hold_ms=_optional_int("ETF_MIN_HOLD_MS", 3_000) or 3_000,
            max_hold_ms=_optional_int("ETF_MAX_HOLD_MS", 12_500) or 12_500,
            major_a_shock_fair_shift_ticks=_optional_int("ETF_MAJOR_A_SHOCK_FAIR_SHIFT_TICKS", 60) or 60,
            major_a_shock_target_inventory=_optional_int("ETF_MAJOR_A_SHOCK_TARGET_INVENTORY", 150) or 150,
            min_target_position_for_major_a_shock=_optional_int("ETF_MIN_TARGET_POSITION_FOR_MAJOR_A_SHOCK", 60) or 60,
            min_book_spread_ticks=_optional_int("ETF_MIN_BOOK_SPREAD_TICKS", 1) or 1,
            reprice_cooldown_ms=_optional_int("ETF_REPRICE_COOLDOWN_MS", 100) or 100,
            reprice_threshold_ticks=_optional_int("ETF_REPRICE_THRESHOLD_TICKS", 2) or 2,
            min_eval_interval_ms=_optional_int("ETF_MIN_EVAL_INTERVAL_MS", 100) or 100,
            unwind_reprice_threshold_ticks=_optional_int("ETF_UNWIND_REPRICE_THRESHOLD_TICKS", 8) or 8,
            entry_retry_window_ms=_optional_int("ETF_ENTRY_RETRY_WINDOW_MS", 2_500) or 2_500,
            entry_force_aggressive_ms=_optional_int("ETF_ENTRY_FORCE_AGGRESSIVE_MS", 250) or 250,
            entry_retry_reprice_ms=_optional_int("ETF_ENTRY_RETRY_REPRICE_MS", 125) or 125,
            churn_window_ms=_optional_int("ETF_CHURN_WINDOW_MS", 250) or 250,
            churn_max_top_of_book_updates=_optional_int("ETF_CHURN_MAX_TOP_OF_BOOK_UPDATES", 25) or 25,
            churn_resume_stable_ms=_optional_int("ETF_CHURN_RESUME_STABLE_MS", 500) or 500,
            enable_c_earnings=_optional_bool("ETF_ENABLE_C_EARNINGS", True),
            alpha_from_c_earnings=_optional_float("ETF_ALPHA_FROM_C_EARNINGS", 0.35) or 0.35,
            min_c_fair_shift_ticks=_optional_int("ETF_MIN_C_FAIR_SHIFT_TICKS", 18) or 18,
            ac_conflict_policy=str(os.getenv("ETF_AC_CONFLICT_POLICY", "suppress") or "suppress").strip().lower(),
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
            trace_snapshot_interval_ms=_optional_int("TRACE_SNAPSHOT_INTERVAL_MS", 2_000) or 2_000,
            trace_book_depth_levels=_optional_int("TRACE_BOOK_DEPTH_LEVELS", 1) or 1,
            trace_markout_windows_ms=_optional_int_tuple("TRACE_MARKOUT_WINDOWS_MS", (250, 1_000, 5_000)),
            trace_write_summary_on_shutdown=_optional_bool("TRACE_WRITE_SUMMARY_ON_SHUTDOWN", True),
            trace_record_book_updates=_optional_bool("TRACE_RECORD_BOOK_UPDATES", False),
            trace_record_observe_only_decisions=_optional_bool("TRACE_RECORD_OBSERVE_ONLY_DECISIONS", False),
        ),
        a_strategy_mode=str(os.getenv("A_STRATEGY_MODE", "ayush_port") or "ayush_port").strip().lower(),
        auto_stop_after_round_complete=_optional_bool("AUTO_STOP_AFTER_ROUND_COMPLETE", True),
        assumed_round_duration_ms=_optional_int("ASSUMED_ROUND_DURATION_MS", 900_000) or 900_000,
        round_completion_grace_ms=_optional_int("ROUND_COMPLETION_GRACE_MS", 5_000) or 5_000,
        auto_stop_on_followup_position_snapshot=_optional_bool("AUTO_STOP_ON_FOLLOWUP_POSITION_SNAPSHOT", False),
        auto_stop_on_market_resolved=_optional_bool("AUTO_STOP_ON_MARKET_RESOLVED", False),
    )
