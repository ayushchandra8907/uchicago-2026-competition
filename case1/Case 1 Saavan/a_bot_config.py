from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal


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
    max_order_qty: int = 39
    earnings_base_budget: int = 120
    mm_base_budget: int = 60
    earnings_shift_budget: int = 180
    mm_shift_budget: int = 0
    startup_assume_fresh_round: bool = True
    pre_news_pullback_ms: int = 4_000
    pre_news_arrival_grace_ms: int = 1_200
    calibration_min_delay_ms: int = 5_000
    calibration_max_delay_ms: int = 20_000
    calibration_sample_period_ms: int = 1_000
    calibration_stability_band_ticks: int = 8
    calibration_tolerance_fraction: float = 0.10
    calibration_min_tolerance_fraction: float = 0.03
    candidate_confirmations: int = 2
    discovery_quote_size: int = 1
    discovery_max_position: int = 4
    discovery_half_spread_ticks: int = 8
    news_caution_quote_size: int = 1
    news_caution_max_position: int = 4
    news_caution_half_spread_ticks: int = 8
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
    earnings_unwind_quote_size: int = 3
    earnings_unwind_aggressive_quote_size: int = 3
    earnings_unwind_rapid_entry: int = 9_999
    earnings_unwind_rapid_exit: int = 9_998
    earnings_unwind_rapid_take_edge: int = 4
    earnings_unwind_aggressive_entry: int = 48
    earnings_unwind_aggressive_exit: int = 24
    earnings_unwind_passive_exit: int = 8
    earnings_unwind_passive_take_edge: int = 8
    shock_quote_size: int = 12
    shock_entry_window_ms: int = 1_000
    shock_entry_quote_size: int = 24
    shock_entry_min_edge: int = 2
    shock_entry_threshold_scale: float = 0.50
    shock_accumulate_target_position: int = 180
    shock_accumulate_min_quote_size: int = 12
    shock_accumulate_max_quote_size: int = 12
    shock_accumulate_min_edge: int = 4
    shock_accumulate_threshold_scale: float = 1.0
    shock_accumulate_window_ms: int = 3_000
    shock_base_max_position: int = 100
    shock_shift_max_position: int = 180
    shock_window_ms: int = 3_000
    shock_take_fraction: float = 0.25
    shock_take_min_edge: int = 4
    shock_settle_min_hold_ms: int = 1_200
    shock_settle_max_hold_ms: int = 4_000
    shock_settle_band_ticks: int = 8
    shock_settle_drift_ticks: int = 4
    shock_settle_confirmations: int = 2
    shock_unwind_quote_size: int = 3
    shock_unwind_aggressive_quote_size: int = 3
    shock_unwind_take_edge: int = 4
    shock_unwind_exit_position: int = 8
    prejump_enabled: bool = True
    prejump_window_ms: int = 1_200
    prejump_low_threshold: float = 0.85
    prejump_high_threshold: float = 1.35
    prejump_max_position: int = 24
    prejump_quote_size: int = 6
    prejump_aggressive_edge: int = 2


@dataclass(frozen=True)
class RiskConfig:
    reprice_cooldown_ms: int = 250
    passive_min_rest_ms: int = 0
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
    trace_detail_level: Literal["lite", "full"] = "lite"
    trace_snapshot_interval_ms: int = 500
    trace_book_depth_levels: int = 10
    trace_markout_windows_ms: tuple[int, ...] = (250, 1_000, 5_000)
    trace_write_summary_on_shutdown: bool = True


@dataclass(frozen=True)
class BotConfig:
    exchange: ExchangeConfig
    market_a: AConfig
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


def _optional_literal(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_csv = ", ".join(allowed)
        raise ConfigError(f"Environment variable {name} must be one of: {allowed_csv}.")
    return normalized


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
            max_order_qty=_optional_int("A_MAX_ORDER_QTY", 39) or 39,
            earnings_base_budget=_optional_int("A_EARNINGS_BASE_BUDGET", 120) or 120,
            mm_base_budget=_optional_int("A_MM_BASE_BUDGET", 60) or 60,
            earnings_shift_budget=_optional_int("A_EARNINGS_SHIFT_BUDGET", 180) or 180,
            mm_shift_budget=_optional_int("A_MM_SHIFT_BUDGET", 0) or 0,
            startup_assume_fresh_round=_optional_bool("A_STARTUP_ASSUME_FRESH_ROUND", True),
            pre_news_pullback_ms=_optional_int("A_PRE_NEWS_PULLBACK_MS", 4_000) or 4_000,
            pre_news_arrival_grace_ms=_optional_int("A_PRE_NEWS_ARRIVAL_GRACE_MS", 1_200) or 1_200,
            calibration_min_delay_ms=_optional_int("A_CALIBRATION_MIN_DELAY_MS", 5_000) or 5_000,
            calibration_max_delay_ms=_optional_int("A_CALIBRATION_MAX_DELAY_MS", 20_000) or 20_000,
            calibration_sample_period_ms=_optional_int("A_CALIBRATION_SAMPLE_PERIOD_MS", 1_000) or 1_000,
            calibration_stability_band_ticks=_optional_int("A_CALIBRATION_STABILITY_BAND_TICKS", 8) or 8,
            calibration_tolerance_fraction=_optional_float("A_CALIBRATION_TOLERANCE_FRACTION", 0.10) or 0.10,
            calibration_min_tolerance_fraction=_optional_float("A_CALIBRATION_MIN_TOLERANCE_FRACTION", 0.03) or 0.03,
            candidate_confirmations=_optional_int("A_CANDIDATE_CONFIRMATIONS", 2) or 2,
            discovery_quote_size=_optional_int("A_DISCOVERY_QUOTE_SIZE", 1) or 1,
            discovery_max_position=_optional_int("A_DISCOVERY_MAX_POSITION", 4) or 4,
            discovery_half_spread_ticks=_optional_int("A_DISCOVERY_HALF_SPREAD_TICKS", 8) or 8,
            news_caution_quote_size=_optional_int("A_NEWS_CAUTION_QUOTE_SIZE", 1) or 1,
            news_caution_max_position=_optional_int("A_NEWS_CAUTION_MAX_POSITION", 4) or 4,
            news_caution_half_spread_ticks=_optional_int("A_NEWS_CAUTION_HALF_SPREAD_TICKS", 8) or 8,
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
            earnings_unwind_quote_size=_optional_int("A_EARNINGS_UNWIND_QUOTE_SIZE", 3) or 3,
            earnings_unwind_aggressive_quote_size=_optional_int("A_EARNINGS_UNWIND_AGGRESSIVE_QUOTE_SIZE", 3) or 3,
            earnings_unwind_rapid_entry=_optional_int("A_EARNINGS_UNWIND_RAPID_ENTRY", 9_999) or 9_999,
            earnings_unwind_rapid_exit=_optional_int("A_EARNINGS_UNWIND_RAPID_EXIT", 9_998) or 9_998,
            earnings_unwind_rapid_take_edge=_optional_int("A_EARNINGS_UNWIND_RAPID_TAKE_EDGE", 4) or 4,
            earnings_unwind_aggressive_entry=_optional_int("A_EARNINGS_UNWIND_AGGRESSIVE_ENTRY", 48) or 48,
            earnings_unwind_aggressive_exit=_optional_int("A_EARNINGS_UNWIND_AGGRESSIVE_EXIT", 24) or 24,
            earnings_unwind_passive_exit=_optional_int("A_EARNINGS_UNWIND_PASSIVE_EXIT", 8) or 8,
            earnings_unwind_passive_take_edge=_optional_int("A_EARNINGS_UNWIND_PASSIVE_TAKE_EDGE", 8) or 8,
            shock_quote_size=_optional_int("A_SHOCK_QUOTE_SIZE", 12) or 12,
            shock_entry_window_ms=_optional_int("A_SHOCK_ENTRY_WINDOW_MS", 1_000) or 1_000,
            shock_entry_quote_size=_optional_int("A_SHOCK_ENTRY_QUOTE_SIZE", 24) or 24,
            shock_entry_min_edge=_optional_int("A_SHOCK_ENTRY_MIN_EDGE", 2) or 2,
            shock_entry_threshold_scale=_optional_float("A_SHOCK_ENTRY_THRESHOLD_SCALE", 0.50) or 0.50,
            shock_accumulate_target_position=_optional_int("A_SHOCK_ACCUMULATE_TARGET_POSITION", 180) or 180,
            shock_accumulate_min_quote_size=_optional_int("A_SHOCK_ACCUMULATE_MIN_QUOTE_SIZE", 12) or 12,
            shock_accumulate_max_quote_size=_optional_int("A_SHOCK_ACCUMULATE_MAX_QUOTE_SIZE", 12) or 12,
            shock_accumulate_min_edge=_optional_int("A_SHOCK_ACCUMULATE_MIN_EDGE", 4) or 4,
            shock_accumulate_threshold_scale=_optional_float("A_SHOCK_ACCUMULATE_THRESHOLD_SCALE", 1.0) or 1.0,
            shock_accumulate_window_ms=_optional_int("A_SHOCK_ACCUMULATE_WINDOW_MS", 3_000) or 3_000,
            shock_base_max_position=_optional_int("A_SHOCK_BASE_MAX_POSITION", 100) or 100,
            shock_shift_max_position=_optional_int("A_SHOCK_SHIFT_MAX_POSITION", 180) or 180,
            shock_window_ms=_optional_int("A_SHOCK_WINDOW_MS", 3_000) or 3_000,
            shock_take_fraction=_optional_float("A_SHOCK_TAKE_FRACTION", 0.25) or 0.25,
            shock_take_min_edge=_optional_int("A_SHOCK_TAKE_MIN_EDGE", 4) or 4,
            shock_settle_min_hold_ms=_optional_int("A_SHOCK_SETTLE_MIN_HOLD_MS", 1_200) or 1_200,
            shock_settle_max_hold_ms=_optional_int("A_SHOCK_SETTLE_MAX_HOLD_MS", 4_000) or 4_000,
            shock_settle_band_ticks=_optional_int("A_SHOCK_SETTLE_BAND_TICKS", 8) or 8,
            shock_settle_drift_ticks=_optional_int("A_SHOCK_SETTLE_DRIFT_TICKS", 4) or 4,
            shock_settle_confirmations=_optional_int("A_SHOCK_SETTLE_CONFIRMATIONS", 2) or 2,
            shock_unwind_quote_size=_optional_int("A_SHOCK_UNWIND_QUOTE_SIZE", 3) or 3,
            shock_unwind_aggressive_quote_size=_optional_int("A_SHOCK_UNWIND_AGGRESSIVE_QUOTE_SIZE", 3) or 3,
            shock_unwind_take_edge=_optional_int("A_SHOCK_UNWIND_TAKE_EDGE", 4) or 4,
            shock_unwind_exit_position=_optional_int("A_SHOCK_UNWIND_EXIT_POSITION", 8) or 8,
            prejump_enabled=_optional_bool("A_PREJUMP_ENABLED", True),
            prejump_window_ms=_optional_int("A_PREJUMP_WINDOW_MS", 1_200) or 1_200,
            prejump_low_threshold=_optional_float("A_PREJUMP_LOW_THRESHOLD", 0.85) or 0.85,
            prejump_high_threshold=_optional_float("A_PREJUMP_HIGH_THRESHOLD", 1.35) or 1.35,
            prejump_max_position=_optional_int("A_PREJUMP_MAX_POSITION", 24) or 24,
            prejump_quote_size=_optional_int("A_PREJUMP_QUOTE_SIZE", 6) or 6,
            prejump_aggressive_edge=_optional_int("A_PREJUMP_AGGRESSIVE_EDGE", 2) or 2,
        ),
        risk=RiskConfig(
            reprice_cooldown_ms=_optional_int("A_REPRICE_COOLDOWN_MS", 250) or 250,
            passive_min_rest_ms=_optional_int("A_PASSIVE_MIN_REST_MS", 0) or 0,
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
            trace_detail_level=_optional_literal("TRACE_DETAIL_LEVEL", "lite", ("lite", "full")),  # type: ignore[arg-type]
            trace_snapshot_interval_ms=_optional_int("TRACE_SNAPSHOT_INTERVAL_MS", 500) or 500,
            trace_book_depth_levels=_optional_int("TRACE_BOOK_DEPTH_LEVELS", 10) or 10,
            trace_markout_windows_ms=_optional_int_tuple("TRACE_MARKOUT_WINDOWS_MS", (250, 1_000, 5_000)),
            trace_write_summary_on_shutdown=_optional_bool("TRACE_WRITE_SUMMARY_ON_SHUTDOWN", True),
        ),
    )
