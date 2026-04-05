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
    pe_ratio: float = 10.0
    price_scale: int = 100
    initial_fair_value: int | None = None
    startup_assume_fresh_round: bool = True
    pre_news_pullback_ms: int = 4_000
    steady_half_spread_ticks: int = 1
    opening_quote_size: int = 1
    opening_max_position: int = 8
    opening_half_spread_ticks: int = 4
    opening_min_book_spread: int = 10
    steady_quote_size: int = 2
    steady_max_position: int = 24
    steady_inventory_skew: float = 0.75
    unwind_inventory_skew: float = 1.50
    unwind_flatten_threshold: int = 2
    shock_quote_size: int = 12
    shock_max_position: int = 80
    shock_window_ms: int = 3_000
    shock_take_fraction: float = 0.25
    shock_take_min_edge: int = 4


@dataclass(frozen=True)
class RiskConfig:
    reprice_cooldown_ms: int = 250

    @property
    def stale_quote_ms(self) -> int:
        """Refresh stale quotes periodically without thrashing the exchange."""
        return max(self.reprice_cooldown_ms * 4, 1_000)


@dataclass(frozen=True)
class BotPaths:
    base_dir: Path
    journal_path: Path


@dataclass(frozen=True)
class BotConfig:
    exchange: ExchangeConfig
    market_a: AConfig
    risk: RiskConfig
    paths: BotPaths
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


def load_bot_config(
    base_dir: str | Path,
    *,
    default_host: str | None = None,
    default_username: str | None = None,
    default_password: str | None = None,
    default_pe_ratio: float | None = 10.0,
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

    return BotConfig(
        exchange=ExchangeConfig(
            host=host,
            username=username,
            password=password,
        ),
        market_a=AConfig(
            pe_ratio=_optional_float("A_PE_RATIO", default=default_pe_ratio) or 10.0,
            price_scale=_optional_int("A_PRICE_SCALE", 100) or 100,
            initial_fair_value=_optional_int("A_INITIAL_FAIR_VALUE", default_initial_fair_value),
            startup_assume_fresh_round=_optional_bool("A_STARTUP_ASSUME_FRESH_ROUND", True),
            pre_news_pullback_ms=_optional_int("A_PRE_NEWS_PULLBACK_MS", 4_000) or 4_000,
            steady_half_spread_ticks=_optional_int("A_STEADY_HALF_SPREAD_TICKS", 1) or 1,
            opening_quote_size=_optional_int("A_OPENING_QUOTE_SIZE", 1) or 1,
            opening_max_position=_optional_int("A_OPENING_MAX_POSITION", 8) or 8,
            opening_half_spread_ticks=_optional_int("A_OPENING_HALF_SPREAD_TICKS", 4) or 4,
            opening_min_book_spread=_optional_int("A_OPENING_MIN_BOOK_SPREAD", 10) or 10,
            steady_quote_size=_optional_int("A_STEADY_QUOTE_SIZE", 2) or 2,
            steady_max_position=_optional_int("A_STEADY_MAX_POSITION", 24) or 24,
            steady_inventory_skew=_optional_float("A_STEADY_INVENTORY_SKEW", 0.75) or 0.75,
            unwind_inventory_skew=_optional_float("A_UNWIND_INVENTORY_SKEW", 1.50) or 1.50,
            unwind_flatten_threshold=_optional_int("A_UNWIND_FLATTEN_THRESHOLD", 2) or 2,
            shock_quote_size=_optional_int("A_SHOCK_QUOTE_SIZE", 12) or 12,
            shock_max_position=_optional_int("A_SHOCK_MAX_POSITION", 80) or 80,
            shock_window_ms=_optional_int("A_SHOCK_WINDOW_MS", 3_000) or 3_000,
            shock_take_fraction=_optional_float("A_SHOCK_TAKE_FRACTION", 0.25) or 0.25,
            shock_take_min_edge=_optional_int("A_SHOCK_TAKE_MIN_EDGE", 4) or 4,
        ),
        risk=RiskConfig(
            reprice_cooldown_ms=_optional_int("A_REPRICE_COOLDOWN_MS", 250) or 250,
        ),
        paths=BotPaths(
            base_dir=base_path,
            journal_path=journal_path,
        ),
    )
