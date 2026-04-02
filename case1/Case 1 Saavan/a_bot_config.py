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
    pe_ratio: float | None
    initial_fair_value: int | None = None
    pe_learning_delay_ms: int = 1_500
    pe_learning_sample_window_ms: int = 750
    pe_learning_min_samples: int = 3
    pe_learning_min_confidence: int = 2
    pe_learning_consistency_tolerance: float = 0.15
    pe_replacement_confirmations: int = 2


@dataclass(frozen=True)
class RiskConfig:
    max_position: int = 80
    quote_size: int = 4
    min_edge: int = 2
    take_edge: int = 4
    inventory_skew: float = 0.35
    reprice_cooldown_ms: int = 750

    @property
    def stale_quote_ms(self) -> int:
        """Reuse the cooldown knob as the stale-quote trigger, but a bit wider."""
        return max(self.reprice_cooldown_ms * 3, 1_500)


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
    trading_enabled: bool
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


def load_bot_config(
    base_dir: str | Path,
    *,
    default_host: str | None = None,
    default_username: str | None = None,
    default_password: str | None = None,
    default_pe_ratio: float | None = None,
    default_initial_fair_value: int | None = None,
) -> BotConfig:
    """Load the exchange, valuation, and risk parameters from env vars or quick-start defaults."""
    base_path = Path(base_dir).resolve()
    host = _required_value("UTC_HOST", default_host)
    username = _required_value("UTC_USERNAME", default_username)
    password = _required_value("UTC_PASSWORD", default_password)
    pe_ratio = _optional_float("A_PE_RATIO", default=default_pe_ratio)
    trading_enabled = pe_ratio is not None
    trading_disabled_reason = None
    if not trading_enabled:
        trading_disabled_reason = (
            "A_PE_RATIO is not set, so the bot will connect and learn A's P/E from earnings before trading."
        )

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
            pe_ratio=pe_ratio,
            initial_fair_value=_optional_int("A_INITIAL_FAIR_VALUE", default_initial_fair_value),
            pe_learning_delay_ms=_optional_int("A_PE_LEARNING_DELAY_MS", 1_500) or 1_500,
            pe_learning_sample_window_ms=_optional_int("A_PE_SAMPLE_WINDOW_MS", 750) or 750,
            pe_learning_min_samples=_optional_int("A_PE_LEARNING_MIN_SAMPLES", 3) or 3,
            pe_learning_min_confidence=_optional_int("A_PE_MIN_CONFIDENCE", 2) or 2,
            pe_learning_consistency_tolerance=_optional_float("A_PE_TOLERANCE", 0.15) or 0.15,
            pe_replacement_confirmations=_optional_int("A_PE_REPLACEMENT_CONFIRMATIONS", 2) or 2,
        ),
        risk=RiskConfig(
            max_position=_optional_int("A_MAX_POSITION", 80) or 80,
            quote_size=_optional_int("A_QUOTE_SIZE", 4) or 4,
            min_edge=_optional_int("A_MIN_EDGE", 2) or 2,
            take_edge=_optional_int("A_TAKE_EDGE", 4) or 4,
            inventory_skew=_optional_float("A_INVENTORY_SKEW", 0.35),
            reprice_cooldown_ms=_optional_int("A_REPRICE_COOLDOWN_MS", 750) or 750,
        ),
        paths=BotPaths(
            base_dir=base_path,
            journal_path=journal_path,
        ),
        trading_enabled=trading_enabled,
        trading_disabled_reason=trading_disabled_reason,
    )
