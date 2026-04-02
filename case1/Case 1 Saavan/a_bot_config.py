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
    pe_ratio: float
    initial_fair_value: int | None = None


@dataclass(frozen=True)
class RiskConfig:
    max_position: int = 40
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


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return value.strip()


def _optional_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
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


def load_bot_config(base_dir: str | Path) -> BotConfig:
    """Load the exchange, valuation, and risk parameters from environment variables."""
    base_path = Path(base_dir).resolve()
    host = _required_env("UTC_HOST")
    username = _required_env("UTC_USERNAME")
    password = _required_env("UTC_PASSWORD")
    pe_ratio = _optional_float("A_PE_RATIO", default=float("nan"))
    if pe_ratio != pe_ratio:
        raise ConfigError("Environment variable A_PE_RATIO is required for market A trading.")

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
            initial_fair_value=_optional_int("A_INITIAL_FAIR_VALUE"),
        ),
        risk=RiskConfig(
            max_position=_optional_int("A_MAX_POSITION", 40) or 40,
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
    )
