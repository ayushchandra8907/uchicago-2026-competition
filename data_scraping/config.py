from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "local_config.json"
DEFAULT_LOG_ROOT = MODULE_DIR / "data"
DEFAULT_MONITORED_SYMBOLS = ("A", "B", "C", "ETF")
DEFAULT_PLOT_SYMBOLS = ("A", "B", "C", "ETF")
DEFAULT_DIRECT_EARNINGS_SYMBOLS = ("A", "C")
DEFAULT_ETF_NEWS_ASSETS = ("A", "C")
DEFAULT_PE_CONSTANTS = {"A": 10.0}
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1.0
DEFAULT_TOP_K_DEPTH = 3
DEFAULT_TOP_N_LEVELS = 5
DEFAULT_POST_NEWS_WINDOW_SECONDS = 5.0
DEFAULT_HISTORY_MAXLEN = 20_000


@dataclass
class ResearchLoggerConfig:
    host: str
    username: str
    password: str
    monitored_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_MONITORED_SYMBOLS))
    plot_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_PLOT_SYMBOLS))
    direct_earnings_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_DIRECT_EARNINGS_SYMBOLS))
    etf_news_assets: list[str] = field(default_factory=lambda: list(DEFAULT_ETF_NEWS_ASSETS))
    pe_constants: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PE_CONSTANTS))
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    top_k_depth: int = DEFAULT_TOP_K_DEPTH
    top_n_levels: int = DEFAULT_TOP_N_LEVELS
    post_news_window_seconds: float = DEFAULT_POST_NEWS_WINDOW_SECONDS
    history_maxlen: int = DEFAULT_HISTORY_MAXLEN
    log_root: Path = DEFAULT_LOG_ROOT
    run_label: str | None = None
    config_path: Path | None = None

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["log_root"] = str(self.log_root)
        payload["config_path"] = str(self.config_path) if self.config_path else None
        return payload


def _load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in config file {path}")
    return data


def _resolve_path_value(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_list(json_data: dict[str, Any], key: str, default: tuple[str, ...]) -> list[str]:
    value = json_data.get(key)
    if value is None:
        return list(default)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a JSON array of strings in {DEFAULT_CONFIG_PATH}")
    return list(value)


def _read_float_dict(json_data: dict[str, Any], key: str, default: dict[str, float]) -> dict[str, float]:
    value = json_data.get(key)
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object in {DEFAULT_CONFIG_PATH}")
    converted: dict[str, float] = {}
    for dict_key, dict_value in value.items():
        if not isinstance(dict_key, str):
            raise ValueError(f"{key} must use string keys in {DEFAULT_CONFIG_PATH}")
        converted[dict_key] = float(dict_value)
    return converted


def load_config() -> ResearchLoggerConfig:
    config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Edit data_scraping/local_config.json and run again."
        )

    json_data = _load_json_config(config_path)
    config_base_dir = config_path.parent

    host = json_data.get("host")
    username = json_data.get("username")
    password = json_data.get("password")
    if not host or not username or not password:
        raise ValueError(
            f"host, username, and password must be filled in inside {DEFAULT_CONFIG_PATH} before running the logger."
        )

    log_root_value = json_data.get("log_root", str(DEFAULT_LOG_ROOT))
    log_root = _resolve_path_value(log_root_value, base_dir=config_base_dir)

    return ResearchLoggerConfig(
        host=str(host),
        username=str(username),
        password=str(password),
        monitored_symbols=_read_list(json_data, "monitored_symbols", DEFAULT_MONITORED_SYMBOLS),
        plot_symbols=_read_list(json_data, "plot_symbols", DEFAULT_PLOT_SYMBOLS),
        direct_earnings_symbols=_read_list(json_data, "direct_earnings_symbols", DEFAULT_DIRECT_EARNINGS_SYMBOLS),
        etf_news_assets=_read_list(json_data, "etf_news_assets", DEFAULT_ETF_NEWS_ASSETS),
        pe_constants=_read_float_dict(json_data, "pe_constants", DEFAULT_PE_CONSTANTS),
        heartbeat_interval_seconds=float(json_data.get("heartbeat_interval_seconds", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)),
        top_k_depth=int(json_data.get("top_k_depth", DEFAULT_TOP_K_DEPTH)),
        top_n_levels=int(json_data.get("top_n_levels", DEFAULT_TOP_N_LEVELS)),
        post_news_window_seconds=float(json_data.get("post_news_window_seconds", DEFAULT_POST_NEWS_WINDOW_SECONDS)),
        history_maxlen=int(json_data.get("history_maxlen", DEFAULT_HISTORY_MAXLEN)),
        log_root=log_root,
        run_label=json_data.get("run_label"),
        config_path=config_path,
    )
