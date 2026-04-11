from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Mapping, MutableMapping


REQUIRED_ENV_KEYS = ("UTC_HOST", "UTC_USERNAME", "UTC_PASSWORD")
PM_GUARD_DEFAULTS: dict[str, str] = {
    "PM4_HYBRID_GUARD_ENABLED": "1",
    "PM4_HYBRID_PAUSE_AFTER_FLATTEN": "0",
    "PM4_HYBRID_GUARD_REARM_SEC": "0.75",
    "PM4_ADVERSE_ONLY_FLATTEN_ENABLED": "1",
    "PM4_ADVERSE_FLATTEN_CONSTANT": "0",
    "PM4_HANDOFF_CPI_ABS": "0.0003",
    "PM4_HANDOFF_FEDSPEAK_MIN_ABS_BIAS_BP": "0.75",
    "PM4_HANDOFF_SKIP_LOW_PRICE_ENABLED": "1",
    "PM4_HANDOFF_SKIP_LOW_PRICE_PX": "100",
    "PM4_HANDOFF_SKIP_HIGH_PRICE_ENABLED": "1",
    "PM4_HANDOFF_SKIP_HIGH_PRICE_PX": "900",
    "PM4_HANDOFF_DING_ENABLED": "1",
    "PM3_MAX_ORDER_SIZE": "40",
    "PM3_MAX_OPEN_ORDERS": "50",
    "PM3_MAX_OUTSTANDING_VOLUME": "120",
    "PM3_MAX_ABS_POSITION": "200",
    "PM3_NEAR_FLAT_THRESHOLD": "2",
    "PM3_TRACE_ENABLED": "1",
}


def validate_required_env(environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    missing = [key for key in REQUIRED_ENV_KEYS if not str(env.get(key, "")).strip()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required competition environment variable(s): {joined}")


def apply_pm_guard_env_defaults(environ: MutableMapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    for key, value in PM_GUARD_DEFAULTS.items():
        env.setdefault(key, value)
    trace_dir = Path(env.get("PM3_TRACE_DIR") or (Path(__file__).resolve().parent / "logs"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("PM3_TRACE_DIR", str(trace_dir))


def load_pm_main():
    try:
        from new_PM4_fedspeak import main as pm_main
    except ModuleNotFoundError:
        from case1.caleb_work.new_PM4_fedspeak import main as pm_main
    return pm_main


async def main() -> None:
    validate_required_env()
    apply_pm_guard_env_defaults()
    pm_main = load_pm_main()
    await pm_main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
