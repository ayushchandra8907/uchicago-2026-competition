from __future__ import annotations

import asyncio
import os

import run_competition
from run_competition import LOGGER


UPGRADED_ENV_DEFAULTS: dict[str, str] = {
    "ETF_ENABLE_C_EARNINGS": "0",
    "A_EARNINGS_TRAP_ENABLED": "1",
    "A_EARNINGS_TRAP_MIN_FAIR_SHIFT_TICKS": "60",
    "A_EARNINGS_TRAP_MIN_ALIGNED_INVENTORY": "40",
    "A_EARNINGS_TRAP_MAX_QTY": "20",
    "A_EARNINGS_TRAP_INVENTORY_RESERVE": "20",
    "A_EARNINGS_TRAP_OFFSET_FRACTION": "0.10",
    "A_EARNINGS_TRAP_OFFSET_MIN_TICKS": "60",
    "A_EARNINGS_TRAP_OFFSET_MAX_TICKS": "120",
    "A_EARNINGS_TRAP_MAX_LIFETIME_MS": "1500",
}


def apply_upgraded_env_defaults(environ=None) -> None:
    env = os.environ if environ is None else environ
    run_competition.apply_competition_env_defaults(env)
    for key, value in UPGRADED_ENV_DEFAULTS.items():
        env.setdefault(key, value)


async def main() -> None:
    run_competition.validate_required_env()
    apply_upgraded_env_defaults()
    LOGGER.info("UPGRADED PRACTICE RUNNER: ETF A-only mode ON")
    LOGGER.info("UPGRADED PRACTICE RUNNER: A earnings trap overlay ON")
    await run_competition.main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down upgraded competition runtime.")
