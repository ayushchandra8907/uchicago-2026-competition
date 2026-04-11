from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, MutableMapping

from testing1 import LOGGER, MarketABot, load_runtime_config


REQUIRED_ENV_KEYS = ("UTC_HOST", "UTC_USERNAME", "UTC_PASSWORD")
COMPETITION_ENV_DEFAULTS: dict[str, str] = {
    "TRACE_ENABLED": "0",
    "TRACE_WRITE_SUMMARY_ON_SHUTDOWN": "0",
    "TRACE_RECORD_BOOK_UPDATES": "0",
    "TRACE_RECORD_OBSERVE_ONLY_DECISIONS": "0",
    "BOT_DISCONNECT_ALERT_ENABLED": "1",
    "BOT_DISCONNECT_ALERT_SOUND": "/System/Library/Sounds/Basso.aiff",
    "PM_GUARD_ENABLED": "1",
}


def validate_required_env(environ: Mapping[str, str] | None = None) -> None:
    env = environ or os.environ
    missing = [key for key in REQUIRED_ENV_KEYS if not str(env.get(key, "")).strip()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required competition environment variable(s): {joined}")


def apply_competition_env_defaults(environ: MutableMapping[str, str] | None = None) -> None:
    env = environ or os.environ
    for key, value in COMPETITION_ENV_DEFAULTS.items():
        env.setdefault(key, value)


def pm_guard_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = environ or os.environ
    raw = str(env.get("PM_GUARD_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def build_pm_guard_command(python_executable: str | None = None) -> list[str]:
    script_path = Path(__file__).resolve().parents[1] / "caleb_work" / "run_pm_guard.py"
    interpreter = python_executable or sys.executable
    return [interpreter, str(script_path)]


def _emit_launcher_alert(reason: str, details: str | None = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] WARNING competition launcher alert: {reason}"
    if details:
        message = f"{message} | {details}"
    try:
        print(message, file=sys.stderr, flush=True)
        print("\a", end="", file=sys.stderr, flush=True)
    except Exception:
        LOGGER.exception("Failed to emit competition launcher alert.")

    afplay = shutil.which("afplay")
    sound_path = (os.getenv("BOT_DISCONNECT_ALERT_SOUND") or "/System/Library/Sounds/Basso.aiff").strip()
    if not afplay or not sound_path:
        return
    sound_file = Path(sound_path)
    if not sound_file.exists():
        return
    try:
        subprocess.Popen(
            [afplay, str(sound_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        LOGGER.exception("Failed to play competition launcher alert sound.")


async def _start_pm_guard_if_enabled(environ: Mapping[str, str]) -> asyncio.subprocess.Process | None:
    if not pm_guard_enabled(environ):
        LOGGER.info("Competition launcher starting without PM guard because PM_GUARD_ENABLED=0.")
        return None

    command = build_pm_guard_command()
    LOGGER.info("Starting PM guard companion process: %s", command[-1])
    return await asyncio.create_subprocess_exec(
        *command,
        env=dict(environ),
    )


async def _terminate_pm_guard(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def main() -> None:
    validate_required_env()
    apply_competition_env_defaults()
    config = load_runtime_config(default_host=None, default_username=None, default_password=None)

    LOGGER.info("Starting competition runtime against %s", config.exchange.host)
    LOGGER.info("Journal path: %s", config.paths.journal_path)

    pm_guard_process = await _start_pm_guard_if_enabled(os.environ)
    bot = MarketABot(config)
    bot_task = asyncio.create_task(bot.start())
    pm_wait_task = asyncio.create_task(pm_guard_process.wait()) if pm_guard_process is not None else None

    try:
        if pm_wait_task is None:
            await bot_task
            return

        done, _ = await asyncio.wait({bot_task, pm_wait_task}, return_when=asyncio.FIRST_COMPLETED)
        if pm_wait_task in done and not bot_task.done():
            exit_code = pm_wait_task.result()
            LOGGER.error("PM guard exited unexpectedly with code %s while main bot is still running.", exit_code)
            _emit_launcher_alert("pm_guard_exited", details=f"exit_code={exit_code}")
            await bot_task
            return
        await bot_task
    finally:
        await _terminate_pm_guard(pm_guard_process)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down competition runtime.")
