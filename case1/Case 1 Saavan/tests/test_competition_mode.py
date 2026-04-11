from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
CASE1_DIR = BOT_DIR.parent
CALEB_DIR = CASE1_DIR / "caleb_work"
for path in (BOT_DIR, CALEB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_competition
import run_pm_guard
from testing1 import MarketABot


class CompetitionModeTests(unittest.TestCase):
    def test_competition_wrapper_requires_explicit_exchange_env(self) -> None:
        with self.assertRaises(SystemExit):
            run_competition.validate_required_env({})

    def test_competition_wrapper_applies_safe_runtime_defaults(self) -> None:
        env = {
            "UTC_HOST": "host",
            "UTC_USERNAME": "user",
            "UTC_PASSWORD": "pass",
        }

        run_competition.apply_competition_env_defaults(env)

        self.assertEqual(env["TRACE_ENABLED"], "0")
        self.assertEqual(env["TRACE_WRITE_SUMMARY_ON_SHUTDOWN"], "0")
        self.assertEqual(env["TRACE_RECORD_BOOK_UPDATES"], "0")
        self.assertEqual(env["TRACE_RECORD_OBSERVE_ONLY_DECISIONS"], "0")
        self.assertEqual(env["BOT_DISCONNECT_ALERT_ENABLED"], "1")
        self.assertEqual(env["PM_GUARD_ENABLED"], "1")

    def test_competition_wrapper_builds_pm_guard_command(self) -> None:
        command = run_competition.build_pm_guard_command("/tmp/test-python")

        self.assertEqual(command[0], "/tmp/test-python")
        self.assertTrue(command[1].endswith("case1/caleb_work/run_pm_guard.py"))

    def test_disconnect_alert_suppressed_for_clean_auto_stop(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        bot._auto_stop_requested = True

        with mock.patch("testing1.shutil.which", return_value="/usr/bin/afplay"), mock.patch(
            "testing1.subprocess.Popen"
        ) as popen_mock, mock.patch("sys.stderr", new=io.StringIO()) as stderr:
            bot._emit_disconnect_alert("eof", "ignored")

        self.assertEqual(stderr.getvalue(), "")
        popen_mock.assert_not_called()

    def test_disconnect_alert_uses_bell_and_sound_when_enabled(self) -> None:
        bot = MarketABot.__new__(MarketABot)
        bot._auto_stop_requested = False
        temp_sound = Path(tempfile.gettempdir()) / "codex_disconnect_test.aiff"
        temp_sound.write_bytes(b"sound")
        self.addCleanup(lambda: temp_sound.unlink(missing_ok=True))

        env = {
            "BOT_DISCONNECT_ALERT_ENABLED": "1",
            "BOT_DISCONNECT_ALERT_SOUND": str(temp_sound),
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch(
            "testing1.shutil.which", return_value="/usr/bin/afplay"
        ), mock.patch("testing1.subprocess.Popen") as popen_mock, mock.patch(
            "sys.stderr", new=io.StringIO()
        ) as stderr:
            bot._emit_disconnect_alert("grpc:UNAVAILABLE", "stream reset")

        self.assertIn("unexpected bot disconnect", stderr.getvalue())
        self.assertIn("\a", stderr.getvalue())
        popen_mock.assert_called_once()

    def test_pm_guard_wrapper_requires_explicit_exchange_env(self) -> None:
        with self.assertRaises(SystemExit):
            run_pm_guard.validate_required_env({})

    def test_pm_guard_wrapper_sets_defaults_but_preserves_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {
                "UTC_HOST": "host",
                "UTC_USERNAME": "user",
                "UTC_PASSWORD": "pass",
                "PM4_HYBRID_GUARD_REARM_SEC": "2.0",
                "PM3_TRACE_DIR": temp_dir,
            }

            run_pm_guard.apply_pm_guard_env_defaults(env)

            self.assertEqual(env["PM4_HYBRID_GUARD_ENABLED"], "1")
            self.assertEqual(env["PM4_HYBRID_GUARD_REARM_SEC"], "2.0")
            self.assertEqual(env["PM3_TRACE_DIR"], temp_dir)
            self.assertTrue(Path(temp_dir).exists())

    def test_pm_guard_wrapper_loads_pm_main(self) -> None:
        pm_main = run_pm_guard.load_pm_main()
        self.assertTrue(callable(pm_main))
