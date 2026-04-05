from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from case1.ayush_work.marketA_v1.data_loader import build_run_catalog


class DataLoaderTests(unittest.TestCase):
    def test_build_run_catalog_supports_legacy_and_new_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            self._write_legacy_run(data_root / "legacy_run")
            self._write_new_run(data_root / "new_run")

            catalog = build_run_catalog(data_root)

            self.assertEqual(len(catalog.sessions), 2)
            session_ids = {session.session_id for session in catalog.sessions}
            self.assertEqual(session_ids, {"legacy_run", "new_run"})
            legacy = next(session for session in catalog.sessions if session.session_id == "legacy_run")
            new = next(session for session in catalog.sessions if session.session_id == "new_run")

            self.assertEqual(legacy.source_layout, "legacy_derived")
            self.assertEqual(new.source_layout, "new_raw")
            self.assertGreaterEqual(legacy.diagnostics["earnings_event_count"], 1)
            self.assertGreaterEqual(new.diagnostics["book_event_count"], 2)

    def _write_legacy_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_book_events.csv").write_text(
            "\n".join(
                [
                    "event_index,event_type,monotonic_ns,symbol,best_bid_px,best_bid_qty,best_ask_px,best_ask_qty,mid_px,spread,bid_levels_json,ask_levels_json",
                    '1,book_update,1000000,A,99,10,101,10,100,2,"[{""px"":99,""qty"":10}]","[{""px"":101,""qty"":10}]"',
                    '2,book_update,2000000,A,100,10,102,10,101,2,"[{""px"":100,""qty"":10}]","[{""px"":102,""qty"":10}]"',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "raw_trade_events.csv").write_text(
            "\n".join(
                [
                    "event_index,monotonic_ns,symbol,trade_px,trade_qty",
                    "3,2500000,A,100,10",
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "raw_news_events.csv").write_text(
            "\n".join(
                [
                    "news_id,monotonic_ns,exchange_tick,kind,symbol,structured_subtype,earnings_value",
                    "legacy-news-1,1500000,10,structured,A,earnings,1.0",
                ]
            ),
            encoding="utf-8",
        )

    def _write_new_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_book_snapshots_A.csv").write_text(
            "\n".join(
                [
                    "message_index,symbol,bids_json,asks_json",
                    '10,A,"[{""px"":99,""qty"":20}]","[{""px"":101,""qty"":20}]"',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "raw_book_updates_A.csv").write_text(
            "\n".join(
                [
                    "message_index,symbol,side,px,dq",
                    "11,A,BUY,100,10",
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "raw_trade_events_A.csv").write_text(
            "\n".join(
                [
                    "message_index,symbol,price,qty",
                    "12,A,100,10",
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "raw_news_events.csv").write_text(
            "\n".join(
                [
                    "message_index,tick,tick_ms,kind,symbol,structured_subtype,earnings_asset,earnings_value",
                    "9,5,1000,structured,A,earnings,A,1.0",
                    "13,6,1200,structured,A,earnings,A,1.1",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
