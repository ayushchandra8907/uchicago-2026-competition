from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from case1.ayush_work.marketA_v1.fair_value import FairValueModel, fit_pe_ratio
from case1.ayush_work.marketA_v1.models import SessionData


class FairValueTests(unittest.TestCase):
    def test_fair_value_model_uses_pe_and_scale(self) -> None:
        model = FairValueModel(pe_ratio=10.0, price_scale=100)
        fair_px = model.update_earnings(1.1)
        self.assertEqual(fair_px, 1100)
        self.assertEqual(model.reference_fair_px(None), 1100)

    def test_fit_pe_ratio_uses_post_earnings_window(self) -> None:
        session = SessionData(
            session_id="synthetic",
            path=Path("."),
            source_layout="test",
            events=(),
            book_rows=pd.DataFrame(
                [
                    {"time_ms": 0.0, "mid_px": 1000.0},
                    {"time_ms": 2200.0, "mid_px": 1100.0},
                    {"time_ms": 3000.0, "mid_px": 1110.0},
                    {"time_ms": 4200.0, "mid_px": 1090.0},
                ]
            ),
            trade_rows=pd.DataFrame(),
            news_rows=pd.DataFrame(
                [
                    {
                        "kind": "structured",
                        "structured_subtype": "earnings",
                        "earnings_asset": "A",
                        "earnings_value": 1.1,
                        "time_ms": 100.0,
                        "source_seq": 1,
                    }
                ]
            ),
            diagnostics={},
        )

        result = fit_pe_ratio((session,), price_scale=100)

        self.assertAlmostEqual(result.pe_ratio, 10.0, places=1)
        self.assertEqual(len(result.observations), 1)


if __name__ == "__main__":
    unittest.main()
