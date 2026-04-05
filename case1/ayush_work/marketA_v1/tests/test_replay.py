from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import pandas as pd

from case1.ayush_work.marketA_v1.backtest.replay_engine import run_session_backtest
from case1.ayush_work.marketA_v1.config import StrategyParameters, build_app_config
from case1.ayush_work.marketA_v1.models import BookState, MarketEvent, SessionData, TradeState


class ReplayTests(unittest.TestCase):
    def test_replay_executes_conservative_passive_fill(self) -> None:
        config = build_app_config(
            strategy=replace(
                StrategyParameters(),
                min_half_spread_px=1,
                base_half_spread_px=1,
                market_spread_weight=0.0,
                vol_widening=0.0,
                toxicity_widening=0.0,
                fill_widening=0.0,
                inventory_penalty=0.0,
                aggressive_edge_px=50,
                normal_order_size=10,
            )
        )
        session = SessionData(
            session_id="replay-test",
            path=Path("."),
            source_layout="test",
            events=(
                MarketEvent(
                    kind="book",
                    session_id="replay-test",
                    seq=1,
                    time_ms=0.0,
                    book=BookState(best_bid_px=99, best_bid_qty=10, best_ask_px=101, best_ask_qty=10, bid_levels=((99, 10),), ask_levels=((101, 10),)),
                ),
                MarketEvent(
                    kind="trade",
                    session_id="replay-test",
                    seq=2,
                    time_ms=100.0,
                    trade=TradeState(price_px=99, qty=15),
                ),
                MarketEvent(
                    kind="book",
                    session_id="replay-test",
                    seq=3,
                    time_ms=200.0,
                    book=BookState(best_bid_px=99, best_bid_qty=5, best_ask_px=101, best_ask_qty=10, bid_levels=((99, 5),), ask_levels=((101, 10),)),
                ),
            ),
            book_rows=pd.DataFrame(),
            trade_rows=pd.DataFrame(),
            news_rows=pd.DataFrame(),
            diagnostics={},
        )

        artifacts = run_session_backtest(session, config)

        self.assertGreaterEqual(artifacts.result.passive_fill_count, 1)
        self.assertGreater(artifacts.result.final_inventory, 0)


if __name__ == "__main__":
    unittest.main()
