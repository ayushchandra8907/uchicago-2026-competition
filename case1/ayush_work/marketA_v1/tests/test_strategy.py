from __future__ import annotations

from dataclasses import replace
import unittest

from case1.ayush_work.marketA_v1.config import StrategyParameters, build_app_config
from case1.ayush_work.marketA_v1.market_maker import StrategyEngine
from case1.ayush_work.marketA_v1.models import BookState, MarketEvent, NewsState


class StrategyEngineTests(unittest.TestCase):
    def test_normal_mode_quotes_two_sided_around_mid_before_earnings(self) -> None:
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
            )
        )
        engine = StrategyEngine(config)
        intent = engine.on_event(
            MarketEvent(
                kind="book",
                session_id="test",
                seq=1,
                time_ms=0.0,
                book=BookState(best_bid_px=99, best_bid_qty=20, best_ask_px=101, best_ask_qty=20),
            )
        )

        self.assertIsNotNone(intent)
        self.assertEqual(intent.mode, "NORMAL_MM")
        self.assertEqual(intent.bid.px, 99)
        self.assertEqual(intent.ask.px, 101)

    def test_earnings_shock_mode_crosses_stale_ask(self) -> None:
        config = build_app_config(
            strategy=replace(
                StrategyParameters(),
                shock_aggressive_edge_px=2,
                aggressive_edge_px=50,
                min_half_spread_px=1,
                base_half_spread_px=1,
            )
        )
        engine = StrategyEngine(config)
        engine.on_event(
            MarketEvent(
                kind="book",
                session_id="test",
                seq=1,
                time_ms=0.0,
                book=BookState(best_bid_px=99, best_bid_qty=20, best_ask_px=101, best_ask_qty=20),
            )
        )
        engine.on_event(
            MarketEvent(
                kind="news",
                session_id="test",
                seq=2,
                time_ms=100.0,
                news=NewsState(kind="structured", symbol="A", structured_subtype="earnings", earnings_asset="A", earnings_value=1.1),
            )
        )
        intent = engine.on_event(
            MarketEvent(
                kind="book",
                session_id="test",
                seq=3,
                time_ms=150.0,
                book=BookState(best_bid_px=104, best_bid_qty=20, best_ask_px=106, best_ask_qty=20),
            )
        )

        self.assertIsNotNone(intent)
        self.assertEqual(intent.mode, "EARNINGS_SHOCK")
        self.assertTrue(any(action.side == "BUY" for action in intent.aggressive_actions))

    def test_inventory_unwind_kicks_in_when_position_stretched(self) -> None:
        engine = StrategyEngine(build_app_config())
        engine.state.inventory = 100
        intent = engine.on_event(
            MarketEvent(
                kind="book",
                session_id="test",
                seq=1,
                time_ms=0.0,
                book=BookState(best_bid_px=99, best_bid_qty=20, best_ask_px=101, best_ask_qty=20),
            )
        )

        self.assertEqual(intent.mode, "INVENTORY_UNWIND")
        self.assertIsNone(intent.bid)
        self.assertIsNotNone(intent.ask)


if __name__ == "__main__":
    unittest.main()
