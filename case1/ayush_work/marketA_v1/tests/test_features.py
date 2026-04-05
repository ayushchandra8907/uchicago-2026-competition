from __future__ import annotations

import unittest

from case1.ayush_work.marketA_v1.features import FeatureEngine, compute_imbalance, compute_microprice, infer_trade_aggressor
from case1.ayush_work.marketA_v1.models import BookState, TradeState


class FeatureEngineTests(unittest.TestCase):
    def test_microprice_imbalance_and_trade_pressure(self) -> None:
        engine = FeatureEngine()
        book = BookState(
            best_bid_px=99,
            best_bid_qty=30,
            best_ask_px=101,
            best_ask_qty=10,
            bid_levels=((99, 30),),
            ask_levels=((101, 10),),
        )
        self.assertAlmostEqual(compute_microprice(book), 100.5)
        self.assertAlmostEqual(compute_imbalance(book), 0.5)

        engine.on_book(book, 0.0)
        engine.on_trade(TradeState(price_px=101, qty=10), 100.0)
        engine.on_trade(TradeState(price_px=101, qty=10), 200.0)
        engine.on_fill("BUY", 5, 250.0)
        snapshot = engine.snapshot(300.0, 100)

        self.assertEqual(snapshot.trade_count_1s, 2)
        self.assertEqual(snapshot.trade_volume_1s, 20)
        self.assertGreater(snapshot.trade_pressure_1s, 0.9)
        self.assertGreater(snapshot.fill_pressure_2s, 0.9)

    def test_infer_trade_aggressor_from_book(self) -> None:
        book = BookState(best_bid_px=99, best_bid_qty=10, best_ask_px=101, best_ask_qty=10)
        self.assertEqual(infer_trade_aggressor(TradeState(price_px=101, qty=5), book), "BUY")
        self.assertEqual(infer_trade_aggressor(TradeState(price_px=99, qty=5), book), "SELL")


if __name__ == "__main__":
    unittest.main()
