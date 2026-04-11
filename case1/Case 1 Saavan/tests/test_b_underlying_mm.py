from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import BConfig, RiskConfig
from b_observer import MarketBObserver
from b_underlying_mm import BUnderlyingMMStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class BUnderlyingMMTests(unittest.TestCase):
    def make_strategy(self) -> BUnderlyingMMStrategy:
        return BUnderlyingMMStrategy(
            BConfig(
                enabled=True,
                trading_enabled=True,
                observe_only=False,
                quote_size=1,
                max_position=8,
                base_half_spread_ticks=2,
                inventory_skew_ticks_per_unit=0.5,
                passive_reduce_start=4,
                passive_reduce_full=8,
                min_book_spread=6,
                max_synthetic_dispersion=4,
                basis_entry_threshold_ticks=1.25,
                basis_strong_threshold_ticks=2.5,
                imbalance_confirmation_threshold=0.15,
                far_side_widen_ticks=4,
            ),
            RiskConfig(
                reprice_cooldown_ms=0,
                passive_reprice_threshold_ticks=2,
                passive_quote_ttl_ms=3_000,
            ),
            book_depth_levels=5,
        )

    @staticmethod
    def seed_consistent_chain(observer: MarketBObserver, *, b_bid: int = 1096, b_ask: int = 1104) -> dict:
        observer.on_book_update("B", FakeOrderBook(bids={b_bid: 10}, asks={b_ask: 10}))
        observer.on_book_update("B_C_950", FakeOrderBook(bids={150: 10}, asks={154: 10}))
        observer.on_book_update("B_P_950", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={100: 10}, asks={104: 10}))
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={50: 10}, asks={54: 10}))
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={0: 10}, asks={4: 10}))
        return observer.compute_residuals()

    @staticmethod
    def seed_positive_basis_chain(observer: MarketBObserver, *, b_bid: int = 1096, b_ask: int = 1104) -> dict:
        observer.on_book_update("B", FakeOrderBook(bids={b_bid: 10}, asks={b_ask: 10}))
        observer.on_book_update("B_C_950", FakeOrderBook(bids={153: 10}, asks={155: 10}))
        observer.on_book_update("B_P_950", FakeOrderBook(bids={0: 10}, asks={2: 10}))
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={103: 10}, asks={105: 10}))
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={0: 10}, asks={2: 10}))
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={53: 10}, asks={55: 10}))
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={0: 10}, asks={2: 10}))
        return observer.compute_residuals()

    @staticmethod
    def seed_negative_basis_chain(observer: MarketBObserver, *, b_bid: int = 1096, b_ask: int = 1104) -> dict:
        observer.on_book_update("B", FakeOrderBook(bids={b_bid: 10}, asks={b_ask: 10}))
        observer.on_book_update("B_C_950", FakeOrderBook(bids={147: 10}, asks={149: 10}))
        observer.on_book_update("B_P_950", FakeOrderBook(bids={2: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={97: 10}, asks={99: 10}))
        observer.on_book_update("B_P_1000", FakeOrderBook(bids={2: 10}, asks={4: 10}))
        observer.on_book_update("B_C_1050", FakeOrderBook(bids={47: 10}, asks={49: 10}))
        observer.on_book_update("B_P_1050", FakeOrderBook(bids={2: 10}, asks={4: 10}))
        return observer.compute_residuals()

    def test_stays_observe_only_when_basis_is_too_small(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        payload = self.seed_consistent_chain(observer)
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertEqual(plan.mode, "OBSERVE_ONLY")
        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "basis_too_small")

    def test_blocks_when_synthetic_dispersion_is_too_wide(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        self.seed_consistent_chain(observer)
        observer.on_book_update("B_C_1000", FakeOrderBook(bids={108: 10}, asks={112: 10}))
        payload = observer.compute_residuals()
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "synthetic_dispersion_wide")

    def test_blocks_when_book_spread_is_too_tight(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        payload = self.seed_consistent_chain(observer, b_bid=1099, b_ask=1101)
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1099: 10}, asks={1101: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertTrue(plan.observe_only)
        self.assertEqual(plan.reason, "book_spread_too_tight")

    def test_reduces_only_when_inventory_nonzero_and_fair_is_missing(self) -> None:
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(3)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=None)

        self.assertEqual(plan.mode, "REDUCE_ONLY")
        self.assertFalse(plan.observe_only)
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.side, "SELL")

    def test_quotes_one_sided_bid_when_basis_is_positive(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        payload = self.seed_positive_basis_chain(observer)
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertEqual(plan.mode, "UNDERLYING_MM")
        self.assertIsNotNone(plan.bid)
        self.assertIsNone(plan.ask)
        self.assertEqual(plan.bid.qty, 1)

    def test_quotes_one_sided_ask_when_basis_is_negative(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        payload = self.seed_negative_basis_chain(observer)
        strategy = self.make_strategy()
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertEqual(plan.mode, "UNDERLYING_MM")
        self.assertIsNone(plan.bid)
        self.assertIsNotNone(plan.ask)
        self.assertEqual(plan.ask.qty, 1)

    def test_reduces_only_when_inventory_opposes_signal(self) -> None:
        observer = MarketBObserver(depth_levels=5)
        payload = self.seed_negative_basis_chain(observer)
        strategy = self.make_strategy()
        strategy.sync_inventory_from_exchange(2)
        strategy.on_book_update_at("B", FakeOrderBook(bids={1096: 10}, asks={1104: 10}), now_ms=1_000)

        plan = strategy.compute_quotes(now_ms=1_000, residual_payload=payload)

        self.assertEqual(plan.mode, "REDUCE_ONLY")
        self.assertIsNotNone(plan.ask)


if __name__ == "__main__":
    unittest.main()
