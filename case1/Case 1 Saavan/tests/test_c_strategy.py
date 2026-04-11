from __future__ import annotations

import sys
from pathlib import Path
import unittest


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from a_bot_config import MarketCConfig, RiskConfig
from c_strategy import MarketCStrategy


class FakeOrderBook:
    def __init__(self, bids: dict[int, int] | None = None, asks: dict[int, int] | None = None):
        self.bids = bids or {}
        self.asks = asks or {}


class MarketCStrategyTests(unittest.TestCase):
    def make_strategy(self) -> MarketCStrategy:
        strategy = MarketCStrategy(
            MarketCConfig(
                enabled=True,
                trading_enabled=True,
                live_earnings_enabled=True,
                live_cpi_enabled=False,
                live_macro_enabled=False,
                mm_enabled=False,
            ),
            RiskConfig(reprice_cooldown_ms=0, passive_reprice_threshold_ticks=1, passive_quote_ttl_ms=3_000),
            book_depth_levels=5,
        )
        strategy.on_book_update_at("C", FakeOrderBook(bids={998: 10}, asks={1002: 10}), now_ms=1_000)
        strategy.on_book_update_at("R_HIKE", FakeOrderBook(bids={29: 10}, asks={31: 10}), now_ms=1_000)
        strategy.on_book_update_at("R_HOLD", FakeOrderBook(bids={39: 10}, asks={41: 10}), now_ms=1_000)
        strategy.on_book_update_at("R_CUT", FakeOrderBook(bids={29: 10}, asks={31: 10}), now_ms=1_000)
        return strategy

    @staticmethod
    def c_earnings_news(value: float, tick: int = 100) -> dict:
        return {
            "tick": tick,
            "kind": "structured",
            "new_data": {
                "structured_subtype": "earnings",
                "asset": "C",
                "value": value,
            },
        }

    def test_rate_context_uses_pm_mids(self) -> None:
        strategy = self.make_strategy()

        context = strategy.rate_context()

        self.assertIsNotNone(context)
        assert context is not None
        self.assertAlmostEqual(context.q_hike, 0.3, places=4)
        self.assertAlmostEqual(context.q_hold, 0.4, places=4)
        self.assertAlmostEqual(context.q_cut, 0.3, places=4)
        self.assertAlmostEqual(context.market_rate_bp, 0.0, places=4)

    def test_first_c_earnings_sets_baseline_without_trade(self) -> None:
        strategy = self.make_strategy()

        result = strategy.on_news(self.c_earnings_news(2.0), now_ms=1_100)
        plan = strategy.compute_quotes(now_ms=1_100)

        self.assertEqual(result["signals"][0].action_class, "baseline_adopted")
        self.assertEqual(plan.mode, "C_OBSERVE_ONLY")
        self.assertEqual(strategy.c_earnings_shock.mode, "IDLE")

    def test_strong_follow_on_c_earnings_creates_live_shock(self) -> None:
        strategy = self.make_strategy()
        strategy.on_news(self.c_earnings_news(2.0), now_ms=1_100)

        result = strategy.on_news(self.c_earnings_news(2.2, tick=200), now_ms=2_000)
        plan = strategy.compute_quotes(now_ms=2_000)

        self.assertEqual(result["signals"][0].strategy_family, "c_earnings")
        self.assertEqual(result["signals"][0].action_class, "earnings_signal")
        self.assertEqual(strategy.c_earnings_shock.mode, "SHOCK")
        self.assertEqual(plan.mode, "C_EARNINGS_SHOCK")
        self.assertEqual(len(plan.aggressive_actions), 1)
        self.assertEqual(plan.aggressive_actions[0].side, "BUY")
        self.assertEqual(plan.aggressive_actions[0].strategy_family, "c_earnings")
        self.assertEqual(plan.aggressive_actions[0].action_class, "shock_take")

    def test_late_weak_earnings_is_blocked(self) -> None:
        strategy = self.make_strategy()
        strategy.on_news(self.c_earnings_news(2.0), now_ms=1_100)

        result = strategy.on_news(self.c_earnings_news(2.015, tick=4400), now_ms=2_000)

        self.assertEqual(result["signals"][0].action_class, "shadow_blocked")
        self.assertEqual(result["signals"][0].payload["blocked_reason"], "late_session_weak_earnings")
        self.assertEqual(strategy.c_earnings_shock.mode, "IDLE")

    def test_cpi_signal_is_shadow_only(self) -> None:
        strategy = self.make_strategy()

        result = strategy.on_news(
            {
                "tick": 300,
                "kind": "structured",
                "new_data": {
                    "actual": 0.0030,
                    "forecast": 0.0010,
                },
            },
            now_ms=3_000,
        )

        families = [signal.strategy_family for signal in result["signals"]]
        self.assertIn("c_cpi_shadow", families)
        cpi_signal = next(signal for signal in result["signals"] if signal.strategy_family == "c_cpi_shadow")
        self.assertEqual(cpi_signal.action_class, "shadow_blocked")
        self.assertFalse(cpi_signal.payload["live_trading_enabled"])


if __name__ == "__main__":
    unittest.main()
