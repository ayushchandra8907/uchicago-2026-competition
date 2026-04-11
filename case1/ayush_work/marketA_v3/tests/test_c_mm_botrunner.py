from __future__ import annotations

from pathlib import Path
import unittest

from case1.ayush_work.marketA_v3.botrunner import BotRunner
from case1.ayush_work.marketA_v3.config import BotConfig, BotPaths, ExchangeConfig, LoggerConfig, StrategyConfig
from case1.ayush_work.marketA_v3.core.types import Decision, DesiredOrder


def build_config() -> BotConfig:
    return BotConfig(
        exchange=ExchangeConfig(host="localhost:3333", username="user", password="pass"),
        strategy=StrategyConfig(),
        c_mm_strategy=StrategyConfig(symbol="C"),
        logger=LoggerConfig(enabled=False),
        paths=BotPaths(base_dir=Path(".")),
    )


class CMMBotRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cm_runner_tracks_only_stock_c(self):
        runner = BotRunner(build_config(), active_strategy="CM")
        self.assertEqual(runner._tracked_symbols, ("C",))

    async def test_apply_decision_for_cm_submits_on_c(self):
        runner = BotRunner(build_config(), active_strategy="CM")
        submitted: list[tuple[str, int, int]] = []

        async def fake_place_order(symbol, qty, side, px):
            submitted.append((str(symbol), int(qty), int(px)))
            return "oid-1"

        async def fake_cancel_order(order_id):
            raise AssertionError("cancel_order should not be called")

        runner.place_order = fake_place_order  # type: ignore[method-assign]
        runner.cancel_order = fake_cancel_order  # type: ignore[method-assign]

        decision = Decision(
            mode="MARKET_MAKING_STUB",
            target_inventory=12,
            desired_order=DesiredOrder(
                side="BUY",
                px=1000,
                qty=12,
                aggressive=False,
                intent="stock_c_market_make",
                reason="test",
                symbol="C",
            ),
            cancel_all=False,
            observe_only=False,
            reason="test",
        )

        await runner._apply_decision(decision)

        self.assertEqual(submitted, [("C", 12, 1000)])


if __name__ == "__main__":
    unittest.main()
