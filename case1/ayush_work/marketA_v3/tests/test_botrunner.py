from __future__ import annotations

from pathlib import Path
import unittest

from case1.ayush_work.marketA_v3.botrunner import BotRunner
from case1.ayush_work.marketA_v3.config import BotConfig, BotPaths, ExchangeConfig, LoggerConfig, StrategyConfig
from case1.ayush_work.marketA_v3.core.types import Decision, DesiredOrder, ManagedOrder
from utcxchangelib import service_pb2


def build_config() -> BotConfig:
    return BotConfig(
        exchange=ExchangeConfig(host="localhost:3333", username="user", password="pass"),
        strategy=StrategyConfig(),
        logger=LoggerConfig(enabled=False),
        paths=BotPaths(base_dir=Path(".")),
    )


class BotRunnerTests(unittest.TestCase):
    def test_normalize_desired_order_preserves_large_target_qty(self):
        runner = BotRunner(build_config())
        desired = DesiredOrder(side="BUY", px=1000, qty=80, aggressive=True, intent="unwind", reason="test")
        normalized = runner._normalize_desired_order(desired)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized.qty, 80)

    def test_next_slice_qty_uses_small_aggressive_clips(self):
        runner = BotRunner(build_config())
        self.assertEqual(runner._next_slice_qty(120), 12)
        self.assertEqual(runner._next_slice_qty(14), 14)
        self.assertEqual(runner._next_slice_qty(19), 12)

    def test_recent_position_update_prevents_fill_double_count(self):
        runner = BotRunner(build_config())
        runner.positions["A"] = 20
        runner._inventory_estimate = 20
        runner._last_position_update_ms = 1_000
        runner._apply_fill_inventory_hint("BUY", 20, 1_020)
        self.assertEqual(runner._inventory(), 20)

    def test_fill_updates_inventory_without_recent_authoritative_update(self):
        runner = BotRunner(build_config())
        runner.positions["A"] = 0
        runner._inventory_estimate = 0
        runner._last_position_update_ms = None
        runner._apply_fill_inventory_hint("SELL", 20, 2_000)
        self.assertEqual(runner._inventory(), -20)

    def test_current_order_includes_cancel_pending(self):
        runner = BotRunner(build_config())
        runner._managed_orders["1"] = ManagedOrder(
            order_id="1",
            side="BUY",
            px=1000,
            qty=20,
            remaining_qty=20,
            submitted_ms=0,
            aggressive=True,
            intent="shock",
            reason="test",
            cancel_pending=True,
        )
        current = runner._current_order()
        self.assertIsNotNone(current)
        self.assertEqual(current.order_id, "1")

    def test_risk_adjusted_order_respects_outstanding_volume(self):
        runner = BotRunner(build_config())
        runner._managed_orders["1"] = runner._managed_orders.get("1") or type("Obj", (), {})()
        runner._managed_orders["1"].remaining_qty = 100
        runner._managed_orders["1"].cancel_pending = False
        runner._managed_orders["1"].is_live = True
        desired = DesiredOrder(side="BUY", px=1000, qty=40, aggressive=True, intent="shock", reason="test")
        adjusted = runner._risk_adjusted_order(desired)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.qty, 20)

    def test_risk_adjusted_order_respects_absolute_position(self):
        config = build_config()
        config = BotConfig(
            exchange=config.exchange,
            strategy=StrategyConfig(max_absolute_position=200),
            logger=config.logger,
            paths=config.paths,
        )
        runner = BotRunner(config)
        runner._inventory_estimate = 190
        desired = DesiredOrder(side="BUY", px=1000, qty=40, aggressive=True, intent="shock", reason="test")
        adjusted = runner._risk_adjusted_order(desired)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.qty, 10)

    def test_risk_adjusted_order_never_allows_41_plus_share_order(self):
        runner = BotRunner(build_config())
        desired = DesiredOrder(side="BUY", px=1000, qty=80, aggressive=True, intent="shock", reason="test")
        adjusted = runner._risk_adjusted_order(desired)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.qty, 40)

    def test_midrun_checkpoint_due_only_after_threshold(self):
        config = build_config()
        config = BotConfig(
            exchange=config.exchange,
            strategy=config.strategy,
            logger=LoggerConfig(enabled=True, midrun_checkpoint_enabled=True, midrun_checkpoint_ms=1_000),
            paths=config.paths,
        )
        runner = BotRunner(config)
        runner._run_started_ms = 5_000
        self.assertFalse(runner._midrun_checkpoint_due(5_999))
        self.assertTrue(runner._midrun_checkpoint_due(6_000))
        runner._midrun_checkpoint_started = True
        self.assertFalse(runner._midrun_checkpoint_due(7_000))

    def test_c_runner_snapshot_exposes_multi_symbol_state(self):
        runner = BotRunner(build_config(), active_strategy="C")
        runner._inventory_estimate_by_symbol["R_HIKE"] = 10
        runner._inventory_estimate_by_symbol["R_HOLD"] = -5
        runner._last_trade_px_by_symbol["R_CUT"] = 444
        snapshot = runner._snapshot(event_symbol="R_HIKE")
        self.assertEqual(snapshot.inventory_for("R_HIKE"), 10)
        self.assertEqual(snapshot.inventory_for("R_HOLD"), -5)
        self.assertEqual(snapshot.last_trade_for("R_CUT"), 444)
        self.assertEqual(snapshot.event_symbol, "R_HIKE")

    def test_c_risk_adjusted_order_respects_per_contract_cap(self):
        config = build_config()
        config = BotConfig(
            exchange=config.exchange,
            strategy=config.strategy,
            logger=config.logger,
            paths=config.paths,
            c_strategy=config.c_strategy.__class__(max_absolute_position_per_contract=40, shared_rate_position_budget=600),
        )
        runner = BotRunner(config, active_strategy="C")
        runner._inventory_estimate_by_symbol["R_HIKE"] = 35
        desired = DesiredOrder(side="BUY", px=500, qty=20, aggressive=True, intent="prediction_market_take", reason="test", symbol="R_HIKE")
        adjusted = runner._risk_adjusted_order(desired)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.qty, 5)


class BotRunnerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_c_runner_stops_immediately_after_market_resolved(self):
        runner = BotRunner(build_config(), active_strategy="C")
        msg = service_pb2.ExchangeMessageToClient(
            index=1,
            market_resolved=service_pb2.MarketResolvedMessage(
                market_id="fed_rates",
                winning_symbol="R_HOLD",
                tick=1234,
            ),
        )

        with self.assertRaises(EOFError):
            await runner.process_message(msg)

        self.assertTrue(runner._stop_after_c_market_resolved)

    async def test_a_runner_does_not_stop_on_prediction_market_resolution(self):
        runner = BotRunner(build_config(), active_strategy="A")
        msg = service_pb2.ExchangeMessageToClient(
            index=1,
            market_resolved=service_pb2.MarketResolvedMessage(
                market_id="fed_rates",
                winning_symbol="R_HOLD",
                tick=1234,
            ),
        )

        await runner.process_message(msg)

        self.assertFalse(runner._stop_after_c_market_resolved)

    async def test_apply_decision_submits_one_small_clip_for_large_target(self):
        runner = BotRunner(build_config())
        submitted: list[int] = []

        async def fake_place_order(symbol, qty, side, px):
            submitted.append(int(qty))
            return f"oid-{len(submitted)}"

        async def fake_cancel_order(order_id):
            raise AssertionError("cancel_order should not be called")

        runner.place_order = fake_place_order  # type: ignore[method-assign]
        runner.cancel_order = fake_cancel_order  # type: ignore[method-assign]

        decision = Decision(
            mode="SHOCK",
            target_inventory=120,
            desired_order=DesiredOrder(
                side="BUY",
                px=1000,
                qty=120,
                aggressive=True,
                intent="post_earnings_shock_take",
                reason="test",
            ),
            cancel_all=False,
            observe_only=False,
            reason="test",
        )

        await runner._apply_decision(decision)

        self.assertEqual(submitted, [12])

    async def test_apply_decision_for_c_submits_on_target_symbol(self):
        runner = BotRunner(build_config(), active_strategy="C")
        submitted: list[tuple[str, int]] = []

        async def fake_place_order(symbol, qty, side, px):
            submitted.append((str(symbol), int(qty)))
            return f"oid-{len(submitted)}"

        async def fake_cancel_order(order_id):
            raise AssertionError("cancel_order should not be called")

        runner.place_order = fake_place_order  # type: ignore[method-assign]
        runner.cancel_order = fake_cancel_order  # type: ignore[method-assign]

        decision = Decision(
            mode="SHOCK",
            target_inventory=30,
            desired_order=DesiredOrder(
                side="BUY",
                px=520,
                qty=30,
                aggressive=True,
                intent="prediction_market_take",
                reason="test",
                symbol="R_HOLD",
            ),
            cancel_all=False,
            observe_only=False,
            reason="test",
        )

        await runner._apply_decision(decision)

        self.assertEqual(submitted, [("R_HOLD", 12)])


if __name__ == "__main__":
    unittest.main()
