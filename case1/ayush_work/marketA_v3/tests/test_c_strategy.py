from __future__ import annotations

import unittest

from case1.ayush_work.marketA_v3.config import MarketCStrategyConfig
from case1.ayush_work.marketA_v3.core.types import BookLevel, BookSnapshot, NewsEvent, StrategySnapshot
from case1.ayush_work.marketA_v3.market_C_strategy import CStrategy
from case1.ayush_work.marketA_v3.market_C_strategy.c_news_sentiment import score_fed_speak_headline


def snapshot(
    *,
    now_ms: int = 1_000,
    books: dict[str, tuple[int, int]] | None = None,
    inventories: dict[str, int] | None = None,
) -> StrategySnapshot:
    default_books = books or {
        "R_HIKE": (398, 402),
        "R_HOLD": (398, 402),
        "R_CUT": (398, 402),
    }
    inventories = inventories or {}
    book_map = {
        symbol: BookSnapshot(best_bid=BookLevel(px=bid, qty=10), best_ask=BookLevel(px=ask, qty=10))
        for symbol, (bid, ask) in default_books.items()
    }
    inventories_by_symbol = {symbol: int(inventories.get(symbol, 0)) for symbol in book_map}
    return StrategySnapshot(
        now_ms=now_ms,
        exchange_tick=1,
        book=book_map["R_HOLD"],
        inventory=inventories_by_symbol["R_HOLD"],
        cash=0,
        fair_value=None,
        trusted_multiplier=None,
        latest_earnings=None,
        mode="IDLE",
        open_orders=(),
        books_by_symbol=book_map,
        inventories_by_symbol=inventories_by_symbol,
        open_orders_by_symbol={symbol: () for symbol in book_map},
        last_trade_px_by_symbol={symbol: None for symbol in book_map},
    )


def cpi_event(actual: float, forecast: float, *, now_ms: int = 1_000) -> NewsEvent:
    return NewsEvent(
        now_ms=now_ms,
        tick=1,
        kind="structured",
        symbol=None,
        structured_subtype="cpi_print",
        forecast=forecast,
        actual=actual,
        raw_payload={"kind": "structured", "new_data": {"structured_subtype": "cpi_print", "forecast": forecast, "actual": actual}},
    )


def fed_event(content: str, *, now_ms: int = 1_000) -> NewsEvent:
    return NewsEvent(
        now_ms=now_ms,
        tick=1,
        kind="unstructured",
        symbol=None,
        content=content,
        news_type="FedSpeak",
        raw_payload={"kind": "unstructured", "new_data": {"type": "FedSpeak", "content": content}},
    )


class CStrategyTests(unittest.TestCase):
    def test_book_updates_do_not_create_positions(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_book(
            snapshot(books={"R_HIKE": (98, 102), "R_HOLD": (398, 402), "R_CUT": (898, 902)})
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.macro_pair_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertTrue(decision.observe_only)

    def test_hawkish_cpi_flattens_all_rate_positions_for_major_release(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(
                inventories={"R_HIKE": 80, "R_HOLD": 25, "R_CUT": -60},
                books={"R_HIKE": (418, 422), "R_HOLD": (398, 402), "R_CUT": (378, 382)},
            ),
            cpi_event(actual=0.0023, forecast=0.0018),
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HOLD"], 0)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(set(strategy.rate_macro_pair_symbols or []), {"R_HIKE", "R_HOLD", "R_CUT"})
        self.assertTrue(strategy.rate_global_flatten_signal)
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")
        self.assertEqual(decision.desired_order.intent, "prediction_market_unwind_macro")

    def test_dovish_cpi_with_no_inventory_does_not_open_trade(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(books={"R_HIKE": (418, 422), "R_HOLD": (398, 402), "R_CUT": (378, 382)}),
            cpi_event(actual=0.0010, forecast=0.0018),
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertTrue(decision.observe_only)
        self.assertEqual(set(strategy.rate_macro_pair_symbols or []), {"R_HIKE", "R_HOLD", "R_CUT"})
        self.assertTrue(strategy.rate_global_flatten_signal)

    def test_hold_fedspeak_that_is_not_major_keeps_positions_unchanged(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(
                inventories={"R_HIKE": 10, "R_HOLD": 55, "R_CUT": -5},
                books={"R_HIKE": (318, 322), "R_HOLD": (398, 402), "R_CUT": (218, 222)},
            ),
            fed_event("Fed chair reiterates data dependence; no clear signal on next move."),
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HOLD"], 55)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], 10)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_CUT"], -5)
        self.assertIsNone(strategy.rate_macro_pair_symbols)
        self.assertFalse(strategy.rate_global_flatten_signal)
        self.assertTrue(decision.observe_only)

    def test_low_relevance_fedspeak_does_nothing(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(inventories={"R_HIKE": 40}),
            fed_event("Markets react to broad geopolitical uncertainty."),
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], 40)
        self.assertIsNone(strategy.rate_macro_pair_symbols)
        self.assertFalse(strategy.rate_global_flatten_signal)
        self.assertTrue(decision.observe_only)

    def test_unaffected_inventory_is_preserved_during_flatten_window(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(
            snapshot(
                now_ms=1_000,
                inventories={"R_HIKE": 70, "R_HOLD": -30, "R_CUT": 45},
                books={"R_HIKE": (418, 422), "R_HOLD": (398, 402), "R_CUT": (378, 382)},
            ),
            cpi_event(actual=0.0023, forecast=0.0018, now_ms=1_000),
        )
        decision = strategy.on_timer(
            snapshot(
                now_ms=2_000,
                inventories={"R_HIKE": 0, "R_HOLD": -30, "R_CUT": 45},
                books={"R_HIKE": (438, 442), "R_HOLD": (398, 402), "R_CUT": (358, 362)},
            )
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HOLD"], 0)
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(decision.desired_order.symbol, "R_CUT")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_flatten_window_expires_and_bot_stops_zeroing_contracts(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(
            snapshot(
                now_ms=1_000,
                inventories={"R_HIKE": 80},
                books={"R_HIKE": (418, 422), "R_HOLD": (398, 402), "R_CUT": (378, 382)},
            ),
            cpi_event(actual=0.0023, forecast=0.0018, now_ms=1_000),
        )
        decision = strategy.on_timer(
            snapshot(
                now_ms=12_500,
                inventories={"R_HIKE": 80},
                books={"R_HIKE": (438, 442), "R_HOLD": (398, 402), "R_CUT": (358, 362)},
            )
        )
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], 80)
        self.assertIsNone(strategy.rate_macro_pair_symbols)
        self.assertTrue(decision.observe_only)

    def test_probe_and_endgame_do_not_create_positions_in_flatten_only_mode(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.session_started_ms = 0

        probe_decision = strategy.on_timer(
            snapshot(
                now_ms=730_000,
                inventories={"R_HOLD": 20},
                books={"R_HIKE": (298, 302), "R_HOLD": (448, 452), "R_CUT": (248, 252)},
            )
        )
        self.assertEqual(strategy.probe_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.endgame_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HOLD"], 20)
        self.assertTrue(probe_decision.observe_only)

        endgame_decision = strategy.on_timer(
            snapshot(
                now_ms=850_000,
                inventories={"R_HIKE": -15},
                books={"R_HIKE": (780, 784), "R_HOLD": (180, 184), "R_CUT": (40, 44)},
            )
        )
        self.assertEqual(strategy.probe_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.endgame_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.trading_phase_targets_by_symbol["R_HIKE"], -15)
        self.assertTrue(endgame_decision.observe_only)

    def test_strong_fedspeak_arms_global_flatten_signal(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(
                inventories={"R_HIKE": 10, "R_HOLD": -20, "R_CUT": 15},
                books={"R_HIKE": (468, 472), "R_HOLD": (388, 392), "R_CUT": (138, 142)},
            ),
            fed_event("Softening data raises expectations of policy easing."),
        )
        self.assertTrue(strategy.rate_global_flatten_signal)
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(decision.mode, "UNWIND")

    def test_probability_state_is_still_exported(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(
            snapshot(books={"R_HIKE": (418, 422), "R_HOLD": (398, 402), "R_CUT": (378, 382)}),
            cpi_event(actual=0.0023, forecast=0.0018),
        )
        state = strategy.probability_state()
        self.assertAlmostEqual(sum(state.posterior_probs.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(state.terminal_probs.values()), 1.0, places=6)

    def test_trace_driven_softening_data_scores_direct_contract_deltas(self):
        result = score_fed_speak_headline("Softening data raises expectations of policy easing.")
        self.assertEqual(result.bucket, "extreme")
        self.assertLess(result.delta_hike, 0.0)
        self.assertGreater(result.delta_hold, 0.0)
        self.assertGreater(result.delta_cut, 0.0)


if __name__ == "__main__":
    unittest.main()
