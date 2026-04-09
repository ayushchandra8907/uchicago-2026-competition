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
    def test_trading_phase_baseline_shorts_high_and_longs_low(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.last_tradeable_macro_ms = 0
        strategy.on_book(
            snapshot(books={"R_HIKE": (98, 102), "R_HOLD": (398, 402), "R_CUT": (898, 902)})
        )
        self.assertGreater(strategy.baseline_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.baseline_targets_by_symbol["R_HOLD"], 0)
        self.assertLess(strategy.baseline_targets_by_symbol["R_CUT"], 0)

    def test_trading_phase_zeroes_center_contracts(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.last_tradeable_macro_ms = 0
        strategy.on_book(
            snapshot(books={"R_HIKE": (358, 362), "R_HOLD": (418, 422), "R_CUT": (698, 702)})
        )
        self.assertEqual(strategy.baseline_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.baseline_targets_by_symbol["R_HOLD"], 0)
        self.assertLess(strategy.baseline_targets_by_symbol["R_CUT"], 0)

    def test_hawkish_cpi_builds_hike_cut_macro_pair(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(snapshot(), cpi_event(actual=0.0023, forecast=0.0018))
        self.assertGreater(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertLess(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HOLD"], 0)
        self.assertEqual(strategy.rate_macro_pair_symbols, ["R_HIKE", "R_CUT"])
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "BUY")

    def test_dovish_cpi_builds_cut_hike_macro_pair(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(snapshot(), cpi_event(actual=0.0010, forecast=0.0018))
        self.assertGreater(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertLess(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HOLD"], 0)
        self.assertEqual(strategy.rate_macro_pair_symbols, ["R_CUT", "R_HIKE"])
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_hold_positive_fedspeak_longs_hold_and_shorts_richer_tail(self):
        strategy = CStrategy(MarketCStrategyConfig())
        decision = strategy.on_news(
            snapshot(books={"R_HIKE": (618, 622), "R_HOLD": (398, 402), "R_CUT": (218, 222)}),
            fed_event("Fed chair reiterates data dependence; no clear signal on next move."),
        )
        self.assertGreater(strategy.macro_pair_targets_by_symbol["R_HOLD"], 0)
        self.assertLess(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(strategy.rate_macro_pair_symbols, ["R_HOLD", "R_HIKE"])
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_macro_pair_overshoot_trim_reduces_target(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(snapshot(), cpi_event(actual=0.0031, forecast=0.0018))
        decision = strategy.on_book(
            snapshot(
                now_ms=2_000,
                books={"R_HIKE": (508, 512), "R_HOLD": (398, 402), "R_CUT": (398, 402)},
                inventories={"R_HIKE": 120, "R_CUT": -120},
            )
        )
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HIKE"], 60)
        self.assertEqual(decision.mode, "SHOCK")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_macro_pair_equilibrium_flatten_fires(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(snapshot(), cpi_event(actual=0.00205, forecast=0.0018))
        strategy.on_book(
            snapshot(
                now_ms=1_500,
                books={"R_HIKE": (424, 426), "R_HOLD": (398, 402), "R_CUT": (374, 376)},
                inventories={"R_HIKE": 40, "R_CUT": -40},
            )
        )
        strategy.on_book(
            snapshot(
                now_ms=1_900,
                books={"R_HIKE": (424, 426), "R_HOLD": (398, 402), "R_CUT": (374, 376)},
                inventories={"R_HIKE": 40, "R_CUT": -40},
            )
        )
        decision = strategy.on_book(
            snapshot(
                now_ms=2_200,
                books={"R_HIKE": (424, 426), "R_HOLD": (398, 402), "R_CUT": (374, 376)},
                inventories={"R_HIKE": 40, "R_CUT": -40},
            )
        )
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_macro_leg_reversal_flatten_fires(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(snapshot(), cpi_event(actual=0.0031, forecast=0.0018))
        strategy.on_book(
            snapshot(
                now_ms=2_000,
                books={"R_HIKE": (448, 452), "R_HOLD": (398, 402), "R_CUT": (398, 402)},
                inventories={"R_HIKE": 120, "R_CUT": -120},
            )
        )
        decision = strategy.on_book(
            snapshot(
                now_ms=2_500,
                books={"R_HIKE": (438, 442), "R_HOLD": (398, 402), "R_CUT": (398, 402)},
                inventories={"R_HIKE": 120, "R_CUT": -120},
            )
        )
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.symbol, "R_HIKE")
        self.assertEqual(decision.desired_order.side, "SELL")

    def test_new_macro_event_replaces_old_pair_after_flatten(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(snapshot(), cpi_event(actual=0.0031, forecast=0.0018))
        decision = strategy.on_news(
            snapshot(
                now_ms=2_000,
                inventories={"R_HIKE": 120, "R_CUT": -120},
            ),
            fed_event("Fed chair highlights cooling labor market and easing inflation pressures.", now_ms=2_000),
        )
        self.assertIsNotNone(strategy.pending_macro_signal)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)
        self.assertEqual(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)
        self.assertEqual(decision.desired_order.side, "SELL")

        strategy.on_book(snapshot(now_ms=2_100))
        self.assertIsNone(strategy.pending_macro_signal)
        self.assertGreater(strategy.macro_pair_targets_by_symbol["R_CUT"], 0)
        self.assertLess(strategy.macro_pair_targets_by_symbol["R_HIKE"], 0)

    def test_probe_phase_uses_small_probe_by_default(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.session_started_ms = 0
        strategy.on_timer(
            snapshot(
                now_ms=730_000,
                books={"R_HIKE": (298, 302), "R_HOLD": (448, 452), "R_CUT": (248, 252)},
            )
        )
        self.assertEqual(strategy.probe_targets_by_symbol["R_HOLD"], 20)
        self.assertEqual(strategy.probe_targets_by_symbol["R_HIKE"], -20)
        self.assertEqual(strategy.probe_targets_by_symbol["R_CUT"], -20)

    def test_probe_phase_scales_when_leader_is_clear(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.session_started_ms = 0
        strategy.on_timer(
            snapshot(
                now_ms=730_000,
                books={"R_HIKE": (60, 64), "R_HOLD": (918, 922), "R_CUT": (4, 8)},
            )
        )
        self.assertEqual(strategy.probe_targets_by_symbol["R_HOLD"], 80)
        self.assertEqual(strategy.probe_targets_by_symbol["R_HIKE"], -80)
        self.assertEqual(strategy.probe_targets_by_symbol["R_CUT"], -80)

    def test_endgame_has_exactly_one_positive_contract(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.session_started_ms = 0
        strategy.on_timer(
            snapshot(
                now_ms=800_000,
                books={"R_HIKE": (780, 784), "R_HOLD": (180, 184), "R_CUT": (40, 44)},
            )
        )
        positives = [symbol for symbol, qty in strategy.endgame_targets_by_symbol.items() if qty > 0]
        negatives = [symbol for symbol, qty in strategy.endgame_targets_by_symbol.items() if qty < 0]
        self.assertEqual(positives, ["R_HIKE"])
        self.assertEqual(len(negatives), 2)
        self.assertEqual(strategy.endgame_targets_by_symbol["R_HIKE"], 200)
        self.assertEqual(strategy.endgame_targets_by_symbol["R_HOLD"], -200)
        self.assertEqual(strategy.endgame_targets_by_symbol["R_CUT"], -200)

    def test_trading_phase_logic_is_disabled_once_probe_phase_starts(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.session_started_ms = 0
        strategy.on_news(snapshot(), cpi_event(actual=0.0031, forecast=0.0018))
        strategy.on_timer(
            snapshot(
                now_ms=730_000,
                books={"R_HIKE": (298, 302), "R_HOLD": (448, 452), "R_CUT": (248, 252)},
            )
        )
        self.assertEqual(strategy.baseline_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.macro_pair_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})

    def test_positions_flatten_ten_seconds_after_tradeable_fedspeak(self):
        strategy = CStrategy(MarketCStrategyConfig())
        strategy.on_news(
            snapshot(now_ms=1_000, books={"R_HIKE": (618, 622), "R_HOLD": (398, 402), "R_CUT": (218, 222)}),
            fed_event("Fed chair reiterates data dependence; no clear signal on next move.", now_ms=1_000),
        )
        decision = strategy.on_timer(
            snapshot(
                now_ms=11_500,
                books={"R_HIKE": (608, 612), "R_HOLD": (408, 412), "R_CUT": (218, 222)},
                inventories={"R_HOLD": 80, "R_HIKE": -80},
            )
        )
        self.assertEqual(strategy.baseline_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.macro_pair_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.rate_no_trade_reason, "macro_signal_stale_flatten")
        self.assertEqual(strategy.last_signal_source, "macro_signal_timeout")
        self.assertEqual(decision.mode, "UNWIND")
        self.assertIsNotNone(decision.desired_order)

    def test_generic_unstructured_news_is_ignored_for_c_trading(self):
        strategy = CStrategy(MarketCStrategyConfig())
        generic_news = NewsEvent(
            now_ms=1_000,
            tick=1,
            kind="unstructured",
            symbol=None,
            content="Markets react to broad geopolitical uncertainty.",
            news_type="WorldNews",
            raw_payload={"kind": "unstructured", "new_data": {"type": "WorldNews", "content": "Markets react to broad geopolitical uncertainty."}},
        )
        decision = strategy.on_news(snapshot(), generic_news)
        self.assertEqual(strategy.macro_pair_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.trading_phase_targets_by_symbol, {"R_HIKE": 0, "R_HOLD": 0, "R_CUT": 0})
        self.assertEqual(strategy.rate_no_trade_reason, "irrelevant_macro_news")
        self.assertTrue(decision.observe_only)

    def test_trace_driven_softening_data_scores_direct_contract_deltas(self):
        result = score_fed_speak_headline("Softening data raises expectations of policy easing.")
        self.assertEqual(result.bucket, "extreme")
        self.assertLess(result.delta_hike, 0.0)
        self.assertGreater(result.delta_hold, 0.0)
        self.assertGreater(result.delta_cut, 0.0)


if __name__ == "__main__":
    unittest.main()
