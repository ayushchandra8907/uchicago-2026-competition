from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from a_bot_strategy import BookSnapshot
from b_parity_opportunist import BParityOpportunist


@dataclass(frozen=True)
class BSignalBundle:
    signal_id: str
    trade_group_id: str
    payload: dict[str, Any]


class MarketBObserver:
    """Observe-only analytics for B and its option chain."""

    STRIKES = (950, 1000, 1050)

    def __init__(
        self,
        *,
        depth_levels: int = 10,
        signal_snapshot_interval_ms: int = 250,
        signal_change_threshold_ticks: int = 1,
    ):
        self.depth_levels = max(1, depth_levels)
        self.signal_snapshot_interval_ms = max(1, int(signal_snapshot_interval_ms))
        self.signal_change_threshold_ticks = max(0, int(signal_change_threshold_ticks))
        self.underlying_symbol = "B"
        self.call_symbols = {strike: f"B_C_{strike}" for strike in self.STRIKES}
        self.put_symbols = {strike: f"B_P_{strike}" for strike in self.STRIKES}
        self.symbols = tuple(
            [self.underlying_symbol]
            + [self.call_symbols[strike] for strike in self.STRIKES]
            + [self.put_symbols[strike] for strike in self.STRIKES]
        )
        self.books: dict[str, BookSnapshot] = {symbol: BookSnapshot() for symbol in self.symbols}
        self.last_book_ms: dict[str, int | None] = {symbol: None for symbol in self.symbols}
        self.positions: dict[str, int] = {symbol: 0 for symbol in self.symbols}
        self.last_trade_px: dict[str, int | None] = {symbol: None for symbol in self.symbols}
        self.last_trade_qty: dict[str, int | None] = {symbol: None for symbol in self.symbols}
        self.last_trade_ms: dict[str, int | None] = {symbol: None for symbol in self.symbols}
        self.signal_seq = 0
        self.last_emitted_payload: dict[str, Any] | None = None
        self.last_emitted_ms: int | None = None

    def on_book_update(self, symbol: str, book, now_ms: int | None = None) -> bool:
        if symbol not in self.books:
            return False
        self.books[symbol] = BookSnapshot.from_order_book(book, depth_levels=self.depth_levels)
        self.last_book_ms[symbol] = self._now_ms() if now_ms is None else int(now_ms)
        return True

    def on_market_trade(self, symbol: str, price: int, qty: int, now_ms: int | None = None) -> bool:
        if symbol not in self.books:
            return False
        self.last_trade_px[symbol] = int(price)
        self.last_trade_qty[symbol] = int(qty)
        self.last_trade_ms[symbol] = self._now_ms() if now_ms is None else int(now_ms)
        return True

    def sync_inventory(self, symbol: str, inventory: int) -> bool:
        if symbol not in self.positions:
            return False
        self.positions[symbol] = int(inventory)
        return True

    def trace_state(self, symbol: str, now_ms: int) -> dict[str, Any]:
        book = self.books.get(symbol, BookSnapshot())
        composite_fair = self.synthetic_fair()
        return {
            "symbol": symbol,
            "market_key": "B",
            "mode": "OBSERVE_ONLY",
            "fair_value": composite_fair if symbol == self.underlying_symbol else None,
            "trusted_multiplier": None,
            "multiplier_confidence": None,
            "latest_earnings": None,
            "inventory": self.positions.get(symbol, 0),
            "earnings_position": 0,
            "mm_position": self.positions.get(symbol, 0),
            "earnings_budget": 0,
            "mm_budget": 0,
            "budget_shift_active": False,
            "unwind_active": False,
            "unwind_aggressive_active": False,
            "buy_exposure": 0,
            "sell_exposure": 0,
            "allowed_buy_size": 0,
            "allowed_sell_size": 0,
            "position_cap": 0,
            "overlay_exposures": {},
            "live_orders": [],
            "book": {
                "best_bid_px": None if book.best_bid is None else book.best_bid.px,
                "best_bid_qty": None if book.best_bid is None else book.best_bid.qty,
                "best_ask_px": None if book.best_ask is None else book.best_ask.px,
                "best_ask_qty": None if book.best_ask is None else book.best_ask.qty,
                "spread": book.spread,
                "mid": book.mid,
                "microprice": book.microprice,
                "top_of_book_imbalance": book.top_of_book_imbalance,
                "bid_levels": [{"px": level.px, "qty": level.qty} for level in book.bid_levels],
                "ask_levels": [{"px": level.px, "qty": level.qty} for level in book.ask_levels],
            },
            "last_trade_px": self.last_trade_px.get(symbol),
            "last_trade_qty": self.last_trade_qty.get(symbol),
            "last_trade_ms": self.last_trade_ms.get(symbol),
            "exchange_tick": None,
            "news_caution_active": False,
            "news_caution_until_ms": 0,
            "news_caution_remaining_ms": 0,
            "current_earnings_signal_id": None,
            "current_news_signal_id": None,
        }

    def derived_signal_bundle(self, *, now_ms: int | None = None) -> BSignalBundle | None:
        event_ms = self._now_ms() if now_ms is None else int(now_ms)
        payload = self.compute_residuals(now_ms=event_ms)
        if payload is None:
            return None
        if not self._should_emit_signal(payload, event_ms):
            return None
        self.signal_seq += 1
        signal_id = f"b_residual_{self.signal_seq}"
        self.last_emitted_payload = payload
        self.last_emitted_ms = event_ms
        return BSignalBundle(signal_id=signal_id, trade_group_id=signal_id, payload=payload)

    def synthetic_fair(self) -> float | None:
        payload = self.compute_residuals()
        if payload is None:
            return None
        return payload.get("composite_synthetic_fair")

    def compute_residuals(self, *, now_ms: int | None = None) -> dict[str, Any] | None:
        underlying_mid = self.books[self.underlying_symbol].mid
        if underlying_mid is None:
            return None

        parity_residual_by_strike: dict[str, float] = {}
        synthetic_forward_by_strike: dict[str, float] = {}
        basis_by_strike: dict[str, float] = {}
        crossing_cost_by_strike: dict[str, float] = {}
        parity_edge_after_cost_by_strike: dict[str, float] = {}
        call_mid_by_strike: dict[int, float] = {}
        put_mid_by_strike: dict[int, float] = {}

        for strike in self.STRIKES:
            call_mid = self.books[self.call_symbols[strike]].mid
            put_mid = self.books[self.put_symbols[strike]].mid
            if call_mid is None or put_mid is None:
                continue
            call_mid_by_strike[strike] = float(call_mid)
            put_mid_by_strike[strike] = float(put_mid)
            synthetic_forward = float(strike) + float(call_mid) - float(put_mid)
            residual = (float(call_mid) - float(put_mid)) - (float(underlying_mid) - float(strike))
            call_spread = max(0.0, float(self.books[self.call_symbols[strike]].spread or 0.0))
            put_spread = max(0.0, float(self.books[self.put_symbols[strike]].spread or 0.0))
            underlying_spread = max(0.0, float(self.books[self.underlying_symbol].spread or 0.0))
            crossing_cost = (call_spread / 2.0) + (put_spread / 2.0) + (underlying_spread / 2.0)
            parity_residual_by_strike[str(strike)] = residual
            synthetic_forward_by_strike[str(strike)] = synthetic_forward
            basis_by_strike[str(strike)] = synthetic_forward - float(underlying_mid)
            crossing_cost_by_strike[str(strike)] = round(crossing_cost, 4)
            parity_edge_after_cost_by_strike[str(strike)] = round(abs(residual) - crossing_cost, 4)

        if not parity_residual_by_strike:
            return None

        call_monotonicity_violations: list[dict[str, Any]] = []
        put_monotonicity_violations: list[dict[str, Any]] = []
        vertical_spread_bound_violations: list[dict[str, Any]] = []

        ordered_strikes = [strike for strike in self.STRIKES if strike in call_mid_by_strike and strike in put_mid_by_strike]
        for lower, upper in zip(ordered_strikes, ordered_strikes[1:]):
            lower_call = call_mid_by_strike[lower]
            upper_call = call_mid_by_strike[upper]
            lower_put = put_mid_by_strike[lower]
            upper_put = put_mid_by_strike[upper]
            strike_gap = upper - lower

            if upper_call > lower_call:
                call_monotonicity_violations.append(
                    {"lower_strike": lower, "upper_strike": upper, "difference": round(upper_call - lower_call, 4)}
                )
            if upper_put < lower_put:
                put_monotonicity_violations.append(
                    {"lower_strike": lower, "upper_strike": upper, "difference": round(lower_put - upper_put, 4)}
                )

            call_spread = lower_call - upper_call
            if call_spread < 0 or call_spread > strike_gap:
                vertical_spread_bound_violations.append(
                    {
                        "instrument": "call",
                        "lower_strike": lower,
                        "upper_strike": upper,
                        "spread_value": round(call_spread, 4),
                        "max_bound": strike_gap,
                    }
                )
            put_spread = upper_put - lower_put
            if put_spread < 0 or put_spread > strike_gap:
                vertical_spread_bound_violations.append(
                    {
                        "instrument": "put",
                        "lower_strike": lower,
                        "upper_strike": upper,
                        "spread_value": round(put_spread, 4),
                        "max_bound": strike_gap,
                    }
                )

        butterfly_convexity_violations: list[dict[str, Any]] = []
        if len(ordered_strikes) == 3:
            low, mid, high = ordered_strikes
            call_bfly = call_mid_by_strike[low] - (2.0 * call_mid_by_strike[mid]) + call_mid_by_strike[high]
            put_bfly = put_mid_by_strike[low] - (2.0 * put_mid_by_strike[mid]) + put_mid_by_strike[high]
            if call_bfly < 0:
                butterfly_convexity_violations.append({"instrument": "call", "value": round(call_bfly, 4)})
            if put_bfly < 0:
                butterfly_convexity_violations.append({"instrument": "put", "value": round(put_bfly, 4)})

        composite_synthetic_fair = sum(synthetic_forward_by_strike.values()) / len(synthetic_forward_by_strike)
        composite_basis = sum(basis_by_strike.values()) / len(basis_by_strike)
        synthetic_values = list(synthetic_forward_by_strike.values())
        synthetic_dispersion = max(synthetic_values) - min(synthetic_values) if synthetic_values else None
        tradeable_parity_by_strike = BParityOpportunist.compute_tradeable_parity(
            books=self.books,
            underlying_symbol=self.underlying_symbol,
            strikes=self.STRIKES,
            call_symbols=self.call_symbols,
            put_symbols=self.put_symbols,
            last_book_ms=self.last_book_ms,
            now_ms=now_ms,
        )
        return {
            "underlying_mid": float(underlying_mid),
            "parity_residual_by_strike": parity_residual_by_strike,
            "estimated_aggressive_crossing_cost_by_strike": crossing_cost_by_strike,
            "parity_edge_after_cost_by_strike": parity_edge_after_cost_by_strike,
            "synthetic_forward_by_strike": synthetic_forward_by_strike,
            "synthetic_dispersion": None if synthetic_dispersion is None else round(synthetic_dispersion, 4),
            "underlying_vs_synthetic_basis": basis_by_strike,
            "composite_synthetic_fair": round(composite_synthetic_fair, 4),
            "composite_basis": round(composite_basis, 4),
            "tradeable_parity_by_strike": tradeable_parity_by_strike,
            "call_monotonicity_violations": call_monotonicity_violations,
            "put_monotonicity_violations": put_monotonicity_violations,
            "vertical_spread_bound_violations": vertical_spread_bound_violations,
            "butterfly_convexity_violations": butterfly_convexity_violations,
        }

    def _should_emit_signal(self, payload: dict[str, Any], now_ms: int) -> bool:
        if self.last_emitted_payload is None or self.last_emitted_ms is None:
            return True
        if now_ms - self.last_emitted_ms >= self.signal_snapshot_interval_ms:
            return True

        if self._violation_signature(payload) != self._violation_signature(self.last_emitted_payload):
            return True

        prior_residuals = self.last_emitted_payload.get("parity_residual_by_strike") or {}
        current_residuals = payload.get("parity_residual_by_strike") or {}
        for strike in set(prior_residuals) | set(current_residuals):
            prior = float(prior_residuals.get(strike, 0.0) or 0.0)
            current = float(current_residuals.get(strike, 0.0) or 0.0)
            if abs(current - prior) >= self.signal_change_threshold_ticks:
                return True

        prior_basis = float(self.last_emitted_payload.get("composite_basis") or 0.0)
        current_basis = float(payload.get("composite_basis") or 0.0)
        return abs(current_basis - prior_basis) >= self.signal_change_threshold_ticks

    @staticmethod
    def _violation_signature(payload: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, len(payload.get(name) or []))
            for name in (
                "call_monotonicity_violations",
                "put_monotonicity_violations",
                "vertical_spread_bound_violations",
                "butterfly_convexity_violations",
            )
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
