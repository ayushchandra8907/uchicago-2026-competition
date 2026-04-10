from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from a_bot_config import BConfig
from a_bot_strategy import BookSnapshot, DesiredOrder


@dataclass(frozen=True)
class ParityLeg:
    symbol: str
    side: str
    qty: int
    px: int
    leg_role: str


@dataclass(frozen=True)
class ParityOpportunity:
    strike: int
    kind: str
    edge: float
    trade_size: int
    legs: tuple[ParityLeg, ...]
    signal_id: str
    trade_group_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "kind": self.kind,
            "edge": round(float(self.edge), 4),
            "trade_size": self.trade_size,
            "signal_id": self.signal_id,
            "trade_group_id": self.trade_group_id,
            "legs": [leg.__dict__ for leg in self.legs],
        }


class BParityOpportunist:
    """Detects large tradeable put-call parity opportunities.

    Live multi-leg execution is intentionally separate from detection. The
    runtime can shadow-log these opportunities first, then route legs once the
    partial-fill recovery policy is battle-tested.
    """

    def __init__(self, b_config: BConfig):
        self.b_config = b_config
        self.signal_seq = 0

    def evaluate(self, payload: dict[str, Any] | None, *, now_ms: int) -> ParityOpportunity | None:
        if payload is None:
            return None
        by_strike = payload.get("tradeable_parity_by_strike") or {}
        best: tuple[int, str, float, dict[str, Any]] | None = None
        for strike_text, metrics in by_strike.items():
            try:
                strike = int(strike_text)
            except (TypeError, ValueError):
                continue
            for kind, edge_key in (("conversion", "conversion_edge"), ("reversal", "reversal_edge")):
                edge = float(metrics.get(edge_key, 0.0) or 0.0)
                if edge < float(self.b_config.parity_edge_threshold_ticks):
                    continue
                if best is None or edge > best[2]:
                    best = (strike, kind, edge, metrics)
        if best is None:
            return None

        strike, kind, edge, metrics = best
        trade_size = max(1, min(int(self.b_config.parity_trade_size), int(self.b_config.parity_max_exposure)))
        self.signal_seq += 1
        signal_id = f"b_parity_{kind}_{strike}_{self.signal_seq}"
        return ParityOpportunity(
            strike=strike,
            kind=kind,
            edge=edge,
            trade_size=trade_size,
            legs=self._legs_for(kind, strike, trade_size, metrics),
            signal_id=signal_id,
            trade_group_id=signal_id,
        )

    def desired_orders_for(self, opportunity: ParityOpportunity) -> tuple[DesiredOrder, ...]:
        action_class = "conversion" if opportunity.kind == "conversion" else "reversal"
        return tuple(
            DesiredOrder(
                side=leg.side,
                px=leg.px,
                qty=leg.qty,
                overlay="mm",
                aggressive=True,
                reason=f"B parity {opportunity.kind} edge {opportunity.edge:.2f} at K={opportunity.strike}",
                intent=f"b_parity_{opportunity.kind}",
                mode_at_submit="B_PARITY_OPPORTUNIST",
                evaluation_reason="parity_edge_exceeded_threshold",
                market_key="B",
                strategy_family="b_parity_opportunist",
                action_class=action_class,
                pnl_owner="b_parity_opportunist",
                signal_id=opportunity.signal_id,
                trade_group_id=opportunity.trade_group_id,
                leg_role=leg.leg_role,
            )
            for leg in opportunity.legs
        )

    @staticmethod
    def compute_tradeable_parity(
        *,
        books: Mapping[str, BookSnapshot],
        underlying_symbol: str,
        strikes: tuple[int, ...],
        call_symbols: Mapping[int, str],
        put_symbols: Mapping[int, str],
        last_book_ms: Mapping[str, int] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        underlying = books.get(underlying_symbol)
        if underlying is None or underlying.best_bid is None or underlying.best_ask is None:
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for strike in strikes:
            call_symbol = call_symbols[strike]
            put_symbol = put_symbols[strike]
            call = books.get(call_symbol)
            put = books.get(put_symbol)
            if (
                call is None
                or put is None
                or call.best_bid is None
                or call.best_ask is None
                or put.best_bid is None
                or put.best_ask is None
            ):
                continue
            conversion_cost = underlying.best_ask.px + put.best_ask.px - call.best_bid.px
            conversion_edge = float(strike - conversion_cost)
            reversal_credit = underlying.best_bid.px + put.best_bid.px - call.best_ask.px
            reversal_edge = float(reversal_credit - strike)
            row = {
                "conversion_cost": int(conversion_cost),
                "conversion_edge": round(conversion_edge, 4),
                "reversal_credit": int(reversal_credit),
                "reversal_edge": round(reversal_edge, 4),
                "conversion_legs": {
                    "B": {"side": "BUY", "px": underlying.best_ask.px},
                    put_symbol: {"side": "BUY", "px": put.best_ask.px},
                    call_symbol: {"side": "SELL", "px": call.best_bid.px},
                },
                "reversal_legs": {
                    "B": {"side": "SELL", "px": underlying.best_bid.px},
                    put_symbol: {"side": "SELL", "px": put.best_bid.px},
                    call_symbol: {"side": "BUY", "px": call.best_ask.px},
                },
            }
            if last_book_ms is not None and now_ms is not None:
                ages = []
                for symbol in (underlying_symbol, call_symbol, put_symbol):
                    stamp = last_book_ms.get(symbol)
                    ages.append(None if stamp is None else max(0, int(now_ms) - int(stamp)))
                row["max_quote_age_ms"] = None if any(age is None for age in ages) else max(int(age) for age in ages if age is not None)
            rows[str(strike)] = row
        return rows

    @staticmethod
    def _legs_for(kind: str, strike: int, qty: int, metrics: dict[str, Any]) -> tuple[ParityLeg, ...]:
        leg_key = "conversion_legs" if kind == "conversion" else "reversal_legs"
        raw_legs = metrics.get(leg_key) or {}
        option_call = f"B_C_{strike}"
        option_put = f"B_P_{strike}"
        leg_roles = {
            "B": "underlying",
            option_put: "put",
            option_call: "call",
        }
        legs: list[ParityLeg] = []
        for symbol in ("B", option_put, option_call):
            raw = raw_legs.get(symbol)
            if not raw:
                continue
            legs.append(
                ParityLeg(
                    symbol=symbol,
                    side=str(raw["side"]),
                    qty=int(qty),
                    px=int(raw["px"]),
                    leg_role=leg_roles[symbol],
                )
            )
        return tuple(legs)
