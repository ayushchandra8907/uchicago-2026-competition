from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import time
from typing import Any

from a_bot_config import BConfig, RiskConfig
from a_bot_strategy import BookSnapshot, DesiredOrder, OrderManager, QuotePlan


@dataclass(frozen=True)
class ParsedOptionSymbol:
    symbol: str
    option_type: str
    strike: int


class BOptionLotteryStrategy:
    """Small buy-only cheap-convexity strategy for B options.

    The goal is not full option market making. We only buy tiny amounts of very
    cheap optionality when the premium budget makes the downside explicitly
    bounded.
    """

    def __init__(self, b_config: BConfig, risk: RiskConfig, *, book_depth_levels: int = 10):
        self.config = b_config
        self.risk = risk
        self.book_depth_levels = max(1, int(book_depth_levels))
        self.option_symbols = tuple(b_config.option_symbols)
        self.parsed: dict[str, ParsedOptionSymbol] = {
            symbol: self._parse_option_symbol(symbol) for symbol in self.option_symbols
        }
        self.books: dict[str, BookSnapshot] = {symbol: BookSnapshot() for symbol in (b_config.underlying_symbol, *self.option_symbols)}
        self.positions: dict[str, int] = {symbol: 0 for symbol in self.option_symbols}
        self.underlying_inventory = 0
        self.order_managers: dict[str, OrderManager] = {
            symbol: OrderManager(symbol=symbol, risk=risk) for symbol in self.option_symbols
        }
        self.last_submit_ms: dict[str, int | None] = {symbol: None for symbol in self.option_symbols}
        self.last_trade_px: dict[str, int | None] = {symbol: None for symbol in self.books}
        self.last_trade_qty: dict[str, int | None] = {symbol: None for symbol in self.books}
        self.last_trade_ms: dict[str, int | None] = {symbol: None for symbol in self.books}
        self.prior_underlying_mid: float | None = None
        self.current_underlying_mid: float | None = None
        self.premium_spent = 0
        self.premium_recovered = 0
        self.hedge_premium_spent = 0
        self.open_cost_basis: dict[str, float] = {symbol: 0.0 for symbol in self.option_symbols}
        self.costed_qty: dict[str, int] = {symbol: 0 for symbol in self.option_symbols}
        self.realized_profit: dict[str, float] = {symbol: 0.0 for symbol in self.option_symbols}
        self.last_mode: dict[str, str] = {symbol: "OBSERVE_ONLY" for symbol in self.option_symbols}
        self.last_block_reason: dict[str, str | None] = {symbol: None for symbol in self.option_symbols}
        self.last_hedge_needed: dict[str, bool] = {symbol: False for symbol in self.option_symbols}
        self.last_hedge_target_qty: dict[str, int] = {symbol: 0 for symbol in self.option_symbols}
        self.last_profit_take_trigger: dict[str, float | None] = {symbol: None for symbol in self.option_symbols}
        self._order_to_symbol: dict[str, str] = {}

    def on_book_update_at(self, symbol: str, book, now_ms: int) -> bool:
        if symbol not in self.books:
            return False
        snapshot = BookSnapshot.from_order_book(book, depth_levels=self.book_depth_levels)
        if symbol == self.config.underlying_symbol and snapshot.mid is not None:
            if self.current_underlying_mid is not None and snapshot.mid != self.current_underlying_mid:
                self.prior_underlying_mid = self.current_underlying_mid
            self.current_underlying_mid = float(snapshot.mid)
        self.books[symbol] = snapshot
        return True

    def on_market_trade(self, symbol: str, price: int, qty: int, *, now_ms: int | None = None) -> bool:
        if symbol not in self.books:
            return False
        self.last_trade_px[symbol] = int(price)
        self.last_trade_qty[symbol] = int(qty)
        self.last_trade_ms[symbol] = self._now_ms() if now_ms is None else int(now_ms)
        return True

    def sync_inventory(self, symbol: str, inventory: int) -> bool:
        if symbol == self.config.underlying_symbol:
            self.underlying_inventory = int(inventory)
            return True
        if symbol not in self.positions:
            return False
        incoming = int(inventory)
        self.positions[symbol] = incoming
        if incoming <= 0:
            self.open_cost_basis[symbol] = 0.0
            self.costed_qty[symbol] = 0
        elif self.costed_qty.get(symbol, 0) > incoming:
            avg_cost = self._average_entry(symbol)
            self.costed_qty[symbol] = incoming
            self.open_cost_basis[symbol] = max(0.0, incoming * avg_cost)
        return True

    def inventory(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def has_order(self, order_id: str) -> bool:
        return order_id in self._order_to_symbol

    def order_for_id(self, order_id: str) -> tuple[str | None, Any | None]:
        symbol = self._order_to_symbol.get(order_id)
        if symbol is None:
            return None, None
        return symbol, self.order_managers[symbol].orders.get(order_id)

    def on_fill(
        self,
        order_id: str,
        qty: int,
        price: int,
        *,
        now_ms: int | None = None,
        authoritative_inventory: int | None = None,
        pre_fill_inventory: int | None = None,
    ) -> tuple[str | None, Any | None]:
        symbol = self._order_to_symbol.get(order_id)
        if symbol is None:
            return None, None
        before_inventory = self.positions.get(symbol, 0) if pre_fill_inventory is None else int(pre_fill_inventory)
        order = self.order_managers[symbol].handle_fill(order_id, qty)
        if order is None:
            return symbol, None
        qty = int(qty)
        price = int(price)
        if order.side == "BUY":
            self.premium_spent += max(0, qty * price)
            if order.action_class == "atm_option_hedge":
                self.hedge_premium_spent += max(0, qty * price)
            self.open_cost_basis[symbol] = float(self.open_cost_basis.get(symbol, 0.0)) + (qty * price)
            self.costed_qty[symbol] = int(self.costed_qty.get(symbol, 0)) + qty
            next_inventory = int(self.positions.get(symbol, 0)) + qty
        else:
            costed_before = int(self.costed_qty.get(symbol, 0))
            closing_qty = min(max(0, costed_before if costed_before > 0 else before_inventory), qty)
            avg_cost = self._average_entry(symbol, position_override=(costed_before if costed_before > 0 else before_inventory))
            cost_reduction = closing_qty * avg_cost
            proceeds = qty * price
            self.premium_recovered += max(0, proceeds)
            self.realized_profit[symbol] = float(self.realized_profit.get(symbol, 0.0)) + (closing_qty * (price - avg_cost))
            self.open_cost_basis[symbol] = max(0.0, float(self.open_cost_basis.get(symbol, 0.0)) - cost_reduction)
            self.costed_qty[symbol] = max(0, costed_before - closing_qty)
            next_inventory = int(self.positions.get(symbol, 0)) - qty
        self.positions[symbol] = int(authoritative_inventory) if authoritative_inventory is not None else next_inventory
        if self.positions[symbol] <= 0:
            self.open_cost_basis[symbol] = 0.0
            self.costed_qty[symbol] = 0
        elif self.costed_qty.get(symbol, 0) > self.positions[symbol]:
            avg_cost = self._average_entry(symbol)
            self.costed_qty[symbol] = self.positions[symbol]
            self.open_cost_basis[symbol] = max(0.0, self.positions[symbol] * avg_cost)
        self.last_trade_px[symbol] = int(price)
        self.last_trade_qty[symbol] = int(qty)
        self.last_trade_ms[symbol] = self._now_ms() if now_ms is None else int(now_ms)
        if order.remaining_qty <= 0:
            self._order_to_symbol.pop(order_id, None)
        return symbol, order

    def on_cancel_response(self, order_id: str, success: bool) -> tuple[str | None, Any | None]:
        symbol = self._order_to_symbol.get(order_id)
        if symbol is None:
            return None, None
        order = self.order_managers[symbol].handle_cancel_response(order_id, success)
        if success:
            self._order_to_symbol.pop(order_id, None)
        return symbol, order

    def on_rejection(self, order_id: str) -> tuple[str | None, Any | None]:
        symbol = self._order_to_symbol.get(order_id)
        if symbol is None:
            return None, None
        order = self.order_managers[symbol].handle_rejection(order_id)
        self._order_to_symbol.pop(order_id, None)
        return symbol, order

    def note_submitted(self, symbol: str, *, order_id: str, desired: DesiredOrder, now_ms: int):
        order = self.order_managers[symbol].note_submitted(
            order_id=order_id,
            side=desired.side,
            px=desired.px,
            qty=desired.qty,
            now_ms=now_ms,
            overlay=desired.overlay,
            aggressive=desired.aggressive,
            intent=desired.intent,
            mode_at_submit=desired.mode_at_submit,
            evaluation_reason=desired.evaluation_reason,
            market_key=desired.market_key,
            strategy_family=desired.strategy_family,
            action_class=desired.action_class,
            pnl_owner=desired.pnl_owner,
            signal_id=desired.signal_id,
            trade_group_id=desired.trade_group_id,
            leg_role=desired.leg_role,
        )
        self._order_to_symbol[order_id] = symbol
        self.last_submit_ms[symbol] = int(now_ms)
        return order

    def compute_quote_plan(self, symbol: str, *, now_ms: int, residual_payload: dict[str, Any] | None) -> QuotePlan:
        if symbol not in self.parsed:
            return QuotePlan("OBSERVE_ONLY", None, None, (), True, "not_a_b_option_lottery_symbol")
        manager = self.order_managers[symbol]
        book = self.books[symbol]
        self.last_hedge_needed[symbol] = False
        self.last_hedge_target_qty[symbol] = 0
        self.last_profit_take_trigger[symbol] = None
        profit_take = self._profit_take_order(symbol, now_ms=now_ms)
        if profit_take is not None:
            self.last_mode[symbol] = "B_OPTION_LOTTERY_PROFIT_TAKE"
            self.last_block_reason[symbol] = None
            return QuotePlan("B_OPTION_LOTTERY_PROFIT_TAKE", None, profit_take, (), False, profit_take.reason)
        existing_profit_take = self._existing_sell_order_as_desired(symbol)
        if existing_profit_take is not None:
            self.last_mode[symbol] = "B_OPTION_LOTTERY_PROFIT_TAKE"
            self.last_block_reason[symbol] = None
            return QuotePlan(
                "B_OPTION_LOTTERY_PROFIT_TAKE",
                None,
                existing_profit_take,
                (),
                False,
                "keeping_existing_option_profit_take_order",
            )
        if book.best_ask is None:
            return self._blocked(symbol, "missing_option_ask")
        ask_px = int(book.best_ask.px)
        hedge_plan = self._hedge_buy_plan(symbol, ask_px=ask_px, now_ms=now_ms)
        if hedge_plan is not None:
            return hedge_plan
        if ask_px > int(self.config.option_lottery_max_ask):
            return self._blocked(symbol, "option_ask_above_lottery_threshold")
        symbol_cap = self._symbol_position_cap(symbol)
        if self.positions[symbol] + manager.buy_exposure() >= symbol_cap:
            return self._blocked(symbol, "option_lottery_symbol_position_full")
        if not self._cooldown_elapsed(symbol, now_ms):
            return self._blocked(symbol, "option_lottery_rebuy_cooldown")

        premium_remaining = self._premium_remaining(symbol)
        if ask_px > 0 and premium_remaining < ask_px:
            return self._blocked(symbol, "option_lottery_premium_budget_spent")

        parsed = self.parsed[symbol]
        tail_wing = symbol in {"B_C_1050", "B_P_950"}
        if (
            ask_px > int(self.config.option_lottery_floor_ask)
            and not tail_wing
            and not self._directional_setup_ok(parsed, residual_payload)
        ):
            return self._blocked(symbol, "option_lottery_direction_filter")

        max_position_qty = symbol_cap - self.positions[symbol] - manager.buy_exposure()
        budget_qty = int(self.config.option_lottery_quote_size) if ask_px <= 0 else premium_remaining // max(1, ask_px)
        qty = min(int(self.config.option_lottery_quote_size), max_position_qty, max(0, budget_qty))
        if qty <= 0:
            return self._blocked(symbol, "option_lottery_no_allowed_qty")

        signal_id = f"b_option_lottery_{symbol}_{int(now_ms)}"
        order = DesiredOrder(
            side="BUY",
            px=ask_px,
            qty=qty,
            overlay="mm",
            aggressive=False,
            reason=f"buying cheap {symbol} optionality at ask {ask_px}",
            intent="b_option_lottery",
            mode_at_submit="B_OPTION_LOTTERY",
            evaluation_reason="cheap_option_convexity",
            market_key="B",
            strategy_family="b_option_lottery",
            action_class="cheap_option_buy",
            pnl_owner=f"b_option_lottery:{symbol}",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role=parsed.option_type.lower(),
        )
        self.last_mode[symbol] = "B_OPTION_LOTTERY"
        self.last_block_reason[symbol] = None
        return QuotePlan("B_OPTION_LOTTERY", order, None, (), False, order.reason)

    def trace_state(self, symbol: str, now_ms: int) -> dict[str, Any]:
        book = self.books.get(symbol, BookSnapshot())
        manager = self.order_managers.get(symbol)
        return {
            "symbol": symbol,
            "market_key": "B",
            "mode": self.last_mode.get(symbol, "OBSERVE_ONLY"),
            "fair_value": None,
            "inventory": int(self.positions.get(symbol, 0)),
            "earnings_position": 0,
            "mm_position": int(self.positions.get(symbol, 0)),
            "buy_exposure": 0 if manager is None else manager.buy_exposure(),
            "sell_exposure": 0,
            "allowed_buy_size": max(0, self._symbol_position_cap(symbol) - int(self.positions.get(symbol, 0))),
            "allowed_sell_size": max(0, int(self.positions.get(symbol, 0)) - (0 if manager is None else manager.sell_exposure())),
            "position_cap": self._symbol_position_cap(symbol),
            "live_orders": [] if manager is None else manager.live_orders_snapshot(),
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
            "block_reason": self.last_block_reason.get(symbol),
            "b_option_lottery_premium_spent": self.premium_spent,
            "b_option_lottery_premium_recovered": self.premium_recovered,
            "b_option_lottery_premium_budget": int(self.config.option_lottery_total_premium_budget),
            "b_option_lottery_symbol_premium_remaining": self._premium_remaining(symbol),
            "b_option_lottery_avg_entry": self._average_entry(symbol),
            "b_option_lottery_realized_profit": self.realized_profit.get(symbol, 0.0),
            "b_option_lottery_underlying_mid": self.current_underlying_mid,
            "b_option_lottery_underlying_momentum": self._underlying_momentum(),
            "b_option_underlying_inventory": self.underlying_inventory,
            "b_option_hedge_needed": self.last_hedge_needed.get(symbol, False),
            "b_option_hedge_target_qty": self.last_hedge_target_qty.get(symbol, 0),
            "b_option_hedge_budget_remaining": self._hedge_premium_remaining(),
            "b_option_profit_take_trigger": self.last_profit_take_trigger.get(symbol),
            "b_option_hedge_premium_spent": self.hedge_premium_spent,
        }

    def _profit_take_order(self, symbol: str, *, now_ms: int) -> DesiredOrder | None:
        if not self.config.option_lottery_profit_take_enabled:
            return None
        manager = self.order_managers[symbol]
        position = int(self.positions.get(symbol, 0))
        if position <= 0:
            return None
        if manager.sell_exposure() > 0:
            return None
        book = self.books[symbol]
        if book.best_bid is None:
            return None
        avg_entry = self._average_entry(symbol)
        if avg_entry <= 0:
            return None
        bid_px = int(book.best_bid.px)
        if avg_entry <= float(self.config.option_lottery_max_ask):
            trigger_px = avg_entry * float(self.config.option_lottery_profit_take_multiple)
        else:
            trigger_px = max(
                avg_entry + float(self.config.option_lottery_profit_take_min_edge),
                avg_entry * float(self.config.option_lottery_profit_take_multiple),
            )
        self.last_profit_take_trigger[symbol] = trigger_px
        if bid_px < trigger_px:
            return None
        if bid_px >= avg_entry * 5.0:
            max_sell = position
        elif bid_px >= avg_entry * 3.0:
            max_sell = max(1, position // 2)
        elif self.underlying_inventory == 0 and symbol in {"B_C_1000", "B_P_1000"}:
            max_sell = max(1, position // 2)
        else:
            max_sell = max(1, int(ceil(position / 4.0)))
        qty = min(
            max_sell,
            max(1, int(self.config.option_lottery_profit_take_quote_size)),
            max(0, position - manager.sell_exposure()),
        )
        if qty <= 0:
            return None
        signal_id = f"b_option_lottery_profit_{symbol}_{int(now_ms)}"
        return DesiredOrder(
            side="SELL",
            px=bid_px,
            qty=qty,
            overlay="mm",
            aggressive=False,
            reason=f"taking profit on cheap {symbol} optionality at bid {bid_px}",
            intent="b_option_lottery_profit_take",
            mode_at_submit="B_OPTION_LOTTERY_PROFIT_TAKE",
            evaluation_reason="cheap_option_profit_take",
            market_key="B",
            strategy_family="b_option_lottery",
            action_class="cheap_option_profit_take",
            pnl_owner=f"b_option_lottery:{symbol}",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role=self.parsed[symbol].option_type.lower(),
        )

    def _existing_sell_order_as_desired(self, symbol: str) -> DesiredOrder | None:
        manager = self.order_managers[symbol]
        live = manager.live_order("SELL")
        if live is None or live.cancel_pending or live.remaining_qty <= 0:
            return None
        return DesiredOrder(
            side="SELL",
            px=int(live.px),
            qty=int(live.remaining_qty),
            overlay=live.overlay,
            aggressive=live.aggressive,
            reason=live.evaluation_reason,
            intent=live.intent,
            mode_at_submit=live.mode_at_submit,
            evaluation_reason=live.evaluation_reason,
            market_key=live.market_key,
            strategy_family=live.strategy_family,
            action_class=live.action_class,
            pnl_owner=live.pnl_owner,
            signal_id=live.signal_id,
            trade_group_id=live.trade_group_id,
            leg_role=live.leg_role,
        )

    def _hedge_buy_plan(self, symbol: str, *, ask_px: int, now_ms: int) -> QuotePlan | None:
        if not self.config.option_hedge_enabled or symbol not in {"B_C_1000", "B_P_1000"}:
            return None
        desired_symbol = None
        if self.underlying_inventory >= int(self.config.option_hedge_min_underlying_inventory):
            desired_symbol = "B_P_1000"
        elif self.underlying_inventory <= -int(self.config.option_hedge_min_underlying_inventory):
            desired_symbol = "B_C_1000"
        if desired_symbol != symbol:
            return None
        target_qty = int(ceil(abs(self.underlying_inventory) * float(self.config.option_hedge_target_ratio)))
        self.last_hedge_needed[symbol] = True
        self.last_hedge_target_qty[symbol] = max(0, target_qty)
        if ask_px > int(self.config.option_hedge_max_ask):
            return None
        manager = self.order_managers[symbol]
        allowed_qty = max(0, min(self._symbol_position_cap(symbol), target_qty) - self.positions[symbol] - manager.buy_exposure())
        if allowed_qty <= 0:
            return None
        remaining_budget = self._hedge_premium_remaining()
        if ask_px > 0 and remaining_budget < ask_px:
            return None
        budget_qty = allowed_qty if ask_px <= 0 else remaining_budget // max(1, ask_px)
        qty = min(max(1, int(self.config.option_lottery_quote_size)), allowed_qty, max(0, budget_qty))
        if qty <= 0:
            return None
        parsed = self.parsed[symbol]
        signal_id = f"b_option_hedge_{symbol}_{int(now_ms)}"
        order = DesiredOrder(
            side="BUY",
            px=int(ask_px),
            qty=int(qty),
            overlay="mm",
            aggressive=False,
            reason=f"buying {symbol} ATM hedge for B inventory {self.underlying_inventory}",
            intent="b_option_hedge",
            mode_at_submit="B_OPTION_HEDGE",
            evaluation_reason="atm_option_hedge",
            market_key="B",
            strategy_family="b_option_hedge",
            action_class="atm_option_hedge",
            pnl_owner=f"b_option_hedge:{symbol}",
            signal_id=signal_id,
            trade_group_id=signal_id,
            leg_role=parsed.option_type.lower(),
        )
        self.last_mode[symbol] = "B_OPTION_HEDGE"
        self.last_block_reason[symbol] = None
        return QuotePlan("B_OPTION_HEDGE", order, None, (), False, order.reason)

    def _directional_setup_ok(self, parsed: ParsedOptionSymbol, residual_payload: dict[str, Any] | None) -> bool:
        underlying_mid = self.current_underlying_mid
        if underlying_mid is None and residual_payload is not None:
            raw_mid = residual_payload.get("underlying_mid")
            underlying_mid = None if raw_mid is None else float(raw_mid)
        if underlying_mid is None:
            return False
        near = abs(float(underlying_mid) - float(parsed.strike)) <= float(self.config.option_lottery_near_strike_ticks)
        basis = 0.0 if residual_payload is None else float(residual_payload.get("composite_basis", 0.0) or 0.0)
        momentum = self._underlying_momentum()
        threshold = float(self.config.option_lottery_min_momentum_ticks)
        if parsed.option_type == "C":
            return near and (momentum >= threshold or basis >= threshold)
        return near and (momentum <= -threshold or basis <= -threshold)

    def _underlying_momentum(self) -> float:
        if self.current_underlying_mid is None or self.prior_underlying_mid is None:
            return 0.0
        return float(self.current_underlying_mid) - float(self.prior_underlying_mid)

    def _cooldown_elapsed(self, symbol: str, now_ms: int) -> bool:
        last = self.last_submit_ms.get(symbol)
        return last is None or int(now_ms) - int(last) >= int(self.config.option_lottery_rebuy_cooldown_ms)

    def _symbol_position_cap(self, symbol: str) -> int:
        if symbol in {"B_C_1050", "B_P_950"}:
            return int(self.config.option_lottery_wing_max_position)
        if symbol in {"B_C_1000", "B_P_1000"}:
            return int(self.config.option_lottery_atm_max_position)
        return int(self.config.option_lottery_max_position_per_symbol)

    def _premium_remaining(self, symbol: str) -> int:
        total_remaining = int(self.config.option_lottery_total_premium_budget) - self.premium_spent - self._outstanding_premium()
        symbol_spent = self._symbol_open_premium(symbol) + self._symbol_outstanding_premium(symbol)
        if symbol in {"B_C_1050", "B_P_950"}:
            symbol_remaining = int(self.config.option_lottery_wing_premium_budget) - symbol_spent
        elif symbol in {"B_C_1000", "B_P_1000"}:
            atm_spent = sum(self._symbol_open_premium(item) + self._symbol_outstanding_premium(item) for item in ("B_C_1000", "B_P_1000"))
            symbol_remaining = int(self.config.option_lottery_atm_total_premium_budget) - atm_spent
        else:
            symbol_remaining = int(self.config.option_lottery_max_ask) * int(self.config.option_lottery_max_position_per_symbol) - symbol_spent
        return max(0, min(total_remaining, symbol_remaining))

    def _hedge_premium_remaining(self) -> int:
        outstanding = 0
        for symbol in ("B_C_1000", "B_P_1000"):
            manager = self.order_managers[symbol]
            for order in manager.orders.values():
                if (
                    order.remaining_qty > 0
                    and not order.cancel_pending
                    and order.side == "BUY"
                    and order.action_class == "atm_option_hedge"
                ):
                    outstanding += int(order.remaining_qty) * max(0, int(order.px))
        return max(0, int(self.config.option_hedge_premium_budget) - int(self.hedge_premium_spent) - outstanding)

    def _symbol_open_premium(self, symbol: str) -> int:
        return int(round(float(self.open_cost_basis.get(symbol, 0.0))))

    def _symbol_outstanding_premium(self, symbol: str) -> int:
        manager = self.order_managers[symbol]
        total = 0
        for order in manager.orders.values():
            if order.remaining_qty > 0 and not order.cancel_pending and order.side == "BUY":
                total += int(order.remaining_qty) * max(0, int(order.px))
        return total

    def _average_entry(self, symbol: str, *, position_override: int | None = None) -> float:
        if position_override is None:
            position = int(self.costed_qty.get(symbol, 0)) or int(self.positions.get(symbol, 0))
        else:
            position = int(position_override)
        if position <= 0:
            return 0.0
        return float(self.open_cost_basis.get(symbol, 0.0)) / position

    def _outstanding_premium(self) -> int:
        total = 0
        for manager in self.order_managers.values():
            for order in manager.orders.values():
                if order.remaining_qty > 0 and not order.cancel_pending and order.side == "BUY":
                    total += int(order.remaining_qty) * max(0, int(order.px))
        return total

    def _blocked(self, symbol: str, reason: str) -> QuotePlan:
        self.last_mode[symbol] = "OBSERVE_ONLY"
        self.last_block_reason[symbol] = reason
        return QuotePlan("OBSERVE_ONLY", None, None, (), True, reason)

    @staticmethod
    def _parse_option_symbol(symbol: str) -> ParsedOptionSymbol:
        _, option_type, strike = symbol.split("_")
        return ParsedOptionSymbol(symbol=symbol, option_type=option_type, strike=int(strike))

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
