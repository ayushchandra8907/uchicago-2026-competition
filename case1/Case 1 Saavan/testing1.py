from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import grpc

from a_bot_config import BotConfig, ConfigError, load_bot_config
from a_bot_journal import TradingJournal, select_recovered_pricing_state
from a_bot_strategy import MarketAStrategy
from a_bot_trace import TraceRecorder
from ayush_a_port import AyushPortStrategy
from b_mean_reversion import BMeanReversionStrategy
from b_observer import MarketBObserver
from b_option_lottery import BOptionLotteryStrategy
from b_parity_opportunist import BParityOpportunist
from b_underlying_mm import BUnderlyingMMStrategy
from b_underlying_mm_v2 import BUnderlyingMMv2
from c_strategy import MarketCStrategy
from etf_a_follower import ETFAFollowerStrategy, ETFShockProjection

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError as exc:
    raise SystemExit(
        "utcxchangelib is not installed. Run `pip install -r requirements.txt` first."
    ) from exc


LOGGER = logging.getLogger("market-a-bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# Quick-start defaults so clicking the IDE Run button works the same way as the
# provided example bot. Environment variables still override these.
DEFAULT_SERVER = "practice.uchicago.exchange:3333"
DEFAULT_USERNAME = "uiuc"
DEFAULT_PASSWORD = "mesa-lynx-octopus"

# A's price-to-earnings multiplier changes by round, so we learn it live.
DEFAULT_A_INITIAL_MULTIPLIER: float | None = None
DEFAULT_A_INITIAL_FAIR_VALUE: int | None = None


class MarketABot(XChangeClient):
    """Live multi-market runtime with switchable A strategy modes and B MM research."""

    _UNEXPECTED_DISCONNECT_RETRY_DELAY_S = 2.0

    def __init__(self, config: BotConfig):
        subscribed_symbols = ["A"]
        if config.etf.enabled and config.etf.symbol not in subscribed_symbols:
            subscribed_symbols.append(config.etf.symbol)
        if config.market_b.enabled:
            subscribed_symbols.extend([config.market_b.underlying_symbol, *config.market_b.option_symbols])
        if getattr(config, "market_c", None) is not None and config.market_c.enabled:
            for symbol in (config.market_c.symbol, *config.market_c.pm_symbols):
                if symbol not in subscribed_symbols:
                    subscribed_symbols.append(symbol)
        super().__init__(
            config.exchange.host,
            config.exchange.username,
            config.exchange.password,
            symbols=subscribed_symbols,
        )
        self.config = config
        self.journal = TradingJournal(config.paths.journal_path)
        self.b_observer = MarketBObserver(
            depth_levels=max(10, config.trace.trace_book_depth_levels),
            signal_snapshot_interval_ms=config.market_b.signal_snapshot_interval_ms,
            signal_change_threshold_ticks=config.market_b.signal_change_threshold_ticks,
        )
        self._ayush_port_mode = config.a_strategy_mode == "ayush_port"
        self._b_observe_only = bool(
            config.market_b.observe_only
            or (self._ayush_port_mode and not config.market_b.allow_trading_with_ayush_port)
        )
        self._b_trading_enabled = bool(config.market_b.enabled and config.market_b.trading_enabled and not self._b_observe_only)
        if config.market_b.meanrev_enabled:
            b_strategy_cls = BMeanReversionStrategy
        else:
            b_strategy_cls = BUnderlyingMMv2 if config.market_b.mm_v2_enabled else BUnderlyingMMStrategy
        b_risk = replace(
            config.risk,
            passive_reprice_threshold_ticks=int(config.market_b.mm_reprice_threshold_ticks),
        )
        self.b_strategy = (
            b_strategy_cls(
                config.market_b,
                b_risk,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
            if self._b_trading_enabled
            else None
        )
        self.b_option_strategy = (
            BOptionLotteryStrategy(
                config.market_b,
                config.risk,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
            if bool(config.market_b.enabled and config.market_b.trading_enabled and config.market_b.option_lottery_enabled)
            else None
        )
        self._last_b_eval_ms: int | None = None
        self._last_c_eval_ms: int | None = None
        self._last_etf_eval_ms: int | None = None
        self._last_position_update_by_symbol: dict[str, tuple[int, int, int]] = {}
        self._etf_trading_enabled = bool(config.etf.enabled and config.etf.trading_enabled)
        etf_risk = replace(
            config.risk,
            reprice_cooldown_ms=int(config.etf.reprice_cooldown_ms),
            passive_reprice_threshold_ticks=int(config.etf.unwind_reprice_threshold_ticks),
        )
        self.etf_strategy = (
            ETFAFollowerStrategy(
                config.etf,
                etf_risk,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
            if self._etf_trading_enabled
            else None
        )
        self._c_trading_enabled = bool(
            getattr(config, "market_c", None) is not None
            and config.market_c.enabled
            and config.market_c.trading_enabled
        )
        self.c_strategy = (
            MarketCStrategy(
                config.market_c,
                config.risk,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
            if getattr(config, "market_c", None) is not None and config.market_c.enabled
            else None
        )
        self.b_parity_opportunist = BParityOpportunist(config.market_b)
        if hasattr(self.journal, "prepare_for_startup"):
            replay_state = self.journal.prepare_for_startup()
        else:
            replay_state = self.journal.load_replay_state()
        (
            recovered_multiplier,
            recovered_multiplier_confidence,
            recovered_fair_value,
            recovered_earnings_value,
        ) = select_recovered_pricing_state(
            replay_state,
            recover_pricing_state=config.market_a.recover_pricing_state,
        )
        if config.a_strategy_mode == "ayush_port":
            self.strategy = AyushPortStrategy(
                a_config=config.market_a,
                risk=config.risk,
                restored_orders=replay_state.live_orders,
                recovered_multiplier=recovered_multiplier,
                recovered_multiplier_confidence=recovered_multiplier_confidence,
                recovered_fair_value=recovered_fair_value,
                recovered_earnings_value=recovered_earnings_value,
                initial_multiplier=config.market_a.initial_multiplier,
                initial_fair_value=config.market_a.initial_fair_value,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
        else:
            self.strategy = MarketAStrategy(
                a_config=config.market_a,
                risk=config.risk,
                restored_orders=replay_state.live_orders,
                recovered_multiplier=recovered_multiplier,
                recovered_multiplier_confidence=recovered_multiplier_confidence,
                recovered_fair_value=recovered_fair_value,
                recovered_earnings_value=recovered_earnings_value,
                book_depth_levels=max(10, config.trace.trace_book_depth_levels),
            )
        self.tracer = TraceRecorder.create_if_enabled(config.trace, session_prefix="a_bot_run", symbol="A")
        self._quote_lock = asyncio.Lock()
        self._position_snapshot_seen = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._refresh_task: asyncio.Task | None = None
        self._round_end_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._last_observe_only_reason: str | None = None
        self._last_mode: str | None = None
        self._position_snapshot_count = 0
        self._auto_stop_requested = False
        self._shutdown_note = "Bot shutdown"
        self._runtime_started_ms = self._now_ms()
        self._connected_once = False
        self._awaiting_reconnect_success = False
        self._reconnect_attempt_count = 0

        if replay_state.live_orders:
            restored_ids = ", ".join(order.order_id for order in replay_state.live_orders)
            LOGGER.warning("Recovered %d local A orders from journal: %s", len(replay_state.live_orders), restored_ids)
        if recovered_multiplier is not None:
            LOGGER.info(
                "Recovered last known A multiplier from journal: %.4f (confidence=%s)",
                recovered_multiplier,
                recovered_multiplier_confidence,
            )
        elif replay_state.multiplier is not None and replay_state.live_orders:
            LOGGER.info(
                "Ignoring recovered A pricing state from journal because A_RECOVER_PRICING_STATE is disabled."
            )
        if recovered_fair_value is not None:
            LOGGER.info("Recovered last known A fair value from journal: %s", recovered_fair_value)
        LOGGER.info("Using A strategy mode: %s", config.a_strategy_mode)
        if config.market_a.initial_multiplier is not None and recovered_multiplier is None:
            LOGGER.info("Seeding A multiplier from config: %.4f", config.market_a.initial_multiplier)
        else:
            LOGGER.info("A valuation will learn a round-specific multiplier from structured earnings.")
        if self.tracer is not None:
            LOGGER.info("Analysis mode enabled; writing trace outputs to %s", self.tracer.run_dir)
        if self.tracer is not None:
            now_ms = self._now_ms()
            self.tracer.record_session_start(
                now_ms=now_ms,
                config_summary={
                    "host": config.exchange.host,
                    "subscribed_symbols": subscribed_symbols,
                    "trace_snapshot_interval_ms": config.trace.trace_snapshot_interval_ms,
                    "trace_book_depth_levels": config.trace.trace_book_depth_levels,
                    "trace_markout_windows_ms": list(config.trace.trace_markout_windows_ms),
                    "trace_record_book_updates": config.trace.trace_record_book_updates,
                    "trace_record_observe_only_decisions": config.trace.trace_record_observe_only_decisions,
                    "journal_path": str(config.paths.journal_path),
                    "trace_root": str(config.trace.trace_root),
                    "recover_pricing_state": config.market_a.recover_pricing_state,
                    "a_strategy_mode": config.a_strategy_mode,
                    "etf_enabled": config.etf.enabled,
                    "etf_trading_enabled": self._etf_trading_enabled,
                    "etf_alpha_from_a": config.etf.alpha_from_a,
                    "etf_alpha_from_a_earnings": config.etf.alpha_from_a_earnings,
                    "etf_alpha_from_a_news": config.etf.alpha_from_a_news,
                    "etf_max_position": config.etf.max_position,
                    "etf_quote_size": config.etf.quote_size,
                    "etf_target_position_per_a_shock_inventory": config.etf.target_position_per_a_shock_inventory,
                    "etf_min_hold_ms": config.etf.min_hold_ms,
                    "etf_min_target_position_for_major_a_shock": config.etf.min_target_position_for_major_a_shock,
                    "market_b_enabled": config.market_b.enabled,
                    "market_b_trading_enabled": self._b_trading_enabled,
                    "market_b_observe_only": self._b_observe_only,
                    "market_b_allow_trading_with_ayush_port": config.market_b.allow_trading_with_ayush_port,
                    "market_b_mm_v2_enabled": config.market_b.mm_v2_enabled,
                    "market_b_parity_enabled": config.market_b.parity_enabled,
                    "market_b_parity_shadow_enabled": config.market_b.parity_shadow_enabled,
                    "market_b_parity_edge_threshold_ticks": config.market_b.parity_edge_threshold_ticks,
                    "market_b_signal_snapshot_interval_ms": config.market_b.signal_snapshot_interval_ms,
                    "market_b_signal_change_threshold_ticks": config.market_b.signal_change_threshold_ticks,
                    "market_b_mm_min_eval_interval_ms": config.market_b.mm_min_eval_interval_ms,
                    "market_b_mm_reprice_threshold_ticks": config.market_b.mm_reprice_threshold_ticks,
                    "market_b_mm_min_valid_spread_ticks": config.market_b.mm_min_valid_spread_ticks,
                    "market_b_mm_min_healthy_book_age_ms": config.market_b.mm_min_healthy_book_age_ms,
                    "market_b_mm_bad_fill_cooldown_ms": config.market_b.mm_bad_fill_cooldown_ms,
                    "market_b_meanrev_enabled": config.market_b.meanrev_enabled,
                    "market_b_meanrev_max_position": config.market_b.meanrev_max_position,
                    "market_b_meanrev_entry_z": config.market_b.meanrev_entry_z,
                    "market_b_meanrev_entry_z2": config.market_b.meanrev_entry_z2,
                    "market_b_meanrev_stop_z": config.market_b.meanrev_stop_z,
                    "market_b_option_lottery_enabled": bool(self.b_option_strategy is not None),
                    "market_b_option_lottery_max_ask": config.market_b.option_lottery_max_ask,
                    "market_b_option_lottery_total_premium_budget": config.market_b.option_lottery_total_premium_budget,
                    "market_b_option_lottery_wing_max_position": config.market_b.option_lottery_wing_max_position,
                    "market_b_option_lottery_atm_max_position": config.market_b.option_lottery_atm_max_position,
                    "market_b_option_hedge_enabled": config.market_b.option_hedge_enabled,
                    "market_b_option_hedge_max_ask": config.market_b.option_hedge_max_ask,
                    "market_b_option_hedge_premium_budget": config.market_b.option_hedge_premium_budget,
                    "market_c_enabled": bool(self.c_strategy is not None),
                    "market_c_trading_enabled": self._c_trading_enabled,
                    "market_c_symbol": None if self.c_strategy is None else config.market_c.symbol,
                    "market_c_pm_symbols": [] if self.c_strategy is None else list(config.market_c.pm_symbols),
                    "market_c_live_earnings_enabled": False if self.c_strategy is None else config.market_c.live_earnings_enabled,
                    "market_c_live_cpi_enabled": False if self.c_strategy is None else config.market_c.live_cpi_enabled,
                    "market_c_live_macro_enabled": False if self.c_strategy is None else config.market_c.live_macro_enabled,
                    "market_c_mm_enabled": False if self.c_strategy is None else config.market_c.mm_enabled,
                    "etf_enable_c_earnings": config.etf.enable_c_earnings,
                    "etf_alpha_from_c_earnings": config.etf.alpha_from_c_earnings,
                    "etf_min_c_fair_shift_ticks": config.etf.min_c_fair_shift_ticks,
                    "etf_ac_conflict_policy": config.etf.ac_conflict_policy,
                    "etf_min_eval_interval_ms": config.etf.min_eval_interval_ms,
                    "etf_unwind_reprice_threshold_ticks": config.etf.unwind_reprice_threshold_ticks,
                    "auto_stop_on_market_resolved": config.auto_stop_on_market_resolved,
                    "auto_stop_on_followup_position_snapshot": config.auto_stop_on_followup_position_snapshot,
                },
                recovered_orders=[self._order_to_dict(order) for order in replay_state.live_orders],
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
            )

    def handle_position_snapshot(self, msg) -> None:
        """Use the exchange snapshot as the anchor for inventory and recovery startup."""
        super().handle_position_snapshot(msg)
        now_ms = self._now_ms()
        should_stop_after_snapshot = self._on_position_snapshot_received(now_ms)
        a_position = int(self.positions.get("A", 0))
        cash_value = int(self.positions.get("cash", 0))
        if self._ayush_port_mode:
            self.strategy.sync_inventory_from_exchange(a_position, now_ms=now_ms)
        else:
            self.strategy.sync_inventory_from_exchange(a_position)
        if self.b_strategy is not None:
            self.b_strategy.sync_inventory_from_exchange(int(self.positions.get(self.config.market_b.underlying_symbol, 0)))
        if self.b_option_strategy is not None:
            self.b_option_strategy.sync_inventory(
                self.config.market_b.underlying_symbol,
                int(self.positions.get(self.config.market_b.underlying_symbol, 0)),
                now_ms=now_ms,
            )
            for symbol in self.config.market_b.option_symbols:
                self.b_option_strategy.sync_inventory(symbol, int(self.positions.get(symbol, 0)), now_ms=now_ms)
        if self.etf_strategy is not None:
            self.etf_strategy.sync_inventory_from_exchange(int(self.positions.get(self.config.etf.symbol, 0)), now_ms=now_ms)
        if self.c_strategy is not None:
            self.c_strategy.sync_inventory_from_exchange(self.config.market_c.symbol, int(self.positions.get(self.config.market_c.symbol, 0)), now_ms=now_ms)
            for pm_symbol in self.config.market_c.pm_symbols:
                self.c_strategy.sync_inventory_from_exchange(pm_symbol, int(self.positions.get(pm_symbol, 0)), now_ms=now_ms)
        for symbol in self.b_observer.symbols:
            self.b_observer.sync_inventory(symbol, int(self.positions.get(symbol, 0)))
        self.journal.record_inventory(a_position, cash=cash_value)
        if self.tracer is not None:
            self.tracer.record_inventory_update(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=cash_value,
                trigger="position_snapshot",
            )
            for symbol in self.b_observer.symbols:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=cash_value,
                    trigger="position_snapshot",
                )
            if self.etf_strategy is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=cash_value,
                    trigger="position_snapshot",
                )
            if self.c_strategy is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=cash_value,
                    trigger="position_snapshot",
                )
        if should_stop_after_snapshot:
            return
        if not self._position_snapshot_seen.is_set():
            self._position_snapshot_seen.set()
            self._recovery_task = asyncio.create_task(self._start_recovery_after_snapshot())

    async def process_message(self, msg) -> None:
        """Mirror exchange position updates into local journaled strategy state."""
        if msg == grpc.aio.EOF:
            raise EOFError("End of gRPC stream")
        self._connected_once = True
        msg_type = msg.WhichOneof("body")
        if self._awaiting_reconnect_success:
            self._awaiting_reconnect_success = False
            self._record_runtime_event(
                "runtime_reconnect_succeeded",
                reason="first message received after reconnect",
                details={"message_type": msg_type, "reconnect_attempt_count": self._reconnect_attempt_count},
            )
        recovery_event: tuple[str, str, bool, str | None, int | None, int | None] | None = None
        if msg_type == "cancel_response":
            order_id = msg.cancel_response.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                result_type = msg.cancel_response.WhichOneof("result")
                recovery_event = (
                    "cancel_response",
                    order_id,
                    result_type == "ok",
                    None if result_type == "ok" else msg.cancel_response.error,
                    None,
                    None,
                )
        elif msg_type == "order_fill":
            order_id = msg.order_fill.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                recovery_event = (
                    "order_fill",
                    order_id,
                    True,
                    None,
                    int(msg.order_fill.qty),
                    int(msg.order_fill.px),
                )
        elif msg_type == "order_rejected":
            order_id = msg.order_rejected.id
            known_open_order = order_id in self.open_orders
            if order_id in self.strategy.recovery_pending and not known_open_order:
                recovery_event = (
                    "order_rejected",
                    order_id,
                    False,
                    msg.order_rejected.reason,
                    None,
                    None,
                )

        await super().process_message(msg)

        if recovery_event is not None:
            event_type, order_id, success, error, qty, price = recovery_event
            if event_type == "cancel_response":
                await self.bot_handle_cancel_response(order_id, success, error)
            elif event_type == "order_fill" and qty is not None and price is not None:
                await self.bot_handle_order_fill(order_id, qty, price)
            elif event_type == "order_rejected" and error is not None:
                await self.bot_handle_order_rejected(order_id, error)

        if msg_type == "position_update" and msg.position_update.symbol == "A":
            now_ms = self._now_ms()
            inventory = int(msg.position_update.value)
            if self._ayush_port_mode:
                self.strategy.sync_inventory_from_exchange(inventory, now_ms=now_ms)
            else:
                self.strategy.sync_inventory_from_exchange(inventory)
            self.journal.record_inventory(inventory, cash=int(self.positions.get("cash", 0)))
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger="position_update",
                )
            await self._evaluate_and_sync("position update")
        elif msg_type == "position_update" and msg.position_update.symbol in self.b_observer.symbols:
            now_ms = self._now_ms()
            symbol = msg.position_update.symbol
            inventory = int(msg.position_update.value)
            before_inventory = self._strategy_inventory_for_symbol(symbol)
            self._last_position_update_by_symbol[symbol] = (now_ms, before_inventory, inventory)
            self.b_observer.sync_inventory(symbol, inventory)
            if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                self.b_strategy.sync_inventory_from_exchange(inventory)
            if self.b_option_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                self.b_option_strategy.sync_inventory(symbol, inventory, now_ms=now_ms)
            if self.b_option_strategy is not None and symbol in self.config.market_b.option_symbols:
                self.b_option_strategy.sync_inventory(symbol, inventory, now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="position_update",
                )
            if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                await self._evaluate_and_sync_b("position update", force=True)
            if self.b_option_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                await self._evaluate_and_sync_b_options(f"position update:{symbol}", force=True)
            if self.b_option_strategy is not None and symbol in self.config.market_b.option_symbols:
                await self._evaluate_and_sync_b_options(f"position update:{symbol}", symbols=(symbol,), force=True)
        elif msg_type == "position_update" and msg.position_update.symbol == self.config.etf.symbol:
            now_ms = self._now_ms()
            inventory = int(msg.position_update.value)
            before_inventory = self._strategy_inventory_for_symbol(self.config.etf.symbol)
            self._last_position_update_by_symbol[self.config.etf.symbol] = (now_ms, before_inventory, inventory)
            if self.etf_strategy is not None:
                accepted = self.etf_strategy.sync_inventory_from_exchange(inventory, now_ms=now_ms)
                if not accepted:
                    self._last_position_update_by_symbol.pop(self.config.etf.symbol, None)
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="position_update",
                )
            if self.etf_strategy is not None:
                await self._evaluate_and_sync_etf("position update", force=True)
        elif self.c_strategy is not None and msg_type == "position_update" and (
            msg.position_update.symbol == self.config.market_c.symbol
            or msg.position_update.symbol in self.config.market_c.pm_symbols
        ):
            now_ms = self._now_ms()
            symbol = msg.position_update.symbol
            inventory = int(msg.position_update.value)
            before_inventory = self._strategy_inventory_for_symbol(symbol)
            self._last_position_update_by_symbol[symbol] = (now_ms, before_inventory, inventory)
            self.c_strategy.sync_inventory_from_exchange(symbol, inventory, now_ms=now_ms)
            if self.tracer is not None and symbol == self.config.market_c.symbol:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="position_update",
                )
            if symbol == self.config.market_c.symbol:
                await self._evaluate_and_sync_c("position update", force=True)
        elif msg_type == "cash_update":
            now_ms = self._now_ms()
            self.journal.record_inventory(
                int(self.strategy.inventory),
                cash=int(msg.cash_update.value),
            )
            if self.tracer is not None:
                self.tracer.record_inventory_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=int(msg.cash_update.value),
                    trigger="cash_update",
                )

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        if self.etf_strategy is not None and order_id in self.etf_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.etf_strategy.on_cancel_response(order_id, success)
            if self.tracer is not None:
                self.tracer.record_cancel_response(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    success=success,
                    error=error,
                    side=None if order is None else order.side,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_etf("cancel response", force=True)
            return

        if self.b_option_strategy is not None and self.b_option_strategy.has_order(order_id):
            now_ms = self._now_ms()
            symbol, order = self.b_option_strategy.on_cancel_response(order_id, success)
            if self.tracer is not None and symbol is not None:
                self.tracer.record_cancel_response(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    success=success,
                    error=error,
                    side=None if order is None else order.side,
                    order=None if order is None else self._order_to_dict(order),
                )
            if symbol is not None:
                await self._evaluate_and_sync_b_options("cancel response", symbols=(symbol,), force=True)
            return

        if self.b_strategy is not None and order_id in self.b_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.b_strategy.on_cancel_response(order_id, success)
            if self.tracer is not None:
                self.tracer.record_cancel_response(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    success=success,
                    error=error,
                    side=None if order is None else order.side,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_b("cancel response", force=True)
            return

        if self.c_strategy is not None and order_id in self.c_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.c_strategy.on_cancel_response(order_id, success)
            if self.tracer is not None:
                self.tracer.record_cancel_response(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    success=success,
                    error=error,
                    side=None if order is None else order.side,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_c("cancel response", force=True)
            return

        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_cancel_response(order_id, success)
        self.journal.record_cancel_response(order_id, success, error)
        if order is not None:
            LOGGER.info("Cancel response for %s success=%s error=%s", order_id, success, error)
        if self.tracer is not None:
            self.tracer.record_cancel_response(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                order_id=order_id,
                success=success,
                error=error,
                side=None if order is None else order.side,
                order=None if order is None else self._order_to_dict(order),
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order cancellation completed")
        await self._evaluate_and_sync("cancel response")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        if self.etf_strategy is not None and order_id in self.etf_strategy.order_manager.orders:
            now_ms = self._now_ms()
            existing_order = self.etf_strategy.order_manager.orders.get(order_id)
            side = "BUY" if existing_order is None else existing_order.side
            strategy_inventory = int(self.etf_strategy.inventory)
            authoritative_inventory, _pre_fill_inventory = self._authoritative_inventory_after_fill(
                self.config.etf.symbol,
                side,
                qty,
                strategy_inventory,
            )
            order = self.etf_strategy.on_fill(
                order_id,
                qty,
                price,
                authoritative_inventory=authoritative_inventory,
                now_ms=now_ms,
            )
            if self.tracer is not None:
                self.tracer.record_fill(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    order=None if order is None else self._order_to_dict(order),
                    order_id=order_id,
                    qty=qty,
                    price=price,
                )
            await self._evaluate_and_sync_etf("fill", force=True)
            return

        if self.b_option_strategy is not None and self.b_option_strategy.has_order(order_id):
            now_ms = self._now_ms()
            symbol, existing_order = self.b_option_strategy.order_for_id(order_id)
            if symbol is None:
                return
            side = "BUY" if existing_order is None else existing_order.side
            strategy_inventory = int(self.b_option_strategy.inventory(symbol))
            authoritative_inventory, pre_fill_inventory = self._authoritative_inventory_after_fill(
                symbol,
                side,
                qty,
                strategy_inventory,
            )
            symbol, order = self.b_option_strategy.on_fill(
                order_id,
                qty,
                price,
                now_ms=now_ms,
                authoritative_inventory=authoritative_inventory,
                pre_fill_inventory=pre_fill_inventory,
            )
            if symbol is not None:
                self.b_observer.sync_inventory(symbol, authoritative_inventory)
                if self.b_option_strategy is not None:
                    self.b_option_strategy.sync_inventory(
                        self.config.market_b.underlying_symbol,
                        int(self.positions.get(self.config.market_b.underlying_symbol, 0)),
                        now_ms=now_ms,
                    )
            if self.tracer is not None and symbol is not None:
                self.tracer.record_fill(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    order=None if order is None else self._order_to_dict(order),
                    order_id=order_id,
                    qty=qty,
                    price=price,
                )
            if symbol is not None:
                await self._evaluate_and_sync_b_options("fill", symbols=(symbol,), force=True)
            return

        if self.b_strategy is not None and order_id in self.b_strategy.order_manager.orders:
            now_ms = self._now_ms()
            existing_order = self.b_strategy.order_manager.orders.get(order_id)
            side = "BUY" if existing_order is None else existing_order.side
            strategy_inventory = int(self.b_strategy.inventory)
            authoritative_inventory, _pre_fill_inventory = self._authoritative_inventory_after_fill(
                self.config.market_b.underlying_symbol,
                side,
                qty,
                strategy_inventory,
            )
            order = self.b_strategy.on_fill(order_id, qty, price, authoritative_inventory=authoritative_inventory)
            self.b_observer.sync_inventory(self.config.market_b.underlying_symbol, authoritative_inventory)
            if self.b_option_strategy is not None:
                self.b_option_strategy.sync_inventory(
                    self.config.market_b.underlying_symbol,
                    authoritative_inventory,
                    now_ms=now_ms,
                )
            if self.tracer is not None:
                self.tracer.record_fill(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                    cash=self._current_cash(),
                    order=None if order is None else self._order_to_dict(order),
                    order_id=order_id,
                    qty=qty,
                    price=price,
                )
            await self._evaluate_and_sync_b("fill", force=True)
            if self.b_option_strategy is not None:
                await self._evaluate_and_sync_b_options("underlying fill", force=True)
            return

        if self.c_strategy is not None and order_id in self.c_strategy.order_manager.orders:
            now_ms = self._now_ms()
            existing_order = self.c_strategy.order_manager.orders.get(order_id)
            side = "BUY" if existing_order is None else existing_order.side
            strategy_inventory = int(self.c_strategy.inventory)
            authoritative_inventory, _pre_fill_inventory = self._authoritative_inventory_after_fill(
                self.config.market_c.symbol,
                side,
                qty,
                strategy_inventory,
            )
            order = self.c_strategy.on_fill(
                order_id,
                qty,
                price,
                authoritative_inventory=authoritative_inventory,
            )
            if self.tracer is not None:
                self.tracer.record_fill(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    order=None if order is None else self._order_to_dict(order),
                    order_id=order_id,
                    qty=qty,
                    price=price,
                )
            await self._evaluate_and_sync_c("fill", force=True)
            return

        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_fill(order_id, qty, price)
        self.journal.record_fill(order_id, qty, price)
        if order is not None:
            LOGGER.info(
                "Fill on %s order %s: %s %s @ %s, estimated inventory=%s",
                order.side,
                order_id,
                qty,
                self.symbol_from_side(order.side),
                price,
                self.strategy.inventory,
            )
        else:
            LOGGER.info("Received fill for unmanaged order %s qty=%s px=%s", order_id, qty, price)
        self.journal.record_inventory(int(self.strategy.inventory), cash=int(self.positions.get("cash", 0)))
        if self.tracer is not None:
            self.tracer.record_fill(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                order=None if order is None else self._order_to_dict(order),
                order_id=order_id,
                qty=qty,
                price=price,
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order filled")
        await self._evaluate_and_sync("fill")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        if self.etf_strategy is not None and order_id in self.etf_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.etf_strategy.on_rejection(order_id)
            LOGGER.warning("ETF order %s rejected: %s", order_id, reason)
            if self.tracer is not None:
                self.tracer.record_rejection(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    reason=reason,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_etf("rejection", force=True)
            return

        if self.b_option_strategy is not None and self.b_option_strategy.has_order(order_id):
            now_ms = self._now_ms()
            symbol, order = self.b_option_strategy.on_rejection(order_id)
            LOGGER.warning("B option order %s rejected: %s", order_id, reason)
            if self.tracer is not None and symbol is not None:
                self.tracer.record_rejection(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    reason=reason,
                    order=None if order is None else self._order_to_dict(order),
                )
            if symbol is not None:
                await self._evaluate_and_sync_b_options("rejection", symbols=(symbol,), force=True)
            return

        if self.b_strategy is not None and order_id in self.b_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.b_strategy.on_rejection(order_id)
            LOGGER.warning("B order %s rejected: %s", order_id, reason)
            if self.tracer is not None:
                self.tracer.record_rejection(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    reason=reason,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_b("rejection", force=True)
            return

        if self.c_strategy is not None and order_id in self.c_strategy.order_manager.orders:
            now_ms = self._now_ms()
            order = self.c_strategy.on_rejection(order_id)
            LOGGER.warning("C order %s rejected: %s", order_id, reason)
            if self.tracer is not None:
                self.tracer.record_rejection(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    order_id=order_id,
                    reason=reason,
                    order=None if order is None else self._order_to_dict(order),
                )
            await self._evaluate_and_sync_c("rejection", force=True)
            return

        was_recovery_active = self.strategy.recovery_active
        now_ms = self._now_ms()
        order = self.strategy.on_rejection(order_id)
        self.journal.record_rejection(order_id, reason)
        LOGGER.warning("Order %s rejected: %s", order_id, reason)
        if self.tracer is not None:
            self.tracer.record_rejection(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                order_id=order_id,
                reason=reason,
                order=None if order is None else self._order_to_dict(order),
            )
            self._trace_recovery_transition(was_recovery_active, "recovery order rejected")
        await self._evaluate_and_sync("rejection")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol == "A":
            now_ms = self._now_ms()
            self.strategy.on_market_trade(price, qty, now_ms=now_ms)
            LOGGER.debug("Trade in A at %s for %s", price, qty)
            if self.tracer is not None:
                self.tracer.record_market_trade(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    price=price,
                    qty=qty,
                )
            return

        if self.config.market_b.enabled and symbol in self.b_observer.symbols:
            now_ms = self._now_ms()
            self.b_observer.on_market_trade(symbol, price, qty, now_ms=now_ms)
            if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                self.b_strategy.on_market_trade(price, qty, now_ms=now_ms)
            if self.b_option_strategy is not None:
                self.b_option_strategy.on_market_trade(symbol, price, qty, now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_market_trade(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    price=price,
                    qty=qty,
                )
                self._record_b_signal(now_ms)
            if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
                await self._evaluate_and_sync_b(f"market trade:{symbol}")
            if self.b_option_strategy is not None:
                option_symbols = self.config.market_b.option_symbols if symbol == self.config.market_b.underlying_symbol else (symbol,)
                await self._evaluate_and_sync_b_options(f"market trade:{symbol}", symbols=option_symbols)
            return

        if self.etf_strategy is not None and symbol == self.config.etf.symbol:
            now_ms = self._now_ms()
            self.etf_strategy.on_market_trade(price, qty, now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_market_trade(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    price=price,
                    qty=qty,
                )
            await self._evaluate_and_sync_etf(f"market trade:{symbol}")
            return

        if self.c_strategy is not None and (
            symbol == self.config.market_c.symbol or symbol in self.config.market_c.pm_symbols
        ):
            now_ms = self._now_ms()
            self.c_strategy.on_market_trade(symbol, price, qty, now_ms=now_ms)
            if self.tracer is not None and symbol == self.config.market_c.symbol:
                self.tracer.record_market_trade(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    price=price,
                    qty=qty,
                )
            await self._evaluate_and_sync_c(f"market trade:{symbol}", force=(symbol == self.config.market_c.symbol))
            return

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol == "A":
            now_ms = self._now_ms()
            self.strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
            if self.tracer is not None:
                self.tracer.record_book_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger="book_update",
                )
            await self._evaluate_and_sync("book update")
            return

        if self.etf_strategy is not None and symbol == self.config.etf.symbol:
            now_ms = self._now_ms()
            self.etf_strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
            if self.tracer is not None:
                self.tracer.record_book_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="book_update",
                )
            await self._evaluate_and_sync_etf(f"book update:{symbol}")
            return

        if self.c_strategy is not None and (
            symbol == self.config.market_c.symbol or symbol in self.config.market_c.pm_symbols
        ):
            now_ms = self._now_ms()
            self.c_strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
            if self.tracer is not None and symbol == self.config.market_c.symbol:
                self.tracer.record_book_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="book_update",
                )
                self.tracer.maybe_record_periodic_snapshot(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    trigger="book_update",
                )
            await self._evaluate_and_sync_c(f"book update:{symbol}", force=(symbol == self.config.market_c.symbol))
            return

        if not self.config.market_b.enabled or symbol not in self.b_observer.symbols:
            return

        now_ms = self._now_ms()
        self.b_observer.on_book_update(symbol, self.order_books[symbol], now_ms=now_ms)
        force_b_eval = False
        if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
            self.b_strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
            force_b_eval = bool(
                getattr(self.b_strategy, "should_force_bad_book_cancel", lambda: False)()
            )
        if self.b_option_strategy is not None:
            self.b_option_strategy.on_book_update_at(symbol, self.order_books[symbol], now_ms)
        if self.tracer is not None:
            self.tracer.record_book_update(
                now_ms=now_ms,
                state=self._trace_state_for_symbol(symbol, now_ms),
                cash=self._current_cash(),
                trigger="book_update",
            )
            self.tracer.maybe_record_periodic_snapshot(
                now_ms=now_ms,
                state=self._trace_state_for_symbol(symbol, now_ms),
                cash=self._current_cash(),
                trigger="book_update",
            )
            self._record_b_signal(now_ms)
        if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
            await self._evaluate_and_sync_b(f"book update:{symbol}", force=force_b_eval)
        if self.b_option_strategy is not None:
            option_symbols = self.config.market_b.option_symbols if symbol == self.config.market_b.underlying_symbol else (symbol,)
            await self._evaluate_and_sync_b_options(f"book update:{symbol}", symbols=option_symbols)
        return

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool) -> None:
        LOGGER.info("Ignoring swap response in A-only bot: %s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict) -> None:
        now_ms = self._now_ms()
        mode_before_news = self.strategy.mode
        reaction = self.strategy.on_news(news_release, now_ms)
        if reaction.note:
            LOGGER.info(reaction.note)
        if self.tracer is not None:
            reaction_dict = self._reaction_to_dict(reaction)
            reaction_dict["mode_before_news"] = mode_before_news
            reaction_dict["mode_after_news"] = self.strategy.mode
            reaction_dict["news_kind"] = news_release.get("kind")
            self.tracer.record_news(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                news_payload=news_release,
                reaction=reaction_dict,
                )
        etf_signal = None
        if self.etf_strategy is not None:
            etf_signal = self.etf_strategy.on_a_news_reaction(reaction, now_ms=now_ms)
            if etf_signal is not None and self.tracer is not None:
                self.tracer.record_signal(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    strategy_family="etf_a_follower",
                    action_class="a_shock_projection",
                    pnl_owner="etf_a_follower",
                    signal_id=etf_signal.signal_id,
                    trade_group_id=etf_signal.signal_id,
                    leg_role="single",
                    payload=self.etf_strategy.signal_payload(etf_signal),
                )
            if etf_signal is not None:
                await self._evaluate_and_sync_etf("A shock signal", force=True)
        c_result = None
        if self.c_strategy is not None:
            c_result = self.c_strategy.on_news(news_release, now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_news(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    news_payload=news_release,
                    reaction={
                        "relevant": bool(c_result.get("signals")),
                        "fair_value_updated": False,
                        "note": c_result.get("note"),
                        "earnings_value": c_result.get("earnings_value"),
                        "resolved_news_text": str((news_release.get("new_data") or {}).get("content") or news_release.get("content") or ""),
                        "news_kind": news_release.get("kind"),
                        "tick": c_result.get("tick"),
                    },
                )
            for parsed_signal in c_result.get("signals", ()):
                if self.tracer is not None:
                    self.tracer.record_signal(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                        cash=self._current_cash(),
                        strategy_family=parsed_signal.strategy_family,
                        action_class=parsed_signal.action_class,
                        pnl_owner=parsed_signal.strategy_family,
                        signal_id=parsed_signal.signal_id,
                        trade_group_id=parsed_signal.signal_id,
                        leg_role="single",
                        payload=parsed_signal.payload,
                    )
            if any(
                parsed_signal.strategy_family == "c_earnings" and parsed_signal.action_class == "earnings_signal"
                for parsed_signal in c_result.get("signals", ())
            ):
                await self._evaluate_and_sync_c("C earnings signal", force=True)
                await self._maybe_emit_c_etf_projection("C earnings signal", now_ms=now_ms)
        if not reaction.relevant:
            if reaction.tick is not None:
                await self._evaluate_and_sync("news tick")
            return
        if reaction.fair_value_updated and self.strategy.fair_value is not None:
            self.journal.record_fair_value(
                fair_value=self.strategy.fair_value,
                source=self.strategy.valuation.last_source,
                earnings_value=self.strategy.valuation.last_earnings_value,
            )
            if self.tracer is not None:
                self.tracer.record_valuation_update(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    source="structured_news_provisional_fair",
                    details=self._reaction_to_dict(reaction),
                )
        await self._evaluate_and_sync("news")

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        LOGGER.info("Received market resolution %s winner=%s tick=%s", market_id, winning_symbol, tick)
        self._record_runtime_event(
            "market_resolved_observed",
            reason="market_resolved",
            details={"market_id": market_id, "winning_symbol": winning_symbol, "tick": int(tick)},
        )
        if self.config.auto_stop_after_round_complete and self.config.auto_stop_on_market_resolved:
            self._request_round_complete_stop("market_resolved")

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        LOGGER.info("Settlement payout user=%s market=%s amount=%s tick=%s", user, market_id, amount, tick)
        self._record_runtime_event(
            "settlement_payout_observed",
            reason="settlement_payout",
            details={"user": user, "market_id": market_id, "amount": int(amount), "tick": int(tick)},
        )
        if self.config.auto_stop_after_round_complete:
            self._request_round_complete_stop("settlement_payout")

    def _on_position_snapshot_received(self, now_ms: int) -> bool:
        self._position_snapshot_count += 1
        if self._position_snapshot_count == 1:
            self._ensure_round_end_watcher()
            return False
        if self.config.auto_stop_after_round_complete and self.config.auto_stop_on_followup_position_snapshot:
            self._request_round_complete_stop("followup_position_snapshot")
            return True
        return False

    def _ensure_round_end_watcher(self) -> None:
        if self._round_end_task is not None or not self.config.auto_stop_after_round_complete:
            return
        if not self.config.market_a.startup_assume_fresh_round:
            return
        self._round_end_task = asyncio.create_task(self._round_completion_watch_loop())

    async def _round_completion_watch_loop(self) -> None:
        timeout_s = max(0.0, (self.config.assumed_round_duration_ms + self.config.round_completion_grace_ms) / 1000.0)
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._request_round_complete_stop("wall_clock_round_timeout")

    def _request_round_complete_stop(self, reason: str) -> None:
        if self._auto_stop_requested:
            return
        self._auto_stop_requested = True
        self._shutdown_note = f"round_complete_auto_stop:{reason}"
        LOGGER.info("Round completion detected via %s; shutting down after finalization.", reason)
        self._shutdown.set()
        if self.call is not None:
            self.call.cancel()

    async def start(self) -> None:
        self._refresh_task = asyncio.create_task(self._quote_refresh_loop())
        try:
            while not self._shutdown.is_set():
                try:
                    await self.connect()
                    break
                except EOFError:
                    self.connected = False
                    self._emit_disconnect_alert(
                        "eof",
                        details=f"elapsed_runtime_ms={self._elapsed_runtime_ms()}",
                    )
                    self._record_runtime_event(
                        "runtime_disconnect",
                        reason="eof",
                        details={"elapsed_runtime_ms": self._elapsed_runtime_ms()},
                    )
                    if self._should_retry_unexpected_disconnect():
                        self._reconnect_attempt_count += 1
                        self._awaiting_reconnect_success = True
                        self._record_runtime_event(
                            "runtime_reconnect_attempt",
                            reason="unexpected eof before round-end cutoff",
                            details={
                                "elapsed_runtime_ms": self._elapsed_runtime_ms(),
                                "attempt": self._reconnect_attempt_count,
                            },
                        )
                        LOGGER.warning(
                            "Exchange stream ended unexpectedly after %.1fs; retrying in %.1fs.",
                            self._elapsed_runtime_ms() / 1000.0,
                            self._UNEXPECTED_DISCONNECT_RETRY_DELAY_S,
                        )
                        await asyncio.sleep(self._UNEXPECTED_DISCONNECT_RETRY_DELAY_S)
                        continue
                    if not self._auto_stop_requested and self._shutdown_note == "Bot shutdown":
                        self._shutdown_note = "exchange_stream_closed"
                    LOGGER.info("Exchange stream closed cleanly (%s).", self._shutdown_note)
                    break
                except grpc.aio.AioRpcError as exc:
                    if self._auto_stop_requested and exc.code() == grpc.StatusCode.CANCELLED:
                        LOGGER.info("Exchange stream cancelled cleanly (%s).", self._shutdown_note)
                        break
                    self.connected = False
                    self._emit_disconnect_alert(
                        f"grpc:{exc.code().name}",
                        details=exc.details(),
                    )
                    self._record_runtime_event(
                        "runtime_disconnect",
                        reason=f"grpc:{exc.code().name}",
                        details={"elapsed_runtime_ms": self._elapsed_runtime_ms(), "details": exc.details()},
                    )
                    if self._is_retryable_stream_error(exc) and self._should_retry_unexpected_disconnect():
                        self._reconnect_attempt_count += 1
                        self._awaiting_reconnect_success = True
                        self._record_runtime_event(
                            "runtime_reconnect_attempt",
                            reason=f"retryable grpc {exc.code().name} before round-end cutoff",
                            details={
                                "elapsed_runtime_ms": self._elapsed_runtime_ms(),
                                "attempt": self._reconnect_attempt_count,
                                "grpc_details": exc.details(),
                            },
                        )
                        LOGGER.warning(
                            "Exchange stream failed with retryable %s after %.1fs: %s. Retrying in %.1fs.",
                            exc.code().name,
                            self._elapsed_runtime_ms() / 1000.0,
                            exc.details(),
                            self._UNEXPECTED_DISCONNECT_RETRY_DELAY_S,
                        )
                        await asyncio.sleep(self._UNEXPECTED_DISCONNECT_RETRY_DELAY_S)
                        continue
                    raise
        except asyncio.CancelledError:
            if not self._auto_stop_requested:
                raise
            LOGGER.info("Exchange stream task cancelled cleanly (%s).", self._shutdown_note)
        finally:
            self._shutdown.set()
            if self._refresh_task is not None:
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
            if self._round_end_task is not None:
                self._round_end_task.cancel()
                try:
                    await self._round_end_task
                except asyncio.CancelledError:
                    pass
            if hasattr(self.journal, "record_session_finished"):
                self.journal.record_session_finished(note=self._shutdown_note)
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.finalize(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    note=self._shutdown_note,
                )

    def _elapsed_runtime_ms(self) -> int:
        return max(0, self._now_ms() - self._runtime_started_ms)

    def _unexpected_disconnect_retry_cutoff_ms(self) -> int:
        assumed = max(0, int(self.config.assumed_round_duration_ms))
        return assumed + max(0, int(self.config.round_completion_grace_ms))

    def _should_retry_unexpected_disconnect(self) -> bool:
        if self._shutdown.is_set() or self._auto_stop_requested:
            return False
        if not self._connected_once or not self._position_snapshot_seen.is_set():
            return True
        return self._elapsed_runtime_ms() < self._unexpected_disconnect_retry_cutoff_ms()

    def _should_alert_disconnect(self) -> bool:
        raw = os.getenv("BOT_DISCONNECT_ALERT_ENABLED")
        if raw is None or raw.strip() == "":
            return False
        return raw.strip().lower() not in {"0", "false", "no", "off"} and not self._auto_stop_requested

    def _emit_disconnect_alert(self, reason: str, details: str | None = None) -> None:
        if not self._should_alert_disconnect():
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] WARNING unexpected bot disconnect: {reason}"
        if details:
            message = f"{message} | {details}"
        try:
            print(message, file=sys.stderr, flush=True)
            print("\a", end="", file=sys.stderr, flush=True)
        except Exception:
            LOGGER.exception("Failed to emit disconnect warning to stderr.")

        afplay = shutil.which("afplay")
        if not afplay:
            return
        sound_path = (os.getenv("BOT_DISCONNECT_ALERT_SOUND") or "/System/Library/Sounds/Basso.aiff").strip()
        if not sound_path:
            return
        sound_file = Path(sound_path)
        if not sound_file.exists():
            return
        try:
            subprocess.Popen(
                [afplay, str(sound_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            LOGGER.exception("Failed to play disconnect alert sound.")

    @staticmethod
    def _is_retryable_stream_error(exc: grpc.aio.AioRpcError) -> bool:
        retryable_codes = {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.UNKNOWN,
            grpc.StatusCode.CANCELLED,
        }
        return exc.code() in retryable_codes

    def _record_runtime_event(self, event_type: str, *, reason: str, details: dict | None = None) -> None:
        if getattr(self, "tracer", None) is None:
            return
        now_ms = self._now_ms()
        self.tracer.record_runtime_event(
            event_type=event_type,
            now_ms=now_ms,
            state=self._trace_state_for_symbol("A", now_ms),
            cash=self._current_cash(),
            reason=reason,
            details=details,
        )

    async def _start_recovery_after_snapshot(self) -> None:
        if self.strategy.recovery_active:
            LOGGER.info("Entering startup recovery mode for A; cancelling restored orders before quoting.")
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.record_recovery_state(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    reason="startup recovery began",
            )
            for order in self.strategy.recovery_orders_to_cancel():
                now_ms = self._now_ms()
                if self._ayush_port_mode:
                    tracked_order = self.strategy.mark_cancel_requested(order.order_id, now_ms)
                else:
                    tracked_order = self.strategy.order_manager.mark_cancel_requested(order.order_id, now_ms)
                self.journal.record_cancel_requested(order.order_id)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol("A", now_ms),
                        cash=self._current_cash(),
                        order_id=order.order_id,
                        side=order.side,
                        cancel_reason="recovery cancel on startup",
                        mode_at_cancel=order.mode_at_submit or "recovery_cancel",
                        order=None if tracked_order is None else self._order_to_dict(tracked_order),
                    )
                await self.cancel_order(order.order_id)
                LOGGER.info("Requested cancel for recovered %s order %s @ %s", order.side, order.order_id, order.px)
        else:
            self.strategy.on_recovery_complete()
            if self.tracer is not None:
                now_ms = self._now_ms()
                self.tracer.record_recovery_state(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    reason="no recovery work required",
                )
        await self._evaluate_and_sync("startup recovery")
        if self.b_strategy is not None:
            await self._evaluate_and_sync_b("startup", force=True)
        if self.b_option_strategy is not None:
            await self._evaluate_and_sync_b_options("startup", force=True)
        if self.etf_strategy is not None:
            await self._evaluate_and_sync_etf("startup", force=True)
        if self.c_strategy is not None:
            await self._evaluate_and_sync_c("startup", force=True)

    async def _quote_refresh_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(max(self.config.risk.reprice_cooldown_ms / 1000.0, 0.25))
            if self._shutdown.is_set():
                break
            await self._evaluate_and_sync("timer")
            if self._b_trading_enabled and self.b_strategy is not None:
                await self._evaluate_and_sync_b("timer")
            if self.b_option_strategy is not None:
                await self._evaluate_and_sync_b_options("timer")
            if self.c_strategy is not None:
                await self._evaluate_and_sync_c("timer")
            if self._etf_trading_enabled and self.etf_strategy is not None:
                await self._evaluate_and_sync_etf("timer")

    async def _evaluate_and_sync(self, reason: str) -> None:
        if self._ayush_port_mode:
            await self._evaluate_and_sync_ayush(reason)
            return
        if self._shutdown.is_set() or not self.connected or not self._position_snapshot_seen.is_set():
            return
        async with self._quote_lock:
            # The strategy only decides what the bot wants to own on each side.
            # The order manager turns that target state into cancel/place actions.
            now_ms = self._now_ms()
            prior_fair = self.strategy.fair_value
            prior_multiplier = self.strategy.trusted_multiplier
            prior_confidence = self.strategy.multiplier_confidence
            plan = self.strategy.compute_quotes(now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )
            for event in self.strategy.drain_learning_events():
                if event.status == "skipped":
                    LOGGER.warning("Skipped A multiplier calibration: %s", event.reason)
                else:
                    LOGGER.info(
                        "A multiplier %s via %s estimate=%.4f settled_mid=%s trusted=%.4f confidence=%s tolerance=%s",
                        event.status,
                        event.method,
                        event.estimate,
                        "n/a" if event.settled_mid is None else f"{event.settled_mid:.2f}",
                        event.trusted_multiplier or 0.0,
                        event.confidence,
                        "n/a" if event.tolerance is None else f"{event.tolerance:.2f}",
                    )
                if event.trusted_multiplier is not None and event.status != "skipped":
                    self.journal.record_multiplier(
                        multiplier=event.trusted_multiplier,
                        confidence=event.confidence,
                        source=event.status,
                        estimate=event.estimate,
                        method=event.method,
                    )
                if self.tracer is not None:
                    self.tracer.record_valuation_update(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol("A", now_ms),
                        cash=self._current_cash(),
                        source="multiplier_learning",
                        details={
                            "old_fair_value": prior_fair,
                            "new_fair_value": self.strategy.fair_value,
                            "earnings_value": self.strategy.valuation.last_earnings_value,
                            "status": event.status,
                            "estimate": event.estimate,
                            "trusted_multiplier": event.trusted_multiplier,
                            "confidence": event.confidence,
                            "method": event.method,
                            "settled_mid": event.settled_mid,
                            "reason": event.reason,
                            "tolerance": event.tolerance,
                        },
                    )
            if self.strategy.fair_value is not None and self.strategy.fair_value != prior_fair:
                self.journal.record_fair_value(
                    fair_value=self.strategy.fair_value,
                    source=self.strategy.valuation.last_source,
                    earnings_value=self.strategy.valuation.last_earnings_value,
                )
                if prior_fair is None:
                    LOGGER.info("A fair value initialized at %s after multiplier learning.", self.strategy.fair_value)
                elif self.strategy.trusted_multiplier != prior_multiplier or self.strategy.multiplier_confidence != prior_confidence:
                    LOGGER.info("A fair value refreshed from %s to %s after multiplier update.", prior_fair, self.strategy.fair_value)
            if plan.mode != self._last_mode:
                LOGGER.info(
                    "A mode -> %s fair=%s inventory=%s news_caution_remaining_ms=%s reason=%s",
                    plan.mode,
                    self.strategy.fair_value,
                    self.strategy.inventory,
                    max(0, self.strategy.news_caution_until_ms - now_ms),
                    plan.reason,
                )
                self._last_mode = plan.mode
            if plan.observe_only:
                if plan.reason != self._last_observe_only_reason:
                    LOGGER.info("A bot observe-only: %s", plan.reason)
                    self._last_observe_only_reason = plan.reason
            else:
                self._last_observe_only_reason = None
            actions = self.strategy.order_manager.build_actions(plan, now_ms)

            for cancel in actions.cancels:
                order = self.strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                self.journal.record_cancel_requested(cancel.order_id)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol("A", now_ms),
                        cash=self._current_cash(),
                        order_id=cancel.order_id,
                        side=cancel.side,
                        cancel_reason=cancel.reason,
                        mode_at_cancel=self.strategy.mode,
                        order=None if order is None else self._order_to_dict(order),
                    )
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling %s order %s because %s", cancel.side, cancel.order_id, cancel.reason)

            for placement in actions.placements:
                side = Side.BUY if placement.side == "BUY" else Side.SELL
                order_id = await self.place_order("A", placement.qty, side, placement.px)
                managed_order = self.strategy.order_manager.note_submitted(
                    order_id=order_id,
                    side=placement.side,
                    px=placement.px,
                    qty=placement.qty,
                    now_ms=now_ms,
                    overlay=placement.overlay,
                    aggressive=placement.aggressive,
                    intent=placement.intent,
                    mode_at_submit=placement.mode_at_submit,
                    evaluation_reason=placement.evaluation_reason,
                    market_key=placement.market_key,
                    strategy_family=placement.strategy_family,
                    action_class=placement.action_class,
                    pnl_owner=placement.pnl_owner,
                    signal_id=placement.signal_id,
                    trade_group_id=placement.trade_group_id,
                    leg_role=placement.leg_role,
                )
                self.journal.record_order_submitted(managed_order)
                if self.tracer is not None:
                    self.tracer.record_order_submitted(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol("A", now_ms),
                        cash=self._current_cash(),
                        order=self._order_to_dict(managed_order),
                    )
                LOGGER.info(
                    "Placed %s %s order %s for A: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )
            if self.tracer is not None:
                self.tracer.maybe_record_periodic_snapshot(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                )

    async def _evaluate_and_sync_ayush(self, reason: str) -> None:
        if self._shutdown.is_set() or not self.connected or not self._position_snapshot_seen.is_set():
            return
        async with self._quote_lock:
            now_ms = self._now_ms()
            prior_fair = self.strategy.fair_value
            prior_multiplier = self.strategy.trusted_multiplier
            prior_confidence = self.strategy.multiplier_confidence
            decision, plan = self.strategy.evaluate_runtime(now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )
            if self.strategy.fair_value is not None and self.strategy.fair_value != prior_fair:
                self.journal.record_fair_value(
                    fair_value=self.strategy.fair_value,
                    source=self.strategy.valuation.last_source,
                    earnings_value=self.strategy.valuation.last_earnings_value,
                )
                if prior_fair is None:
                    LOGGER.info("Ayush A fair value initialized at %s.", self.strategy.fair_value)
                elif self.strategy.trusted_multiplier != prior_multiplier or self.strategy.multiplier_confidence != prior_confidence:
                    LOGGER.info(
                        "Ayush A fair value refreshed from %s to %s after multiplier update.",
                        prior_fair,
                        self.strategy.fair_value,
                    )
            if plan.mode != self._last_mode:
                LOGGER.info(
                    "Ayush A mode -> %s fair=%s inventory=%s reason=%s",
                    plan.mode,
                    self.strategy.fair_value,
                    self.strategy.inventory,
                    plan.reason,
                )
                self._last_mode = plan.mode
            if plan.observe_only:
                if plan.reason != self._last_observe_only_reason:
                    LOGGER.info("Ayush A observe-only: %s", plan.reason)
                    self._last_observe_only_reason = plan.reason
            else:
                self._last_observe_only_reason = None
            await self._apply_ayush_decision(decision, now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.maybe_record_periodic_snapshot(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                )

    async def _apply_ayush_decision(self, decision, *, now_ms: int) -> None:
        if self.strategy.recovery_active:
            return
        current_orders = self.strategy.current_orders()
        desired = None if decision is None else self.strategy.normalize_desired_order(decision)

        if self.strategy.any_cancel_pending():
            return

        if current_orders:
            cancel_reason = getattr(decision, "reason", "") if desired is None else "replace"
            if desired is None or bool(getattr(decision, "cancel_all", False)) or not self.strategy.can_keep_orders(current_orders, desired, now_ms):
                await self._cancel_ayush_orders(current_orders, now_ms=now_ms, cancel_reason=cancel_reason)
                return

        if desired is None:
            return

        live_matching_qty = sum(
            order.remaining_qty
            for order in current_orders
            if self.strategy.order_matches(order, desired, now_ms)
        )
        remaining_target_qty = max(0, int(desired.qty) - live_matching_qty)
        if remaining_target_qty <= 0:
            return

        slice_qty = self.strategy.next_slice_qty(remaining_target_qty)
        staged_desired = replace(desired, qty=slice_qty)
        adjusted = self.strategy.risk_adjusted_order(staged_desired)
        if adjusted is None:
            return
        await self._submit_ayush_order(adjusted, now_ms=now_ms)

    async def _cancel_ayush_orders(self, orders, *, now_ms: int, cancel_reason: str) -> None:
        for order in orders:
            if order.cancel_pending:
                continue
            tracked_order = self.strategy.mark_cancel_requested(order.order_id, now_ms)
            self.journal.record_cancel_requested(order.order_id)
            if self.tracer is not None:
                self.tracer.record_cancel_requested(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol("A", now_ms),
                    cash=self._current_cash(),
                    order_id=order.order_id,
                    side=order.side,
                    cancel_reason=cancel_reason,
                    mode_at_cancel=self.strategy.mode,
                    order=None if tracked_order is None else self._order_to_dict(tracked_order),
                )
            await self.cancel_order(order.order_id)
            LOGGER.info("Cancelling Ayush A %s order %s because %s", order.side, order.order_id, cancel_reason)

    async def _submit_ayush_order(self, desired, *, now_ms: int) -> None:
        side = Side.BUY if desired.side == "BUY" else Side.SELL
        order_id = await self.place_order("A", desired.qty, side, desired.px)
        managed_order = self.strategy.note_submitted(order_id=order_id, desired=desired, now_ms=now_ms)
        self.journal.record_order_submitted(managed_order)
        if self.tracer is not None:
            self.tracer.record_order_submitted(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                order=self._order_to_dict(managed_order),
            )
        LOGGER.info(
            "Placed %s %s order %s for Ayush A: qty=%s px=%s reason=%s",
            "aggressive" if desired.aggressive else "passive",
            desired.side,
            order_id,
            desired.qty,
            desired.px,
            desired.reason,
        )

    async def _evaluate_and_sync_c(self, reason: str, *, force: bool = False) -> None:
        if (
            self._shutdown.is_set()
            or self.c_strategy is None
            or not self.connected
            or not self._position_snapshot_seen.is_set()
            or not self.config.market_c.enabled
        ):
            return
        now_ms = self._now_ms()
        min_interval = max(0, int(self.config.market_c.min_eval_interval_ms))
        if not force and self._last_c_eval_ms is not None and now_ms - self._last_c_eval_ms < min_interval:
            return
        async with self._quote_lock:
            now_ms = self._now_ms()
            if not force and self._last_c_eval_ms is not None and now_ms - self._last_c_eval_ms < min_interval:
                return
            self._last_c_eval_ms = now_ms
            plan = self.c_strategy.compute_quotes(now_ms=now_ms)
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )
            actions = self.c_strategy.order_manager.build_actions(plan, now_ms)
            for cancel in actions.cancels:
                order = self.c_strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                        cash=self._current_cash(),
                        order_id=cancel.order_id,
                        side=cancel.side,
                        cancel_reason=cancel.reason,
                        mode_at_cancel=plan.mode,
                        order=None if order is None else self._order_to_dict(order),
                    )
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling C %s order %s because %s", cancel.side, cancel.order_id, cancel.reason)
            if not self._c_trading_enabled:
                return
            for placement in actions.placements:
                side = Side.BUY if placement.side == "BUY" else Side.SELL
                order_id = await self.place_order(self.config.market_c.symbol, placement.qty, side, placement.px)
                managed_order = self.c_strategy.order_manager.note_submitted(
                    order_id=order_id,
                    side=placement.side,
                    px=placement.px,
                    qty=placement.qty,
                    now_ms=now_ms,
                    overlay=placement.overlay,
                    aggressive=placement.aggressive,
                    intent=placement.intent,
                    mode_at_submit=placement.mode_at_submit,
                    evaluation_reason=placement.evaluation_reason,
                    market_key=placement.market_key,
                    strategy_family=placement.strategy_family,
                    action_class=placement.action_class,
                    pnl_owner=placement.pnl_owner,
                    signal_id=placement.signal_id,
                    trade_group_id=placement.trade_group_id,
                    leg_role=placement.leg_role,
                )
                if self.tracer is not None:
                    self.tracer.record_order_submitted(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.market_c.symbol, now_ms),
                        cash=self._current_cash(),
                        order=self._order_to_dict(managed_order),
                    )
                LOGGER.info(
                    "Placed %s %s order %s for C: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )

    async def _maybe_emit_c_etf_projection(self, reason: str, *, now_ms: int) -> None:
        if (
            self.etf_strategy is None
            or self.c_strategy is None
            or not self._etf_trading_enabled
            or not bool(self.config.etf.enable_c_earnings)
        ):
            return
        projection_data = self.c_strategy.active_etf_projection()
        if projection_data is None:
            return
        fair_shift = float(projection_data.get("fair_shift_ticks") or 0.0)
        if abs(fair_shift) < max(0, int(self.config.etf.min_c_fair_shift_ticks)):
            return

        c_projection = ETFShockProjection(
            source_market="C",
            source_kind=str(projection_data.get("source_kind") or "structured_earnings"),
            source_signal_id=projection_data.get("source_signal_id"),
            fair_shift_ticks=fair_shift,
            alpha=float(self.config.etf.alpha_from_c_earnings),
            source_target_inventory=projection_data.get("source_target_inventory"),
            source_combo="C_only",
            source_direction=projection_data.get("source_direction"),
        )
        preview = self.etf_strategy.preview_projection(c_projection)
        if preview is None:
            return

        a_state = self._trace_state_for_symbol("A", now_ms)
        a_mode = str(a_state.get("mode") or "")
        a_active = a_mode in {"POST_EARNINGS_SHOCK", "POST_NEWS_SHOCK"}
        a_direction = int(a_state.get("shock_direction") or 0)
        c_direction = int(preview.get("direction") or 0)
        action_class = "c_shock_projection"
        eval_reason = reason
        projection_to_emit = c_projection

        if a_active and a_direction and c_direction and a_direction != c_direction:
            if str(self.config.etf.ac_conflict_policy or "suppress").lower() == "suppress":
                suppressed_signal_id = (
                    str(projection_data.get("source_signal_id") or "")
                    or f"etf_c_conflict_{int(now_ms)}"
                )
                if self.tracer is not None:
                    self.tracer.record_signal(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                        cash=self._current_cash(),
                        strategy_family="etf_a_follower",
                        action_class="c_shock_projection_suppressed",
                        pnl_owner="etf_a_follower",
                        signal_id=suppressed_signal_id,
                        trade_group_id=suppressed_signal_id,
                        leg_role="single",
                        payload={
                            "source_market": "C",
                            "source_combo": "A_C_conflict_suppressed",
                            "source_signal_id": projection_data.get("source_signal_id"),
                            "source_kind": projection_data.get("source_kind"),
                            "source_fair_shift": fair_shift,
                            "a_direction": a_direction,
                            "c_direction": c_direction,
                            "suppressed_reason": "conflicting_ac_suppressed",
                        },
                    )
                return
        elif a_active and a_direction and c_direction and a_direction == c_direction:
            stronger_target = abs(int(preview.get("target_inventory") or 0))
            if (
                self.etf_strategy.active_signal is not None
                and int(self.etf_strategy.active_signal.source_direction) == int(a_direction)
            ):
                stronger_target = max(stronger_target, abs(int(self.etf_strategy.active_signal.target_inventory)))
            projection_to_emit = ETFShockProjection(
                source_market="AC",
                source_kind="aligned_earnings",
                source_signal_id=str(projection_data.get("source_signal_id") or f"ac_{int(now_ms)}"),
                fair_shift_ticks=fair_shift,
                alpha=float(self.config.etf.alpha_from_c_earnings),
                source_target_inventory=projection_data.get("source_target_inventory"),
                source_combo="A_C_aligned",
                target_inventory_override=int(a_direction) * min(int(self.config.etf.max_position), int(stronger_target)),
                source_direction=int(a_direction),
            )
            action_class = "ac_shock_projection"
            eval_reason = "aligned A+C earnings signal"

        etf_signal = self.etf_strategy.on_shock_projection(projection_to_emit, now_ms=now_ms)
        if etf_signal is None:
            return
        etf_signal = self.etf_strategy.activate_signal(etf_signal)
        if self.tracer is not None:
            self.tracer.record_signal(
                now_ms=now_ms,
                state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                cash=self._current_cash(),
                strategy_family="etf_a_follower",
                action_class=action_class,
                pnl_owner="etf_a_follower",
                signal_id=etf_signal.signal_id,
                trade_group_id=etf_signal.signal_id,
                leg_role="single",
                payload=self.etf_strategy.signal_payload(etf_signal),
            )
        await self._evaluate_and_sync_etf(eval_reason, force=True)

    async def _evaluate_and_sync_etf(self, reason: str, *, force: bool = False) -> None:
        if (
            self._shutdown.is_set()
            or self.etf_strategy is None
            or not self.connected
            or not self._position_snapshot_seen.is_set()
            or not self.config.etf.enabled
            or not self.config.etf.trading_enabled
        ):
            return
        now_ms = self._now_ms()
        pending_etf_entry = (
            self.etf_strategy.active_signal is not None
            and self.etf_strategy.inventory == 0
            and self.etf_strategy.entry_diagnostics.get(self.etf_strategy.active_signal.signal_id) is not None
            and self.etf_strategy.entry_diagnostics[self.etf_strategy.active_signal.signal_id].first_fill_ms is None
        )
        min_interval = max(
            0,
            int(self.config.etf.entry_retry_reprice_ms if pending_etf_entry else self.config.etf.min_eval_interval_ms),
        )
        if not force and self._last_etf_eval_ms is not None and now_ms - self._last_etf_eval_ms < min_interval:
            return
        async with self._quote_lock:
            now_ms = self._now_ms()
            pending_etf_entry = (
                self.etf_strategy.active_signal is not None
                and self.etf_strategy.inventory == 0
                and self.etf_strategy.entry_diagnostics.get(self.etf_strategy.active_signal.signal_id) is not None
                and self.etf_strategy.entry_diagnostics[self.etf_strategy.active_signal.signal_id].first_fill_ms is None
            )
            min_interval = max(
                0,
                int(self.config.etf.entry_retry_reprice_ms if pending_etf_entry else self.config.etf.min_eval_interval_ms),
            )
            if not force and self._last_etf_eval_ms is not None and now_ms - self._last_etf_eval_ms < min_interval:
                return
            self._last_etf_eval_ms = now_ms
            c_state = None if self.c_strategy is None else self._trace_state_for_symbol(self.config.market_c.symbol, now_ms)
            plan = self.etf_strategy.compute_quotes(
                now_ms=now_ms,
                a_state=self._trace_state_for_symbol("A", now_ms),
                c_state=c_state,
            )
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )

            actions = self.etf_strategy.order_manager.build_actions(plan, now_ms)
            for cancel in actions.cancels:
                order = self.etf_strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                        cash=self._current_cash(),
                        order_id=cancel.order_id,
                        side=cancel.side,
                        cancel_reason=cancel.reason,
                        mode_at_cancel=plan.mode,
                        order=None if order is None else self._order_to_dict(order),
                    )
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling ETF %s order %s because %s", cancel.side, cancel.order_id, cancel.reason)

            for placement in actions.placements:
                side = Side.BUY if placement.side == "BUY" else Side.SELL
                order_id = await self.place_order(self.config.etf.symbol, placement.qty, side, placement.px)
                managed_order = self.etf_strategy.order_manager.note_submitted(
                    order_id=order_id,
                    side=placement.side,
                    px=placement.px,
                    qty=placement.qty,
                    now_ms=now_ms,
                    overlay=placement.overlay,
                    aggressive=placement.aggressive,
                    intent=placement.intent,
                    mode_at_submit=placement.mode_at_submit,
                    evaluation_reason=placement.evaluation_reason,
                    market_key=placement.market_key,
                    strategy_family=placement.strategy_family,
                    action_class=placement.action_class,
                    pnl_owner=placement.pnl_owner,
                    signal_id=placement.signal_id,
                    trade_group_id=placement.trade_group_id,
                    leg_role=placement.leg_role,
                )
                if self.tracer is not None:
                    self.tracer.record_order_submitted(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.etf.symbol, now_ms),
                        cash=self._current_cash(),
                        order=self._order_to_dict(managed_order),
                    )
                LOGGER.info(
                    "Placed %s %s order %s for ETF: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )

    async def _evaluate_and_sync_b(self, reason: str, *, force: bool = False) -> None:
        if (
            self._shutdown.is_set()
            or
            self.b_strategy is None
            or not self.connected
            or not self._position_snapshot_seen.is_set()
            or not self.config.market_b.enabled
            or not self.config.market_b.trading_enabled
        ):
            return
        now_ms = self._now_ms()
        min_interval = max(0, int(self.config.market_b.mm_min_eval_interval_ms))
        if not force and self._last_b_eval_ms is not None and now_ms - self._last_b_eval_ms < min_interval:
            return
        async with self._quote_lock:
            now_ms = self._now_ms()
            if not force and self._last_b_eval_ms is not None and now_ms - self._last_b_eval_ms < min_interval:
                return
            self._last_b_eval_ms = now_ms
            residual_payload = self.b_observer.compute_residuals(now_ms=now_ms)
            plan = self.b_strategy.compute_quotes(now_ms=now_ms, residual_payload=residual_payload)
            if self.tracer is not None:
                self.tracer.record_decision(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                    plan=plan,
                )

            if hasattr(self.b_strategy, "build_actions"):
                actions = self.b_strategy.build_actions(plan, now_ms)
            else:
                actions = self.b_strategy.order_manager.build_actions(plan, now_ms)
            for cancel in actions.cancels:
                order = self.b_strategy.order_manager.mark_cancel_requested(cancel.order_id, now_ms)
                if self.tracer is not None:
                    self.tracer.record_cancel_requested(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                        cash=self._current_cash(),
                        order_id=cancel.order_id,
                        side=cancel.side,
                        cancel_reason=cancel.reason,
                        mode_at_cancel=plan.mode,
                        order=None if order is None else self._order_to_dict(order),
                    )
                await self.cancel_order(cancel.order_id)
                LOGGER.info("Cancelling B %s order %s because %s", cancel.side, cancel.order_id, cancel.reason)

            for placement in actions.placements:
                side = Side.BUY if placement.side == "BUY" else Side.SELL
                order_id = await self.place_order(self.config.market_b.underlying_symbol, placement.qty, side, placement.px)
                managed_order = self.b_strategy.order_manager.note_submitted(
                    order_id=order_id,
                    side=placement.side,
                    px=placement.px,
                    qty=placement.qty,
                    now_ms=now_ms,
                    overlay=placement.overlay,
                    aggressive=placement.aggressive,
                    intent=placement.intent,
                    mode_at_submit=placement.mode_at_submit,
                    evaluation_reason=placement.evaluation_reason,
                    market_key=placement.market_key,
                    strategy_family=placement.strategy_family,
                    action_class=placement.action_class,
                    pnl_owner=placement.pnl_owner,
                    signal_id=placement.signal_id,
                    trade_group_id=placement.trade_group_id,
                    leg_role=placement.leg_role,
                )
                if self.tracer is not None:
                    self.tracer.record_order_submitted(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                        cash=self._current_cash(),
                        order=self._order_to_dict(managed_order),
                    )
                LOGGER.info(
                    "Placed %s %s order %s for B: qty=%s px=%s reason=%s",
                    "aggressive" if placement.aggressive else "passive",
                    placement.side,
                    order_id,
                    placement.qty,
                    placement.px,
                    placement.reason,
                )
            if self.tracer is not None:
                self.tracer.maybe_record_periodic_snapshot(
                    now_ms=now_ms,
                    state=self._trace_state_for_symbol(self.config.market_b.underlying_symbol, now_ms),
                    cash=self._current_cash(),
                    trigger=reason,
                )

    async def _evaluate_and_sync_b_options(
        self,
        reason: str,
        *,
        symbols: tuple[str, ...] | None = None,
        force: bool = False,
    ) -> None:
        if (
            self._shutdown.is_set()
            or self.b_option_strategy is None
            or not self.connected
            or not self._position_snapshot_seen.is_set()
            or not self.config.market_b.enabled
            or not self.config.market_b.trading_enabled
            or not self.config.market_b.option_lottery_enabled
        ):
            return
        option_symbols = tuple(symbols or self.config.market_b.option_symbols)
        async with self._quote_lock:
            now_ms = self._now_ms()
            residual_payload = self.b_observer.compute_residuals(now_ms=now_ms)
            for symbol in option_symbols:
                if symbol not in self.config.market_b.option_symbols:
                    continue
                plan = self.b_option_strategy.compute_quote_plan(symbol, now_ms=now_ms, residual_payload=residual_payload)
                if self.tracer is not None:
                    self.tracer.record_decision(
                        now_ms=now_ms,
                        state=self._trace_state_for_symbol(symbol, now_ms),
                        cash=self._current_cash(),
                        trigger=reason,
                        plan=plan,
                    )

                manager = self.b_option_strategy.order_managers[symbol]
                actions = manager.build_actions(plan, now_ms)
                for cancel in actions.cancels:
                    order = manager.mark_cancel_requested(cancel.order_id, now_ms)
                    if self.tracer is not None:
                        self.tracer.record_cancel_requested(
                            now_ms=now_ms,
                            state=self._trace_state_for_symbol(symbol, now_ms),
                            cash=self._current_cash(),
                            order_id=cancel.order_id,
                            side=cancel.side,
                            cancel_reason=cancel.reason,
                            mode_at_cancel=plan.mode,
                            order=None if order is None else self._order_to_dict(order),
                        )
                    await self.cancel_order(cancel.order_id)
                    LOGGER.info("Cancelling %s %s order %s because %s", symbol, cancel.side, cancel.order_id, cancel.reason)

                for placement in actions.placements:
                    side = Side.BUY if placement.side == "BUY" else Side.SELL
                    order_id = await self.place_order(symbol, placement.qty, side, placement.px)
                    managed_order = self.b_option_strategy.note_submitted(
                        symbol,
                        order_id=order_id,
                        desired=placement,
                        now_ms=now_ms,
                    )
                    if self.tracer is not None:
                        self.tracer.record_order_submitted(
                            now_ms=now_ms,
                            state=self._trace_state_for_symbol(symbol, now_ms),
                            cash=self._current_cash(),
                            order=self._order_to_dict(managed_order),
                        )
                    LOGGER.info(
                        "Placed B option lottery BUY order %s for %s: qty=%s px=%s reason=%s",
                        order_id,
                        symbol,
                        placement.qty,
                        placement.px,
                        placement.reason,
                    )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    @staticmethod
    def symbol_from_side(side: str) -> str:
        return "shares" if side in {"BUY", "SELL"} else side

    def _exchange_side_for_order(self, order_id: str) -> str | None:
        order_info = self.open_orders.get(order_id)
        if not order_info:
            return None
        raw_side = getattr(order_info[0], "side", None)
        if raw_side == Side.BUY:
            return "BUY"
        if raw_side == Side.SELL:
            return "SELL"
        side_text = str(raw_side).upper()
        if "BUY" in side_text:
            return "BUY"
        if "SELL" in side_text:
            return "SELL"
        return None

    def _strategy_inventory_for_symbol(self, symbol: str) -> int:
        if symbol == "A":
            return int(self.strategy.inventory)
        if self.etf_strategy is not None and symbol == self.config.etf.symbol:
            return int(self.etf_strategy.inventory)
        if self.c_strategy is not None and symbol == self.config.market_c.symbol:
            return int(self.c_strategy.inventory)
        if self.c_strategy is not None and symbol in self.config.market_c.pm_symbols:
            return int(self.c_strategy.pm_positions.get(symbol, 0))
        if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
            return int(self.b_strategy.inventory)
        if self.b_option_strategy is not None and symbol in self.config.market_b.option_symbols:
            return int(self.b_option_strategy.inventory(symbol))
        return int(self.positions.get(symbol, 0))

    def _authoritative_inventory_after_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        strategy_inventory_before_fill: int,
    ) -> tuple[int, int]:
        signed_qty = int(qty) if side == "BUY" else -int(qty)
        exchange_inventory = int(self.positions.get(symbol, strategy_inventory_before_fill))
        update = self._last_position_update_by_symbol.get(symbol)
        if update is not None:
            _update_ms, inventory_before_update, inventory_after_update = update
            if inventory_after_update == inventory_before_update + signed_qty and exchange_inventory == inventory_after_update:
                return exchange_inventory, inventory_before_update
        if exchange_inventory == strategy_inventory_before_fill + signed_qty:
            return exchange_inventory, strategy_inventory_before_fill
        return strategy_inventory_before_fill + signed_qty, strategy_inventory_before_fill

    def _current_cash(self) -> int | None:
        return int(self.positions.get("cash", 0)) if self._position_snapshot_seen.is_set() else None

    def _trace_state(self, now_ms: int) -> dict:
        return self.strategy.trace_state(now_ms)

    def _trace_state_for_symbol(self, symbol: str, now_ms: int) -> dict:
        if symbol == "A":
            return self.strategy.trace_state(now_ms)
        exchange_inventory = int(self.positions.get(symbol, 0))
        if self.etf_strategy is not None and symbol == self.config.etf.symbol:
            state = self.etf_strategy.trace_state(now_ms)
            state["strategy_inventory"] = int(state.get("inventory") or 0)
            state["exchange_inventory"] = exchange_inventory
            return state
        if self.c_strategy is not None and symbol == self.config.market_c.symbol:
            state = self.c_strategy.trace_state_at(symbol, now_ms)
            state["strategy_inventory"] = int(state.get("inventory") or 0)
            state["exchange_inventory"] = exchange_inventory
            return state
        if self.c_strategy is not None and symbol in self.config.market_c.pm_symbols:
            state = self.c_strategy.trace_state_at(symbol, now_ms)
            state["strategy_inventory"] = int(state.get("inventory") or 0)
            state["exchange_inventory"] = exchange_inventory
            return state
        if self.b_strategy is not None and symbol == self.config.market_b.underlying_symbol:
            state = self.b_strategy.trace_state(now_ms)
            state["strategy_inventory"] = int(state.get("inventory") or 0)
            state["exchange_inventory"] = exchange_inventory
            return state
        if self.b_option_strategy is not None and symbol in self.config.market_b.option_symbols:
            state = self.b_option_strategy.trace_state(symbol, now_ms)
            state["strategy_inventory"] = int(state.get("inventory") or 0)
            state["exchange_inventory"] = exchange_inventory
            return state
        state = self.b_observer.trace_state(symbol, now_ms)
        state["strategy_inventory"] = int(state.get("inventory") or 0)
        state["exchange_inventory"] = exchange_inventory
        return state

    def _record_b_signal(self, now_ms: int) -> None:
        if self.tracer is None or not self.config.market_b.enabled:
            return
        bundle = self.b_observer.derived_signal_bundle(now_ms=now_ms)
        if bundle is None:
            return
        self.tracer.record_signal(
            now_ms=now_ms,
            state=self._trace_state_for_symbol(self.b_observer.underlying_symbol, now_ms),
            cash=self._current_cash(),
            strategy_family="b_observe_only",
            action_class="observe_only",
            pnl_owner="b_observe_only",
            signal_id=bundle.signal_id,
            trade_group_id=bundle.trade_group_id,
            leg_role="composite",
            payload=bundle.payload,
        )
        if not self.config.market_b.parity_shadow_enabled:
            return
        opportunity = self.b_parity_opportunist.evaluate(bundle.payload, now_ms=now_ms)
        if opportunity is None:
            return
        self.tracer.record_signal(
            now_ms=now_ms,
            state=self._trace_state_for_symbol(self.b_observer.underlying_symbol, now_ms),
            cash=self._current_cash(),
            strategy_family="b_parity_opportunist",
            action_class="parity_shadow",
            pnl_owner="b_parity_opportunist",
            signal_id=opportunity.signal_id,
            trade_group_id=opportunity.trade_group_id,
            leg_role="multi_leg",
            payload={
                "enabled_for_live_trading": bool(self.config.market_b.parity_enabled),
                "threshold_ticks": self.config.market_b.parity_edge_threshold_ticks,
                "opportunity": opportunity.to_dict(),
            },
        )

    @staticmethod
    def _order_to_dict(order) -> dict:
        return {
            "order_id": order.order_id,
            "side": order.side,
            "px": order.px,
            "qty": order.qty,
            "remaining_qty": order.remaining_qty,
            "submitted_ms": order.submitted_ms,
            "overlay": getattr(order, "overlay", "mm"),
            "aggressive": order.aggressive,
            "cancel_pending": getattr(order, "cancel_pending", False),
            "restored": getattr(order, "restored", False),
            "intent": getattr(order, "intent", ""),
            "mode_at_submit": getattr(order, "mode_at_submit", ""),
            "evaluation_reason": getattr(order, "evaluation_reason", ""),
            "market_key": getattr(order, "market_key", "A"),
            "strategy_family": getattr(order, "strategy_family", ""),
            "action_class": getattr(order, "action_class", ""),
            "pnl_owner": getattr(order, "pnl_owner", ""),
            "signal_id": getattr(order, "signal_id", ""),
            "trade_group_id": getattr(order, "trade_group_id", ""),
            "leg_role": getattr(order, "leg_role", "single"),
        }

    @staticmethod
    def _reaction_to_dict(reaction) -> dict:
        return {
            "relevant": reaction.relevant,
            "fair_value_updated": reaction.fair_value_updated,
            "note": reaction.note,
            "earnings_value": reaction.earnings_value,
            "news_sentiment_score": reaction.news_sentiment_score,
            "news_sentiment_bucket": reaction.news_sentiment_bucket,
            "old_fair_value": reaction.old_fair_value,
            "new_fair_value": reaction.new_fair_value,
            "base_fair_value": reaction.base_fair_value,
            "news_fair_value": reaction.news_fair_value,
            "news_target_inventory": reaction.news_target_inventory,
            "pending_news_target_inventory": reaction.pending_news_target_inventory,
            "news_confirmation_state": reaction.news_confirmation_state,
            "active_news_signal_id": reaction.active_news_signal_id,
            "news_matched_phrases": reaction.news_matched_phrases,
            "news_matched_unigrams": reaction.news_matched_unigrams,
            "news_matched_bigrams": reaction.news_matched_bigrams,
            "unknown_candidate_phrases": reaction.unknown_candidate_phrases,
            "unknown_candidate_unigrams": reaction.unknown_candidate_unigrams,
            "unknown_candidate_bigrams": reaction.unknown_candidate_bigrams,
            "resolved_news_text": reaction.resolved_news_text,
            "resolved_news_text_source": reaction.resolved_news_text_source,
            "shock_direction": reaction.shock_direction,
            "shock_target_inventory": reaction.shock_target_inventory,
            "shock_threshold": reaction.shock_threshold,
            "tick": reaction.tick,
        }

    def _trace_recovery_transition(self, was_recovery_active: bool, reason: str) -> None:
        if self.tracer is None:
            return
        if was_recovery_active != self.strategy.recovery_active:
            now_ms = self._now_ms()
            self.tracer.record_recovery_state(
                now_ms=now_ms,
                state=self._trace_state_for_symbol("A", now_ms),
                cash=self._current_cash(),
                reason=reason if self.strategy.recovery_active else f"{reason}; recovery complete",
            )


def load_runtime_config(
    *,
    default_host: str | None = DEFAULT_SERVER,
    default_username: str | None = DEFAULT_USERNAME,
    default_password: str | None = DEFAULT_PASSWORD,
) -> BotConfig:
    base_dir = Path(__file__).resolve().parent
    try:
        return load_bot_config(
            base_dir,
            default_host=default_host,
            default_username=default_username,
            default_password=default_password,
            default_initial_multiplier=DEFAULT_A_INITIAL_MULTIPLIER,
            default_initial_fair_value=DEFAULT_A_INITIAL_FAIR_VALUE,
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


async def run_bot(
    *,
    default_host: str | None = DEFAULT_SERVER,
    default_username: str | None = DEFAULT_USERNAME,
    default_password: str | None = DEFAULT_PASSWORD,
) -> None:
    config = load_runtime_config(
        default_host=default_host,
        default_username=default_username,
        default_password=default_password,
    )

    LOGGER.info("Starting multi-market runtime against %s", config.exchange.host)
    LOGGER.info("Journal path: %s", config.paths.journal_path)
    bot = MarketABot(config)
    await bot.start()


async def main() -> None:
    await run_bot()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down multi-market runtime.")
