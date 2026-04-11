from __future__ import annotations

import asyncio
from dataclasses import replace
import grpc
import logging
import time
from pathlib import Path

from .analyze_run import write_run_outputs
from .config import BotConfig, ConfigError, MarketCStrategyConfig, StrategyConfig, load_bot_config
from .core.logger import RunLogger, load_trace_events
from .core.types import BookSnapshot, Decision, DesiredOrder, ManagedOrder, NewsEvent, StrategySnapshot
from .market_A_strategy import AStrategy
from .market_C_maker_strategy import CMarketMakerStrategy
from .market_C_strategy import CStrategy

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError as exc:
    raise SystemExit("utcxchangelib is not installed. Use the repo venv before running marketA_v3.") from exc


LOGGER = logging.getLogger("market-a-v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

A_TRADING_ENABLE = True
C_TRADING_ENABLE = False
ACTIVE_STRATEGY = "A"
GLOBAL_MACRO_FLATTEN_INTENT = "global_macro_risk_off_flatten"


def _inspect_message_body(msg) -> tuple[str | None, bool]:
    if msg == grpc.aio.EOF:
        raise EOFError("End of gRPC stream")
    which_oneof = getattr(msg, "WhichOneof", None)
    if which_oneof is None:
        return None, False
    return which_oneof("body"), True


class BotRunner(XChangeClient):
    """Shared exchange runner for A equity trading and C prediction-market trading."""

    _CONNECT_RETRY_DELAY_S = 2.0

    def __init__(
        self,
        config: BotConfig,
        *,
        a_trading_enable: bool = True,
        c_trading_enable: bool = True,
        active_strategy: str = "A",
    ):
        self.config = config
        self.active_strategy = str(active_strategy or "A").upper()
        if self.active_strategy == "A":
            strategy_symbols = (config.strategy.symbol,)
            tracked_symbols = strategy_symbols
            self.strategy = AStrategy(config.strategy)
            self._primary_symbol = config.strategy.symbol
            session_prefix = "marketA_v3"
        elif self.active_strategy == "CM":
            strategy_symbols = (config.c_mm_strategy.symbol,)
            tracked_symbols = (
                config.c_mm_strategy.symbol,
                config.c_mm_strategy.rate_hike_symbol,
                config.c_mm_strategy.rate_hold_symbol,
                config.c_mm_strategy.rate_cut_symbol,
            )
            self.strategy = CMarketMakerStrategy(config.c_mm_strategy)
            self._primary_symbol = config.c_mm_strategy.symbol
            session_prefix = "marketCmm_v1"
        elif self.active_strategy == "C":
            strategy_symbols = config.c_strategy.tracked_symbols
            tracked_symbols = (config.strategy.symbol, *config.c_strategy.tracked_symbols)
            self.strategy = CStrategy(config.c_strategy)
            self._primary_symbol = config.c_strategy.fed_hold
            session_prefix = "marketC_v1"
        else:
            raise ValueError(f"Unsupported active strategy {active_strategy!r}")

        self._tracked_symbols = tuple(tracked_symbols)
        self._tracked_symbol_set = set(self._tracked_symbols)
        self._strategy_symbols = tuple(strategy_symbols)
        self._strategy_symbol_set = set(self._strategy_symbols)

        super().__init__(
            config.exchange.host,
            config.exchange.username,
            config.exchange.password,
            symbols=list(self._tracked_symbols),
        )

        self.logger = RunLogger.create_if_enabled(
            config.logger,
            session_prefix=session_prefix,
            symbol=self._primary_symbol,
        )
        self._order_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._timer_task: asyncio.Task | None = None
        self._position_snapshot_seen = asyncio.Event()
        self._managed_orders: dict[str, ManagedOrder] = {}
        self._last_trade_px_by_symbol: dict[str, int | None] = {symbol: None for symbol in self._tracked_symbols}
        self._last_trade_px: int | None = None
        self._current_message_index: int | None = None
        self._inventory_estimate_by_symbol: dict[str, int] = {symbol: 0 for symbol in self._tracked_symbols}
        self._inventory_estimate: int = 0
        self._last_position_update_ms_by_symbol: dict[str, int | None] = {symbol: None for symbol in self._tracked_symbols}
        self._last_position_update_ms: int | None = None
        self._run_started_ms: int = self._now_ms()
        self._midrun_checkpoint_started = False
        self._midrun_checkpoint_task: asyncio.Task | None = None
        self._connected_once = False
        self._stop_after_c_market_resolved = False
        self._global_macro_flatten_until_ms: int | None = None
        self._global_macro_flatten_reason: str | None = None
        self._global_macro_flatten_context_id: str | None = None
        self._global_macro_flatten_source: str | None = None
        self.a_trading_enable = a_trading_enable
        self.c_trading_enable = c_trading_enable

    async def start(self) -> None:
        if not self._execution_enabled():
            LOGGER.info("%s trading is disabled in botrunner; running in observe-only mode for execution.", self.active_strategy)
        if self.logger is not None:
            self.logger.record_event(
                "runner_started",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                symbol=self._primary_symbol,
                active_strategy=self.active_strategy,
                tracked_symbols=list(self._tracked_symbols),
                a_trading_enable=self.a_trading_enable,
                c_trading_enable=self.c_trading_enable,
            )
        self._timer_task = asyncio.create_task(self._timer_loop())
        try:
            await self._connect_with_retry()
        except EOFError:
            LOGGER.info("Exchange stream closed cleanly.")
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.UNAVAILABLE:
                LOGGER.info("Exchange connection became unavailable: %s", exc.details())
            else:
                raise
        finally:
            self._shutdown.set()
            if self._timer_task is not None:
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass
            if self._midrun_checkpoint_task is not None:
                try:
                    await self._midrun_checkpoint_task
                except Exception:
                    LOGGER.exception("Mid-run checkpoint generation failed.")
            run_dir = None if self.logger is None else self.logger.run_dir
            if self.logger is not None:
                self.logger.close()
            if run_dir is not None:
                try:
                    _, generated = write_run_outputs(run_dir, write_summary=True, write_graphs=True)
                    if generated:
                        LOGGER.info("Generated post-run artifacts for %s", run_dir)
                except Exception:
                    LOGGER.exception("Failed to generate post-run outputs for %s", run_dir)

    async def _connect_with_retry(self) -> None:
        while True:
            try:
                await self.connect()
                return
            except EOFError:
                if self._connected_once or self._position_snapshot_seen.is_set():
                    raise
                LOGGER.warning(
                    "Exchange stream closed before initialization completed. Retrying in %.1fs.",
                    self._CONNECT_RETRY_DELAY_S,
                )
            except grpc.aio.AioRpcError as exc:
                if exc.code() != grpc.StatusCode.UNAVAILABLE:
                    raise
                if self._connected_once or self._position_snapshot_seen.is_set():
                    raise
                LOGGER.warning(
                    "Initial exchange connection failed: %s. Retrying in %.1fs.",
                    exc.details(),
                    self._CONNECT_RETRY_DELAY_S,
                )
            await asyncio.sleep(self._CONNECT_RETRY_DELAY_S)

    async def process_message(self, msg) -> None:
        msg_type, is_proto = _inspect_message_body(msg)
        if not is_proto:
            return
        self._connected_once = True
        self._current_message_index = int(getattr(msg, "index", 0) or 0)
        await super().process_message(msg)

        if msg_type == "position_update":
            symbol = str(msg.position_update.symbol)
            if symbol in self._tracked_symbol_set:
                self._inventory_estimate_by_symbol[symbol] = int(msg.position_update.value)
                self._last_position_update_ms_by_symbol[symbol] = self._now_ms()
                if symbol == self.config.strategy.symbol:
                    self._inventory_estimate = self._inventory_estimate_by_symbol[symbol]
                    self._last_position_update_ms = self._last_position_update_ms_by_symbol[symbol]
        elif msg_type == "position_snapshot":
            now_ms = self._now_ms()
            for symbol in self._tracked_symbols:
                self._inventory_estimate_by_symbol[symbol] = int(self.positions.get(symbol, 0))
                self._last_position_update_ms_by_symbol[symbol] = now_ms
            self._inventory_estimate = self._inventory_estimate_by_symbol.get(self.config.strategy.symbol, self._inventory_estimate)
            self._last_position_update_ms = now_ms

        if msg_type == "market_resolved" and self._stop_after_c_market_resolved:
            raise EOFError("C prediction market resolved; stopping runner early for post-run analysis.")

        if msg_type == "cash_update":
            await self._evaluate_and_sync("cash_update")
        elif msg_type == "position_update" and str(msg.position_update.symbol) in self._tracked_symbol_set:
            await self._evaluate_and_sync("position_update", event_symbol=str(msg.position_update.symbol))

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        now_ms = self._now_ms()
        for symbol in self._tracked_symbols:
            self._inventory_estimate_by_symbol[symbol] = int(self.positions.get(symbol, 0))
            self._last_position_update_ms_by_symbol[symbol] = now_ms
        self._inventory_estimate = self._inventory_estimate_by_symbol.get(self.config.strategy.symbol, self._inventory_estimate)
        self._last_position_update_ms = now_ms
        if self.logger is not None:
            self.logger.record_event(
                "position_snapshot",
                now_ms=now_ms,
                message_index=self._current_message_index,
                inventory=self._inventory(self._primary_symbol),
                cash=self._cash(),
                symbol=self._primary_symbol,
            )
        self._position_snapshot_seen.set()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol not in self._tracked_symbol_set:
            return
        now_ms = self._now_ms()
        book = self._book_snapshot(symbol)
        if self.logger is not None:
            self.logger.record_event(
                "book_update",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=symbol,
                best_bid_px=None if book.best_bid is None else book.best_bid.px,
                best_bid_qty=None if book.best_bid is None else book.best_bid.qty,
                best_ask_px=None if book.best_ask is None else book.best_ask.px,
                best_ask_qty=None if book.best_ask is None else book.best_ask.qty,
                mid=book.mid,
                spread=book.spread,
                inventory=self._inventory(symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("book", event_symbol=symbol)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol not in self._tracked_symbol_set:
            return
        self._last_trade_px_by_symbol[symbol] = int(price)
        if symbol == self.config.strategy.symbol:
            self._last_trade_px = int(price)
        now_ms = self._now_ms()
        if self.logger is not None:
            self.logger.record_event(
                "trade_print",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=symbol,
                price=int(price),
                qty=int(qty),
                inventory=self._inventory(symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._sync_decision(self.strategy.on_trade(self._snapshot(event_symbol=symbol)), trigger="trade", event_symbol=symbol)

    async def bot_handle_news(self, news_release: dict) -> None:
        now_ms = self._now_ms()
        news_event = self._to_news_event(now_ms, news_release)
        event_symbol = news_event.symbol if news_event.symbol in self._tracked_symbol_set else None
        decision = self.strategy.on_news(self._snapshot(event_symbol=event_symbol), news_event)
        strategy_state = self.strategy.export_state()
        if self._should_arm_global_macro_flatten(strategy_state):
            self._arm_global_macro_flatten(now_ms, strategy_state)
        rate_contract_snapshot = self._rate_contract_snapshot() if self.active_strategy == "C" else None
        if self.logger is not None:
            self.logger.record_event(
                "news_received",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=news_event.tick,
                symbol=event_symbol,
                news_kind=news_event.kind,
                raw_payload=news_release,
                content=news_event.content,
                active_signal_kind=strategy_state.get("active_signal_kind"),
                news_sentiment_score=strategy_state.get("news_sentiment_score"),
                news_sentiment_bucket=strategy_state.get("news_sentiment_bucket"),
                news_matched_phrases=strategy_state.get("news_matched_phrases"),
                news_matched_unigrams=strategy_state.get("news_matched_unigrams"),
                news_matched_bigrams=strategy_state.get("news_matched_bigrams"),
                unknown_candidate_phrases=strategy_state.get("unknown_candidate_phrases"),
                unknown_candidate_unigrams=strategy_state.get("unknown_candidate_unigrams"),
                unknown_candidate_bigrams=strategy_state.get("unknown_candidate_bigrams"),
                base_fair_value=strategy_state.get("base_fair_value"),
                news_fair_value=strategy_state.get("news_fair_value"),
                pending_news=strategy_state.get("pending_news"),
                pending_news_target_inventory=strategy_state.get("pending_news_target_inventory"),
                news_confirmation_state=strategy_state.get("news_confirmation_state"),
                news_confirmation_deadline_ms=strategy_state.get("news_confirmation_deadline_ms"),
                news_takeover_started_ms=strategy_state.get("news_takeover_started_ms"),
                posterior_hike=strategy_state.get("posterior_hike"),
                posterior_hold=strategy_state.get("posterior_hold"),
                posterior_cut=strategy_state.get("posterior_cut"),
                terminal_hike=strategy_state.get("terminal_hike"),
                terminal_hold=strategy_state.get("terminal_hold"),
                terminal_cut=strategy_state.get("terminal_cut"),
                fair_value_hike=strategy_state.get("fair_value_hike"),
                fair_value_hold=strategy_state.get("fair_value_hold"),
                fair_value_cut=strategy_state.get("fair_value_cut"),
                prior_hike=strategy_state.get("prior_hike"),
                prior_hold=strategy_state.get("prior_hold"),
                prior_cut=strategy_state.get("prior_cut"),
                market_prior_hike=strategy_state.get("market_prior_hike"),
                market_prior_hold=strategy_state.get("market_prior_hold"),
                market_prior_cut=strategy_state.get("market_prior_cut"),
                memory_logit_hike=strategy_state.get("memory_logit_hike"),
                memory_logit_hold=strategy_state.get("memory_logit_hold"),
                memory_logit_cut=strategy_state.get("memory_logit_cut"),
                dominant_regime=strategy_state.get("dominant_regime"),
                expected_rate_delta_bp=strategy_state.get("expected_rate_delta_bp"),
                rate_macro_event_id=strategy_state.get("rate_macro_event_id"),
                rate_signal_source=strategy_state.get("rate_signal_source"),
                rate_no_trade_reason=strategy_state.get("rate_no_trade_reason"),
                rate_relevance_score=strategy_state.get("rate_relevance_score"),
                rate_bucket=strategy_state.get("rate_bucket"),
                rate_target_symbol=strategy_state.get("rate_target_symbol"),
                rate_target_inventory=strategy_state.get("rate_target_inventory"),
                rate_chosen_edge_ticks=strategy_state.get("rate_chosen_edge_ticks"),
                rate_no_arb_gap_ticks=strategy_state.get("rate_no_arb_gap_ticks"),
                rate_dominant_regime=strategy_state.get("rate_dominant_regime"),
                rate_contrary_signal_score=strategy_state.get("rate_contrary_signal_score"),
                rate_regime_break_score=strategy_state.get("rate_regime_break_score"),
                rate_long_edge_hike=strategy_state.get("rate_long_edge_hike"),
                rate_long_edge_hold=strategy_state.get("rate_long_edge_hold"),
                rate_long_edge_cut=strategy_state.get("rate_long_edge_cut"),
                rate_short_edge_hike=strategy_state.get("rate_short_edge_hike"),
                rate_short_edge_hold=strategy_state.get("rate_short_edge_hold"),
                rate_short_edge_cut=strategy_state.get("rate_short_edge_cut"),
                rate_hawk_score=strategy_state.get("rate_hawk_score"),
                rate_hold_score=strategy_state.get("rate_hold_score"),
                rate_cut_score=strategy_state.get("rate_cut_score"),
                rate_matched_phrases=strategy_state.get("rate_matched_phrases"),
                rate_matched_unigrams=strategy_state.get("rate_matched_unigrams"),
                rate_matched_bigrams=strategy_state.get("rate_matched_bigrams"),
                rate_matched_hike_terms=strategy_state.get("rate_matched_hike_terms"),
                rate_matched_hold_terms=strategy_state.get("rate_matched_hold_terms"),
                rate_matched_cut_terms=strategy_state.get("rate_matched_cut_terms"),
                rate_unknown_candidate_phrases=strategy_state.get("rate_unknown_candidate_phrases"),
                rate_unknown_candidate_unigrams=strategy_state.get("rate_unknown_candidate_unigrams"),
                rate_unknown_candidate_bigrams=strategy_state.get("rate_unknown_candidate_bigrams"),
                rate_baseline_targets_by_symbol=strategy_state.get("rate_baseline_targets_by_symbol"),
                rate_macro_targets_by_symbol=strategy_state.get("rate_macro_targets_by_symbol"),
                rate_macro_pair_targets_by_symbol=strategy_state.get("rate_macro_pair_targets_by_symbol"),
                rate_trading_phase_targets_by_symbol=strategy_state.get("rate_trading_phase_targets_by_symbol"),
                rate_probe_targets_by_symbol=strategy_state.get("rate_probe_targets_by_symbol"),
                rate_reversion_targets_by_symbol=strategy_state.get("rate_reversion_targets_by_symbol"),
                rate_pair_targets_by_symbol=strategy_state.get("rate_pair_targets_by_symbol"),
                rate_endgame_targets_by_symbol=strategy_state.get("rate_endgame_targets_by_symbol"),
                rate_final_phase_targets_by_symbol=strategy_state.get("rate_final_phase_targets_by_symbol"),
                rate_combined_targets_by_symbol=strategy_state.get("rate_combined_targets_by_symbol"),
                rate_macro_pair_symbols=strategy_state.get("rate_macro_pair_symbols"),
                rate_macro_leg_reference_mids=strategy_state.get("rate_macro_leg_reference_mids"),
                rate_macro_leg_fairs=strategy_state.get("rate_macro_leg_fairs"),
                rate_macro_leg_bucket=strategy_state.get("rate_macro_leg_bucket"),
                rate_reversion_active_symbols=strategy_state.get("rate_reversion_active_symbols"),
                rate_reversion_entry_px_by_symbol=strategy_state.get("rate_reversion_entry_px_by_symbol"),
                rate_reversion_reason_by_symbol=strategy_state.get("rate_reversion_reason_by_symbol"),
                rate_pair_active_pair=strategy_state.get("rate_pair_active_pair"),
                rate_pair_entry_px_by_symbol=strategy_state.get("rate_pair_entry_px_by_symbol"),
                rate_pair_reason_by_symbol=strategy_state.get("rate_pair_reason_by_symbol"),
                rate_pair_move_by_symbol=strategy_state.get("rate_pair_move_by_symbol"),
                rate_pair_last_event_id=strategy_state.get("rate_pair_last_event_id"),
                rate_pair_last_event_kind=strategy_state.get("rate_pair_last_event_kind"),
                rate_pair_last_event_pair=strategy_state.get("rate_pair_last_event_pair"),
                rate_pair_last_event_reason=strategy_state.get("rate_pair_last_event_reason"),
                rate_reversion_last_event_id=strategy_state.get("rate_reversion_last_event_id"),
                rate_reversion_last_event_kind=strategy_state.get("rate_reversion_last_event_kind"),
                rate_reversion_last_event_symbol=strategy_state.get("rate_reversion_last_event_symbol"),
                rate_reversion_last_event_reason=strategy_state.get("rate_reversion_last_event_reason"),
                rate_reversion_last_entry_px=strategy_state.get("rate_reversion_last_entry_px"),
                rate_reversion_last_exit_px=strategy_state.get("rate_reversion_last_exit_px"),
                rate_global_flatten_signal=strategy_state.get("rate_global_flatten_signal"),
                rate_global_flatten_reason=strategy_state.get("rate_global_flatten_reason"),
                global_macro_flatten_active=self._global_macro_flatten_active(now_ms),
                global_macro_flatten_until_ms=self._global_macro_flatten_until_ms,
                global_macro_flatten_reason=self._global_macro_flatten_reason,
                rate_contract_snapshot_t0=rate_contract_snapshot,
                inventory=self._inventory(event_symbol or self._primary_symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._sync_decision(decision, trigger="news", event_symbol=event_symbol)

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int) -> None:
        now_ms = self._now_ms()
        resolved_symbol = str(winning_symbol)
        focus_symbol = resolved_symbol if resolved_symbol in self._tracked_symbol_set else self._primary_symbol
        if self.logger is not None:
            self.logger.record_event(
                "market_resolved",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=int(tick),
                symbol=focus_symbol,
                market_id=market_id,
                winning_symbol=resolved_symbol,
                inventory=self._inventory(focus_symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        if self.active_strategy == "C" and resolved_symbol in self._tracked_symbol_set:
            LOGGER.info(
                "Received market resolution for %s with winner %s at tick %s. Closing C runner early for post-run analysis.",
                market_id,
                resolved_symbol,
                tick,
            )
            self._stop_after_c_market_resolved = True

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int) -> None:
        if self.logger is None:
            return
        self.logger.record_event(
            "settlement_payout",
            now_ms=self._now_ms(),
            message_index=self._current_message_index,
            exchange_tick=int(tick),
            symbol=self._primary_symbol,
            market_id=market_id,
            settlement_user=user,
            settlement_amount=int(amount),
            cash=self._cash(),
            mtm_pnl_estimate=self._total_mtm(),
            shock_pnl=self._total_mtm(),
            mm_pnl=0.0,
        )

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        order = self._managed_orders.get(order_id)
        now_ms = self._now_ms()
        fill_symbol = None if order is None else order.symbol
        if order is not None:
            self._apply_fill_inventory_hint(order.symbol, order.side, int(qty), now_ms)
            order.remaining_qty = max(0, order.remaining_qty - int(qty))
            if order.remaining_qty == 0:
                self._managed_orders.pop(order_id, None)
        if self.logger is not None:
            self.logger.record_event(
                "order_filled",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=fill_symbol,
                order_id=order_id,
                side=None if order is None else order.side,
                fill_qty=int(qty),
                fill_price=int(price),
                aggressive=None if order is None else order.aggressive,
                intent=None if order is None else order.intent,
                reason=None if order is None else order.reason,
                rate_macro_event_id=None if order is None else order.context_id,
                inventory=self._inventory(fill_symbol or self._primary_symbol),
                cash=self._cash(),
                mid=self._book_snapshot(fill_symbol or self._primary_symbol).mid,
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._sync_decision(self.strategy.on_fill(self._snapshot(event_symbol=fill_symbol)), trigger="fill", event_symbol=fill_symbol)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        order = self._managed_orders.pop(order_id, None)
        symbol = None if order is None else order.symbol
        if self.logger is not None:
            self.logger.record_event(
                "order_rejected",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=symbol,
                order_id=order_id,
                rejection_reason=reason,
                rate_macro_event_id=None if order is None else order.context_id,
                inventory=self._inventory(symbol or self._primary_symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("order_rejected", event_symbol=symbol)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: str | None) -> None:
        order = self._managed_orders.get(order_id)
        symbol = None if order is None else order.symbol
        if success:
            self._managed_orders.pop(order_id, None)
        elif order is not None:
            order.cancel_pending = False
        if self.logger is not None:
            self.logger.record_event(
                "cancel_response",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=symbol,
                order_id=order_id,
                success=success,
                error=error,
                rate_macro_event_id=None if order is None else order.context_id,
                inventory=self._inventory(symbol or self._primary_symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("cancel_response", event_symbol=symbol)

    async def _timer_loop(self) -> None:
        interval_s = max(0.05, self._strategy_order_config(self._primary_symbol).timer_interval_ms / 1000.0)
        while not self._shutdown.is_set():
            await asyncio.sleep(interval_s)
            if not self._position_snapshot_seen.is_set():
                continue
            await self._sync_decision(self.strategy.on_timer(self._snapshot()), trigger="timer")
            self._maybe_schedule_midrun_checkpoint(self._now_ms())

    def _midrun_checkpoint_due(self, now_ms: int) -> bool:
        if self.logger is None:
            return False
        if not self.config.logger.midrun_checkpoint_enabled:
            return False
        if self._midrun_checkpoint_started:
            return False
        return (now_ms - self._run_started_ms) >= self.config.logger.midrun_checkpoint_ms

    def _maybe_schedule_midrun_checkpoint(self, now_ms: int) -> None:
        if not self._midrun_checkpoint_due(now_ms):
            return
        self._midrun_checkpoint_started = True
        self._midrun_checkpoint_task = asyncio.create_task(
            asyncio.to_thread(self._write_checkpoint_outputs, "halfway"),
            name=f"{self.active_strategy.lower()}-halfway-checkpoint",
        )

    def _write_checkpoint_outputs(self, checkpoint_name: str) -> None:
        if self.logger is None:
            return
        self.logger.flush()
        run_dir = self.logger.run_dir
        checkpoint_dir = run_dir / "checkpoints" / checkpoint_name
        events = load_trace_events(run_dir)
        _, generated = write_run_outputs(
            run_dir,
            output_dir=checkpoint_dir,
            events=events,
            write_summary=True,
            write_graphs=True,
        )
        LOGGER.info(
            "Generated %s checkpoint artifacts for %s%s",
            checkpoint_name,
            run_dir,
            "" if not generated else f" ({len(generated)} graph(s))",
        )

    async def _evaluate_and_sync(self, trigger: str, *, event_symbol: str | None = None) -> None:
        if not self._position_snapshot_seen.is_set():
            return
        await self._sync_decision(self.strategy.on_book(self._snapshot(event_symbol=event_symbol)), trigger=trigger, event_symbol=event_symbol)

    async def _sync_decision(self, decision: Decision, *, trigger: str, event_symbol: str | None = None) -> None:
        focus_symbol = self._decision_focus_symbol(decision, event_symbol=event_symbol)
        snapshot = self._snapshot(event_symbol=event_symbol, focus_symbol=focus_symbol)
        strategy_state = self.strategy.export_state()
        if self._global_macro_flatten_active(snapshot.now_ms, snapshot=snapshot):
            decision = self._global_flatten_decision(snapshot, event_symbol=event_symbol, current_decision=decision)
            focus_symbol = self._decision_focus_symbol(decision, event_symbol=event_symbol)
            snapshot = self._snapshot(event_symbol=event_symbol, focus_symbol=focus_symbol)
        current_edge_ticks = None
        if decision.fair_value is not None and snapshot.book.mid is not None:
            current_edge_ticks = float(decision.fair_value) - float(snapshot.book.mid)
        if self.logger is not None:
            desired = decision.desired_order
            self.logger.record_decision_snapshot(
                now_ms=snapshot.now_ms,
                exchange_tick=snapshot.exchange_tick,
                message_index=snapshot.message_index,
                symbol=focus_symbol,
                row={
                    "mode": decision.mode,
                    "trigger": trigger,
                    "reason": decision.reason,
                    "observe_only": decision.observe_only,
                    "event_symbol": snapshot.event_symbol,
                    "desired_symbol": None if desired is None else desired.symbol,
                    "inventory": snapshot.inventory,
                    "target_inventory": decision.target_inventory,
                    "best_bid_px": None if snapshot.book.best_bid is None else snapshot.book.best_bid.px,
                    "best_bid_qty": None if snapshot.book.best_bid is None else snapshot.book.best_bid.qty,
                    "best_ask_px": None if snapshot.book.best_ask is None else snapshot.book.best_ask.px,
                    "best_ask_qty": None if snapshot.book.best_ask is None else snapshot.book.best_ask.qty,
                    "mid": snapshot.book.mid,
                    "spread": snapshot.book.spread,
                    "fair_value": decision.fair_value,
                    "base_fair_value": strategy_state.get("base_fair_value"),
                    "news_fair_value": strategy_state.get("news_fair_value"),
                    "trusted_multiplier": decision.trusted_multiplier,
                    "latest_earnings": decision.latest_earnings,
                    "fair_change_ticks": strategy_state.get("fair_change_ticks"),
                    "active_signal_kind": strategy_state.get("active_signal_kind"),
                    "pe_frozen": strategy_state.get("pe_frozen"),
                    "clean_multiplier_sample_count": strategy_state.get("clean_multiplier_sample_count"),
                    "equilibrium_reached": decision.equilibrium_reached,
                    "current_edge_ticks": current_edge_ticks,
                    "overshoot_active": strategy_state.get("overshoot_active"),
                    "overshoot_stage_index": strategy_state.get("overshoot_stage_index"),
                    "overshoot_trimmed_qty_total": strategy_state.get("overshoot_trimmed_qty_total"),
                    "overshoot_trigger_ticks": strategy_state.get("overshoot_trigger_ticks"),
                    "news_sentiment_score": strategy_state.get("news_sentiment_score"),
                    "news_sentiment_bucket": strategy_state.get("news_sentiment_bucket"),
                    "news_matched_phrases": strategy_state.get("news_matched_phrases"),
                    "news_matched_unigrams": strategy_state.get("news_matched_unigrams"),
                    "news_matched_bigrams": strategy_state.get("news_matched_bigrams"),
                    "unknown_candidate_phrases": strategy_state.get("unknown_candidate_phrases"),
                    "unknown_candidate_unigrams": strategy_state.get("unknown_candidate_unigrams"),
                    "unknown_candidate_bigrams": strategy_state.get("unknown_candidate_bigrams"),
                    "pending_news": strategy_state.get("pending_news"),
                    "pending_news_target_inventory": strategy_state.get("pending_news_target_inventory"),
                    "news_confirmation_state": strategy_state.get("news_confirmation_state"),
                    "news_confirmation_deadline_ms": strategy_state.get("news_confirmation_deadline_ms"),
                    "news_takeover_started_ms": strategy_state.get("news_takeover_started_ms"),
                    "posterior_hike": strategy_state.get("posterior_hike"),
                    "posterior_hold": strategy_state.get("posterior_hold"),
                    "posterior_cut": strategy_state.get("posterior_cut"),
                    "terminal_hike": strategy_state.get("terminal_hike"),
                    "terminal_hold": strategy_state.get("terminal_hold"),
                    "terminal_cut": strategy_state.get("terminal_cut"),
                    "fair_value_hike": strategy_state.get("fair_value_hike"),
                    "fair_value_hold": strategy_state.get("fair_value_hold"),
                    "fair_value_cut": strategy_state.get("fair_value_cut"),
                    "prior_hike": strategy_state.get("prior_hike"),
                    "prior_hold": strategy_state.get("prior_hold"),
                    "prior_cut": strategy_state.get("prior_cut"),
                    "market_prior_hike": strategy_state.get("market_prior_hike"),
                    "market_prior_hold": strategy_state.get("market_prior_hold"),
                    "market_prior_cut": strategy_state.get("market_prior_cut"),
                    "memory_logit_hike": strategy_state.get("memory_logit_hike"),
                    "memory_logit_hold": strategy_state.get("memory_logit_hold"),
                    "memory_logit_cut": strategy_state.get("memory_logit_cut"),
                    "dominant_regime": strategy_state.get("dominant_regime"),
                    "expected_rate_delta_bp": strategy_state.get("expected_rate_delta_bp"),
                    "rate_macro_event_id": strategy_state.get("rate_macro_event_id"),
                    "rate_signal_source": strategy_state.get("rate_signal_source"),
                    "rate_no_trade_reason": strategy_state.get("rate_no_trade_reason"),
                    "rate_relevance_score": strategy_state.get("rate_relevance_score"),
                    "rate_bucket": strategy_state.get("rate_bucket"),
                    "rate_target_symbol": strategy_state.get("rate_target_symbol"),
                    "rate_target_inventory": strategy_state.get("rate_target_inventory"),
                    "rate_chosen_edge_ticks": strategy_state.get("rate_chosen_edge_ticks"),
                    "rate_no_arb_gap_ticks": strategy_state.get("rate_no_arb_gap_ticks"),
                    "rate_dominant_regime": strategy_state.get("rate_dominant_regime"),
                    "rate_contrary_signal_score": strategy_state.get("rate_contrary_signal_score"),
                    "rate_regime_break_score": strategy_state.get("rate_regime_break_score"),
                    "rate_long_edge_hike": strategy_state.get("rate_long_edge_hike"),
                    "rate_long_edge_hold": strategy_state.get("rate_long_edge_hold"),
                    "rate_long_edge_cut": strategy_state.get("rate_long_edge_cut"),
                    "rate_short_edge_hike": strategy_state.get("rate_short_edge_hike"),
                    "rate_short_edge_hold": strategy_state.get("rate_short_edge_hold"),
                    "rate_short_edge_cut": strategy_state.get("rate_short_edge_cut"),
                    "rate_hawk_score": strategy_state.get("rate_hawk_score"),
                    "rate_hold_score": strategy_state.get("rate_hold_score"),
                    "rate_cut_score": strategy_state.get("rate_cut_score"),
                    "rate_matched_phrases": strategy_state.get("rate_matched_phrases"),
                    "rate_matched_unigrams": strategy_state.get("rate_matched_unigrams"),
                    "rate_matched_bigrams": strategy_state.get("rate_matched_bigrams"),
                    "rate_matched_hike_terms": strategy_state.get("rate_matched_hike_terms"),
                    "rate_matched_hold_terms": strategy_state.get("rate_matched_hold_terms"),
                    "rate_matched_cut_terms": strategy_state.get("rate_matched_cut_terms"),
                    "rate_unknown_candidate_phrases": strategy_state.get("rate_unknown_candidate_phrases"),
                    "rate_unknown_candidate_unigrams": strategy_state.get("rate_unknown_candidate_unigrams"),
                    "rate_unknown_candidate_bigrams": strategy_state.get("rate_unknown_candidate_bigrams"),
                    "rate_baseline_targets_by_symbol": strategy_state.get("rate_baseline_targets_by_symbol"),
                    "rate_macro_targets_by_symbol": strategy_state.get("rate_macro_targets_by_symbol"),
                    "rate_macro_pair_targets_by_symbol": strategy_state.get("rate_macro_pair_targets_by_symbol"),
                    "rate_trading_phase_targets_by_symbol": strategy_state.get("rate_trading_phase_targets_by_symbol"),
                    "rate_probe_targets_by_symbol": strategy_state.get("rate_probe_targets_by_symbol"),
                    "rate_reversion_targets_by_symbol": strategy_state.get("rate_reversion_targets_by_symbol"),
                    "rate_pair_targets_by_symbol": strategy_state.get("rate_pair_targets_by_symbol"),
                    "rate_endgame_targets_by_symbol": strategy_state.get("rate_endgame_targets_by_symbol"),
                    "rate_final_phase_targets_by_symbol": strategy_state.get("rate_final_phase_targets_by_symbol"),
                    "rate_combined_targets_by_symbol": strategy_state.get("rate_combined_targets_by_symbol"),
                    "rate_macro_pair_symbols": strategy_state.get("rate_macro_pair_symbols"),
                    "rate_macro_leg_reference_mids": strategy_state.get("rate_macro_leg_reference_mids"),
                    "rate_macro_leg_fairs": strategy_state.get("rate_macro_leg_fairs"),
                    "rate_macro_leg_bucket": strategy_state.get("rate_macro_leg_bucket"),
                    "rate_reversion_active_symbols": strategy_state.get("rate_reversion_active_symbols"),
                    "rate_reversion_entry_px_by_symbol": strategy_state.get("rate_reversion_entry_px_by_symbol"),
                    "rate_reversion_reason_by_symbol": strategy_state.get("rate_reversion_reason_by_symbol"),
                    "rate_pair_active_pair": strategy_state.get("rate_pair_active_pair"),
                    "rate_pair_entry_px_by_symbol": strategy_state.get("rate_pair_entry_px_by_symbol"),
                    "rate_pair_reason_by_symbol": strategy_state.get("rate_pair_reason_by_symbol"),
                    "rate_pair_move_by_symbol": strategy_state.get("rate_pair_move_by_symbol"),
                    "rate_pair_last_event_id": strategy_state.get("rate_pair_last_event_id"),
                    "rate_pair_last_event_kind": strategy_state.get("rate_pair_last_event_kind"),
                    "rate_pair_last_event_pair": strategy_state.get("rate_pair_last_event_pair"),
                    "rate_pair_last_event_reason": strategy_state.get("rate_pair_last_event_reason"),
                    "rate_reversion_last_event_id": strategy_state.get("rate_reversion_last_event_id"),
                    "rate_reversion_last_event_kind": strategy_state.get("rate_reversion_last_event_kind"),
                    "rate_reversion_last_event_symbol": strategy_state.get("rate_reversion_last_event_symbol"),
                    "rate_reversion_last_event_reason": strategy_state.get("rate_reversion_last_event_reason"),
                    "rate_reversion_last_entry_px": strategy_state.get("rate_reversion_last_entry_px"),
                    "rate_reversion_last_exit_px": strategy_state.get("rate_reversion_last_exit_px"),
                    "rate_global_flatten_signal": strategy_state.get("rate_global_flatten_signal"),
                    "rate_global_flatten_reason": strategy_state.get("rate_global_flatten_reason"),
                    "global_macro_flatten_active": self._global_macro_flatten_active(snapshot.now_ms, snapshot=snapshot),
                    "global_macro_flatten_until_ms": self._global_macro_flatten_until_ms,
                    "global_macro_flatten_reason": self._global_macro_flatten_reason,
                    "mm_microprice": strategy_state.get("mm_microprice"),
                    "mm_recent_trade_count": strategy_state.get("mm_recent_trade_count"),
                    "mm_flow_score": strategy_state.get("mm_flow_score"),
                    "mm_burst_active": strategy_state.get("mm_burst_active"),
                    "mm_quote_side": strategy_state.get("mm_quote_side"),
                    "mm_quote_px": strategy_state.get("mm_quote_px"),
                    "mm_quote_reason": strategy_state.get("mm_quote_reason"),
                    "desired_side": None if desired is None else desired.side,
                    "desired_px": None if desired is None else desired.px,
                    "desired_qty": None if desired is None else desired.qty,
                    "desired_intent": None if desired is None else desired.intent,
                    "cash": snapshot.cash,
                    "mtm_pnl_estimate": self._total_mtm(),
                    "shock_pnl": self._total_mtm(),
                    "mm_pnl": 0.0,
                },
            )
        await self._apply_decision(decision)

    async def _apply_decision(self, decision: Decision) -> None:
        async with self._order_lock:
            current_orders = self._current_orders()
            if not self._execution_enabled():
                if current_orders:
                    await self._cancel_managed_orders(current_orders, cancel_reason=f"{self.active_strategy} trading disabled in botrunner")
                return

            desired = self._normalize_desired_order(decision.desired_order)

            if any(order.cancel_pending for order in current_orders):
                return

            if current_orders:
                if desired is None or decision.cancel_all or not self._can_keep_orders(current_orders, desired):
                    await self._cancel_managed_orders(
                        current_orders,
                        cancel_reason=decision.reason if desired is None else "replace",
                    )
                    return

            if desired is None:
                return

            if not self._trading_enabled_for_symbol(desired.symbol) and not self._is_global_macro_flatten_order(desired):
                if current_orders:
                    await self._cancel_managed_orders(current_orders, cancel_reason=f"{desired.symbol} execution disabled")
                return

            live_matching_qty = sum(order.remaining_qty for order in current_orders if self._order_matches(order, desired))
            remaining_target_qty = max(0, int(desired.qty) - live_matching_qty)
            if remaining_target_qty <= 0:
                return

            slice_qty = self._next_slice_qty(desired.symbol, remaining_target_qty)
            staged_desired = replace(desired, qty=slice_qty)
            adjusted = self._risk_adjusted_order(staged_desired)
            if adjusted is None:
                return
            await self._submit_managed_order(adjusted)

    def _normalize_desired_order(self, desired):
        if desired is None:
            return None
        normalized_qty = int(desired.qty)
        if normalized_qty <= 0:
            return None
        if normalized_qty == desired.qty:
            return desired
        return replace(desired, qty=normalized_qty)

    def _risk_adjusted_order(self, desired):
        cfg = self._strategy_order_config(desired.symbol)
        live_orders = [order for order in self._managed_orders.values() if order.is_live]
        if len(live_orders) >= cfg.max_open_orders:
            return None

        outstanding_volume = sum(order.remaining_qty for order in live_orders)
        available_volume = max(0, cfg.max_outstanding_volume - outstanding_volume)
        if available_volume <= 0:
            return None

        current_inventory = self._inventory(desired.symbol)
        max_qty_from_position = self._max_qty_from_position_limit(desired.symbol, desired.side, current_inventory)
        if max_qty_from_position <= 0:
            return None

        shared_budget_qty = self._max_qty_from_shared_budget(desired.symbol, desired.side, int(desired.qty))
        allowed_qty = min(
            int(desired.qty),
            available_volume,
            max_qty_from_position,
            shared_budget_qty,
            cfg.max_exchange_order_qty,
        )
        if allowed_qty <= 0:
            return None
        if allowed_qty == desired.qty:
            return desired
        return replace(desired, qty=allowed_qty)

    def _max_qty_from_position_limit(self, symbol: str, side: str, current_inventory: int) -> int:
        if symbol == self.config.strategy.symbol:
            limit = self.config.strategy.max_absolute_position
        else:
            limit = self.config.c_strategy.max_absolute_position_per_contract
        if side == "BUY":
            return max(0, limit - current_inventory)
        return max(0, limit + current_inventory)

    def _max_qty_from_shared_budget(self, symbol: str, side: str, requested_qty: int) -> int:
        if symbol not in set(self.config.c_strategy.tracked_symbols):
            return requested_qty
        others_abs = sum(abs(self._inventory(other)) for other in self.config.c_strategy.tracked_symbols if other != symbol)
        current = self._inventory(symbol)
        max_qty = 0
        for candidate in range(1, requested_qty + 1):
            future_inventory = current + candidate if side == "BUY" else current - candidate
            if abs(future_inventory) + others_abs <= self.config.c_strategy.shared_rate_position_budget:
                max_qty = candidate
        return max_qty

    def _apply_fill_inventory_hint(self, *args) -> None:
        if len(args) == 4:
            symbol, side, qty, now_ms = args
        elif len(args) == 3:
            symbol = self.config.strategy.symbol
            side, qty, now_ms = args
        else:
            raise TypeError("_apply_fill_inventory_hint expects (symbol, side, qty, now_ms) or legacy (side, qty, now_ms)")
        authoritative_inventory = int(self.positions.get(symbol, self._inventory_estimate_by_symbol.get(symbol, 0)))
        last_update_ms = self._last_position_update_ms if symbol == self.config.strategy.symbol else self._last_position_update_ms_by_symbol.get(symbol)
        if last_update_ms is not None and (now_ms - last_update_ms) <= 50:
            self._inventory_estimate_by_symbol[symbol] = authoritative_inventory
            if symbol == self.config.strategy.symbol:
                self._inventory_estimate = authoritative_inventory
            return
        if authoritative_inventory != self._inventory_estimate_by_symbol.get(symbol, 0):
            self._inventory_estimate_by_symbol[symbol] = authoritative_inventory
            if symbol == self.config.strategy.symbol:
                self._inventory_estimate = authoritative_inventory
            return
        signed_qty = qty if side == "BUY" else -qty
        self._inventory_estimate_by_symbol[symbol] = self._inventory_estimate_by_symbol.get(symbol, 0) + signed_qty
        if symbol == self.config.strategy.symbol:
            self._inventory_estimate = self._inventory_estimate_by_symbol[symbol]

    def _can_keep_orders(self, active_orders: list[ManagedOrder], desired) -> bool:
        if not active_orders:
            return True
        if any(not self._order_matches(active, desired) for active in active_orders):
            return False
        total_live_qty = sum(active.remaining_qty for active in active_orders if active.is_live)
        tolerance = self._strategy_order_config(desired.symbol).replace_qty_tolerance
        return total_live_qty <= (int(desired.qty) + tolerance)

    def _current_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        return [
            order
            for order in self._managed_orders.values()
            if order.remaining_qty > 0 and (symbol is None or order.symbol == symbol)
        ]

    def _current_order(self, symbol: str | None = None) -> ManagedOrder | None:
        for order in self._managed_orders.values():
            if order.remaining_qty > 0 and (symbol is None or order.symbol == symbol):
                return order
        return None

    def _order_matches(self, active: ManagedOrder, desired) -> bool:
        if active.cancel_pending:
            return False
        if active.symbol != desired.symbol:
            return False
        if active.side != desired.side:
            return False
        cfg = self._strategy_order_config(active.symbol)
        if abs(active.px - desired.px) > cfg.replace_price_tolerance_ticks:
            age_ms = self._now_ms() - active.submitted_ms
            return age_ms < cfg.min_order_live_ms
        return True

    def _next_slice_qty(self, *args) -> int:
        if len(args) == 2:
            symbol, remaining_target_qty = args
        elif len(args) == 1:
            symbol = self.config.strategy.symbol
            remaining_target_qty = args[0]
        else:
            raise TypeError("_next_slice_qty expects (symbol, remaining_target_qty) or legacy (remaining_target_qty)")
        cfg = self._strategy_order_config(symbol)
        max_legal_qty = min(cfg.max_exchange_order_qty, cfg.order_slice_max_qty)
        if remaining_target_qty <= max_legal_qty:
            return remaining_target_qty

        slice_qty = min(max_legal_qty, cfg.order_slice_target_qty)
        remainder = remaining_target_qty - slice_qty
        min_slice = max(1, cfg.order_slice_min_qty)
        if 0 < remainder < min_slice:
            slice_qty = min(max_legal_qty, slice_qty + (min_slice - remainder))
        return max(1, slice_qty)

    async def _submit_managed_order(self, desired) -> None:
        order_id = await self.place_order(
            desired.symbol,
            qty=desired.qty,
            side=Side.BUY if desired.side == "BUY" else Side.SELL,
            px=desired.px,
        )
        self._managed_orders[order_id] = ManagedOrder(
            order_id=order_id,
            side=desired.side,
            px=desired.px,
            qty=desired.qty,
            remaining_qty=desired.qty,
            submitted_ms=self._now_ms(),
            aggressive=desired.aggressive,
            intent=desired.intent,
            reason=desired.reason,
            symbol=desired.symbol,
            context_id=desired.context_id,
        )
        if self.logger is not None:
            self.logger.record_event(
                "order_submitted",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                symbol=desired.symbol,
                order_id=order_id,
                side=desired.side,
                px=desired.px,
                qty=desired.qty,
                aggressive=desired.aggressive,
                intent=desired.intent,
                reason=desired.reason,
                rate_macro_event_id=desired.context_id,
                inventory=self._inventory(desired.symbol),
                cash=self._cash(),
                mtm_pnl_estimate=self._total_mtm(),
                shock_pnl=self._total_mtm(),
                mm_pnl=0.0,
            )

    async def _cancel_managed_orders(self, orders: list[ManagedOrder], *, cancel_reason: str) -> None:
        for order in orders:
            if order.cancel_pending:
                continue
            order.cancel_pending = True
            await self.cancel_order(order.order_id)
            if self.logger is not None:
                self.logger.record_event(
                    "order_cancel_requested",
                    now_ms=self._now_ms(),
                    message_index=self._current_message_index,
                    exchange_tick=self._estimated_tick(),
                    symbol=order.symbol,
                    order_id=order.order_id,
                    cancel_reason=cancel_reason,
                    rate_macro_event_id=order.context_id,
                    inventory=self._inventory(order.symbol),
                    cash=self._cash(),
                    mtm_pnl_estimate=self._total_mtm(),
                    shock_pnl=self._total_mtm(),
                    mm_pnl=0.0,
                )

    def _snapshot(self, *, event_symbol: str | None = None, focus_symbol: str | None = None) -> StrategySnapshot:
        chosen_symbol = focus_symbol or self._decision_focus_symbol(None, event_symbol=event_symbol)
        books_by_symbol = {symbol: self._book_snapshot(symbol) for symbol in self._tracked_symbols}
        inventories_by_symbol = {symbol: self._inventory(symbol) for symbol in self._tracked_symbols}
        open_orders_by_symbol = {
            symbol: tuple(order for order in self._managed_orders.values() if order.remaining_qty > 0 and order.symbol == symbol)
            for symbol in self._tracked_symbols
        }
        last_trade_px_by_symbol = {symbol: self._last_trade_px_by_symbol.get(symbol) for symbol in self._tracked_symbols}
        return StrategySnapshot(
            now_ms=self._now_ms(),
            exchange_tick=self._estimated_tick(),
            book=books_by_symbol.get(chosen_symbol, BookSnapshot()),
            inventory=inventories_by_symbol.get(chosen_symbol, 0),
            cash=self._cash(),
            fair_value=getattr(self.strategy, "fair_value", None),
            trusted_multiplier=getattr(self.strategy, "trusted_multiplier", None),
            latest_earnings=getattr(self.strategy, "latest_earnings", None),
            mode=getattr(self.strategy, "mode", "IDLE"),
            open_orders=open_orders_by_symbol.get(chosen_symbol, ()),
            last_trade_px=last_trade_px_by_symbol.get(chosen_symbol),
            message_index=self._current_message_index,
            books_by_symbol=books_by_symbol,
            inventories_by_symbol=inventories_by_symbol,
            open_orders_by_symbol=open_orders_by_symbol,
            last_trade_px_by_symbol=last_trade_px_by_symbol,
            event_symbol=event_symbol,
        )

    def _rate_contract_snapshot(self) -> dict[str, dict[str, int | float | None]]:
        if self.active_strategy != "C":
            return {}
        snapshot: dict[str, dict[str, int | float | None]] = {}
        for symbol in self.config.c_strategy.tracked_symbols:
            book = self._book_snapshot(symbol)
            snapshot[symbol] = {
                "best_bid_px": None if book.best_bid is None else book.best_bid.px,
                "best_bid_qty": None if book.best_bid is None else book.best_bid.qty,
                "best_ask_px": None if book.best_ask is None else book.best_ask.px,
                "best_ask_qty": None if book.best_ask is None else book.best_ask.qty,
                "mid": book.mid,
                "inventory": self._inventory(symbol),
                "open_order_count": len(self._current_orders(symbol)),
            }
        return snapshot

    def _book_snapshot(self, symbol: str) -> BookSnapshot:
        book = self.order_books.get(symbol)
        if book is None:
            return BookSnapshot()
        return BookSnapshot.from_order_book(book)

    def _estimated_tick(self) -> int | None:
        return None

    def _inventory(self, symbol: str | None = None) -> int:
        chosen_symbol = self._primary_symbol if symbol is None else symbol
        if chosen_symbol == self.config.strategy.symbol:
            return int(getattr(self, "_inventory_estimate", self._inventory_estimate_by_symbol.get(chosen_symbol, self.positions.get(chosen_symbol, 0))))
        return int(self._inventory_estimate_by_symbol.get(chosen_symbol, self.positions.get(chosen_symbol, 0)))

    def _cash(self) -> int:
        return int(self.positions.get("cash", 0))

    def _total_mtm(self) -> float | None:
        total = float(self._cash())
        saw_mark = False
        for symbol in self._tracked_symbols:
            mark = self._book_snapshot(symbol).mid
            inventory = self._inventory(symbol)
            if mark is None:
                continue
            saw_mark = True
            total += float(inventory) * float(mark)
        if saw_mark:
            return total
        return float(self._cash())

    def _decision_focus_symbol(self, decision: Decision | None, *, event_symbol: str | None = None) -> str:
        if decision is not None and decision.desired_order is not None:
            return decision.desired_order.symbol
        if event_symbol is not None and event_symbol in self._strategy_symbol_set:
            return event_symbol
        active_symbol = getattr(self.strategy, "active_target_symbol", None)
        if active_symbol in self._strategy_symbol_set:
            return str(active_symbol)
        return self._primary_symbol

    def _should_arm_global_macro_flatten(self, strategy_state: dict) -> bool:
        if self.active_strategy != "C":
            return False
        return bool(strategy_state.get("rate_global_flatten_signal"))

    def _arm_global_macro_flatten(self, now_ms: int, strategy_state: dict) -> None:
        window_ms = max(1, int(self.config.c_strategy.macro_signal_timeout_ms))
        until_ms = int(now_ms) + window_ms
        self._global_macro_flatten_until_ms = until_ms if self._global_macro_flatten_until_ms is None else max(self._global_macro_flatten_until_ms, until_ms)
        self._global_macro_flatten_reason = str(
            strategy_state.get("rate_global_flatten_reason")
            or strategy_state.get("rate_no_trade_reason")
            or "major macro shock; flattening all positions"
        )
        context_id = strategy_state.get("rate_macro_event_id")
        signal_source = strategy_state.get("rate_signal_source")
        self._global_macro_flatten_context_id = None if context_id is None else str(context_id)
        self._global_macro_flatten_source = None if signal_source is None else str(signal_source)

    def _clear_global_macro_flatten(self) -> None:
        self._global_macro_flatten_until_ms = None
        self._global_macro_flatten_reason = None
        self._global_macro_flatten_context_id = None
        self._global_macro_flatten_source = None

    def _global_macro_flatten_active(self, now_ms: int, *, snapshot: StrategySnapshot | None = None) -> bool:
        if self._global_macro_flatten_until_ms is None:
            return False
        if int(now_ms) <= int(self._global_macro_flatten_until_ms):
            return True
        if snapshot is not None:
            has_residual_inventory = any(snapshot.inventory_for(symbol) != 0 for symbol in self._tracked_symbols)
            has_open_orders = any(snapshot.open_orders_for(symbol) for symbol in self._tracked_symbols)
            if has_residual_inventory or has_open_orders:
                return True
        self._clear_global_macro_flatten()
        return False

    @staticmethod
    def _is_global_macro_flatten_order(desired: DesiredOrder) -> bool:
        return str(desired.intent) == GLOBAL_MACRO_FLATTEN_INTENT

    def _global_flatten_decision(
        self,
        snapshot: StrategySnapshot,
        *,
        event_symbol: str | None = None,
        current_decision: Decision | None = None,
    ) -> Decision:
        reason = self._global_macro_flatten_reason or "major macro shock; flattening all positions"
        positions = {symbol: snapshot.inventory_for(symbol) for symbol in self._tracked_symbols}
        candidate_symbols: list[str] = []
        if event_symbol in self._tracked_symbol_set and positions.get(str(event_symbol), 0) != 0:
            candidate_symbols.append(str(event_symbol))
        for symbol, qty in sorted(positions.items(), key=lambda item: abs(item[1]), reverse=True):
            if qty != 0 and symbol not in candidate_symbols:
                candidate_symbols.append(symbol)

        for symbol in candidate_symbols:
            qty = int(positions[symbol])
            book = snapshot.book_for(symbol)
            if qty > 0 and book.best_bid is not None:
                return Decision(
                    mode="UNWIND",
                    target_inventory=0,
                    desired_order=DesiredOrder(
                        side="SELL",
                        px=book.best_bid.px,
                        qty=abs(qty),
                        aggressive=True,
                        intent=GLOBAL_MACRO_FLATTEN_INTENT,
                        reason=reason,
                        symbol=symbol,
                        context_id=self._global_macro_flatten_context_id,
                    ),
                    cancel_all=True,
                    observe_only=False,
                    reason=reason,
                    fair_value=None,
                )
            if qty < 0 and book.best_ask is not None:
                return Decision(
                    mode="UNWIND",
                    target_inventory=0,
                    desired_order=DesiredOrder(
                        side="BUY",
                        px=book.best_ask.px,
                        qty=abs(qty),
                        aggressive=True,
                        intent=GLOBAL_MACRO_FLATTEN_INTENT,
                        reason=reason,
                        symbol=symbol,
                        context_id=self._global_macro_flatten_context_id,
                    ),
                    cancel_all=True,
                    observe_only=False,
                    reason=reason,
                    fair_value=None,
                )

        if any(snapshot.open_orders_for(symbol) for symbol in self._tracked_symbols):
            return Decision(
                mode="UNWIND",
                target_inventory=0,
                desired_order=None,
                cancel_all=True,
                observe_only=True,
                reason=reason,
                fair_value=None,
            )
        return Decision(
            mode="IDLE",
            target_inventory=0,
            desired_order=None,
            cancel_all=False,
            observe_only=True,
            reason=reason,
            fair_value=None,
        )

    def _execution_enabled(self) -> bool:
        return self.a_trading_enable if self.active_strategy == "A" else self.c_trading_enable

    def _trading_enabled_for_symbol(self, symbol: str) -> bool:
        if symbol == self.config.strategy.symbol:
            return self.a_trading_enable
        if symbol == self.config.c_mm_strategy.symbol:
            return self.c_trading_enable
        if symbol in set(self.config.c_strategy.tracked_symbols):
            return self.c_trading_enable
        return False

    def _strategy_order_config(self, symbol: str) -> StrategyConfig | MarketCStrategyConfig:
        if symbol == self.config.strategy.symbol:
            return self.config.strategy
        if symbol == self.config.c_mm_strategy.symbol:
            return self.config.c_mm_strategy
        if symbol in set(self.config.c_strategy.tracked_symbols):
            return self.config.c_strategy
        return self.config.strategy

    @staticmethod
    def _to_news_event(now_ms: int, news_release: dict) -> NewsEvent:
        raw = dict(news_release)
        new_data = raw.get("new_data") or {}
        structured_subtype = new_data.get("structured_subtype")
        value = None if new_data.get("value") is None else float(new_data.get("value"))
        forecast = None if new_data.get("forecast") is None else float(new_data.get("forecast"))
        actual = None if new_data.get("actual") is None else float(new_data.get("actual"))
        return NewsEvent(
            now_ms=now_ms,
            tick=raw.get("tick"),
            kind=str(raw.get("kind") or "unknown"),
            symbol=None if raw.get("symbol") is None else str(raw.get("symbol")),
            structured_subtype=structured_subtype,
            asset=None if new_data.get("asset") is None else str(new_data.get("asset")),
            value=value,
            content=new_data.get("content") or raw.get("content"),
            news_type=new_data.get("type") or raw.get("type"),
            forecast=forecast,
            actual=actual,
            raw_payload=raw,
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)


async def _async_main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_bot_config(base_dir)
    await BotRunner(
        config,
        a_trading_enable=A_TRADING_ENABLE,
        c_trading_enable=C_TRADING_ENABLE,
        active_strategy=ACTIVE_STRATEGY,
    ).start()


def main() -> None:
    try:
        asyncio.run(_async_main())
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
