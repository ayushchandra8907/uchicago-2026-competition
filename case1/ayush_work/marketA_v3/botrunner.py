from __future__ import annotations

import asyncio
from dataclasses import replace
import grpc
import logging
import time
from pathlib import Path

from .A_strategy import AStrategy
from .analyze_run import write_run_outputs
from .config import BotConfig, ConfigError, load_bot_config
from .core.logger import RunLogger, estimate_mtm, load_trace_events
from .core.types import BookSnapshot, Decision, ManagedOrder, NewsEvent, StrategySnapshot

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError as exc:
    raise SystemExit("utcxchangelib is not installed. Use the repo venv before running marketA_v3.") from exc


LOGGER = logging.getLogger("market-a-v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _inspect_message_body(msg) -> tuple[str | None, bool]:
    if msg == grpc.aio.EOF:
        raise EOFError("End of gRPC stream")
    which_oneof = getattr(msg, "WhichOneof", None)
    if which_oneof is None:
        return None, False
    return which_oneof("body"), True


class BotRunner(XChangeClient):
    """Thin exchange runner that delegates all A decisions to AStrategy."""

    _CONNECT_RETRY_DELAY_S = 2.0

    def __init__(self, config: BotConfig):
        super().__init__(
            config.exchange.host,
            config.exchange.username,
            config.exchange.password,
            symbols=[config.strategy.symbol],
        )
        self.config = config
        self.strategy = AStrategy(config.strategy)
        self.logger = RunLogger.create_if_enabled(config.logger, session_prefix="marketA_v3", symbol=config.strategy.symbol)
        self._order_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._timer_task: asyncio.Task | None = None
        self._position_snapshot_seen = asyncio.Event()
        self._managed_orders: dict[str, ManagedOrder] = {}
        self._last_trade_px: int | None = None
        self._current_message_index: int | None = None
        self._inventory_estimate: int = 0
        self._last_position_update_ms: int | None = None
        self._run_started_ms: int = self._now_ms()
        self._midrun_checkpoint_started = False
        self._midrun_checkpoint_task: asyncio.Task | None = None
        self._connected_once = False

    async def start(self) -> None:
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
        if msg_type == "position_update" and msg.position_update.symbol == self.config.strategy.symbol:
            self._inventory_estimate = int(msg.position_update.value)
            self._last_position_update_ms = self._now_ms()
        elif msg_type == "position_snapshot":
            self._inventory_estimate = int(self.positions.get(self.config.strategy.symbol, 0))
            self._last_position_update_ms = self._now_ms()
        if msg_type == "cash_update":
            await self._evaluate_and_sync("cash_update")
        elif msg_type == "position_update" and msg.position_update.symbol == self.config.strategy.symbol:
            await self._evaluate_and_sync("position_update")

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        self._inventory_estimate = int(self.positions.get(self.config.strategy.symbol, 0))
        self._last_position_update_ms = self._now_ms()
        now_ms = self._now_ms()
        if self.logger is not None:
            self.logger.record_event(
                "position_snapshot",
                now_ms=now_ms,
                message_index=self._current_message_index,
                inventory=self._inventory(),
                cash=self._cash(),
            )
        self._position_snapshot_seen.set()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol != self.config.strategy.symbol:
            return
        now_ms = self._now_ms()
        book = self._book_snapshot()
        if self.logger is not None:
            self.logger.record_event(
                "book_update",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                best_bid_px=None if book.best_bid is None else book.best_bid.px,
                best_bid_qty=None if book.best_bid is None else book.best_bid.qty,
                best_ask_px=None if book.best_ask is None else book.best_ask.px,
                best_ask_qty=None if book.best_ask is None else book.best_ask.qty,
                mid=book.mid,
                spread=book.spread,
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(book.mid),
                shock_pnl=self._mtm(book.mid),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("book")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol != self.config.strategy.symbol:
            return
        self._last_trade_px = int(price)
        now_ms = self._now_ms()
        if self.logger is not None:
            self.logger.record_event(
                "trade_print",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                price=int(price),
                qty=int(qty),
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid or float(price)),
                shock_pnl=self._mtm(self._book_snapshot().mid or float(price)),
                mm_pnl=0.0,
            )
        await self._sync_decision(self.strategy.on_trade(self._snapshot()), trigger="trade")

    async def bot_handle_news(self, news_release: dict) -> None:
        now_ms = self._now_ms()
        news_event = self._to_news_event(now_ms, news_release)
        decision = self.strategy.on_news(self._snapshot(), news_event)
        strategy_state = self.strategy.export_state()
        if self.logger is not None:
            self.logger.record_event(
                "news_received",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=news_event.tick,
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
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                shock_pnl=self._mtm(self._book_snapshot().mid),
                mm_pnl=0.0,
            )
        await self._sync_decision(decision, trigger="news")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        order = self._managed_orders.get(order_id)
        now_ms = self._now_ms()
        if order is not None:
            self._apply_fill_inventory_hint(order.side, int(qty), now_ms)
            order.remaining_qty = max(0, order.remaining_qty - int(qty))
            if order.remaining_qty == 0:
                self._managed_orders.pop(order_id, None)
        if self.logger is not None:
            self.logger.record_event(
                "order_filled",
                now_ms=now_ms,
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                order_id=order_id,
                side=None if order is None else order.side,
                fill_qty=int(qty),
                fill_price=int(price),
                aggressive=None if order is None else order.aggressive,
                intent=None if order is None else order.intent,
                reason=None if order is None else order.reason,
                inventory=self._inventory(),
                cash=self._cash(),
                mid=self._book_snapshot().mid,
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                shock_pnl=self._mtm(self._book_snapshot().mid),
                mm_pnl=0.0,
            )
        await self._sync_decision(self.strategy.on_fill(self._snapshot()), trigger="fill")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self._managed_orders.pop(order_id, None)
        if self.logger is not None:
            self.logger.record_event(
                "order_rejected",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                order_id=order_id,
                rejection_reason=reason,
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                shock_pnl=self._mtm(self._book_snapshot().mid),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("order_rejected")

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: str | None) -> None:
        order = self._managed_orders.get(order_id)
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
                order_id=order_id,
                success=success,
                error=error,
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                shock_pnl=self._mtm(self._book_snapshot().mid),
                mm_pnl=0.0,
            )
        await self._evaluate_and_sync("cancel_response")

    async def _timer_loop(self) -> None:
        interval_s = max(0.05, self.config.strategy.timer_interval_ms / 1000.0)
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
            name="market-a-v3-halfway-checkpoint",
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
        LOGGER.info("Generated %s checkpoint artifacts for %s%s", checkpoint_name, run_dir, "" if not generated else f" ({len(generated)} graph(s))")

    async def _evaluate_and_sync(self, trigger: str) -> None:
        if not self._position_snapshot_seen.is_set():
            return
        await self._sync_decision(self.strategy.on_book(self._snapshot()), trigger=trigger)

    async def _sync_decision(self, decision: Decision, *, trigger: str) -> None:
        snapshot = self._snapshot()
        strategy_state = self.strategy.export_state()
        current_edge_ticks = None
        if decision.fair_value is not None and snapshot.book.mid is not None:
            current_edge_ticks = float(decision.fair_value) - float(snapshot.book.mid)
        if self.logger is not None:
            desired = decision.desired_order
            self.logger.record_decision_snapshot(
                now_ms=snapshot.now_ms,
                exchange_tick=snapshot.exchange_tick,
                message_index=snapshot.message_index,
                row={
                    "mode": decision.mode,
                    "trigger": trigger,
                    "reason": decision.reason,
                    "observe_only": decision.observe_only,
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
                    "desired_side": None if desired is None else desired.side,
                    "desired_px": None if desired is None else desired.px,
                    "desired_qty": None if desired is None else desired.qty,
                    "desired_intent": None if desired is None else desired.intent,
                    "cash": snapshot.cash,
                    "mtm_pnl_estimate": self._mtm(snapshot.book.mid),
                    "shock_pnl": self._mtm(snapshot.book.mid),
                    "mm_pnl": 0.0,
                },
            )
        await self._apply_decision(decision)

    async def _apply_decision(self, decision: Decision) -> None:
        async with self._order_lock:
            current_orders = self._current_orders()
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

            live_matching_qty = sum(
                order.remaining_qty for order in current_orders if self._order_matches(order, desired)
            )
            remaining_target_qty = max(0, int(desired.qty) - live_matching_qty)
            if remaining_target_qty <= 0:
                return

            slice_qty = self._next_slice_qty(remaining_target_qty)
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
        live_orders = [order for order in self._managed_orders.values() if order.is_live]
        if len(live_orders) >= self.config.strategy.max_open_orders:
            return None

        outstanding_volume = sum(order.remaining_qty for order in live_orders)
        available_volume = max(0, self.config.strategy.max_outstanding_volume - outstanding_volume)
        if available_volume <= 0:
            return None

        current_inventory = self._inventory()
        if desired.side == "BUY":
            max_qty_from_position = max(0, self.config.strategy.max_absolute_position - current_inventory)
        else:
            max_qty_from_position = max(0, self.config.strategy.max_absolute_position + current_inventory)
        if max_qty_from_position <= 0:
            return None

        allowed_qty = min(int(desired.qty), available_volume, max_qty_from_position, self.config.strategy.max_exchange_order_qty)
        if allowed_qty <= 0:
            return None
        if allowed_qty == desired.qty:
            return desired
        return replace(desired, qty=allowed_qty)

    def _apply_fill_inventory_hint(self, side: str, qty: int, now_ms: int) -> None:
        authoritative_inventory = int(self.positions.get(self.config.strategy.symbol, self._inventory_estimate))
        if self._last_position_update_ms is not None and (now_ms - self._last_position_update_ms) <= 50:
            self._inventory_estimate = authoritative_inventory
            return
        if authoritative_inventory != self._inventory_estimate:
            self._inventory_estimate = authoritative_inventory
            return
        signed_qty = qty if side == "BUY" else -qty
        self._inventory_estimate += signed_qty

    def _can_keep_orders(self, active_orders: list[ManagedOrder], desired) -> bool:
        if not active_orders:
            return True
        if any(not self._order_matches(active, desired) for active in active_orders):
            return False
        total_live_qty = sum(active.remaining_qty for active in active_orders if active.is_live)
        return total_live_qty <= (int(desired.qty) + self.config.strategy.replace_qty_tolerance)

    def _active_order(self) -> ManagedOrder | None:
        for order in self._managed_orders.values():
            if order.is_live:
                return order
        return None

    def _current_orders(self) -> list[ManagedOrder]:
        return [order for order in self._managed_orders.values() if order.remaining_qty > 0]

    def _current_order(self) -> ManagedOrder | None:
        for order in self._managed_orders.values():
            if order.remaining_qty > 0:
                return order
        return None

    def _order_matches(self, active: ManagedOrder, desired) -> bool:
        if active.cancel_pending:
            return False
        if active.side != desired.side:
            return False
        if abs(active.px - desired.px) > self.config.strategy.replace_price_tolerance_ticks:
            age_ms = self._now_ms() - active.submitted_ms
            return age_ms < self.config.strategy.min_order_live_ms
        return True

    def _next_slice_qty(self, remaining_target_qty: int) -> int:
        max_legal_qty = min(
            self.config.strategy.max_exchange_order_qty,
            self.config.strategy.order_slice_max_qty,
        )
        if remaining_target_qty <= max_legal_qty:
            return remaining_target_qty

        slice_qty = min(max_legal_qty, self.config.strategy.order_slice_target_qty)
        remainder = remaining_target_qty - slice_qty
        min_slice = max(1, self.config.strategy.order_slice_min_qty)
        if 0 < remainder < min_slice:
            slice_qty = min(max_legal_qty, slice_qty + (min_slice - remainder))
        return max(1, slice_qty)

    async def _submit_managed_order(self, desired) -> None:
        order_id = await self.place_order(
            self.config.strategy.symbol,
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
        )
        if self.logger is not None:
            self.logger.record_event(
                "order_submitted",
                now_ms=self._now_ms(),
                message_index=self._current_message_index,
                exchange_tick=self._estimated_tick(),
                order_id=order_id,
                side=desired.side,
                px=desired.px,
                qty=desired.qty,
                aggressive=desired.aggressive,
                intent=desired.intent,
                reason=desired.reason,
                inventory=self._inventory(),
                cash=self._cash(),
                mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                shock_pnl=self._mtm(self._book_snapshot().mid),
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
                    order_id=order.order_id,
                    cancel_reason=cancel_reason,
                    inventory=self._inventory(),
                    cash=self._cash(),
                    mtm_pnl_estimate=self._mtm(self._book_snapshot().mid),
                    shock_pnl=self._mtm(self._book_snapshot().mid),
                    mm_pnl=0.0,
                )

    def _snapshot(self) -> StrategySnapshot:
        book = self._book_snapshot()
        return StrategySnapshot(
            now_ms=self._now_ms(),
            exchange_tick=self._estimated_tick(),
            book=book,
            inventory=self._inventory(),
            cash=self._cash(),
            fair_value=self.strategy.fair_value,
            trusted_multiplier=self.strategy.trusted_multiplier,
            latest_earnings=self.strategy.latest_earnings,
            mode=self.strategy.mode,
            open_orders=tuple(order for order in self._managed_orders.values() if order.remaining_qty > 0),
            last_trade_px=self._last_trade_px,
            message_index=self._current_message_index,
        )

    def _book_snapshot(self) -> BookSnapshot:
        return BookSnapshot.from_order_book(self.order_books[self.config.strategy.symbol])

    def _estimated_tick(self) -> int | None:
        return None

    def _inventory(self) -> int:
        return int(self._inventory_estimate)

    def _cash(self) -> int:
        return int(self.positions.get("cash", 0))

    def _mtm(self, mark_price: float | None) -> float | None:
        return estimate_mtm(inventory=self._inventory(), cash=self._cash(), mark_price=mark_price)

    @staticmethod
    def _to_news_event(now_ms: int, news_release: dict) -> NewsEvent:
        raw = dict(news_release)
        new_data = raw.get("new_data") or {}
        return NewsEvent(
            now_ms=now_ms,
            tick=raw.get("tick"),
            kind=str(raw.get("kind") or "unknown"),
            symbol=raw.get("symbol"),
            structured_subtype=new_data.get("structured_subtype"),
            asset=new_data.get("asset"),
            value=None if new_data.get("value") is None else float(new_data.get("value")),
            content=new_data.get("content"),
            raw_payload=raw,
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)


async def _async_main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_bot_config(base_dir)
    await BotRunner(config).start()


def main() -> None:
    try:
        asyncio.run(_async_main())
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
