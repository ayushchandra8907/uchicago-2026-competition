from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from utcxchangelib import Side, XChangeClient
except ModuleNotFoundError:
    class Side(Enum):
        BUY = "BUY"
        SELL = "SELL"

    class XChangeClient:  # pragma: no cover - fallback for offline replay analysis mode
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "utcxchangelib is required to run the live bot. Replay analysis mode can run without it."
            )


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("prediction-c")


@dataclass(frozen=True)
class BotConfig:
    symbol_c: str = "C"
    fed_hike: str = "R_HIKE"
    fed_hold: str = "R_HOLD"
    fed_cut: str = "R_CUT"

    payout_scale: float = 100.0
    default_eps_c: float = 2.0

    c_ops_weight: float = 0.72
    c_bond_weight: float = 0.28
    c_pe_yield_gamma: float = 13.0
    c_bond_duration: float = 4.5
    c_bond_convexity: float = 30.0

    cpi_to_rate_bp: float = 4000.0
    max_temp_rate_bias_bp: float = 8.0
    cpi_bias_ttl_secs: float = 2.5
    headline_bias_ttl_secs: float = 2.0

    c_earnings_ignore_delta: float = 0.010
    c_earnings_small_delta: float = 0.025
    c_earnings_medium_delta: float = 0.045
    c_earnings_fresh_secs: float = 4.0
    c_earnings_hold_secs: float = 2.5
    c_initial_shock_window_secs: float = 4.0
    c_initial_shock_gap_ticks: float = 14.0
    c_initial_shock_initial_size: int = 60
    c_initial_shock_add_size: int = 60
    c_initial_shock_cap: int = 120
    c_initial_shock_compress_ticks: float = 10.0
    c_initial_shock_flatten_ticks: float = 6.0

    post_baseline_entry_secs: float = 10.0
    post_baseline_gap_ticks: float = 14.0
    post_baseline_add_ticks: float = 20.0

    lead_lag_bp_trigger: float = 3.0
    lead_lag_entry_ticks: float = 16.0
    lead_lag_add_ticks: float = 22.0

    c_hard_position_limit: int = 200
    rate_hard_position_limit: int = 150
    c_max_order_size: int = 40
    rate_max_order_size: int = 40

    rate_normal_size: int = 50
    rate_strong_size: int = 100
    rate_extreme_size: int = 150
    rate_entry_edge_bp: float = 2.5
    rate_exit_edge_bp: float = 1.0
    rate_add_edge_step_bp: float = 1.5
    rate_reentry_block_secs: float = 1.0

    c_tier1_initial_size: int = 60
    c_tier1_add_size: int = 40
    c_tier1_cap: int = 100
    c_tier2_initial_size: int = 80
    c_tier2_add_size: int = 70
    c_tier2_cap: int = 150
    c_tier3_initial_size: int = 100
    c_tier3_add_size: int = 100
    c_tier3_cap: int = 200
    c_baseline_initial_size: int = 60
    c_baseline_add_size: int = 60
    c_baseline_cap: int = 120
    c_leadlag_initial_size: int = 50
    c_leadlag_add_size: int = 50
    c_leadlag_cap: int = 100

    c_add_edge_step_ticks: float = 6.0
    c_hard_flip_min_ticks: float = 6.0
    c_compression_min_ticks: float = 6.0
    c_compression_frac: float = 0.35
    c_rate_reversal_bp: float = 1.5
    c_reentry_block_secs: float = 1.5
    c_flat_entry_cooldown_secs: float = 0.8
    c_add_cooldown_secs: float = 0.35

    max_active_orders_per_symbol: int = 1
    order_stale_secs: float = 0.50
    anchor_reprice_secs: float = 2.0
    loop_sleep_secs: float = 0.20
    status_log_interval_secs: float = 3.0

    startup_flatten_chunk_c: int = 25
    startup_flatten_chunk_rate: int = 50
    startup_flatten_sleep_secs: float = 0.25

    @property
    def tracked_symbols(self) -> tuple[str, str, str, str]:
        return (self.symbol_c, self.fed_hike, self.fed_hold, self.fed_cut)

    @property
    def rate_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


@dataclass
class TopOfBook:
    bid: Optional[int] = None
    bid_qty: int = 0
    ask: Optional[int] = None
    ask_qty: int = 0
    updated_ts: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        if self.bid is not None:
            return float(self.bid)
        if self.ask is not None:
            return float(self.ask)
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return float(self.ask - self.bid)

    @property
    def usable_spread(self) -> float:
        spread = self.spread
        if spread is None:
            return 0.0
        return max(1.0, abs(spread))


@dataclass
class TrackedOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    role: str
    engine: str
    reason: str
    thesis: Optional[str]
    signal_strength: float
    created_at: float = field(default_factory=time.time)


@dataclass
class RateSnapshot:
    q_hike: float
    q_hold: float
    q_cut: float
    market_expected_rate_bp: float
    effective_expected_rate_bp: float
    delta_market_bp: float
    delta_effective_bp: float
    bias_bp: float
    urgent: bool
    fresh_macro_event: bool
    macro_source: Optional[str]


@dataclass
class EarningsContext:
    delta: float
    abs_delta: float
    age: float
    tier: int
    is_initial: bool
    side: Optional[Side]
    hold_active: bool


@dataclass
class CSignal:
    bid: int
    bid_qty: int
    ask: int
    ask_qty: int
    mid: float
    spread: float
    fair: float
    gap: float
    gap_abs: float
    bp_dislocation: float
    fair_change: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class CEntryDecision:
    side: Side
    thesis: str
    edge_ticks: float
    initial_size: int
    add_size: int
    thesis_cap: int


@dataclass
class RatesEntryDecision:
    direction: str
    edge_bp: float
    target_size: int


@dataclass
class MarketState:
    books: Dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: Dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)
    last_trade_price: Dict[str, int] = field(default_factory=dict)

    current_eps_c: float = 2.0
    have_real_eps_c: bool = False
    last_c_earnings_delta: float = 0.0
    last_c_earnings_ts: float = 0.0
    last_c_earnings_is_initial: bool = False
    last_c_initial_baseline_ts: float = 0.0
    c_initial_shock_consumed: bool = False

    anchor_price: Optional[float] = None
    anchor_eps: Optional[float] = None
    anchor_yield_bp: Optional[float] = None
    anchor_has_real_eps: bool = False
    last_anchor_update_ts: float = 0.0

    temp_rate_bias_bp: float = 0.0
    temp_rate_bias_started_at: float = 0.0
    temp_rate_bias_expires_at: float = 0.0
    last_macro_event_ts: float = 0.0
    last_macro_source: Optional[str] = None
    last_macro_bias_bp: float = 0.0
    news_urgency_until: float = 0.0

    last_market_expected_rate_bp: Optional[float] = None
    last_effective_expected_rate_bp: Optional[float] = None
    last_fair_c: Optional[float] = None

    startup_flatten_complete: bool = False
    session_start_cash: Optional[float] = None
    session_start_mtm: Optional[float] = None

    last_status_log_ts: float = 0.0

    c_regime_side: Optional[Side] = None
    c_regime_thesis: Optional[str] = None
    c_entry_stage: int = 0
    c_last_entry_edge: float = 0.0
    c_last_add_ts: float = 0.0
    c_blocked_side: Optional[Side] = None
    c_blocked_until: float = 0.0
    c_flat_entry_cooldown_until: float = 0.0

    rates_regime_direction: Optional[str] = None
    rates_entry_stage: int = 0
    rates_last_entry_edge: float = 0.0
    rates_last_add_ts: float = 0.0
    rates_blocked_direction: Optional[str] = None
    rates_blocked_until: float = 0.0

    def clear_c_regime(self) -> None:
        self.c_regime_side = None
        self.c_regime_thesis = None
        self.c_entry_stage = 0
        self.c_last_entry_edge = 0.0
        self.c_last_add_ts = 0.0

    def clear_rates_regime(self) -> None:
        self.rates_regime_direction = None
        self.rates_entry_stage = 0
        self.rates_last_entry_edge = 0.0
        self.rates_last_add_ts = 0.0


class RunDataLogger:
    def __init__(self, cfg: BotConfig, *, host: str, username: str, log_root: Optional[Path] = None):
        self.cfg = cfg
        self.host = host
        self.username = username
        self.root = Path(log_root) if log_root is not None else _default_replay_root()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000:06d}"
        self.run_id = f"market_research_live_{stamp}_{suffix}"
        self.run_dir = (self.root / self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._seq = 0
        self._csv_writers: dict[str, tuple[Any, csv.DictWriter, tuple[str, ...]]] = {}
        self._events_handle = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")

        metadata = {
            "run_id": self.run_id,
            "started_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "started_time_ns": time.time_ns(),
            "started_monotonic_ns": time.monotonic_ns(),
            "host": host,
            "username": username,
            "config": asdict(cfg),
        }
        (self.run_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Run data logger active: %s", self.run_dir)

    @staticmethod
    def _side_name(side: Optional[Side]) -> str:
        return "" if side is None else side.name

    def _next_meta(self) -> dict[str, Any]:
        self._seq += 1
        return {
            "run_id": self.run_id,
            "event_seq": self._seq,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
        }

    def _writer(self, filename: str, fieldnames: tuple[str, ...]) -> tuple[Any, csv.DictWriter]:
        cached = self._csv_writers.get(filename)
        if cached is not None:
            handle, writer, _ = cached
            return handle, writer

        path = self.run_dir / filename
        handle = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        handle.flush()
        self._csv_writers[filename] = (handle, writer, fieldnames)
        return handle, writer

    def _write_csv(self, filename: str, fieldnames: tuple[str, ...], row: dict[str, Any]) -> None:
        handle, writer = self._writer(filename, fieldnames)
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()

    def _context(self, client: "MyXchangeClient") -> dict[str, Any]:
        cash, mtm = client.cash_and_total_mtm()
        session_cash, session_mtm = client.session_pnl_snapshot(cash, mtm)
        state = client.state
        return {
            "pos_c": client.get_position(client.cfg.symbol_c),
            "pos_r_hike": client.get_position(client.cfg.fed_hike),
            "pos_r_hold": client.get_position(client.cfg.fed_hold),
            "pos_r_cut": client.get_position(client.cfg.fed_cut),
            "cash": cash,
            "mtm": mtm,
            "session_cash": session_cash,
            "session_mtm": session_mtm,
            "eps_c": state.current_eps_c,
            "have_real_eps_c": state.have_real_eps_c,
            "last_c_earnings_delta": state.last_c_earnings_delta,
            "anchor_price": state.anchor_price,
            "anchor_eps": state.anchor_eps,
            "anchor_yield_bp": state.anchor_yield_bp,
            "c_regime_side": self._side_name(state.c_regime_side),
            "c_regime_thesis": state.c_regime_thesis or "",
            "c_entry_stage": state.c_entry_stage,
            "rates_regime_direction": state.rates_regime_direction or "",
            "rates_entry_stage": state.rates_entry_stage,
            "temp_rate_bias_bp": state.temp_rate_bias_bp,
            "last_macro_source": state.last_macro_source or "",
            "live_order_count": len(state.live_orders),
        }

    def log_event(self, client: "MyXchangeClient", event_type: str, **payload: Any) -> None:
        row = self._next_meta()
        row["event_type"] = event_type
        row.update(self._context(client))
        row.update(payload)
        self._events_handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
        self._events_handle.flush()

    def log_book_update(self, client: "MyXchangeClient", symbol: str, source: str) -> None:
        book = client.top(symbol)
        ctx = self._context(client)
        meta = self._next_meta()
        row = {
            **meta,
            "source": source,
            "symbol": symbol,
            "bid_px": book.bid,
            "bid_qty": book.bid_qty,
            "ask_px": book.ask,
            "ask_qty": book.ask_qty,
            "mid_px": book.mid,
            "spread": book.spread,
            "last_trade_px": client.state.last_trade_price.get(symbol, ""),
            "pos": client.get_position(symbol),
            "cash": ctx["cash"],
            "mtm": ctx["mtm"],
            "session_cash": ctx["session_cash"],
            "session_mtm": ctx["session_mtm"],
        }
        self._write_csv(
            f"raw_book_events_{symbol}.csv",
            (
                "run_id",
                "event_seq",
                "wall_time_ns",
                "monotonic_ns",
                "source",
                "symbol",
                "bid_px",
                "bid_qty",
                "ask_px",
                "ask_qty",
                "mid_px",
                "spread",
                "last_trade_px",
                "pos",
                "cash",
                "mtm",
                "session_cash",
                "session_mtm",
            ),
            row,
        )
        self.log_event(
            client,
            "book_update",
            source=source,
            symbol=symbol,
            bid_px=book.bid,
            bid_qty=book.bid_qty,
            ask_px=book.ask,
            ask_qty=book.ask_qty,
            mid_px=book.mid,
            spread=book.spread,
            last_trade_px=client.state.last_trade_price.get(symbol),
        )

    def log_trade_event(self, client: "MyXchangeClient", symbol: str, price: int, qty: int, source: str) -> None:
        book = client.top(symbol)
        ctx = self._context(client)
        meta = self._next_meta()
        row = {
            **meta,
            "source": source,
            "symbol": symbol,
            "price": price,
            "qty": qty,
            "mid_px": book.mid,
            "spread": book.spread,
            "pos": client.get_position(symbol),
            "cash": ctx["cash"],
            "mtm": ctx["mtm"],
            "session_cash": ctx["session_cash"],
            "session_mtm": ctx["session_mtm"],
        }
        self._write_csv(
            f"raw_trade_events_{symbol}.csv",
            (
                "run_id",
                "event_seq",
                "wall_time_ns",
                "monotonic_ns",
                "source",
                "symbol",
                "price",
                "qty",
                "mid_px",
                "spread",
                "pos",
                "cash",
                "mtm",
                "session_cash",
                "session_mtm",
            ),
            row,
        )
        self.log_event(client, "trade_msg", source=source, symbol=symbol, price=price, qty=qty, mid_px=book.mid, spread=book.spread)

    def log_news_callback(
        self,
        client: "MyXchangeClient",
        news_release: dict[str, Any],
        *,
        kept_in_raw_news_events: bool,
        computed_bias_bp: Optional[float] = None,
        note: str = "",
    ) -> None:
        new_data = news_release.get("new_data", {}) or {}
        row = {
            **self._next_meta(),
            "tick": news_release.get("tick", ""),
            "kind": news_release.get("kind", ""),
            "structured_subtype": new_data.get("structured_subtype", ""),
            "asset": new_data.get("asset", "") or new_data.get("symbol", ""),
            "value": new_data.get("value", ""),
            "actual": new_data.get("actual", ""),
            "forecast": new_data.get("forecast", ""),
            "content": new_data.get("content", ""),
            "new_data_json": json.dumps(new_data, sort_keys=True, default=str),
            "computed_bias_bp": computed_bias_bp,
            "kept_in_raw_news_events": kept_in_raw_news_events,
            "note": note,
        }
        fieldnames = (
            "run_id",
            "event_seq",
            "wall_time_ns",
            "monotonic_ns",
            "tick",
            "kind",
            "structured_subtype",
            "asset",
            "value",
            "actual",
            "forecast",
            "content",
            "new_data_json",
            "computed_bias_bp",
            "kept_in_raw_news_events",
            "note",
        )
        self._write_csv("raw_all_news_callbacks.csv", fieldnames, row)
        if kept_in_raw_news_events:
            self._write_csv("raw_news_events.csv", fieldnames, row)
        self.log_event(
            client,
            "news_callback",
            tick=news_release.get("tick"),
            kind=news_release.get("kind"),
            structured_subtype=new_data.get("structured_subtype"),
            asset=new_data.get("asset") or new_data.get("symbol"),
            value=new_data.get("value"),
            actual=new_data.get("actual"),
            forecast=new_data.get("forecast"),
            content=new_data.get("content"),
            kept_in_raw_news_events=kept_in_raw_news_events,
            computed_bias_bp=computed_bias_bp,
            note=note,
        )

    def log_order_event(
        self,
        client: "MyXchangeClient",
        event_name: str,
        *,
        tracked: Optional[TrackedOrder] = None,
        order_id: Optional[str] = None,
        qty: Optional[int] = None,
        price: Optional[int] = None,
        error: str = "",
    ) -> None:
        self.log_event(
            client,
            "order_event",
            order_event=event_name,
            order_id=(tracked.order_id if tracked is not None else order_id or ""),
            symbol=(tracked.symbol if tracked is not None else ""),
            side=(self._side_name(tracked.side) if tracked is not None else ""),
            qty=(tracked.qty if tracked is not None else qty),
            price=(tracked.price if tracked is not None else price),
            role=(tracked.role if tracked is not None else ""),
            engine=(tracked.engine if tracked is not None else ""),
            reason=(tracked.reason if tracked is not None else ""),
            thesis=(tracked.thesis if tracked is not None else ""),
            signal_strength=(tracked.signal_strength if tracked is not None else ""),
            fill_qty=qty,
            fill_price=price,
            error=error,
        )

    def log_anchor_event(self, client: "MyXchangeClient", event_name: str, source: str) -> None:
        self.log_event(
            client,
            "anchor_event",
            anchor_event=event_name,
            source=source,
            anchor_price=client.state.anchor_price,
            anchor_eps=client.state.anchor_eps,
            anchor_yield_bp=client.state.anchor_yield_bp,
            anchor_has_real_eps=client.state.anchor_has_real_eps,
        )

    def log_strategy_event(self, client: "MyXchangeClient", event_name: str, **payload: Any) -> None:
        self.log_event(client, "strategy_event", strategy_event=event_name, **payload)

    def log_decision_snapshot(
        self,
        client: "MyXchangeClient",
        reason: str,
        *,
        rate_snapshot: Optional[RateSnapshot] = None,
        c_signal: Optional[CSignal] = None,
        c_entry_decision: Optional[CEntryDecision] = None,
    ) -> None:
        earnings = client.c_fair_engine.earnings_context()
        initial_shock_window_remaining = client.c_trading_engine.initial_shock_window_remaining()
        payload: dict[str, Any] = {
            "reason": reason,
            "missing_prereqs": client.missing_prereqs(),
            "earnings_delta": earnings.delta,
            "earnings_age": earnings.age,
            "earnings_tier": earnings.tier,
            "earnings_side": self._side_name(earnings.side),
            "earnings_hold_active": earnings.hold_active,
            "initial_shock_window_active": initial_shock_window_remaining > 0.0,
            "initial_shock_window_remaining": initial_shock_window_remaining,
            "initial_shock_consumed": client.state.c_initial_shock_consumed,
        }
        if rate_snapshot is not None:
            payload.update(
                {
                    "q_hike": rate_snapshot.q_hike,
                    "q_hold": rate_snapshot.q_hold,
                    "q_cut": rate_snapshot.q_cut,
                    "market_expected_rate_bp": rate_snapshot.market_expected_rate_bp,
                    "effective_expected_rate_bp": rate_snapshot.effective_expected_rate_bp,
                    "delta_market_bp": rate_snapshot.delta_market_bp,
                    "delta_effective_bp": rate_snapshot.delta_effective_bp,
                    "bias_bp": rate_snapshot.bias_bp,
                    "urgent": rate_snapshot.urgent,
                    "fresh_macro_event": rate_snapshot.fresh_macro_event,
                    "macro_source": rate_snapshot.macro_source,
                }
            )
        if c_signal is not None:
            payload.update(
                {
                    "c_bid": c_signal.bid,
                    "c_bid_qty": c_signal.bid_qty,
                    "c_ask": c_signal.ask,
                    "c_ask_qty": c_signal.ask_qty,
                    "c_mid": c_signal.mid,
                    "c_spread": c_signal.spread,
                    "c_fair": c_signal.fair,
                    "c_gap": c_signal.gap,
                    "c_gap_abs": c_signal.gap_abs,
                    "c_bp_dislocation": c_signal.bp_dislocation,
                    "c_fair_change": c_signal.fair_change,
                    "c_entry_threshold": c_signal.entry_threshold,
                    "c_exit_threshold": c_signal.exit_threshold,
                }
            )
        if c_entry_decision is not None:
            payload.update(
                {
                    "c_candidate_side": self._side_name(c_entry_decision.side),
                    "c_candidate_thesis": c_entry_decision.thesis,
                    "c_candidate_edge_ticks": c_entry_decision.edge_ticks,
                    "c_candidate_initial_size": c_entry_decision.initial_size,
                    "c_candidate_add_size": c_entry_decision.add_size,
                    "c_candidate_cap": c_entry_decision.thesis_cap,
                }
            )
        self.log_event(client, "decision_snapshot", **payload)


class OrderManager:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def has_live_order(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        engine: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> bool:
        for order in self.state.live_orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if engine is not None and order.engine != engine:
                continue
            if side is not None and order.side != side:
                continue
            return True
        return False

    def pending_qty(self, symbol: str, side: Side) -> int:
        total = 0
        for order in self.state.live_orders.values():
            if order.symbol == symbol and order.side == side:
                total += int(order.qty)
        return total

    async def cancel_order_if_present(self, order_id: str) -> None:
        order_key = str(order_id)
        if order_key in self.state.pending_cancels:
            return

        tracked = self.state.live_orders.get(order_key)
        self.state.pending_cancels.add(order_key)
        try:
            await self.client.cancel_order(order_id)
        except Exception as exc:
            message = str(exc)
            if "No such order" in message:
                self.state.live_orders.pop(order_key, None)
            else:
                LOGGER.warning("Cancel failed for %s: %s", order_id, exc)
                if tracked is not None:
                    self.state.live_orders[order_key] = tracked
        finally:
            self.state.pending_cancels.discard(order_key)

    async def cancel_live_orders(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        engine: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> None:
        order_ids: list[str] = []
        for order_id, order in self.state.live_orders.items():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if engine is not None and order.engine != engine:
                continue
            if side is not None and order.side != side:
                continue
            order_ids.append(order_id)

        for order_id in order_ids:
            await self.cancel_order_if_present(order_id)

    async def cancel_stale_orders(self) -> None:
        now = time.time()
        stale = [
            order_id
            for order_id, order in self.state.live_orders.items()
            if now - order.created_at >= self.cfg.order_stale_secs
        ]
        for order_id in stale:
            await self.cancel_order_if_present(order_id)

    async def place_tracked_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: Side,
        price: int,
        role: str,
        engine: str,
        reason: str,
        thesis: Optional[str] = None,
        signal_strength: float = 0.0,
    ) -> bool:
        if qty <= 0:
            return False

        if self.has_live_order(symbol=symbol):
            return False

        same_symbol_orders = [order for order in self.state.live_orders.values() if order.symbol == symbol]
        if len(same_symbol_orders) >= self.cfg.max_active_orders_per_symbol:
            await self.cancel_order_if_present(same_symbol_orders[0].order_id)
            return False

        order_id = await self.client.place_order(symbol, int(qty), side, int(price))
        if order_id is None:
            return False

        tracked = TrackedOrder(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            qty=int(qty),
            price=int(price),
            role=role,
            engine=engine,
            reason=reason,
            thesis=thesis,
            signal_strength=float(signal_strength),
        )
        self.state.live_orders[str(order_id)] = tracked
        LOGGER.info(
            "Placed %s order: symbol=%s side=%s qty=%s price=%s thesis=%s strength=%.2f pos=%s",
            reason,
            symbol,
            side.name,
            qty,
            price,
            thesis or "none",
            signal_strength,
            self.client.get_position(symbol),
        )
        if getattr(self.client, "data_logger", None) is not None:
            self.client.data_logger.log_order_event(self.client, "placed", tracked=tracked)
        return True

    def sync_fill(self, order_id: str) -> Optional[TrackedOrder]:
        order_key = str(order_id)
        tracked = self.state.live_orders.get(order_key)
        if tracked is None:
            return None

        if order_key in self.client.open_orders:
            remaining_qty = int(self.client.open_orders[order_key][1])
            tracked.qty = remaining_qty
            if remaining_qty <= 0:
                self.state.live_orders.pop(order_key, None)
        else:
            self.state.live_orders.pop(order_key, None)
        return tracked

    def sync_rejected(self, order_id: str) -> Optional[TrackedOrder]:
        return self.state.live_orders.pop(str(order_id), None)

    def sync_cancel_response(self, order_id: str, success: bool) -> Optional[TrackedOrder]:
        if success:
            return self.state.live_orders.pop(str(order_id), None)
        return self.state.live_orders.get(str(order_id))


class RiskManager:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState, orders: OrderManager):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders

    def clip_c_qty(self, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(self.cfg.symbol_c)
        pending = self.orders.pending_qty(self.cfg.symbol_c, side)

        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.c_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.c_max_order_size))

    def clip_rate_qty(self, symbol: str, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(symbol)
        pending = self.orders.pending_qty(symbol, side)

        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.rate_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.rate_max_order_size))

    def arm_session_baseline_if_ready(self) -> bool:
        if self.state.session_start_cash is not None and self.state.session_start_mtm is not None:
            return False

        for symbol in self.cfg.tracked_symbols:
            if self.client.get_position(symbol) != 0:
                return False
        if self.client.open_orders:
            return False
        if self.state.live_orders:
            return False

        cash, mtm = self.client.cash_and_total_mtm()
        self.state.session_start_cash = cash
        self.state.session_start_mtm = mtm
        LOGGER.info(
            "Session baseline armed: cash=%.2f mtm=%.2f session_pnl=0.00",
            cash,
            mtm,
        )
        if getattr(self.client, "data_logger", None) is not None:
            self.client.data_logger.log_strategy_event(self.client, "session_baseline_armed")
        return True

    async def startup_flatten_step(self) -> bool:
        inherited_order_ids = [str(order_id) for order_id in self.client.open_orders.keys() if str(order_id) not in self.state.live_orders]
        for order_id in inherited_order_ids:
            await self.orders.cancel_order_if_present(order_id)

        for symbol in self.cfg.tracked_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue

            book = self.client.top(symbol)
            if book.bid is None or book.ask is None:
                continue

            side = Side.SELL if pos > 0 else Side.BUY
            chunk = self.cfg.startup_flatten_chunk_c if symbol == self.cfg.symbol_c else self.cfg.startup_flatten_chunk_rate
            qty = min(abs(pos), chunk)
            price = int(book.bid if side == Side.SELL else book.ask)
            if self.orders.has_live_order(symbol=symbol):
                continue
            await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="flatten",
                engine="risk",
                reason="startup_flatten",
                signal_strength=float(abs(pos)),
            )

        all_flat = all(self.client.get_position(symbol) == 0 for symbol in self.cfg.tracked_symbols)
        no_orders = not self.client.open_orders and not self.state.live_orders
        if all_flat and no_orders:
            self.state.startup_flatten_complete = True
            self.arm_session_baseline_if_ready()
            LOGGER.info("Startup flatten complete; all tracked positions and inherited orders are flat.")
            if getattr(self.client, "data_logger", None) is not None:
                self.client.data_logger.log_strategy_event(self.client, "startup_flatten_complete")
            return True
        return False


class RatesSignalEngine:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def current_temp_bias(self) -> float:
        now = time.time()
        if now >= self.state.temp_rate_bias_expires_at or self.state.temp_rate_bias_expires_at <= self.state.temp_rate_bias_started_at:
            self.state.temp_rate_bias_bp = 0.0
            return 0.0

        duration = self.state.temp_rate_bias_expires_at - self.state.temp_rate_bias_started_at
        remaining = max(0.0, self.state.temp_rate_bias_expires_at - now)
        if duration <= 0:
            return 0.0
        return self.state.temp_rate_bias_bp * (remaining / duration)

    def mark_news_urgent(self, ttl_secs: float) -> None:
        self.state.news_urgency_until = max(self.state.news_urgency_until, time.time() + ttl_secs)

    def is_news_urgent(self) -> bool:
        return time.time() < self.state.news_urgency_until

    def apply_temp_rate_bias(self, delta_bp: float, ttl_secs: float, source: str) -> None:
        current_bias = self.current_temp_bias()
        next_bias = self.client.clip(current_bias + delta_bp, -self.cfg.max_temp_rate_bias_bp, self.cfg.max_temp_rate_bias_bp)
        now = time.time()
        self.state.temp_rate_bias_bp = next_bias
        self.state.temp_rate_bias_started_at = now
        self.state.temp_rate_bias_expires_at = now + ttl_secs
        self.state.last_macro_event_ts = now
        self.state.last_macro_source = source
        self.state.last_macro_bias_bp = next_bias
        self.mark_news_urgent(ttl_secs)

    def headline_rate_bias_bp(self, content: str) -> float:
        text = content.lower()
        score = 0.0

        hawkish = {
            "inflation risks": 1.5,
            "reassess path of cuts": 1.5,
            "higher for longer": 2.0,
            "keeps options open": 1.0,
            "keep options open": 1.0,
            "strong demand and sticky prices": 2.0,
            "sticky prices": 1.5,
            "stay restrictive for longer": 2.0,
            "policy may stay restrictive": 2.0,
            "persistent inflation": 1.5,
            "concerned about wage growth": 1.5,
            "pressure on the fed": 1.5,
            "emphasizes inflation risks": 2.0,
        }
        dovish = {
            "moving back to target": -1.5,
            "cooling inflation": -1.0,
            "softer inflation": -1.0,
            "disinflation": -1.0,
            "confidence inflation is moving back to target": -2.0,
            "softening data": -1.5,
            "policy easing": -1.5,
            "expectations of policy easing": -2.0,
            "cooling labor market": -1.25,
            "easing inflation pressures": -1.5,
            "cooling labor market and easing inflation pressures": -2.0,
            "increasing confidence inflation is moving back to target": -2.0,
        }

        for phrase, value in hawkish.items():
            if phrase in text:
                score += value
        for phrase, value in dovish.items():
            if phrase in text:
                score += value

        if (
            "balanced risks" in text
            or "mixed economic indicators" in text
            or "communication remains cautious" in text
            or "await upcoming data" in text
        ):
            score *= 0.5

        return self.client.clip(score, -2.0, 2.0)

    def fed_probs(self) -> Optional[tuple[float, float, float]]:
        mid_hike = self.client.mid(self.cfg.fed_hike)
        mid_hold = self.client.mid(self.cfg.fed_hold)
        mid_cut = self.client.mid(self.cfg.fed_cut)
        if mid_hike is None or mid_hold is None or mid_cut is None:
            return None

        q_hike = mid_hike / self.cfg.payout_scale
        q_hold = mid_hold / self.cfg.payout_scale
        q_cut = mid_cut / self.cfg.payout_scale
        total = q_hike + q_hold + q_cut
        if total <= 1e-9:
            return None
        return q_hike / total, q_hold / total, q_cut / total

    def expected_rate_bp(self) -> Optional[float]:
        probs = self.fed_probs()
        if probs is None:
            return None
        q_hike, _, q_cut = probs
        return 25.0 * q_hike - 25.0 * q_cut

    def snapshot(self) -> Optional[RateSnapshot]:
        probs = self.fed_probs()
        if probs is None:
            return None

        market_expected_rate_bp = self.expected_rate_bp()
        if market_expected_rate_bp is None:
            return None

        bias_bp = self.current_temp_bias()
        effective_expected_rate_bp = market_expected_rate_bp + bias_bp
        delta_market_bp = (
            0.0
            if self.state.last_market_expected_rate_bp is None
            else market_expected_rate_bp - self.state.last_market_expected_rate_bp
        )
        delta_effective_bp = (
            0.0
            if self.state.last_effective_expected_rate_bp is None
            else effective_expected_rate_bp - self.state.last_effective_expected_rate_bp
        )
        fresh_macro = False
        if self.state.last_macro_event_ts > 0.0:
            ttl = self.cfg.cpi_bias_ttl_secs if self.state.last_macro_source == "cpi_print" else self.cfg.headline_bias_ttl_secs
            fresh_macro = time.time() - self.state.last_macro_event_ts <= ttl

        return RateSnapshot(
            q_hike=probs[0],
            q_hold=probs[1],
            q_cut=probs[2],
            market_expected_rate_bp=market_expected_rate_bp,
            effective_expected_rate_bp=effective_expected_rate_bp,
            delta_market_bp=delta_market_bp,
            delta_effective_bp=delta_effective_bp,
            bias_bp=bias_bp,
            urgent=self.is_news_urgent(),
            fresh_macro_event=fresh_macro,
            macro_source=self.state.last_macro_source,
        )


class CFairValueEngine:
    def __init__(self, client: "MyXchangeClient", cfg: BotConfig, state: MarketState, rates: RatesSignalEngine):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.rates = rates

    def earnings_context(self) -> EarningsContext:
        if self.state.last_c_earnings_ts <= 0.0:
            return EarningsContext(0.0, 0.0, float("inf"), 0, False, None, False)

        age = max(0.0, time.time() - self.state.last_c_earnings_ts)
        delta = self.state.last_c_earnings_delta
        abs_delta = abs(delta)
        side: Optional[Side] = None
        if delta > 0:
            side = Side.BUY
        elif delta < 0:
            side = Side.SELL

        tier = 0
        if not self.state.last_c_earnings_is_initial:
            if abs_delta >= self.cfg.c_earnings_medium_delta:
                tier = 3
            elif abs_delta >= self.cfg.c_earnings_small_delta:
                tier = 2
            elif abs_delta >= self.cfg.c_earnings_ignore_delta:
                tier = 1

        hold_secs = {
            0: 0.0,
            1: 1.50,
            2: self.cfg.c_earnings_hold_secs,
            3: self.cfg.c_earnings_hold_secs + 0.75,
        }[tier]

        return EarningsContext(
            delta=delta,
            abs_delta=abs_delta,
            age=age,
            tier=tier,
            is_initial=self.state.last_c_earnings_is_initial,
            side=side,
            hold_active=tier > 0 and age <= hold_secs,
        )

    def initialize_anchor(self, rate_snapshot: Optional[RateSnapshot], *, force: bool = False) -> bool:
        book = self.client.top(self.cfg.symbol_c)
        mid_c = book.mid
        if mid_c is None or rate_snapshot is None:
            return False

        if (
            self.state.anchor_price is None
            or self.state.anchor_eps is None
            or self.state.anchor_yield_bp is None
            or force
        ):
            self.state.anchor_price = mid_c
            self.state.anchor_eps = self.state.current_eps_c
            self.state.anchor_yield_bp = rate_snapshot.effective_expected_rate_bp
            self.state.anchor_has_real_eps = self.state.have_real_eps_c
            self.state.last_anchor_update_ts = time.time()
            LOGGER.info(
                "Initialized C anchor: price=%.2f eps=%.4f exp_bp=%.2f source=%s",
                self.state.anchor_price,
                self.state.anchor_eps,
                self.state.anchor_yield_bp,
                "real_earnings" if self.state.anchor_has_real_eps else "default_eps",
            )
            if getattr(self.client, "data_logger", None) is not None:
                self.client.data_logger.log_anchor_event(
                    self.client,
                    "initialize_anchor",
                    "real_earnings" if self.state.anchor_has_real_eps else "default_eps",
                )
            return True
        return False

    def maybe_refresh_anchor_after_first_real_eps(self, rate_snapshot: Optional[RateSnapshot]) -> None:
        if not self.state.have_real_eps_c:
            return
        if self.state.anchor_has_real_eps:
            return
        if self.initialize_anchor(rate_snapshot, force=True):
            now = time.time()
            self.state.last_c_initial_baseline_ts = now
            self.state.c_initial_shock_consumed = False
            LOGGER.info(
                "Adopted first real C EPS as new baseline anchor: price=%.2f eps=%.4f exp_bp=%.2f",
                self.state.anchor_price or 0.0,
                self.state.anchor_eps or 0.0,
                self.state.anchor_yield_bp or 0.0,
            )
            if getattr(self.client, "data_logger", None) is not None:
                self.client.data_logger.log_anchor_event(self.client, "baseline_anchor_refresh", "first_real_eps")
                self.client.data_logger.log_strategy_event(
                    self.client,
                    "initial_earnings_shock_window_started",
                    started_at=now,
                    window_secs=self.cfg.c_initial_shock_window_secs,
                )

    def maybe_reanchor(self, signal: Optional[CSignal], rate_snapshot: Optional[RateSnapshot]) -> None:
        if signal is None or rate_snapshot is None:
            return
        if self.client.get_position(self.cfg.symbol_c) != 0:
            return
        if self.orders_live_for_c():
            return
        if self.rates.is_news_urgent():
            return
        if time.time() - self.state.last_anchor_update_ts < self.cfg.anchor_reprice_secs:
            return
        if signal.gap_abs > max(signal.spread, signal.exit_threshold):
            return

        self.state.anchor_price = 0.98 * float(self.state.anchor_price) + 0.02 * signal.mid
        self.state.anchor_yield_bp = (
            0.98 * float(self.state.anchor_yield_bp) + 0.02 * rate_snapshot.effective_expected_rate_bp
        )
        self.state.last_anchor_update_ts = time.time()
        if getattr(self.client, "data_logger", None) is not None:
            self.client.data_logger.log_anchor_event(self.client, "soft_reanchor", "idle_reprice")

    def orders_live_for_c(self) -> bool:
        return any(order.symbol == self.cfg.symbol_c for order in self.state.live_orders.values())

    def fair_value(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[float]:
        if (
            self.state.anchor_price is None
            or self.state.anchor_eps is None
            or self.state.anchor_yield_bp is None
            or rate_snapshot is None
        ):
            return None

        if self.state.anchor_eps == 0:
            return None

        anchor_price = float(self.state.anchor_price)
        anchor_eps = float(self.state.anchor_eps)
        anchor_yield_bp = float(self.state.anchor_yield_bp)
        yield_now = float(rate_snapshot.effective_expected_rate_bp)
        dy = (yield_now - anchor_yield_bp) / 10000.0

        ops_anchor = self.cfg.c_ops_weight * anchor_price
        bond_anchor = self.cfg.c_bond_weight * anchor_price

        ops_fair = ops_anchor * (self.state.current_eps_c / anchor_eps) * math.exp(-self.cfg.c_pe_yield_gamma * dy)
        bond_fair = bond_anchor * (1.0 - self.cfg.c_bond_duration * dy + 0.5 * self.cfg.c_bond_convexity * dy * dy)
        return ops_fair + bond_fair

    def snapshot(self, rate_snapshot: Optional[RateSnapshot]) -> Optional[CSignal]:
        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return None

        fair = self.fair_value(rate_snapshot)
        if fair is None:
            return None
        if self.state.anchor_yield_bp is None or rate_snapshot is None:
            return None

        mid = book.mid
        if mid is None:
            return None

        fair_change = 0.0 if self.state.last_fair_c is None else fair - self.state.last_fair_c
        spread = book.usable_spread
        return CSignal(
            bid=int(book.bid),
            bid_qty=int(book.bid_qty),
            ask=int(book.ask),
            ask_qty=int(book.ask_qty),
            mid=float(mid),
            spread=float(spread),
            fair=float(fair),
            gap=float(fair - mid),
            gap_abs=float(abs(fair - mid)),
            bp_dislocation=float(rate_snapshot.effective_expected_rate_bp - self.state.anchor_yield_bp),
            fair_change=float(fair_change),
            entry_threshold=max(8.0, 1.25 * spread),
            exit_threshold=max(6.0, 0.75 * spread),
        )


class RatesTradingEngine:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk

    def current_pair_direction(self) -> Optional[str]:
        pos_hike = self.client.get_position(self.cfg.fed_hike)
        pos_cut = self.client.get_position(self.cfg.fed_cut)
        if pos_hike > 0 and pos_cut < 0:
            return "hawkish"
        if pos_hike < 0 and pos_cut > 0:
            return "dovish"
        return None

    def current_pair_abs(self) -> int:
        pos_hike = abs(self.client.get_position(self.cfg.fed_hike))
        pos_cut = abs(self.client.get_position(self.cfg.fed_cut))
        return max(pos_hike, pos_cut)

    async def flatten_all_rates(self, reason: str) -> bool:
        acted = False
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None:
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.rate_extreme_size)
            price = int(book.bid if side == Side.SELL else book.ask)
            if self.orders.has_live_order(symbol=symbol):
                continue
            placed = await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="exit",
                engine="rates",
                reason=reason,
                thesis=self.state.rates_regime_direction,
                signal_strength=float(abs(pos)),
            )
            acted = acted or placed
        if acted:
            self.state.rates_blocked_direction = self.state.rates_regime_direction
            self.state.rates_blocked_until = time.time() + self.cfg.rate_reentry_block_secs
            self.state.clear_rates_regime()
            LOGGER.info("Rates exit trigger: %s", reason)
        return acted

    async def handle_missing_signal_exit(self) -> bool:
        if any(self.client.get_position(symbol) != 0 for symbol in self.cfg.rate_symbols):
            return await self.flatten_all_rates("signal_lost")
        return False

    async def maybe_exit(self, snapshot: RateSnapshot) -> bool:
        direction = self.current_pair_direction()
        if direction is None:
            if self.client.get_position(self.cfg.fed_hold) != 0:
                return await self.flatten_all_rates("signal_lost")
            return False

        edge_bp = abs(snapshot.bias_bp)
        compressed = edge_bp <= max(self.cfg.rate_exit_edge_bp, self.state.rates_last_entry_edge * 0.30)
        opposite = (direction == "hawkish" and snapshot.bias_bp <= -self.cfg.rate_exit_edge_bp) or (
            direction == "dovish" and snapshot.bias_bp >= self.cfg.rate_exit_edge_bp
        )
        stale = edge_bp < self.cfg.rate_exit_edge_bp

        if stale or opposite or compressed:
            return await self.flatten_all_rates("bias_decay" if stale or compressed else "macro_reversal")
        return False

    def compute_entry_decision(self, snapshot: RateSnapshot) -> Optional[RatesEntryDecision]:
        if not snapshot.fresh_macro_event:
            return None
        if snapshot.bias_bp >= self.cfg.rate_entry_edge_bp:
            direction = "hawkish"
        elif snapshot.bias_bp <= -self.cfg.rate_entry_edge_bp:
            direction = "dovish"
        else:
            return None

        edge_bp = abs(snapshot.bias_bp)
        if snapshot.macro_source == "cpi_print" and edge_bp >= 5.0:
            target_size = self.cfg.rate_extreme_size
        elif edge_bp >= 3.5 or snapshot.urgent:
            target_size = self.cfg.rate_strong_size
        else:
            target_size = self.cfg.rate_normal_size
        return RatesEntryDecision(direction=direction, edge_bp=edge_bp, target_size=target_size)

    async def maybe_enter(self, snapshot: RateSnapshot) -> bool:
        decision = self.compute_entry_decision(snapshot)
        if decision is None:
            return False

        now = time.time()
        if self.state.rates_blocked_direction == decision.direction and now < self.state.rates_blocked_until:
            return False

        current_direction = self.current_pair_direction()
        if current_direction is not None and current_direction != decision.direction:
            return False

        if current_direction is not None:
            if self.state.rates_entry_stage >= 2:
                return False
            if decision.edge_bp < self.state.rates_last_entry_edge + self.cfg.rate_add_edge_step_bp:
                return False
            if now - self.state.rates_last_add_ts < self.cfg.c_add_cooldown_secs:
                return False

        if decision.direction == "hawkish":
            buy_symbol = self.cfg.fed_hike
            sell_symbol = self.cfg.fed_cut
        else:
            buy_symbol = self.cfg.fed_cut
            sell_symbol = self.cfg.fed_hike

        buy_book = self.client.top(buy_symbol)
        sell_book = self.client.top(sell_symbol)
        if buy_book.ask is None or sell_book.bid is None:
            return False

        current_leg_abs = self.current_pair_abs()
        if current_leg_abs >= decision.target_size:
            return False
        desired_leg = decision.target_size - current_leg_abs

        buy_qty = self.risk.clip_rate_qty(buy_symbol, Side.BUY, desired_leg, decision.target_size)
        sell_qty = self.risk.clip_rate_qty(sell_symbol, Side.SELL, desired_leg, decision.target_size)
        qty = min(buy_qty, sell_qty)
        if qty <= 0:
            return False

        if self.orders.has_live_order(symbol=buy_symbol) or self.orders.has_live_order(symbol=sell_symbol):
            return False

        placed_buy = await self.orders.place_tracked_order(
            symbol=buy_symbol,
            qty=qty,
            side=Side.BUY,
            price=int(buy_book.ask),
            role="entry",
            engine="rates",
            reason="rates_entry",
            thesis=decision.direction,
            signal_strength=decision.edge_bp,
        )
        placed_sell = await self.orders.place_tracked_order(
            symbol=sell_symbol,
            qty=qty,
            side=Side.SELL,
            price=int(sell_book.bid),
            role="entry",
            engine="rates",
            reason="rates_entry",
            thesis=decision.direction,
            signal_strength=decision.edge_bp,
        )
        placed = placed_buy or placed_sell
        if placed:
            self.state.rates_regime_direction = decision.direction
            self.state.rates_entry_stage = 1 if current_direction is None else self.state.rates_entry_stage + 1
            self.state.rates_last_entry_edge = decision.edge_bp
            self.state.rates_last_add_ts = now
            LOGGER.info(
                "Rates entry: direction=%s qty=%s edge_bp=%.2f target=%s",
                decision.direction,
                qty,
                decision.edge_bp,
                decision.target_size,
            )
        return placed


class CTradingEngine:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        fair_engine: CFairValueEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.fair_engine = fair_engine

    def initial_shock_window_remaining(self, now: Optional[float] = None) -> float:
        if self.state.last_c_initial_baseline_ts <= 0.0:
            return 0.0
        if now is None:
            now = time.time()
        elapsed = max(0.0, now - self.state.last_c_initial_baseline_ts)
        return max(0.0, self.cfg.c_initial_shock_window_secs - elapsed)

    def initial_shock_window_active(self, now: Optional[float] = None) -> bool:
        return self.initial_shock_window_remaining(now) > 0.0

    def compute_entry_decision(self, signal: CSignal, rate_snapshot: RateSnapshot) -> Optional[CEntryDecision]:
        if not self.state.have_real_eps_c:
            return None

        now = time.time()
        earnings = self.fair_engine.earnings_context()
        fresh_earnings = earnings.tier >= 1 and not earnings.is_initial and earnings.age <= self.cfg.c_earnings_fresh_secs
        initial_shock_window = self.initial_shock_window_active(now)

        rate_tailwind = (
            (signal.gap > 0 and signal.bp_dislocation <= -self.cfg.lead_lag_bp_trigger)
            or (signal.gap < 0 and signal.bp_dislocation >= self.cfg.lead_lag_bp_trigger)
        )

        if fresh_earnings and earnings.side is not None:
            if earnings.side == Side.BUY and signal.gap <= 0:
                return None
            if earnings.side == Side.SELL and signal.gap >= 0:
                return None

            if earnings.tier == 1:
                threshold = max(self.cfg.lead_lag_entry_ticks - 2.0, 1.25 * signal.spread, 14.0)
                if signal.gap_abs >= threshold:
                    return CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=self.cfg.c_tier1_initial_size,
                        add_size=self.cfg.c_tier1_add_size,
                        thesis_cap=self.cfg.c_tier1_cap,
                    )
            elif earnings.tier == 2:
                initial_size = self.cfg.c_tier2_initial_size
                add_size = self.cfg.c_tier2_add_size
                cap = self.cfg.c_tier2_cap
                if rate_tailwind:
                    initial_size = self.cfg.c_tier3_initial_size
                    add_size = self.cfg.c_tier3_add_size
                    cap = self.cfg.c_tier3_cap
                threshold = max(10.0, signal.entry_threshold)
                if signal.gap_abs >= threshold:
                    return CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=initial_size,
                        add_size=add_size,
                        thesis_cap=cap,
                    )
            else:
                threshold = max(8.0, signal.entry_threshold)
                if signal.gap_abs >= threshold:
                    return CEntryDecision(
                        side=earnings.side,
                        thesis="earnings",
                        edge_ticks=signal.gap_abs,
                        initial_size=self.cfg.c_tier3_initial_size,
                        add_size=self.cfg.c_tier3_add_size,
                        thesis_cap=self.cfg.c_tier3_cap,
                    )

        if not fresh_earnings and initial_shock_window:
            threshold = max(self.cfg.c_initial_shock_gap_ticks, 1.25 * signal.spread)
            if signal.gap_abs >= threshold and (
                not self.state.c_initial_shock_consumed or self.state.c_regime_thesis == "initial_earnings_shock"
            ):
                side = Side.BUY if signal.gap > 0 else Side.SELL
                return CEntryDecision(
                    side=side,
                    thesis="initial_earnings_shock",
                    edge_ticks=signal.gap_abs,
                    initial_size=self.cfg.c_initial_shock_initial_size,
                    add_size=self.cfg.c_initial_shock_add_size,
                    thesis_cap=self.cfg.c_initial_shock_cap,
                )
            return None

        if (
            not fresh_earnings
            and abs(signal.bp_dislocation) >= self.cfg.lead_lag_bp_trigger
            and signal.gap_abs >= max(self.cfg.lead_lag_entry_ticks, 1.5 * signal.spread)
        ):
            side = Side.BUY if signal.gap > 0 else Side.SELL
            return CEntryDecision(
                side=side,
                thesis="rates_lead_lag",
                edge_ticks=signal.gap_abs,
                initial_size=self.cfg.c_leadlag_initial_size,
                add_size=self.cfg.c_leadlag_add_size,
                thesis_cap=self.cfg.c_leadlag_cap,
            )

        return None

    async def handle_missing_signal_exit(self) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False

        side = Side.SELL if pos > 0 else Side.BUY
        price = int(book.bid if side == Side.SELL else book.ask)
        qty = abs(pos)
        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=side,
            price=price,
            role="exit",
            engine="c",
            reason="signal_lost",
            thesis=self.state.c_regime_thesis,
            signal_strength=float(abs(pos)),
        )
        if placed:
            self.state.c_blocked_side = self.state.c_regime_side
            self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
            self.state.clear_c_regime()
            LOGGER.info("C exit trigger: signal_lost")
        return placed

    async def maybe_exit(self, signal: CSignal, rate_snapshot: RateSnapshot) -> bool:
        pos = self.client.get_position(self.cfg.symbol_c)
        if pos == 0:
            self.state.clear_c_regime()
            return False

        earnings = self.fair_engine.earnings_context()
        side = None
        reason = None
        qty = 0
        long_pos = pos > 0
        favorable_gap = signal.gap if long_pos else -signal.gap
        adverse_gap = -signal.gap if long_pos else signal.gap
        entry_edge = max(self.state.c_last_entry_edge, signal.entry_threshold)
        compression_band = max(self.cfg.c_compression_min_ticks, self.cfg.c_compression_frac * entry_edge)
        initial_shock_active = self.initial_shock_window_active()

        earnings_opposite = (
            earnings.tier >= 1
            and earnings.side is not None
            and earnings.side != (Side.BUY if long_pos else Side.SELL)
            and earnings.age <= self.cfg.c_earnings_fresh_secs
        )

        if earnings_opposite and earnings.tier >= 2:
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "earnings_reversal"
        elif adverse_gap >= max(self.cfg.c_hard_flip_min_ticks, signal.exit_threshold):
            side = Side.SELL if long_pos else Side.BUY
            qty = abs(pos)
            reason = "gap_flip"
        elif self.state.c_regime_thesis == "initial_earnings_shock":
            compress_threshold = max(self.cfg.c_initial_shock_compress_ticks, signal.exit_threshold)
            if not initial_shock_active:
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "initial_shock_unwind"
            elif favorable_gap <= compress_threshold:
                side = Side.SELL if long_pos else Side.BUY
                if favorable_gap <= self.cfg.c_initial_shock_flatten_ticks:
                    qty = abs(pos)
                    reason = "initial_shock_compress_full"
                else:
                    qty = min(abs(pos), max(10, abs(pos) // 2))
                    reason = "initial_shock_compress_trim"
        elif self.state.c_regime_thesis in {"post_baseline_gap", "rates_lead_lag"}:
            rate_reversal = (
                (long_pos and rate_snapshot.delta_effective_bp >= self.cfg.c_rate_reversal_bp)
                or ((not long_pos) and rate_snapshot.delta_effective_bp <= -self.cfg.c_rate_reversal_bp)
            )
            if rate_reversal and favorable_gap <= max(entry_edge * 0.75, compression_band):
                side = Side.SELL if long_pos else Side.BUY
                qty = abs(pos)
                reason = "rate_reversal"

        if side is None and favorable_gap <= compression_band:
            side = Side.SELL if long_pos else Side.BUY
            if favorable_gap <= 0:
                qty = abs(pos)
                reason = "gap_compress_full"
            else:
                qty = min(abs(pos), max(10, abs(pos) // 2))
                reason = "gap_compress_trim"

        if side is None or qty <= 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False

        await self.orders.cancel_live_orders(symbol=self.cfg.symbol_c, role="entry", engine="c")

        price = int(book.bid if side == Side.SELL else book.ask)
        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=side,
            price=price,
            role="exit",
            engine="c",
            reason=reason,
            thesis=self.state.c_regime_thesis,
            signal_strength=signal.gap_abs,
        )
        if placed:
            if qty >= abs(pos):
                self.state.c_blocked_side = self.state.c_regime_side
                self.state.c_blocked_until = time.time() + self.cfg.c_reentry_block_secs
                self.state.clear_c_regime()
            LOGGER.info(
                "C exit trigger: %s pos=%s gap=%.2f delta_effective_bp=%.2f",
                reason,
                pos,
                signal.gap,
                rate_snapshot.delta_effective_bp,
            )
        return placed

    async def maybe_enter(self, decision: Optional[CEntryDecision], signal: Optional[CSignal]) -> bool:
        if decision is None or signal is None:
            return False

        now = time.time()
        pos = self.client.get_position(self.cfg.symbol_c)
        same_direction = (pos > 0 and decision.side == Side.BUY) or (pos < 0 and decision.side == Side.SELL)

        if self.state.c_blocked_side == decision.side and now < self.state.c_blocked_until:
            return False
        if pos == 0 and now < self.state.c_flat_entry_cooldown_until:
            return False
        if self.orders.has_live_order(symbol=self.cfg.symbol_c):
            return False
        if pos != 0 and not same_direction:
            return False

        if same_direction:
            if self.state.c_regime_thesis != decision.thesis:
                return False
            if self.state.c_entry_stage >= 2:
                return False
            if now - self.state.c_last_add_ts < self.cfg.c_add_cooldown_secs:
                return False
            if decision.edge_ticks < self.state.c_last_entry_edge + self.cfg.c_add_edge_step_ticks:
                return False
            raw_qty = decision.add_size
        else:
            raw_qty = decision.initial_size

        qty = self.risk.clip_c_qty(decision.side, raw_qty, decision.thesis_cap)
        if qty <= 0:
            return False

        book = self.client.top(self.cfg.symbol_c)
        if book.bid is None or book.ask is None:
            return False
        price = int(book.ask if decision.side == Side.BUY else book.bid)

        placed = await self.orders.place_tracked_order(
            symbol=self.cfg.symbol_c,
            qty=qty,
            side=decision.side,
            price=price,
            role="entry",
            engine="c",
            reason="add_entry" if same_direction else "initial_entry",
            thesis=decision.thesis,
            signal_strength=decision.edge_ticks,
        )
        if placed:
            self.state.c_regime_side = decision.side
            self.state.c_regime_thesis = decision.thesis
            self.state.c_entry_stage = 1 if not same_direction else self.state.c_entry_stage + 1
            self.state.c_last_entry_edge = decision.edge_ticks
            self.state.c_last_add_ts = now
            if decision.thesis == "initial_earnings_shock":
                self.state.c_initial_shock_consumed = True
            if pos == 0:
                self.state.c_flat_entry_cooldown_until = now + self.cfg.c_flat_entry_cooldown_secs
            LOGGER.info(
                "C entry: thesis=%s side=%s qty=%s gap=%.2f fair=%.2f mid=%.2f",
                decision.thesis,
                decision.side.name,
                qty,
                signal.gap,
                signal.fair,
                signal.mid,
            )
        return placed


class Coordinator:
    def __init__(
        self,
        client: "MyXchangeClient",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        rates_signals: RatesSignalEngine,
        c_fair: CFairValueEngine,
        rates_trading: RatesTradingEngine,
        c_trading: CTradingEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.rates_signals = rates_signals
        self.c_fair = c_fair
        self.rates_trading = rates_trading
        self.c_trading = c_trading

    def sync_regimes_to_positions(self) -> None:
        pos_c = self.client.get_position(self.cfg.symbol_c)
        if pos_c == 0 and not self.orders.has_live_order(symbol=self.cfg.symbol_c):
            self.state.clear_c_regime()
        if all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                self.state.clear_rates_regime()

    async def evaluate(self) -> None:
        self.client.refresh_all_books()
        await self.orders.cancel_stale_orders()
        self.sync_regimes_to_positions()

        if not self.state.startup_flatten_complete:
            await self.risk.startup_flatten_step()
            self.record_decision_snapshot("startup_flatten")
            self.log_status("startup_flatten")
            return

        self.risk.arm_session_baseline_if_ready()

        rate_snapshot = self.rates_signals.snapshot()
        self.c_fair.initialize_anchor(rate_snapshot)
        self.c_fair.maybe_refresh_anchor_after_first_real_eps(rate_snapshot)
        c_signal = self.c_fair.snapshot(rate_snapshot)
        self.c_fair.maybe_reanchor(c_signal, rate_snapshot)

        if rate_snapshot is None:
            rates_acted = await self.rates_trading.handle_missing_signal_exit()
            c_acted = await self.c_trading.handle_missing_signal_exit()
            self.record_decision_snapshot("signal_not_ready")
            self.log_status("signal_not_ready" if rates_acted or c_acted else "signal_not_ready")
            return

        if c_signal is None:
            c_acted = await self.c_trading.handle_missing_signal_exit()
            self.record_decision_snapshot("waiting_real_eps", rate_snapshot=rate_snapshot)
            self.log_status("waiting_real_eps" if not self.state.have_real_eps_c else "signal_not_ready")
            self.state.last_market_expected_rate_bp = rate_snapshot.market_expected_rate_bp
            self.state.last_effective_expected_rate_bp = rate_snapshot.effective_expected_rate_bp
            return

        c_entry_decision = self.c_trading.compute_entry_decision(c_signal, rate_snapshot)

        if await self.rates_trading.maybe_exit(rate_snapshot):
            self.record_decision_snapshot("rates_exit", rate_snapshot=rate_snapshot, c_signal=c_signal, c_entry_decision=c_entry_decision)
            self.update_last_signals(rate_snapshot, c_signal)
            return
        if await self.c_trading.maybe_exit(c_signal, rate_snapshot):
            self.record_decision_snapshot("c_exit", rate_snapshot=rate_snapshot, c_signal=c_signal, c_entry_decision=c_entry_decision)
            self.update_last_signals(rate_snapshot, c_signal)
            return

        rates_entered = await self.rates_trading.maybe_enter(rate_snapshot)
        if rates_entered:
            self.record_decision_snapshot("rates_entry", rate_snapshot=rate_snapshot, c_signal=c_signal, c_entry_decision=c_entry_decision)

        c_is_high_priority = c_entry_decision is not None and c_entry_decision.thesis in {
            "earnings",
            "initial_earnings_shock",
            "post_baseline_gap",
        }
        if (not rates_entered) or c_is_high_priority:
            if await self.c_trading.maybe_enter(c_entry_decision, c_signal):
                self.record_decision_snapshot("c_entry", rate_snapshot=rate_snapshot, c_signal=c_signal, c_entry_decision=c_entry_decision)
                self.update_last_signals(rate_snapshot, c_signal)
                return

        self.record_decision_snapshot("no_trade", rate_snapshot=rate_snapshot, c_signal=c_signal, c_entry_decision=c_entry_decision)
        self.log_status("no_trade", rate_snapshot, c_signal)
        self.update_last_signals(rate_snapshot, c_signal)

    def update_last_signals(self, rate_snapshot: RateSnapshot, c_signal: CSignal) -> None:
        self.state.last_market_expected_rate_bp = rate_snapshot.market_expected_rate_bp
        self.state.last_effective_expected_rate_bp = rate_snapshot.effective_expected_rate_bp
        self.state.last_fair_c = c_signal.fair

    def record_decision_snapshot(
        self,
        reason: str,
        *,
        rate_snapshot: Optional[RateSnapshot] = None,
        c_signal: Optional[CSignal] = None,
        c_entry_decision: Optional[CEntryDecision] = None,
    ) -> None:
        if getattr(self.client, "data_logger", None) is None:
            return
        self.client.data_logger.log_decision_snapshot(
            self.client,
            reason,
            rate_snapshot=rate_snapshot,
            c_signal=c_signal,
            c_entry_decision=c_entry_decision,
        )

    def log_status(
        self,
        reason: str,
        rate_snapshot: Optional[RateSnapshot] = None,
        c_signal: Optional[CSignal] = None,
    ) -> None:
        now = time.time()
        if now - self.state.last_status_log_ts < self.cfg.status_log_interval_secs:
            return
        self.state.last_status_log_ts = now

        cash, mtm = self.client.cash_and_total_mtm()
        session_cash, session_mtm = self.client.session_pnl_snapshot(cash, mtm)

        if rate_snapshot is None or c_signal is None:
            LOGGER.info(
                "Idle: %s missing=%s pos_C=%s pos_rates=%s/%s/%s eps=%.4f cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
                reason,
                self.client.missing_prereqs(),
                self.client.get_position(self.cfg.symbol_c),
                self.client.get_position(self.cfg.fed_hike),
                self.client.get_position(self.cfg.fed_hold),
                self.client.get_position(self.cfg.fed_cut),
                self.state.current_eps_c,
                cash,
                mtm,
                session_cash,
                session_mtm,
            )
            return

        earnings = self.c_fair.earnings_context()
        LOGGER.info(
            "Idle: %s exp_bp=%.2f market_bp=%.2f bias=%.2f eps=%.4f eps_delta=%+.4f eps_tier=%s fair=%.2f C=%s/%s gap=%.2f bp_disloc=%.2f pos_C=%s rates=%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            reason,
            rate_snapshot.effective_expected_rate_bp,
            rate_snapshot.market_expected_rate_bp,
            rate_snapshot.bias_bp,
            self.state.current_eps_c,
            earnings.delta,
            earnings.tier,
            c_signal.fair,
            c_signal.bid,
            c_signal.ask,
            c_signal.gap,
            c_signal.bp_dislocation,
            self.client.get_position(self.cfg.symbol_c),
            self.client.get_position(self.cfg.fed_hike),
            self.client.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )


class MyXchangeClient(XChangeClient):
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        log_root: Optional[Path] = None,
        enable_run_logger: bool = True,
    ):
        self.cfg = BotConfig()
        super().__init__(host, username, password, silent=True, symbols=list(self.cfg.tracked_symbols))

        self.state = MarketState(current_eps_c=self.cfg.default_eps_c)
        self._decision_lock = asyncio.Lock()
        self.data_logger = RunDataLogger(self.cfg, host=host, username=username, log_root=log_root) if enable_run_logger else None

        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)
        self.rates_signal_engine = RatesSignalEngine(self, self.cfg, self.state)
        self.c_fair_engine = CFairValueEngine(self, self.cfg, self.state, self.rates_signal_engine)
        self.rates_trading_engine = RatesTradingEngine(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
        )
        self.c_trading_engine = CTradingEngine(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.c_fair_engine,
        )
        self.coordinator = Coordinator(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.rates_signal_engine,
            self.c_fair_engine,
            self.rates_trading_engine,
            self.c_trading_engine,
        )

    @staticmethod
    def clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def refresh_book(self, symbol: str) -> TopOfBook:
        book = self.order_books.get(symbol)
        bids = []
        asks = []
        if book is not None:
            bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
            asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]

        best_bid = max(bids, key=lambda level: level[0]) if bids else None
        best_ask = min(asks, key=lambda level: level[0]) if asks else None
        snapshot = TopOfBook(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.state.books[symbol] = snapshot
        return snapshot

    def refresh_all_books(self) -> None:
        for symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)

    def top(self, symbol: str) -> TopOfBook:
        if symbol not in self.state.books:
            return self.refresh_book(symbol)
        return self.state.books[symbol]

    def mid(self, symbol: str) -> Optional[float]:
        return self.top(symbol).mid

    def mark_price(self, symbol: str) -> Optional[float]:
        return self.mid(symbol)

    def cash_and_total_mtm(self) -> tuple[float, float]:
        cash = float(self.positions.get("cash", 0))
        mtm = cash
        for symbol in self.cfg.tracked_symbols:
            pos = self.get_position(symbol)
            if pos == 0:
                continue
            mark = self.mark_price(symbol)
            if mark is not None:
                mtm += pos * mark
        return cash, mtm

    def session_pnl_snapshot(self, cash: float, mtm: float) -> tuple[float, float]:
        if self.state.session_start_cash is None or self.state.session_start_mtm is None:
            return 0.0, 0.0
        return cash - self.state.session_start_cash, mtm - self.state.session_start_mtm

    def missing_prereqs(self) -> list[str]:
        missing: list[str] = []
        for symbol in self.cfg.tracked_symbols:
            book = self.top(symbol)
            if book.bid is None or book.ask is None:
                missing.append(f"{symbol}_book")
        if self.state.anchor_price is None or self.state.anchor_eps is None or self.state.anchor_yield_bp is None:
            missing.append("anchor")
        return missing

    async def evaluate(self) -> None:
        async with self._decision_lock:
            await self.coordinator.evaluate()

    async def bot_handle_cancel_response(
        self,
        order_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        tracked = self.order_manager.sync_cancel_response(order_id, success)
        self.state.pending_cancels.discard(str(order_id))
        if success:
            LOGGER.info("Cancel acknowledged: order_id=%s", order_id)
            if self.data_logger is not None:
                self.data_logger.log_order_event(self, "cancel_ack", tracked=tracked, order_id=str(order_id))
        else:
            LOGGER.warning("Cancel failed: order_id=%s error=%s", order_id, error)
            if tracked is not None:
                self.state.live_orders[str(order_id)] = tracked
            if self.data_logger is not None:
                self.data_logger.log_order_event(self, "cancel_failed", tracked=tracked, order_id=str(order_id), error=error or "")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        tracked = self.order_manager.sync_fill(order_id)
        pos_c = self.get_position(self.cfg.symbol_c)
        if pos_c == 0 and not self.order_manager.has_live_order(symbol=self.cfg.symbol_c):
            self.state.clear_c_regime()

        if self.get_position(self.cfg.fed_hike) == 0 and self.get_position(self.cfg.fed_cut) == 0 and self.get_position(self.cfg.fed_hold) == 0:
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                self.state.clear_rates_regime()

        cash, mtm = self.cash_and_total_mtm()
        session_cash, session_mtm = self.session_pnl_snapshot(cash, mtm)
        LOGGER.info(
            "Fill: order_id=%s symbol=%s qty=%s price=%s pos_C=%s pos_rates=%s/%s/%s cash=%.2f mtm=%.2f session_cash=%.2f session_mtm=%.2f",
            order_id,
            tracked.symbol if tracked is not None else "unknown",
            qty,
            price,
            self.get_position(self.cfg.symbol_c),
            self.get_position(self.cfg.fed_hike),
            self.get_position(self.cfg.fed_hold),
            self.get_position(self.cfg.fed_cut),
            cash,
            mtm,
            session_cash,
            session_mtm,
        )
        if self.data_logger is not None:
            self.data_logger.log_order_event(self, "filled", tracked=tracked, order_id=str(order_id), qty=qty, price=price)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.state.pending_cancels.discard(str(order_id))
        reason_text = (reason or "").lower()
        limit_rejection = "exceeds limits" in reason_text or "limit" in reason_text
        if tracked is not None:
            if tracked.engine == "c" and tracked.role == "entry":
                self.state.c_blocked_side = tracked.side
                cooldown = self.cfg.c_reentry_block_secs * (2.0 if limit_rejection else 1.0)
                self.state.c_blocked_until = time.time() + cooldown
            if tracked.engine == "rates" and tracked.role == "entry":
                self.state.rates_blocked_direction = tracked.thesis
                cooldown = self.cfg.rate_reentry_block_secs * (2.0 if limit_rejection else 1.0)
                self.state.rates_blocked_until = time.time() + cooldown
        LOGGER.warning("Order rejected: order_id=%s reason=%s", order_id, reason)
        if self.data_logger is not None:
            self.data_logger.log_order_event(self, "rejected", tracked=tracked, order_id=str(order_id), error=reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.tracked_symbols:
            self.state.last_trade_price[symbol] = int(price)
            if self.data_logger is not None:
                self.data_logger.log_trade_event(self, symbol, int(price), int(qty), "trade_msg")
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.tracked_symbols:
            self.refresh_book(symbol)
            if self.data_logger is not None:
                self.data_logger.log_book_update(self, symbol, "book_update")
            await self.evaluate()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        LOGGER.info("Ignoring swap response: swap=%s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        tick = news_release.get("tick")
        kept_in_raw_news_events = kind in {"structured", "unstructured"}
        note = ""
        computed_bias_bp: Optional[float] = None

        rate_snapshot = self.rates_signal_engine.snapshot()

        if kind == "structured":
            subtype = new_data.get("structured_subtype")

            if subtype == "earnings":
                asset = str(new_data.get("asset", "")).upper()
                value = float(new_data["value"])
                if asset == self.cfg.symbol_c:
                    previous_eps = self.state.current_eps_c
                    had_real_eps = self.state.have_real_eps_c
                    self.state.current_eps_c = value
                    self.state.have_real_eps_c = True
                    self.state.last_c_earnings_delta = 0.0 if not had_real_eps else value - previous_eps
                    self.state.last_c_earnings_ts = time.time()
                    self.state.last_c_earnings_is_initial = not had_real_eps

                    earnings_delta = abs(self.state.last_c_earnings_delta)
                    if not self.state.last_c_earnings_is_initial:
                        if earnings_delta >= self.cfg.c_earnings_medium_delta:
                            self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs + 0.75)
                        elif earnings_delta >= self.cfg.c_earnings_small_delta:
                            self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs)
                        elif earnings_delta >= self.cfg.c_earnings_ignore_delta:
                            self.rates_signal_engine.mark_news_urgent(1.50)
                    else:
                        self.rates_signal_engine.mark_news_urgent(self.cfg.c_earnings_hold_secs)

                    self.c_fair_engine.maybe_refresh_anchor_after_first_real_eps(rate_snapshot)
                    LOGGER.info(
                        "C earnings update at tick %s: %.4f -> %.4f delta=%+.4f initial=%s",
                        tick,
                        previous_eps,
                        value,
                        self.state.last_c_earnings_delta,
                        self.state.last_c_earnings_is_initial,
                    )
                    note = "c_earnings"
                else:
                    LOGGER.info("Ignoring %s earnings in C-only mode at tick %s: value=%.4f", asset, tick, value)
                    note = "ignored_non_c_earnings"

            elif subtype == "cpi_print":
                forecast = float(new_data["forecast"])
                actual = float(new_data["actual"])
                surprise = actual - forecast
                bias_bp = self.clip(
                    surprise * self.cfg.cpi_to_rate_bp,
                    -self.cfg.max_temp_rate_bias_bp,
                    self.cfg.max_temp_rate_bias_bp,
                )
                computed_bias_bp = bias_bp
                if abs(bias_bp) >= 0.25:
                    self.rates_signal_engine.apply_temp_rate_bias(bias_bp, self.cfg.cpi_bias_ttl_secs, "cpi_print")
                    LOGGER.info(
                        "CPI surprise at tick %s: actual=%.6f forecast=%.6f bias_bp=%.2f",
                        tick,
                        actual,
                        forecast,
                        bias_bp,
                    )
                    note = "cpi_applied"
                else:
                    note = "cpi_ignored_small_bias"

        elif kind == "unstructured":
            content = str(new_data.get("content", ""))
            bias_bp = self.rates_signal_engine.headline_rate_bias_bp(content)
            computed_bias_bp = bias_bp
            if abs(bias_bp) >= 0.25:
                self.rates_signal_engine.apply_temp_rate_bias(bias_bp, self.cfg.headline_bias_ttl_secs, "headline")
                LOGGER.info("Fed headline at tick %s: bias_bp=%.2f content=%s", tick, bias_bp, content)
                note = "headline_applied"
            else:
                note = "headline_ignored_small_bias"

        if self.data_logger is not None:
            self.data_logger.log_news_callback(
                self,
                news_release,
                kept_in_raw_news_events=kept_in_raw_news_events,
                computed_bias_bp=computed_bias_bp,
                note=note,
            )

        await self.evaluate()

    async def trade(self):
        await asyncio.sleep(2.0)
        while True:
            try:
                await self.evaluate()
            except Exception as exc:
                LOGGER.exception("trade loop error: %s", exc)
                if self.data_logger is not None:
                    self.data_logger.log_strategy_event(self, "trade_loop_error", error=str(exc))
            await asyncio.sleep(self.cfg.loop_sleep_secs)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _default_replay_root() -> Path:
    here = Path(__file__).resolve()
    candidates = []
    for parent in (here, *here.parents):
        candidates.append(parent / "data_scraping" / "data")
        candidates.append(parent / "uchicago-2026-competition" / "data_scraping" / "data")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return here.parents[2] / "data_scraping" / "data"


def summarize_saved_replays(replay_root: Optional[Path] = None) -> None:
    root = replay_root or _default_replay_root()
    runs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("market_research_"))
    if not runs:
        print(f"No market_research_* runs found under {root}")
        return

    event_counts = Counter()
    agg: Dict[str, dict[str, float]] = {}

    def record_event(label: str, move_1s: float, move_5s: Optional[float], spread: Optional[float]) -> None:
        event_counts[label] += 1
        bucket = agg.setdefault(label, {"sum_1s": 0.0, "sum_abs_1s": 0.0, "sum_5s": 0.0, "count_5s": 0.0, "sum_spread": 0.0})
        bucket["sum_1s"] += move_1s
        bucket["sum_abs_1s"] += abs(move_1s)
        bucket["sum_spread"] += 0.0 if spread is None else spread
        if move_5s is not None:
            bucket["sum_5s"] += move_5s
            bucket["count_5s"] += 1.0

    for run in runs:
        book_rows = _load_csv_rows(run / "raw_book_events_C.csv")
        if not book_rows:
            continue
        books = sorted(
            (
                _as_int(row.get("monotonic_ns")),
                _as_float(row.get("mid_px")),
                _as_float(row.get("spread")),
            )
            for row in book_rows
        )
        books = [(ns, mid, spread) for ns, mid, spread in books if ns is not None and mid is not None]
        if not books:
            continue

        def nearest_before(target_ns: int) -> Optional[tuple[int, float, Optional[float]]]:
            out = None
            for item in books:
                if item[0] <= target_ns:
                    out = item
                else:
                    break
            return out

        def nearest_after(target_ns: int) -> Optional[tuple[int, float, Optional[float]]]:
            for item in books:
                if item[0] >= target_ns:
                    return item
            return None

        news_rows = _load_csv_rows(run / "raw_news_events.csv")
        all_news_rows = _load_csv_rows(run / "raw_all_news_callbacks.csv")

        combined_events: list[tuple[str, int]] = []
        for row in news_rows:
            ns = _as_int(row.get("monotonic_ns"))
            if ns is None:
                continue
            kind = row.get("kind", "")
            subtype = row.get("structured_subtype", "")
            asset = row.get("earnings_asset") or row.get("symbol") or ""
            if kind == "structured" and subtype == "earnings" and asset == "C":
                combined_events.append(("C_earnings", ns))
            elif kind == "structured" and subtype == "cpi_print":
                combined_events.append(("CPI", ns))
            elif kind == "unstructured":
                combined_events.append(("Fed_headline", ns))

        for row in all_news_rows:
            ns = _as_int(row.get("monotonic_ns"))
            if ns is None:
                continue
            kept = str(row.get("kept_in_raw_news_events", "")).lower()
            if kept in {"true", "1", "yes"}:
                continue
            kind = row.get("kind", "")
            subtype = row.get("structured_subtype", "")
            if kind == "structured" and subtype == "cpi_print":
                combined_events.append(("CPI", ns))
            elif kind == "unstructured":
                combined_events.append(("Fed_headline", ns))

        for label, ns in combined_events:
            before = nearest_before(ns)
            after_1s = nearest_after(ns + 1_000_000_000)
            after_5s = nearest_after(ns + 5_000_000_000)
            if before is None or after_1s is None:
                continue
            move_1s = after_1s[1] - before[1]
            move_5s = None if after_5s is None else after_5s[1] - before[1]
            record_event(label, move_1s, move_5s, before[2])

    print(f"Replay root: {root}")
    print(f"Runs scanned: {len(runs)}")
    for label in sorted(event_counts):
        stats = agg[label]
        n = max(1, event_counts[label])
        avg_1s = stats["sum_1s"] / n
        avg_abs_1s = stats["sum_abs_1s"] / n
        avg_spread = stats["sum_spread"] / n
        avg_5s = stats["sum_5s"] / stats["count_5s"] if stats["count_5s"] else 0.0
        print(
            f"{label}: n={event_counts[label]} avg_move_1s={avg_1s:.3f} "
            f"avg_abs_move_1s={avg_abs_1s:.3f} avg_move_5s={avg_5s:.3f} avg_spread={avg_spread:.3f}"
        )


def _resolve_events_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "events.jsonl"
    if candidate.name != "events.jsonl" or not candidate.exists():
        raise FileNotFoundError(f"Could not find events.jsonl for reference run: {path}")
    return candidate


def _iter_event_rows(events_path: Path):
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return step * round(value / step)


def _dedupe_sorted(values: list[float], *, digits: int = 4) -> list[float]:
    seen: set[float] = set()
    ordered: list[float] = []
    for value in sorted(values):
        rounded = round(float(value), digits)
        if rounded in seen:
            continue
        seen.add(rounded)
        ordered.append(rounded)
    return ordered


def _collect_rates_pair_episodes(events_path: Path) -> list[dict[str, float | str]]:
    fills = [
        row
        for row in _iter_event_rows(events_path)
        if row.get("event_type") == "order_event"
        and row.get("order_event") == "filled"
        and row.get("engine") == "rates"
        and row.get("symbol") in {"R_HIKE", "R_CUT"}
    ]

    pos = {"R_HIKE": 0, "R_CUT": 0}
    avg = {"R_HIKE": 0.0, "R_CUT": 0.0}
    realized = 0.0
    current_thesis: Optional[str] = None
    start_ns: Optional[int] = None
    entry_edge: Optional[float] = None
    episodes: list[dict[str, float | str]] = []

    for row in fills:
        symbol = str(row["symbol"])
        side = str(row["side"])
        qty = int(row.get("fill_qty") or 0)
        price = float(row.get("fill_price") or 0.0)
        signed_qty = qty if side == "BUY" else -qty
        prev_pos = pos[symbol]

        if current_thesis is None and row.get("role") == "entry":
            current_thesis = str(row.get("thesis") or "")
            start_ns = _as_int(row.get("monotonic_ns"))
            entry_edge = _as_float(row.get("signal_strength"))

        if prev_pos == 0 or (prev_pos > 0 and signed_qty > 0) or (prev_pos < 0 and signed_qty < 0):
            new_abs = abs(prev_pos) + abs(signed_qty)
            avg[symbol] = price if new_abs == 0 else ((abs(prev_pos) * avg[symbol]) + (abs(signed_qty) * price)) / new_abs
            pos[symbol] = prev_pos + signed_qty
        else:
            close_qty = min(abs(prev_pos), abs(signed_qty))
            if prev_pos > 0:
                realized += close_qty * (price - avg[symbol])
            else:
                realized += close_qty * (avg[symbol] - price)
            pos[symbol] = prev_pos + signed_qty
            if pos[symbol] == 0:
                avg[symbol] = 0.0

        if pos["R_HIKE"] == 0 and pos["R_CUT"] == 0 and current_thesis is not None and start_ns is not None:
            end_ns = _as_int(row.get("monotonic_ns")) or start_ns
            episodes.append(
                {
                    "thesis": current_thesis,
                    "pnl": realized,
                    "entry_edge": 0.0 if entry_edge is None else entry_edge,
                    "duration_s": max(0.0, (end_ns - start_ns) / 1_000_000_000.0),
                }
            )
            realized = 0.0
            current_thesis = None
            start_ns = None
            entry_edge = None

    return episodes


def _collect_c_closed_episodes(events_path: Path) -> list[dict[str, float | str]]:
    fills = [
        row
        for row in _iter_event_rows(events_path)
        if row.get("event_type") == "order_event"
        and row.get("order_event") == "filled"
        and row.get("symbol") == "C"
    ]

    pos = 0
    avg = 0.0
    realized = 0.0
    current_thesis: Optional[str] = None
    start_ns: Optional[int] = None
    entry_edge: Optional[float] = None
    entry_eps_delta: Optional[float] = None
    episodes: list[dict[str, float | str]] = []

    for row in fills:
        side = str(row["side"])
        qty = int(row.get("fill_qty") or 0)
        price = float(row.get("fill_price") or 0.0)
        signed_qty = qty if side == "BUY" else -qty
        prev_pos = pos

        if prev_pos == 0 and signed_qty != 0:
            current_thesis = str(row.get("thesis") or "")
            start_ns = _as_int(row.get("monotonic_ns"))
            entry_edge = _as_float(row.get("signal_strength"))
            entry_eps_delta = _as_float(row.get("last_c_earnings_delta"))

        if prev_pos == 0 or (prev_pos > 0 and signed_qty > 0) or (prev_pos < 0 and signed_qty < 0):
            new_abs = abs(prev_pos) + abs(signed_qty)
            avg = price if new_abs == 0 else ((abs(prev_pos) * avg) + (abs(signed_qty) * price)) / new_abs
            pos = prev_pos + signed_qty
        else:
            close_qty = min(abs(prev_pos), abs(signed_qty))
            if prev_pos > 0:
                realized += close_qty * (price - avg)
            else:
                realized += close_qty * (avg - price)
            pos = prev_pos + signed_qty
            if pos == 0 and current_thesis is not None and start_ns is not None:
                end_ns = _as_int(row.get("monotonic_ns")) or start_ns
                episodes.append(
                    {
                        "thesis": current_thesis,
                        "pnl": realized,
                        "entry_edge": 0.0 if entry_edge is None else entry_edge,
                        "earnings_delta": 0.0 if entry_eps_delta is None else entry_eps_delta,
                        "duration_s": max(0.0, (end_ns - start_ns) / 1_000_000_000.0),
                    }
                )
                realized = 0.0
                current_thesis = None
                start_ns = None
                entry_edge = None
                entry_eps_delta = None
                avg = 0.0
            elif (prev_pos > 0 and pos < 0) or (prev_pos < 0 and pos > 0):
                current_thesis = str(row.get("thesis") or current_thesis or "")
                start_ns = _as_int(row.get("monotonic_ns"))
                entry_edge = _as_float(row.get("signal_strength"))
                entry_eps_delta = _as_float(row.get("last_c_earnings_delta"))
                avg = price
                realized = 0.0

    return episodes


def recommend_parameter_sweep(
    reference_runs: list[Path],
    *,
    replay_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    if not reference_runs:
        root = replay_root or _default_replay_root()
        candidates = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("market_research_live_"))
        if not candidates:
            print(f"No market_research_live_* runs found under {root}")
            return
        reference_runs = [candidates[-1]]

    event_paths = [_resolve_events_path(path) for path in reference_runs]
    cfg = BotConfig()

    rates_episodes: list[dict[str, float | str]] = []
    c_episodes: list[dict[str, float | str]] = []
    run_labels: list[str] = []
    for events_path in event_paths:
        run_labels.append(events_path.parent.name)
        rates_episodes.extend(_collect_rates_pair_episodes(events_path))
        c_episodes.extend(_collect_c_closed_episodes(events_path))

    winning_rates = [ep for ep in rates_episodes if float(ep["pnl"]) > 0.0]
    positive_earnings = [ep for ep in c_episodes if ep["thesis"] == "earnings" and float(ep["pnl"]) > 0.0]
    losing_leadlag = [ep for ep in c_episodes if ep["thesis"] == "rates_lead_lag" and float(ep["pnl"]) < 0.0]

    rate_edges = sorted(float(ep["entry_edge"]) for ep in winning_rates)
    rate_durations = sorted(float(ep["duration_s"]) for ep in winning_rates)
    earnings_deltas = sorted(abs(float(ep["earnings_delta"])) for ep in positive_earnings if abs(float(ep["earnings_delta"])) > 0)

    edge_anchor = _round_to_step(rate_edges[0], 0.25) if rate_edges else cfg.rate_entry_edge_bp
    duration_anchor = statistics.median(rate_durations) if rate_durations else cfg.headline_bias_ttl_secs
    earnings_anchor = statistics.median(earnings_deltas) if earnings_deltas else cfg.c_earnings_small_delta

    entry_candidates = _dedupe_sorted(
        [
            cfg.rate_entry_edge_bp,
            max(1.0, edge_anchor),
            max(1.0, edge_anchor + 0.25),
        ],
        digits=2,
    )
    exit_candidates = _dedupe_sorted([0.8, cfg.rate_exit_edge_bp, 1.2], digits=2)
    headline_ttl_candidates = _dedupe_sorted(
        [
            max(1.25, _round_to_step(duration_anchor - 0.25, 0.25)),
            cfg.headline_bias_ttl_secs,
            min(3.0, _round_to_step(duration_anchor + 0.25, 0.25)),
        ],
        digits=2,
    )
    cpi_ttl_candidates = _dedupe_sorted(
        [
            max(1.5, cfg.cpi_bias_ttl_secs - 0.25),
            cfg.cpi_bias_ttl_secs,
            min(3.5, cfg.cpi_bias_ttl_secs + 0.25),
        ],
        digits=2,
    )
    earnings_small_candidates = _dedupe_sorted(
        [
            max(0.01, round(earnings_anchor - 0.005, 3)),
            cfg.c_earnings_small_delta,
            round(max(cfg.c_earnings_small_delta + 0.005, earnings_anchor + 0.005), 3),
        ],
        digits=3,
    )
    earnings_medium_candidates = _dedupe_sorted(
        [
            max(0.02, cfg.c_earnings_medium_delta - 0.005),
            cfg.c_earnings_medium_delta,
            cfg.c_earnings_medium_delta + 0.005,
        ],
        digits=3,
    )
    earnings_hold_candidates = _dedupe_sorted([2.0, cfg.c_earnings_hold_secs, 3.0], digits=2)
    leadlag_variants = [
        {"lead_lag_bp_trigger": cfg.lead_lag_bp_trigger, "lead_lag_entry_ticks": cfg.lead_lag_entry_ticks},
        {"lead_lag_bp_trigger": 4.0, "lead_lag_entry_ticks": 18.0},
        {"lead_lag_bp_trigger": 5.0, "lead_lag_entry_ticks": 20.0},
    ]

    control = {
        "name": "control_85k_baseline",
        "overrides": {},
        "focus": "Use the restored 85k-era defaults as the control.",
    }
    grid = [control]
    grid.extend(
        {
            "name": f"rates_entry_{value:.2f}".replace(".", "_"),
            "overrides": {"rate_entry_edge_bp": value},
            "focus": "Tighten or loosen the macro entry threshold.",
        }
        for value in entry_candidates
        if value != cfg.rate_entry_edge_bp
    )
    grid.extend(
        {
            "name": f"rates_exit_{value:.2f}".replace(".", "_"),
            "overrides": {"rate_exit_edge_bp": value},
            "focus": "Exit faster or let macro bias run slightly longer.",
        }
        for value in exit_candidates
        if value != cfg.rate_exit_edge_bp
    )
    grid.extend(
        {
            "name": f"headline_ttl_{value:.2f}".replace(".", "_"),
            "overrides": {"headline_bias_ttl_secs": value},
            "focus": "Adjust how long headline bias stays active.",
        }
        for value in headline_ttl_candidates
        if value != cfg.headline_bias_ttl_secs
    )
    grid.extend(
        {
            "name": f"cpi_ttl_{value:.2f}".replace(".", "_"),
            "overrides": {"cpi_bias_ttl_secs": value},
            "focus": "Adjust how long CPI bias stays active.",
        }
        for value in cpi_ttl_candidates
        if value != cfg.cpi_bias_ttl_secs
    )
    grid.extend(
        {
            "name": f"earnings_small_{small:.3f}_medium_{medium:.3f}".replace(".", "_"),
            "overrides": {
                "c_earnings_small_delta": small,
                "c_earnings_medium_delta": medium,
            },
            "focus": "Change which earnings deltas qualify as tier-2/tier-3 trades.",
        }
        for small, medium in zip(earnings_small_candidates, earnings_medium_candidates)
        if small != cfg.c_earnings_small_delta or medium != cfg.c_earnings_medium_delta
    )
    grid.extend(
        {
            "name": f"earnings_hold_{value:.2f}".replace(".", "_"),
            "overrides": {"c_earnings_hold_secs": value},
            "focus": "Shorten or extend the earnings hold window.",
        }
        for value in earnings_hold_candidates
        if value != cfg.c_earnings_hold_secs
    )
    grid.extend(
        {
            "name": f"leadlag_bp_{variant['lead_lag_bp_trigger']:.1f}_ticks_{variant['lead_lag_entry_ticks']:.0f}".replace(".", "_"),
            "overrides": variant,
            "focus": "Make rates lead-lag stricter to cut the main C loser.",
        }
        for variant in leadlag_variants[1:]
    )

    summary = {
        "reference_runs": run_labels,
        "rates_closed_episodes": len(rates_episodes),
        "rates_total_pnl": round(sum(float(ep["pnl"]) for ep in rates_episodes), 2),
        "rates_winning_edges_bp": [round(edge, 3) for edge in rate_edges],
        "rates_winning_durations_s": [round(duration, 3) for duration in rate_durations],
        "c_total_pnl": round(sum(float(ep["pnl"]) for ep in c_episodes), 2),
        "c_earnings_total_pnl": round(sum(float(ep["pnl"]) for ep in c_episodes if ep["thesis"] == "earnings"), 2),
        "c_initial_shock_total_pnl": round(sum(float(ep["pnl"]) for ep in c_episodes if ep["thesis"] == "initial_earnings_shock"), 2),
        "c_leadlag_total_pnl": round(sum(float(ep["pnl"]) for ep in c_episodes if ep["thesis"] == "rates_lead_lag"), 2),
        "c_leadlag_loser_count": len(losing_leadlag),
        "candidate_grid": grid,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote parameter sweep recommendations to {output_path}")
        return

    print(json.dumps(summary, indent=2))


async def run_bot(
    host: str,
    username: str,
    password: str,
    *,
    log_root: Optional[Path] = None,
    enable_run_logger: bool = True,
) -> None:
    client = MyXchangeClient(
        host,
        username,
        password,
        log_root=log_root,
        enable_run_logger=enable_run_logger,
    )
    await client.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C + Fed event bot with optional replay summary helper.")
    parser.add_argument("--analyze-replay", action="store_true", help="Summarize saved replay logs instead of running the bot.")
    parser.add_argument("--recommend-sweep", action="store_true", help="Derive a narrow parameter test grid from reference live runs.")
    parser.add_argument("--replay-root", type=Path, default=None, help="Optional override for the replay data root.")
    parser.add_argument(
        "--reference-run",
        type=Path,
        action="append",
        default=[],
        help="Reference live run directory or events.jsonl path for parameter recommendations. Can be passed multiple times.",
    )
    parser.add_argument("--sweep-output", type=Path, default=None, help="Optional JSON output path for recommended parameter variants.")
    parser.add_argument("--host", default="practice.uchicago.exchange:3333")
    parser.add_argument("--username", default="uiuc")
    parser.add_argument("--password", default="mesa-lynx-octopus")
    parser.add_argument("--log-root", type=Path, default=None, help="Optional override for structured live run logging output.")
    parser.add_argument("--disable-run-logger", action="store_true", help="Disable structured per-run data logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_replay:
        summarize_saved_replays(args.replay_root)
        return
    if args.recommend_sweep:
        recommend_parameter_sweep(args.reference_run, replay_root=args.replay_root, output_path=args.sweep_output)
        return
    asyncio.run(
        run_bot(
            args.host,
            args.username,
            args.password,
            log_root=args.log_root,
            enable_run_logger=not args.disable_run_logger,
        )
    )


if __name__ == "__main__":
    main()
