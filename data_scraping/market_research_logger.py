from __future__ import annotations

import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOCAL_UTCXCHANGE_PATH = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_UTCXCHANGE_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_UTCXCHANGE_PATH))

from config import ResearchLoggerConfig, load_config
from csv_writer import AppendSafeCsvWriter
from feature_extractor import (
    RETURN_WINDOWS_SECONDS,
    MarketSnapshot,
    bucket_post_news_elapsed,
    compute_book_features,
    compute_realized_volatility,
    compute_return,
    mark_post_news_window,
    naive_fair_price,
    nearest_mid_at_or_before,
    nearest_snapshot_at_or_after,
    nearest_snapshot_at_or_before,
    news_mentions_symbol,
    regime_for_post_news,
    seconds_since,
    signed_distance_to_fair,
    trailing_trade_stats,
)


LOGGER = logging.getLogger("market-research-logger")

try:
    from utcxchangelib import XChangeClient
except ModuleNotFoundError as exc:
    XChangeClient = None  # type: ignore[assignment]
    UTCXCHANGE_IMPORT_ERROR = exc
else:
    UTCXCHANGE_IMPORT_ERROR = None

BaseXChangeClient = XChangeClient if XChangeClient is not None else object

RAW_BOOK_FIELDNAMES = [
    "event_index",
    "event_type",
    "session_id",
    "run_id",
    "wall_time_iso",
    "wall_time_ns",
    "monotonic_ns",
    "exchange_tick",
    "round",
    "day",
    "symbol",
    "best_bid_px",
    "best_bid_qty",
    "best_ask_px",
    "best_ask_qty",
    "mid_px",
    "spread",
    "microprice",
    "top_of_book_imbalance",
    "total_bid_depth_top_k",
    "total_ask_depth_top_k",
    "bid_levels_json",
    "ask_levels_json",
    "current_position_symbol",
    "current_cash",
    "last_trade_px_for_symbol",
    "last_trade_qty_for_symbol",
    "most_recent_news_id_affecting_symbol",
    "seconds_since_last_news_for_symbol",
    "inside_post_news_reaction_window",
]

RAW_TRADE_FIELDNAMES = [
    "event_index",
    "session_id",
    "run_id",
    "wall_time_iso",
    "wall_time_ns",
    "monotonic_ns",
    "exchange_tick",
    "round",
    "day",
    "symbol",
    "trade_px",
    "trade_qty",
    "mid_at_trade",
    "spread_at_trade",
    "time_since_last_symbol_news",
    "latest_known_eps_for_symbol",
]

RAW_NEWS_FIELDNAMES = [
    "news_id",
    "session_id",
    "run_id",
    "wall_time_iso",
    "wall_time_ns",
    "monotonic_ns",
    "exchange_tick",
    "round",
    "day",
    "kind",
    "symbol",
    "structured_subtype",
    "earnings_asset",
    "earnings_value",
    "affected_symbols_json",
    "raw_content",
    "normalized_content",
    "parsed_sentiment",
    "parsed_signal_strength",
    "previous_known_eps_for_asset",
    "new_known_eps_for_asset",
    "inferred_fair_price_before",
    "inferred_fair_price_after",
    "note",
]

DERIVED_FIELDNAMES = [
    "derived_row_id",
    "source_event_type",
    "source_event_index",
    "source_news_id",
    "session_id",
    "run_id",
    "wall_time_iso",
    "wall_time_ns",
    "monotonic_ns",
    "exchange_tick",
    "round",
    "day",
    "symbol",
    "best_bid_px",
    "best_bid_qty",
    "best_ask_px",
    "best_ask_qty",
    "mid_px",
    "spread",
    "microprice",
    "top_of_book_imbalance",
    "latest_known_eps_for_symbol",
    "naive_fair_price_for_symbol",
    "signed_distance_mid_to_fair",
    "seconds_since_last_news_for_symbol",
    "post_news_elapsed_bucket",
    "earnings_regime",
    "market_fully_incorporated_placeholder",
    "recent_trade_count_250ms",
    "recent_trade_count_1s",
    "recent_trade_count_5s",
    "recent_trade_volume_250ms",
    "recent_trade_volume_1s",
    "recent_trade_volume_5s",
    "recent_realized_vol_250ms",
    "recent_realized_vol_1s",
    "recent_realized_vol_5s",
    "past_return_100ms",
    "past_return_250ms",
    "past_return_500ms",
    "past_return_1s",
    "past_return_2s",
    "past_return_5s",
    "future_return_100ms",
    "future_return_250ms",
    "future_return_500ms",
    "future_return_1s",
    "future_return_2s",
    "future_return_5s",
    "pre_news_mid_1s",
    "pre_news_spread_1s",
    "post_news_mid_100ms",
    "post_news_mid_250ms",
    "post_news_mid_500ms",
    "post_news_mid_1s",
    "post_news_mid_2s",
    "post_news_mid_5s",
    "post_news_spread_100ms",
    "post_news_spread_250ms",
    "post_news_spread_500ms",
    "post_news_spread_1s",
    "post_news_spread_2s",
    "post_news_spread_5s",
]


def utc_now() -> tuple[str, int]:
    wall_time_ns = time.time_ns()
    wall_iso = datetime.fromtimestamp(wall_time_ns / 1_000_000_000, tz=timezone.utc).isoformat()
    return wall_iso, wall_time_ns


@dataclass
class PendingDerivedRow:
    payload: dict[str, Any]
    finalize_after_ns: int


class MarketResearchLogger(BaseXChangeClient):  # type: ignore[misc, valid-type]
    def __init__(self, config: ResearchLoggerConfig, run_dir: Path):
        super().__init__(config.host, config.username, config.password, symbols=config.monitored_symbols)
        self.config = config
        self.monitored_symbols = list(dict.fromkeys(config.monitored_symbols))
        self.plot_symbols = list(dict.fromkeys(config.plot_symbols))
        self.direct_earnings_symbols = set(config.direct_earnings_symbols)
        self.etf_news_assets = set(config.etf_news_assets)
        self.run_dir = run_dir
        self.session_id = run_dir.name
        self.run_id = uuid.uuid4().hex
        self.event_index = 0
        self.news_index = 0
        self.derived_row_index = 0
        self.latest_exchange_tick: int | None = None
        self.latest_known_eps_by_asset: dict[str, float] = {}
        self.shutdown_event = asyncio.Event()
        self.heartbeat_task: asyncio.Task[None] | None = None

        self.last_trade_by_symbol: dict[str, dict[str, int | None]] = {
            symbol: {"price": None, "qty": None} for symbol in self.monitored_symbols
        }
        self.last_news_by_symbol: dict[str, dict[str, Any]] = {
            symbol: {"news_id": None, "monotonic_ns": None} for symbol in self.monitored_symbols
        }
        self.market_history_by_symbol: dict[str, deque[MarketSnapshot]] = {
            symbol: deque(maxlen=config.history_maxlen) for symbol in self.monitored_symbols
        }
        self.trade_history_by_symbol: dict[str, deque[dict[str, Any]]] = {
            symbol: deque(maxlen=config.history_maxlen) for symbol in self.monitored_symbols
        }
        self.news_history: deque[dict[str, Any]] = deque(maxlen=config.history_maxlen)
        self.pending_derived_rows: deque[PendingDerivedRow] = deque()

        self.book_writer = AppendSafeCsvWriter(run_dir / "raw_book_events.csv", RAW_BOOK_FIELDNAMES)
        self.trade_writer = AppendSafeCsvWriter(run_dir / "raw_trade_events.csv", RAW_TRADE_FIELDNAMES)
        self.book_writers_by_symbol: dict[str, AppendSafeCsvWriter] = {
            symbol: AppendSafeCsvWriter(run_dir / f"raw_book_events_{symbol}.csv", RAW_BOOK_FIELDNAMES)
            for symbol in self.monitored_symbols
        }
        self.trade_writers_by_symbol: dict[str, AppendSafeCsvWriter] = {
            symbol: AppendSafeCsvWriter(run_dir / f"raw_trade_events_{symbol}.csv", RAW_TRADE_FIELDNAMES)
            for symbol in self.monitored_symbols
        }
        self.news_writer = AppendSafeCsvWriter(run_dir / "raw_news_events.csv", RAW_NEWS_FIELDNAMES)
        self.derived_writer = AppendSafeCsvWriter(run_dir / "derived_feature_rows.csv", DERIVED_FIELDNAMES)

    async def start(self) -> None:
        self.write_session_metadata(status="starting")
        self.install_signal_handlers()
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="market-research-heartbeat")
        try:
            await self.connect()
        finally:
            self.shutdown_event.set()
            if self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            self.flush_pending_derived_rows(force=True)
            self.write_session_metadata(status="stopped")
            self.close_writers()
            self.run_post_analysis()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signame in ("SIGINT", "SIGTERM"):
            if hasattr(signal, signame):
                try:
                    loop.add_signal_handler(getattr(signal, signame), self.shutdown_event.set)
                except NotImplementedError:
                    pass

    def close_writers(self) -> None:
        self.book_writer.close()
        self.trade_writer.close()
        for writer in self.book_writers_by_symbol.values():
            writer.close()
        for writer in self.trade_writers_by_symbol.values():
            writer.close()
        self.news_writer.close()
        self.derived_writer.close()

    def run_post_analysis(self) -> None:
        analyze_script = ROOT / "analyze_logs.py"
        cmd = [sys.executable, str(analyze_script), "--run-dir", str(self.run_dir), "--plot"]
        LOGGER.info("Running post-processing: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            LOGGER.exception("Automatic post-run analysis failed for %s", self.run_dir)

    def write_session_metadata(self, *, status: str) -> None:
        wall_iso, wall_time_ns = utc_now()
        payload = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": status,
            "written_at_iso": wall_iso,
            "written_at_ns": wall_time_ns,
            "monitored_symbols": self.monitored_symbols,
            "plot_symbols": self.plot_symbols,
            "latest_exchange_tick": self.latest_exchange_tick,
            "latest_known_eps_by_asset": self.latest_known_eps_by_asset,
            "positions": dict(self.positions),
            "open_orders_count": len(self.open_orders),
            "notes": {
                "round_day_inference": "The current utcxchangelib client does not expose round/day directly. Raw tick and timestamps are preserved for offline inference.",
                "no_trading": "This listener never calls place_order, cancel_order, or place_swap_order.",
            },
            "config": self.config.to_metadata(),
        }
        with (self.run_dir / "session_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    async def _heartbeat_loop(self) -> None:
        while not self.shutdown_event.is_set():
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            for symbol in self.monitored_symbols:
                self._log_book_like_event(symbol=symbol, event_type="heartbeat")

    def _next_event_index(self) -> int:
        self.event_index += 1
        return self.event_index

    def _next_news_id(self) -> str:
        self.news_index += 1
        return f"{self.session_id}-news-{self.news_index}"

    def _next_derived_row_id(self) -> str:
        self.derived_row_index += 1
        return f"{self.session_id}-derived-{self.derived_row_index}"

    def _current_book_features(self, symbol: str):
        return compute_book_features(
            self.order_books[symbol],
            top_k_depth=self.config.top_k_depth,
            top_n_levels=self.config.top_n_levels,
        )

    def _common_state(self, *, now_ns: int, wall_iso: str, wall_time_ns: int) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "wall_time_iso": wall_iso,
            "wall_time_ns": wall_time_ns,
            "monotonic_ns": now_ns,
            "exchange_tick": self.latest_exchange_tick,
            "round": None,
            "day": None,
        }

    def _latest_eps_for_symbol(self, symbol: str) -> float | None:
        return self.latest_known_eps_by_asset.get(symbol)

    def _fair_price_for_symbol(self, symbol: str) -> float | None:
        pe_constant = self.config.pe_constants.get(symbol)
        latest_eps = self._latest_eps_for_symbol(symbol)
        if pe_constant is None or latest_eps is None:
            return None
        return naive_fair_price(latest_eps, pe_constant)

    def _last_news_state(self, symbol: str) -> dict[str, Any]:
        return self.last_news_by_symbol[symbol]

    def _log_book_like_event(self, *, symbol: str, event_type: str) -> None:
        now_ns = time.monotonic_ns()
        wall_iso, wall_time_ns = utc_now()
        features = self._current_book_features(symbol)
        last_news_state = self._last_news_state(symbol)
        seconds_since_news = seconds_since(last_news_state["monotonic_ns"], now_ns)
        last_trade_state = self.last_trade_by_symbol[symbol]
        event_index = self._next_event_index()

        row = {
            "event_index": event_index,
            "event_type": event_type,
            **self._common_state(now_ns=now_ns, wall_iso=wall_iso, wall_time_ns=wall_time_ns),
            "symbol": symbol,
            "best_bid_px": features.best_bid_px,
            "best_bid_qty": features.best_bid_qty,
            "best_ask_px": features.best_ask_px,
            "best_ask_qty": features.best_ask_qty,
            "mid_px": features.mid_px,
            "spread": features.spread,
            "microprice": features.microprice,
            "top_of_book_imbalance": features.top_of_book_imbalance,
            "total_bid_depth_top_k": features.total_bid_depth_top_k,
            "total_ask_depth_top_k": features.total_ask_depth_top_k,
            "bid_levels_json": features.bid_levels_json,
            "ask_levels_json": features.ask_levels_json,
            "current_position_symbol": self.positions.get(symbol, 0),
            "current_cash": self.positions.get("cash", 0),
            "last_trade_px_for_symbol": last_trade_state["price"],
            "last_trade_qty_for_symbol": last_trade_state["qty"],
            "most_recent_news_id_affecting_symbol": last_news_state["news_id"],
            "seconds_since_last_news_for_symbol": seconds_since_news,
            "inside_post_news_reaction_window": mark_post_news_window(
                now_ns,
                last_news_state["monotonic_ns"],
                self.config.post_news_window_seconds,
            ),
        }
        self.book_writer.write_row(row)
        self.book_writers_by_symbol[symbol].write_row(row)
        self._record_market_snapshot(symbol=symbol, row=row, event_kind=event_type)
        self._enqueue_derived_from_market_row(symbol=symbol, row=row, source_event_type=event_type)
        self.flush_pending_derived_rows(force=False)

    def _record_market_snapshot(self, *, symbol: str, row: dict[str, Any], event_kind: str) -> None:
        snapshot = MarketSnapshot(
            monotonic_ns=int(row["monotonic_ns"]),
            wall_time_ns=int(row["wall_time_ns"]),
            exchange_tick=row["exchange_tick"],
            event_kind=event_kind,
            event_index=int(row["event_index"]),
            mid_px=row["mid_px"],
            spread=row["spread"],
            microprice=row["microprice"],
            imbalance=row["top_of_book_imbalance"],
            best_bid_px=row["best_bid_px"],
            best_bid_qty=row["best_bid_qty"],
            best_ask_px=row["best_ask_px"],
            best_ask_qty=row["best_ask_qty"],
        )
        self.market_history_by_symbol[symbol].append(snapshot)

    def _enqueue_derived_from_market_row(self, *, symbol: str, row: dict[str, Any], source_event_type: str) -> None:
        payload = self._build_common_derived_payload(
            symbol=symbol,
            monotonic_ns=int(row["monotonic_ns"]),
            wall_time_ns=int(row["wall_time_ns"]),
            wall_time_iso=str(row["wall_time_iso"]),
            exchange_tick=row["exchange_tick"],
            source_event_type=source_event_type,
            source_event_index=int(row["event_index"]),
            source_news_id=None,
            best_bid_px=row["best_bid_px"],
            best_bid_qty=row["best_bid_qty"],
            best_ask_px=row["best_ask_px"],
            best_ask_qty=row["best_ask_qty"],
            mid_px=row["mid_px"],
            spread=row["spread"],
            microprice=row["microprice"],
            imbalance=row["top_of_book_imbalance"],
        )
        self.pending_derived_rows.append(
            PendingDerivedRow(
                payload=payload,
                finalize_after_ns=int(row["monotonic_ns"]) + int(max(RETURN_WINDOWS_SECONDS) * 1_000_000_000),
            )
        )

    def _build_common_derived_payload(
        self,
        *,
        symbol: str,
        monotonic_ns: int,
        wall_time_ns: int,
        wall_time_iso: str,
        exchange_tick: int | None,
        source_event_type: str,
        source_event_index: int | None,
        source_news_id: str | None,
        best_bid_px: int | None,
        best_bid_qty: int | None,
        best_ask_px: int | None,
        best_ask_qty: int | None,
        mid_px: float | None,
        spread: float | None,
        microprice: float | None,
        imbalance: float | None,
    ) -> dict[str, Any]:
        last_news_state = self._last_news_state(symbol)
        seconds_since_news = seconds_since(last_news_state["monotonic_ns"], monotonic_ns)
        fair_px = self._fair_price_for_symbol(symbol)
        trade_history = self.trade_history_by_symbol[symbol]
        market_history = self.market_history_by_symbol[symbol]
        trade_count_250ms, trade_volume_250ms = trailing_trade_stats(trade_history, monotonic_ns, 0.25)
        trade_count_1s, trade_volume_1s = trailing_trade_stats(trade_history, monotonic_ns, 1.0)
        trade_count_5s, trade_volume_5s = trailing_trade_stats(trade_history, monotonic_ns, 5.0)

        payload: dict[str, Any] = {
            "derived_row_id": self._next_derived_row_id(),
            "source_event_type": source_event_type,
            "source_event_index": source_event_index,
            "source_news_id": source_news_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "wall_time_iso": wall_time_iso,
            "wall_time_ns": wall_time_ns,
            "monotonic_ns": monotonic_ns,
            "exchange_tick": exchange_tick,
            "round": None,
            "day": None,
            "symbol": symbol,
            "best_bid_px": best_bid_px,
            "best_bid_qty": best_bid_qty,
            "best_ask_px": best_ask_px,
            "best_ask_qty": best_ask_qty,
            "mid_px": mid_px,
            "spread": spread,
            "microprice": microprice,
            "top_of_book_imbalance": imbalance,
            "latest_known_eps_for_symbol": self._latest_eps_for_symbol(symbol),
            "naive_fair_price_for_symbol": fair_px,
            "signed_distance_mid_to_fair": signed_distance_to_fair(mid_px, fair_px),
            "seconds_since_last_news_for_symbol": seconds_since_news,
            "post_news_elapsed_bucket": bucket_post_news_elapsed(seconds_since_news),
            "earnings_regime": regime_for_post_news(seconds_since_news),
            "market_fully_incorporated_placeholder": self._market_fully_incorporated_placeholder(mid_px, fair_px, seconds_since_news),
            "recent_trade_count_250ms": trade_count_250ms,
            "recent_trade_count_1s": trade_count_1s,
            "recent_trade_count_5s": trade_count_5s,
            "recent_trade_volume_250ms": trade_volume_250ms,
            "recent_trade_volume_1s": trade_volume_1s,
            "recent_trade_volume_5s": trade_volume_5s,
            "recent_realized_vol_250ms": compute_realized_volatility(market_history, monotonic_ns, 0.25),
            "recent_realized_vol_1s": compute_realized_volatility(market_history, monotonic_ns, 1.0),
            "recent_realized_vol_5s": compute_realized_volatility(market_history, monotonic_ns, 5.0),
        }

        for window in RETURN_WINDOWS_SECONDS:
            payload[f"past_return_{self._window_label(window)}"] = self._past_return(symbol, monotonic_ns, mid_px, window)
            payload[f"future_return_{self._window_label(window)}"] = None

        for window in RETURN_WINDOWS_SECONDS:
            payload[f"post_news_mid_{self._window_label(window)}"] = None
            payload[f"post_news_spread_{self._window_label(window)}"] = None

        payload["pre_news_mid_1s"] = None
        payload["pre_news_spread_1s"] = None
        return payload

    def _window_label(self, window_seconds: float) -> str:
        mapping = {
            0.1: "100ms",
            0.25: "250ms",
            0.5: "500ms",
            1.0: "1s",
            2.0: "2s",
            5.0: "5s",
        }
        return mapping[window_seconds]

    def _past_return(self, symbol: str, now_ns: int, current_mid: float | None, window_seconds: float) -> float | None:
        reference_mid = nearest_mid_at_or_before(
            self.market_history_by_symbol[symbol],
            now_ns - int(window_seconds * 1_000_000_000),
        )
        return compute_return(current_mid, reference_mid)

    def _market_fully_incorporated_placeholder(
        self,
        mid_px: float | None,
        fair_px: float | None,
        seconds_since_news: float | None,
    ) -> bool | None:
        if mid_px is None or fair_px is None or seconds_since_news is None:
            return None
        if seconds_since_news < 1.0:
            return False
        return abs(mid_px - fair_px) <= max(0.01 * fair_px, 1.0)

    def flush_pending_derived_rows(self, *, force: bool) -> None:
        now_ns = time.monotonic_ns()
        while self.pending_derived_rows:
            pending = self.pending_derived_rows[0]
            if not force and pending.finalize_after_ns > now_ns:
                break
            self.pending_derived_rows.popleft()
            finalized = self._finalize_derived_payload(pending.payload)
            self.derived_writer.write_row(finalized)

    def _finalize_derived_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload["symbol"])
        market_history = self.market_history_by_symbol[symbol]
        event_ns = int(payload["monotonic_ns"])
        event_mid = payload["mid_px"]
        for window in RETURN_WINDOWS_SECONDS:
            label = self._window_label(window)
            future_snapshot = nearest_snapshot_at_or_after(market_history, event_ns + int(window * 1_000_000_000))
            future_mid = future_snapshot.mid_px if future_snapshot else None
            payload[f"future_return_{label}"] = compute_return(future_mid, event_mid)

        if payload["source_event_type"] == "news" and payload["source_news_id"]:
            pre_snapshot = nearest_snapshot_at_or_before(market_history, event_ns - 1_000_000_000)
            if pre_snapshot is not None:
                payload["pre_news_mid_1s"] = pre_snapshot.mid_px
                payload["pre_news_spread_1s"] = pre_snapshot.spread
            for window in RETURN_WINDOWS_SECONDS:
                label = self._window_label(window)
                snapshot = nearest_snapshot_at_or_after(market_history, event_ns + int(window * 1_000_000_000))
                if snapshot is not None:
                    payload[f"post_news_mid_{label}"] = snapshot.mid_px
                    payload[f"post_news_spread_{label}"] = snapshot.spread
        return payload

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol not in self.monitored_symbols:
            return
        self._log_book_like_event(symbol=symbol, event_type="book_update")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol not in self.monitored_symbols:
            return
        now_ns = time.monotonic_ns()
        wall_iso, wall_time_ns = utc_now()
        features = self._current_book_features(symbol)
        self.last_trade_by_symbol[symbol] = {"price": price, "qty": qty}
        last_news_state = self._last_news_state(symbol)
        event_index = self._next_event_index()

        row = {
            "event_index": event_index,
            **self._common_state(now_ns=now_ns, wall_iso=wall_iso, wall_time_ns=wall_time_ns),
            "symbol": symbol,
            "trade_px": price,
            "trade_qty": qty,
            "mid_at_trade": features.mid_px,
            "spread_at_trade": features.spread,
            "time_since_last_symbol_news": seconds_since(last_news_state["monotonic_ns"], now_ns),
            "latest_known_eps_for_symbol": self._latest_eps_for_symbol(symbol),
        }
        self.trade_writer.write_row(row)
        self.trade_writers_by_symbol[symbol].write_row(row)
        self.trade_history_by_symbol[symbol].append(row)
        self.flush_pending_derived_rows(force=False)

    async def bot_handle_news(self, news_release: dict) -> None:
        self.latest_exchange_tick = news_release.get("tick")
        affected_symbols, note = self._affected_symbols_for_news(news_release)
        if not affected_symbols:
            return

        now_ns = time.monotonic_ns()
        wall_iso, wall_time_ns = utc_now()
        news_id = self._next_news_id()
        new_data = news_release.get("new_data", {})
        structured_subtype = new_data.get("structured_subtype")
        earnings_asset = new_data.get("asset") if structured_subtype == "earnings" else None
        earnings_value = float(new_data.get("value")) if structured_subtype == "earnings" and new_data.get("value") is not None else None

        previous_eps = self.latest_known_eps_by_asset.get(earnings_asset) if earnings_asset else None
        new_eps = previous_eps
        if earnings_asset is not None and earnings_value is not None:
            new_eps = earnings_value
            self.latest_known_eps_by_asset[earnings_asset] = earnings_value

        for symbol in affected_symbols:
            self.last_news_by_symbol[symbol] = {"news_id": news_id, "monotonic_ns": now_ns}

        raw_content: str | None = None
        normalized_content = json.dumps(new_data, sort_keys=True, separators=(",", ":"))
        if news_release["kind"] == "unstructured":
            raw_content = new_data.get("content")

        fair_before = None
        fair_after = None
        if earnings_asset is not None:
            pe_constant = self.config.pe_constants.get(earnings_asset)
            if pe_constant is not None:
                fair_before = naive_fair_price(previous_eps, pe_constant)
                fair_after = naive_fair_price(new_eps, pe_constant)

        row = {
            "news_id": news_id,
            **self._common_state(now_ns=now_ns, wall_iso=wall_iso, wall_time_ns=wall_time_ns),
            "kind": news_release["kind"],
            "symbol": news_release.get("symbol"),
            "structured_subtype": structured_subtype,
            "earnings_asset": earnings_asset,
            "earnings_value": earnings_value,
            "affected_symbols_json": json.dumps(sorted(affected_symbols), separators=(",", ":")),
            "raw_content": raw_content,
            "normalized_content": normalized_content,
            "parsed_sentiment": None,
            "parsed_signal_strength": None,
            "previous_known_eps_for_asset": previous_eps,
            "new_known_eps_for_asset": new_eps,
            "inferred_fair_price_before": fair_before,
            "inferred_fair_price_after": fair_after,
            "note": note,
        }
        self.news_writer.write_row(row)
        self.news_history.append(row)

        for symbol in affected_symbols:
            features = self._current_book_features(symbol)
            payload = self._build_common_derived_payload(
                symbol=symbol,
                monotonic_ns=now_ns,
                wall_time_ns=wall_time_ns,
                wall_time_iso=wall_iso,
                exchange_tick=self.latest_exchange_tick,
                source_event_type="news",
                source_event_index=None,
                source_news_id=news_id,
                best_bid_px=features.best_bid_px,
                best_bid_qty=features.best_bid_qty,
                best_ask_px=features.best_ask_px,
                best_ask_qty=features.best_ask_qty,
                mid_px=features.mid_px,
                spread=features.spread,
                microprice=features.microprice,
                imbalance=features.top_of_book_imbalance,
            )
            self.pending_derived_rows.append(
                PendingDerivedRow(
                    payload=payload,
                    finalize_after_ns=now_ns + int(max(RETURN_WINDOWS_SECONDS) * 1_000_000_000),
                )
            )
        self.flush_pending_derived_rows(force=False)

    def _affected_symbols_for_news(self, news_release: dict[str, Any]) -> tuple[list[str], str]:
        symbol = news_release.get("symbol")
        new_data = news_release.get("new_data", {})
        kind = news_release.get("kind")
        affected_symbols: set[str] = set()
        notes: list[str] = []

        if kind == "structured":
            if new_data.get("structured_subtype") != "earnings":
                return [], "Structured news ignored because it is not earnings."
            earnings_asset = new_data.get("asset")
            if isinstance(earnings_asset, str):
                if earnings_asset in self.monitored_symbols:
                    affected_symbols.add(earnings_asset)
                if "ETF" in self.monitored_symbols and earnings_asset in self.etf_news_assets:
                    affected_symbols.add("ETF")
                if affected_symbols:
                    notes.append(f"Structured earnings for {earnings_asset}.")
            return sorted(affected_symbols), " ".join(notes) if notes else "Structured earnings ignored."

        if isinstance(symbol, str) and symbol in self.monitored_symbols:
            affected_symbols.add(symbol)
            notes.append(f"Unstructured news tagged to {symbol}.")

        content = new_data.get("content")
        mentioned_symbols = [candidate for candidate in self.monitored_symbols if news_mentions_symbol(content, candidate)]
        for mentioned_symbol in mentioned_symbols:
            affected_symbols.add(mentioned_symbol)
        if mentioned_symbols:
            notes.append(f"Unstructured news mentions {', '.join(sorted(set(mentioned_symbols)))}.")

        return sorted(affected_symbols), " ".join(notes) if notes else "Unstructured news ignored."

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        LOGGER.warning("Received order fill for passive logger. order_id=%s qty=%s price=%s", order_id, qty, price)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        LOGGER.warning("Unexpected order rejection in passive logger. order_id=%s reason=%s", order_id, reason)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        LOGGER.warning(
            "Unexpected cancel response in passive logger. order_id=%s success=%s error=%s",
            order_id,
            success,
            error,
        )

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        LOGGER.warning("Unexpected swap response in passive logger. swap=%s qty=%s success=%s", swap, qty, success)

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        LOGGER.info("Market resolved market_id=%s winning_symbol=%s tick=%s", market_id, winning_symbol, tick)

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        LOGGER.info("Settlement payout user=%s market_id=%s amount=%s tick=%s", user, market_id, amount, tick)


def make_run_dir(config: ResearchLoggerConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    label_suffix = f"_{config.run_label}" if config.run_label else ""
    run_dir = config.log_root / f"market_research_{timestamp}{label_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def async_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config()
    if UTCXCHANGE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "utcxchangelib dependencies are not available. Install grpcio/protobuf and the client library "
            "before running the live logger."
        ) from UTCXCHANGE_IMPORT_ERROR
    run_dir = make_run_dir(config)
    LOGGER.info("Writing market research logs to %s", run_dir)
    client = MarketResearchLogger(config, run_dir)
    await client.start()


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "This logger reads settings from data_scraping/local_config.json. "
            "Edit that file, then run `python3 data_scraping/market_research_logger.py` with no extra arguments."
        )
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
