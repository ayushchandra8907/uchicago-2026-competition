from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class BotConfig:
    fed_hike: str
    fed_hold: str
    fed_cut: str
    cpi_trigger_abs: float
    trade_qty: int
    max_abs_position: int
    max_order_size: int
    corr_lookback_points: int
    min_corr_points: int
    min_pair_corr: float
    max_book_age_sec: float
    order_stale_sec: float
    status_interval_sec: float
    loop_sleep_sec: float
    trace_enabled: bool
    trace_dir: str

    @property
    def rate_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


def load_config() -> BotConfig:
    return BotConfig(
        fed_hike=env_str("PM2_FED_HIKE_SYMBOL", "R_HIKE"),
        fed_hold=env_str("PM2_FED_HOLD_SYMBOL", "R_HOLD"),
        fed_cut=env_str("PM2_FED_CUT_SYMBOL", "R_CUT"),
        cpi_trigger_abs=env_float("PM2_CPI_TRIGGER_ABS", 0.0003),
        trade_qty=env_int("PM2_TRADE_QTY", 40),
        max_abs_position=env_int("PM2_MAX_ABS_POSITION", 200),
        max_order_size=env_int("PM2_MAX_ORDER_SIZE", 40),
        corr_lookback_points=env_int("PM2_CORR_LOOKBACK_POINTS", 80),
        min_corr_points=env_int("PM2_MIN_CORR_POINTS", 12),
        min_pair_corr=env_float("PM2_MIN_PAIR_CORR", 0.55),
        max_book_age_sec=env_float("PM2_MAX_BOOK_AGE_SEC", 1.0),
        order_stale_sec=env_float("PM2_ORDER_STALE_SEC", 1.0),
        status_interval_sec=env_float("PM2_STATUS_INTERVAL_SEC", 3.0),
        loop_sleep_sec=env_float("PM2_LOOP_SLEEP_SEC", 0.20),
        trace_enabled=env_bool("PM2_TRACE_ENABLED", True),
        trace_dir=env_str("PM2_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
    )


def parse_cpi(kind: Any, new_data: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
    if kind == "structured":
        subtype = str(new_data.get("structured_subtype") or "")
        if subtype == "cpi_print" and "actual" in new_data and "forecast" in new_data:
            return float(new_data["actual"]), float(new_data["forecast"]), "structured"
        return None, None, "none"

    if kind != "unstructured":
        return None, None, "none"

    content = str(new_data.get("content", "") or "")
    patterns = [
        r"cpi\s*actual\s*([+-]?\d*\.?\d+)\s*vs\.?\s*forecast\s*([+-]?\d*\.?\d+)",
        r"actual\s*cpi\s*([+-]?\d*\.?\d+)\s*vs\.?\s*forecast(?:ed)?\s*cpi\s*([+-]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2)), "unstructured_text"
    return None, None, "none"


@dataclass
class Top:
    bid: Optional[int] = None
    bid_qty: int = 0
    ask: Optional[int] = None
    ask_qty: int = 0
    updated_ts: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass
class MarketLabels:
    important_1: str
    important_2: str
    irrelevant: str
    pair_corr: float = 0.0
    updated_ts: float = field(default_factory=time.time)


class NewPM2HeuristicClient(XChangeClient):
    def __init__(self, host: str, username: str, password: str, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or load_config()
        super().__init__(host, username, password, silent=False, symbols=list(self.cfg.rate_symbols))

        self.books: dict[str, Top] = {sym: Top() for sym in self.cfg.rate_symbols}
        self.mid_hist: dict[str, deque[float]] = {
            sym: deque(maxlen=max(16, self.cfg.corr_lookback_points)) for sym in self.cfg.rate_symbols
        }
        self.labels = MarketLabels(
            important_1=self.cfg.fed_hike,
            important_2=self.cfg.fed_hold,
            irrelevant=self.cfg.fed_cut,
            pair_corr=0.0,
        )

        self.live_order_birth: dict[str, float] = {}
        self.last_status_ts = 0.0
        self.last_cpi_trade_tick: Optional[int] = None
        self.current_tick: Optional[int] = None

        self._trace_fp = None
        self._trace_path: Optional[Path] = None

    def _trace(self, event_type: str, **payload: Any) -> None:
        if not self.cfg.trace_enabled:
            return
        if self._trace_fp is None:
            trace_dir = Path(self.cfg.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = trace_dir / f"new_PM2_heuristic_{int(time.time())}.jsonl"
            self._trace_fp = self._trace_path.open("a", encoding="utf-8")
        row = {"event_type": event_type, "ts": time.time(), **payload}
        self._trace_fp.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._trace_fp.flush()

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def refresh_book(self, symbol: str) -> Top:
        book = self.order_books.get(symbol)
        bids = []
        asks = []
        if book is not None:
            bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
            asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]
        best_bid = max(bids, key=lambda lvl: lvl[0]) if bids else None
        best_ask = min(asks, key=lambda lvl: lvl[0]) if asks else None
        top = Top(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.books[symbol] = top
        if top.mid is not None:
            self.mid_hist[symbol].append(top.mid)
        return top

    def top(self, symbol: str) -> Top:
        top = self.books.get(symbol)
        if top is None or (time.time() - top.updated_ts) > self.cfg.max_book_age_sec:
            return self.refresh_book(symbol)
        return top

    def has_open_order_on(self, symbol: str) -> bool:
        for _, order_data in self.open_orders.items():
            if order_data and str(order_data[0]) == symbol:
                return True
        return False

    def returns(self, symbol: str) -> list[float]:
        mids = list(self.mid_hist[symbol])
        if len(mids) < 3:
            return []
        rets = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        return rets[-self.cfg.corr_lookback_points :]

    def pearson(self, a: list[float], b: list[float]) -> Optional[float]:
        n = min(len(a), len(b))
        if n < self.cfg.min_corr_points:
            return None
        a = a[-n:]
        b = b[-n:]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((x - mean_b) ** 2 for x in b)
        if var_a <= 1e-9 or var_b <= 1e-9:
            return None
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        return cov / ((var_a * var_b) ** 0.5)

    def update_labels(self) -> None:
        syms = list(self.cfg.rate_symbols)
        pairs = [(syms[0], syms[1]), (syms[0], syms[2]), (syms[1], syms[2])]
        best_pair: Optional[tuple[str, str]] = None
        best_corr = -2.0
        for a, b in pairs:
            corr = self.pearson(self.returns(a), self.returns(b))
            if corr is None:
                continue
            if corr > best_corr:
                best_corr = corr
                best_pair = (a, b)
        if best_pair is None or best_corr < self.cfg.min_pair_corr:
            return

        a, b = best_pair
        c = next(sym for sym in syms if sym not in {a, b})
        self.labels = MarketLabels(important_1=a, important_2=b, irrelevant=c, pair_corr=best_corr, updated_ts=time.time())
        self._trace(
            "labels",
            tick=self.current_tick,
            important_1=a,
            important_2=b,
            irrelevant=c,
            pair_corr=best_corr,
        )

    def choose_counterpart(self, winner_symbol: str) -> str:
        imp_a = self.labels.important_1
        imp_b = self.labels.important_2
        if winner_symbol == imp_a:
            return imp_b
        if winner_symbol == imp_b:
            return imp_a
        corr_a = self.pearson(self.returns(winner_symbol), self.returns(imp_a))
        corr_b = self.pearson(self.returns(winner_symbol), self.returns(imp_b))
        score_a = -2.0 if corr_a is None else corr_a
        score_b = -2.0 if corr_b is None else corr_b
        return imp_a if score_a >= score_b else imp_b

    def clip_qty(self, symbol: str, side: Side, requested: int) -> int:
        requested = max(0, min(int(requested), self.cfg.max_order_size))
        if requested <= 0:
            return 0
        pos = self.get_position(symbol)
        if side == Side.BUY:
            room = self.cfg.max_abs_position - max(0, pos)
        else:
            room = self.cfg.max_abs_position - max(0, -pos)
        if room <= 0:
            return 0
        return max(0, min(requested, room))

    async def place_heuristic_pair(self, winner_symbol: str, loser_symbol: str, qty: int, reason: str, surprise: float) -> None:
        win_book = self.top(winner_symbol)
        lose_book = self.top(loser_symbol)
        if win_book.ask is None or lose_book.bid is None:
            self._trace("decision", tick=self.current_tick, reason="skip_no_liquidity", winner=winner_symbol, loser=loser_symbol)
            return
        if self.has_open_order_on(winner_symbol) or self.has_open_order_on(loser_symbol):
            self._trace("decision", tick=self.current_tick, reason="skip_open_order_exists", winner=winner_symbol, loser=loser_symbol)
            return

        buy_qty = self.clip_qty(winner_symbol, Side.BUY, qty)
        sell_qty = self.clip_qty(loser_symbol, Side.SELL, qty)
        pair_qty = min(buy_qty, sell_qty)
        if pair_qty <= 0:
            self._trace(
                "decision",
                tick=self.current_tick,
                reason="skip_risk_cap",
                winner=winner_symbol,
                loser=loser_symbol,
                buy_qty=buy_qty,
                sell_qty=sell_qty,
            )
            return

        buy_id = await self.place_order(winner_symbol, pair_qty, Side.BUY, int(win_book.ask))
        sell_id = await self.place_order(loser_symbol, pair_qty, Side.SELL, int(lose_book.bid))
        now = time.time()
        if buy_id is not None:
            self.live_order_birth[str(buy_id)] = now
        if sell_id is not None:
            self.live_order_birth[str(sell_id)] = now
        self._trace(
            "decision",
            tick=self.current_tick,
            reason=reason,
            winner=winner_symbol,
            loser=loser_symbol,
            qty=pair_qty,
            surprise=surprise,
            buy_px=win_book.ask,
            sell_px=lose_book.bid,
            labels=vars(self.labels),
            positions={sym: self.get_position(sym) for sym in self.cfg.rate_symbols},
        )

    async def maybe_cancel_stale_orders(self) -> None:
        now = time.time()
        stale_ids = [oid for oid, born in self.live_order_birth.items() if now - born >= self.cfg.order_stale_sec]
        for oid in stale_ids:
            try:
                await self.cancel_order(oid)
            except Exception:
                pass

    def print_status(self) -> None:
        now = time.time()
        if now - self.last_status_ts < self.cfg.status_interval_sec:
            return
        self.last_status_ts = now
        mids = {}
        for sym in self.cfg.rate_symbols:
            mids[sym] = self.top(sym).mid
        print(
            "[PM2] "
            f"labels=({self.labels.important_1},{self.labels.important_2},{self.labels.irrelevant}) "
            f"pair_corr={self.labels.pair_corr:.3f} mids={mids} "
            f"pos={{'{self.cfg.fed_hike}':{self.get_position(self.cfg.fed_hike)},'{self.cfg.fed_hold}':{self.get_position(self.cfg.fed_hold)},'{self.cfg.fed_cut}':{self.get_position(self.cfg.fed_cut)}}}"
        )

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            self.update_labels()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            self.update_labels()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self.live_order_birth.pop(str(order_id), None)
        self._trace("fill", tick=self.current_tick, order_id=str(order_id), qty=qty, price=price)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self.live_order_birth.pop(str(order_id), None)
        self._trace("reject", tick=self.current_tick, order_id=str(order_id), reason=reason)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        self.live_order_birth.pop(str(order_id), None)
        self._trace("cancel", tick=self.current_tick, order_id=str(order_id), success=success, error=error)

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")

        actual, forecast, source = parse_cpi(kind, new_data)
        if actual is None or forecast is None:
            return
        surprise = actual - forecast
        self._trace(
            "news",
            tick=self.current_tick,
            kind=kind,
            source=source,
            actual=actual,
            forecast=forecast,
            surprise=surprise,
            labels=vars(self.labels),
        )

        if self.last_cpi_trade_tick == self.current_tick:
            return
        if abs(surprise) <= self.cfg.cpi_trigger_abs:
            return

        self.update_labels()
        winner = self.cfg.fed_cut if surprise > 0 else self.cfg.fed_hike
        loser = self.choose_counterpart(winner)
        self.last_cpi_trade_tick = self.current_tick
        reason = "cpi_positive_buy_cut_sell_counterpart" if surprise > 0 else "cpi_negative_buy_hike_sell_counterpart"
        await self.place_heuristic_pair(winner, loser, self.cfg.trade_qty, reason, surprise)

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        self.current_tick = tick
        self._trace("round_end", tick=tick, winning_symbol=winning_symbol, positions={sym: self.get_position(sym) for sym in self.cfg.rate_symbols})

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        self.current_tick = tick
        self._trace("payout", tick=tick, user=user, amount=amount)

    async def trade(self):
        await asyncio.sleep(2.0)
        while True:
            try:
                for symbol in self.cfg.rate_symbols:
                    self.refresh_book(symbol)
                self.update_labels()
                await self.maybe_cancel_stale_orders()
                self.print_status()
            except Exception as exc:
                self._trace("loop_error", tick=self.current_tick, error=repr(exc))
            await asyncio.sleep(self.cfg.loop_sleep_sec)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    client = NewPM2HeuristicClient(
        env_str("UTC_HOST", "34.197.188.76:3333"),
        env_str("UTC_USERNAME", "uiuc"),
        env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
