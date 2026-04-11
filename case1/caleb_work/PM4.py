from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
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
    return default if value is None or value.strip() == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


@dataclass(frozen=True)
class Config:
    fed_hike: str
    fed_hold: str
    fed_cut: str
    cpi_trigger_abs: float
    qty: int
    max_abs_position: int
    order_ttl_sec: float
    hold_sec: float
    event_cooldown_sec: float
    loop_sleep_sec: float
    status_sec: float
    trace_dir: str

    @property
    def symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


def load_config() -> Config:
    return Config(
        fed_hike=env_str("PM4_FED_HIKE_SYMBOL", "R_HIKE"),
        fed_hold=env_str("PM4_FED_HOLD_SYMBOL", "R_HOLD"),
        fed_cut=env_str("PM4_FED_CUT_SYMBOL", "R_CUT"),
        cpi_trigger_abs=env_float("PM4_CPI_TRIGGER_ABS", 0.0003),
        qty=env_int("PM4_QTY", 40),
        max_abs_position=env_int("PM4_MAX_ABS_POSITION", 200),
        order_ttl_sec=env_float("PM4_ORDER_TTL_SEC", 0.8),
        hold_sec=env_float("PM4_HOLD_SEC", 2.0),
        event_cooldown_sec=env_float("PM4_EVENT_COOLDOWN_SEC", 2.0),
        loop_sleep_sec=env_float("PM4_LOOP_SLEEP_SEC", 0.15),
        status_sec=env_float("PM4_STATUS_SEC", 2.0),
        trace_dir=env_str("PM4_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
    )


def parse_cpi(kind: Any, new_data: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
    if kind == "structured":
        if str(new_data.get("structured_subtype") or "") == "cpi_print":
            if "actual" in new_data and "forecast" in new_data:
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
        m = re.search(pattern, content, flags=re.IGNORECASE)
        if m:
            return float(m.group(1)), float(m.group(2)), "unstructured_text"
    return None, None, "none"


class PM4Client(XChangeClient):
    def __init__(self, host: str, username: str, password: str, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        super().__init__(host, username, password, silent=False, symbols=list(self.cfg.symbols))

        self.current_tick: Optional[int] = None
        self._snapshot_seen = asyncio.Event()
        self._trade_lock = asyncio.Lock()
        self._last_status_ts = 0.0
        self._last_cpi_ts = 0.0

        self.active = False
        self.exit_due_ts = 0.0
        self.last_pair: tuple[str, str, str] | None = None

        self.live_orders: dict[str, tuple[str, str, float, bool]] = {}

        trace_dir = Path(self.cfg.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"PM4_{int(time.time())}.jsonl"
        self._trace_fp = self.trace_path.open("a", encoding="utf-8")

    def _trace(self, event_type: str, **payload: Any) -> None:
        self._trace_fp.write(json.dumps({"event_type": event_type, "ts": time.time(), **payload}, ensure_ascii=True) + "\n")
        self._trace_fp.flush()

    def get_pos(self, sym: str) -> int:
        return int(self.positions.get(sym, 0))

    def top(self, sym: str) -> tuple[Optional[int], Optional[int]]:
        book = self.order_books.get(sym)
        if book is None:
            return None, None
        bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
        asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]
        bid = max(bids, key=lambda x: x[0])[0] if bids else None
        ask = min(asks, key=lambda x: x[0])[0] if asks else None
        return bid, ask

    def probs(self) -> Optional[tuple[float, float, float]]:
        mids = {}
        for s in self.cfg.symbols:
            bid, ask = self.top(s)
            if bid is None or ask is None:
                return None
            mids[s] = (bid + ask) / 2.0
        total = mids[self.cfg.fed_hike] + mids[self.cfg.fed_hold] + mids[self.cfg.fed_cut]
        if total <= 1e-9:
            return None
        return mids[self.cfg.fed_hike] / total, mids[self.cfg.fed_hold] / total, mids[self.cfg.fed_cut] / total

    def choose_pair(self, surprise: float) -> Optional[tuple[str, str, str, float, float, float]]:
        pq = self.probs()
        if pq is None:
            return None
        qh, qo, qc = pq
        if surprise > 0:
            # Positive CPI surprise usually lifts HIKE, except when HIKE is near-zero
            # and CUT is saturated; in that regime HOLD is typically the fast repricer.
            if qh <= 0.05 and qc >= 0.75:
                winner = self.cfg.fed_hold
                loser = self.cfg.fed_cut
            else:
                winner = self.cfg.fed_hike
                loser = self.cfg.fed_hold if qo >= qc else self.cfg.fed_cut
        else:
            cut = (qc >= 0.07) or (qo >= 0.70) or (abs(surprise) >= 0.0005 and qh >= 0.60)
            winner = self.cfg.fed_cut if cut else self.cfg.fed_hold
            loser = self.cfg.fed_hold if winner == self.cfg.fed_cut else self.cfg.fed_hike
        irrelevant = next(s for s in self.cfg.symbols if s not in {winner, loser})
        return winner, loser, irrelevant, qh, qo, qc

    def clip_qty(self, sym: str, side: Side, qty: int) -> int:
        qty = max(0, min(int(qty), self.cfg.qty))
        pos = self.get_pos(sym)
        if side == Side.BUY:
            room = self.cfg.max_abs_position - max(0, pos)
        else:
            room = self.cfg.max_abs_position - max(0, -pos)
        return max(0, min(qty, room))

    async def submit_aggressive(self, sym: str, side: Side, qty: int, reason: str) -> bool:
        qty = self.clip_qty(sym, side, qty)
        if qty <= 0:
            return False
        bid, ask = self.top(sym)
        px = ask if side == Side.BUY else bid
        if px is None:
            return False
        oid = await self.place_order(sym, qty, side, int(px))
        if oid is None:
            return False
        self.live_orders[str(oid)] = (sym, "BUY" if side == Side.BUY else "SELL", time.time(), False)
        self._trace("order_submitted", tick=self.current_tick, reason=reason, order_id=str(oid), symbol=sym, side=("BUY" if side == Side.BUY else "SELL"), qty=qty, px=int(px))
        return True

    async def cancel_stale(self) -> None:
        now = time.time()
        for oid, (sym, side, born, canceling) in list(self.live_orders.items()):
            if canceling:
                continue
            if now - born < self.cfg.order_ttl_sec:
                continue
            try:
                await self.cancel_order(oid)
                self.live_orders[oid] = (sym, side, born, True)
            except Exception:
                pass

    def is_flat(self) -> bool:
        return all(self.get_pos(s) == 0 for s in self.cfg.symbols)

    async def flatten_all(self, reason: str) -> bool:
        if self.open_orders:
            return False
        acted = False
        for s in self.cfg.symbols:
            p = self.get_pos(s)
            if p == 0:
                continue
            side = Side.SELL if p > 0 else Side.BUY
            acted = (await self.submit_aggressive(s, side, min(abs(p), self.cfg.qty), reason)) or acted
        return acted

    async def enter_cpi_trade(self, surprise: float, source: str) -> None:
        pair = self.choose_pair(surprise)
        if pair is None:
            self._trace("decision", tick=self.current_tick, reason="skip_missing_books", surprise=surprise)
            return
        winner, loser, irrelevant, qh, qo, qc = pair
        if self.open_orders:
            self._trace("decision", tick=self.current_tick, reason="skip_open_orders", surprise=surprise)
            return
        b1 = await self.submit_aggressive(winner, Side.BUY, self.cfg.qty, "cpi_entry")
        b2 = await self.submit_aggressive(loser, Side.SELL, self.cfg.qty, "cpi_entry")
        if b1 or b2:
            self.active = True
            self.exit_due_ts = time.time() + self.cfg.hold_sec
            self.last_pair = (winner, loser, irrelevant)
            self._last_cpi_ts = time.time()
            self._trace(
                "cpi_trade",
                tick=self.current_tick,
                surprise=surprise,
                source=source,
                winner=winner,
                loser=loser,
                irrelevant=irrelevant,
                q_hike=qh,
                q_hold=qo,
                q_cut=qc,
                exit_due_ts=self.exit_due_ts,
            )

    async def on_timer(self) -> None:
        await self.cancel_stale()
        if not self._snapshot_seen.is_set():
            return
        if not self.active:
            return
        if time.time() < self.exit_due_ts:
            return
        await self.flatten_all("cpi_exit_flatten")
        if self.is_flat() and not self.open_orders:
            self.active = False
            self._trace("transition", tick=self.current_tick, reason="back_to_flat")

    def print_status(self) -> None:
        now = time.time()
        if now - self._last_status_ts < self.cfg.status_sec:
            return
        self._last_status_ts = now
        q = self.probs()
        qstr = "n/a" if q is None else f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"
        print(
            f"[PM4] tick={self.current_tick} active={self.active} q={qstr} "
            f"pos={{'{self.cfg.fed_hike}':{self.get_pos(self.cfg.fed_hike)},'{self.cfg.fed_hold}':{self.get_pos(self.cfg.fed_hold)},'{self.cfg.fed_cut}':{self.get_pos(self.cfg.fed_cut)}}} "
            f"open_orders={len(self.open_orders)} pair={self.last_pair}"
        )

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        if not self._snapshot_seen.is_set():
            self._snapshot_seen.set()
            self._trace("position_snapshot", tick=self.current_tick, positions={s: self.get_pos(s) for s in self.cfg.symbols})

    async def bot_handle_news(self, news_release: dict) -> None:
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")
        actual, forecast, source = parse_cpi(kind, new_data)
        if actual is None or forecast is None:
            return
        surprise = actual - forecast
        self._trace("news", tick=self.current_tick, kind=kind, source=source, actual=actual, forecast=forecast, surprise=surprise)
        if abs(surprise) < self.cfg.cpi_trigger_abs:
            return
        if (time.time() - self._last_cpi_ts) < self.cfg.event_cooldown_sec:
            self._trace("decision", tick=self.current_tick, reason="skip_event_cooldown", surprise=surprise)
            return
        async with self._trade_lock:
            await self.enter_cpi_trade(surprise, source)

    async def bot_handle_book_update(self, symbol: str) -> None:
        return

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int) -> None:
        return

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        self.live_orders.pop(str(order_id), None)
        self._trace("fill", tick=self.current_tick, order_id=str(order_id), qty=qty, price=price)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self.live_orders.pop(str(order_id), None)
        self._trace("reject", tick=self.current_tick, order_id=str(order_id), reason=reason)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        self.live_orders.pop(str(order_id), None)
        self._trace("cancel", tick=self.current_tick, order_id=str(order_id), success=success, error=error)

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool) -> None:
        return

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int) -> None:
        self.current_tick = tick
        self._trace("round_end", tick=tick, winner=winning_symbol)

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int) -> None:
        self.current_tick = tick
        self._trace("payout", tick=tick, user=user, amount=amount)

    async def timer_loop(self) -> None:
        await self._snapshot_seen.wait()
        while True:
            async with self._trade_lock:
                await self.on_timer()
                self.print_status()
            await asyncio.sleep(self.cfg.loop_sleep_sec)

    async def start(self) -> None:
        asyncio.create_task(self.timer_loop())
        await self.connect()


async def main() -> None:
    client = PM4Client(
        env_str("UTC_HOST", "34.197.188.76:3333"),
        env_str("UTC_USERNAME", "uiuc"),
        env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
