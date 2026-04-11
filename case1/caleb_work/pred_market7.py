from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient


@dataclass(frozen=True)
class BotConfig:
    fed_hike: str = "R_HIKE"
    fed_hold: str = "R_HOLD"
    fed_cut: str = "R_CUT"

    payout_scale: float = 1000.0
    event_off_secs: float = 5.0
    ewma_tau_secs: float = 60.0
    dead_tail_mid: float = 120.0
    dead_tail_gap: float = 180.0

    pair_entry_z: float = 2.0
    pair_exit_z: float = 0.5
    pair_entry_abs_dev: float = 24.0
    pair_spread_cap: int = 12
    pair_quote_size: int = 20
    pair_max_leg: int = 80
    pair_max_hold_secs: float = 6.0
    pair_hedge_timeout_secs: float = 0.50

    package_edge_threshold: float = 24.0
    package_leg_size: int = 20
    package_max_leg: int = 60

    symbol_hard_position_limit: int = 150
    max_active_orders_per_symbol_side: int = 1
    order_stale_secs: float = 0.50
    min_pair_obs: int = 20
    min_pair_sigma: float = 4.0

    @property
    def symbols(self) -> tuple[str, str, str]:
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
        return None

    @property
    def spread(self) -> Optional[int]:
        if self.bid is None or self.ask is None:
            return None
        return int(self.ask - self.bid)

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None


@dataclass
class TrackedOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    strategy: str
    role: str
    reason: str
    group_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class PairStats:
    mean: float = 0.0
    var: float = 0.0
    last_ts: float = 0.0
    obs: int = 0


@dataclass
class PairSignal:
    left: str
    right: str
    spread: float
    mean: float
    sigma: float
    zscore: Optional[float]
    raw_deviation: float
    eligible: bool
    side: Optional[str]
    winner: str
    challenger: str
    dead: str
    dead_tail_regime: bool


@dataclass
class PackageSignal:
    buy_edge: Optional[float]
    sell_edge: Optional[float]
    action: Optional[str]


@dataclass
class PairGroup:
    group_id: str
    left: str
    right: str
    left_side: Side
    right_side: Side
    qty: int
    created_at: float = field(default_factory=time.time)
    left_order_id: Optional[str] = None
    right_order_id: Optional[str] = None
    left_filled: int = 0
    right_filled: int = 0
    hedge_deadline: Optional[float] = None
    hedge_symbol: Optional[str] = None
    hedge_side: Optional[Side] = None
    hedge_qty: int = 0
    hedge_order_id: Optional[str] = None


@dataclass
class MarketState:
    books: dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)
    last_macro_event_ts: float = 0.0
    pair_stats: dict[tuple[str, str], PairStats] = field(default_factory=dict)
    pair_positions: dict[str, int] = field(default_factory=lambda: {"R_CUT": 0, "R_HOLD": 0, "R_HIKE": 0})
    package_positions: dict[str, int] = field(default_factory=lambda: {"R_CUT": 0, "R_HOLD": 0, "R_HIKE": 0})
    pair_entry_left: Optional[str] = None
    pair_entry_right: Optional[str] = None
    pair_entry_ts: float = 0.0
    pair_groups: dict[str, PairGroup] = field(default_factory=dict)
    next_group_seq: int = 1

    def next_group_id(self) -> str:
        group_id = f"g{self.next_group_seq}"
        self.next_group_seq += 1
        return group_id

    def pair_flat(self) -> bool:
        return all(v == 0 for v in self.pair_positions.values())

    def clear_pair_state(self) -> None:
        self.pair_entry_left = None
        self.pair_entry_right = None
        self.pair_entry_ts = 0.0


class OrderManager:
    def __init__(self, client: "PredictionMarketsBot7", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def has_live_order(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[Side] = None,
        strategy: Optional[str] = None,
        role: Optional[str] = None,
    ) -> bool:
        for order in self.state.live_orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if side is not None and order.side != side:
                continue
            if strategy is not None and order.strategy != strategy:
                continue
            if role is not None and order.role != role:
                continue
            return True
        return False

    def pending_same_side_qty(self, symbol: str, side: Side) -> int:
        return sum(order.qty for order in self.state.live_orders.values() if order.symbol == symbol and order.side == side)

    async def cancel_order_if_present(self, order_id: str) -> None:
        order_key = str(order_id)
        if order_key in self.state.pending_cancels or order_key not in self.state.live_orders:
            return
        self.state.pending_cancels.add(order_key)
        try:
            await self.client.cancel_order(order_id)
        finally:
            self.state.pending_cancels.discard(order_key)

    async def cancel_orders(
        self,
        *,
        strategy: Optional[str] = None,
        exclude_roles: tuple[str, ...] = (),
    ) -> None:
        to_cancel = [
            order.order_id
            for order in self.state.live_orders.values()
            if (strategy is None or order.strategy == strategy) and order.role not in exclude_roles
        ]
        for order_id in to_cancel:
            await self.cancel_order_if_present(order_id)

    async def cancel_stale_orders(self) -> None:
        now = time.time()
        stale = [order.order_id for order in self.state.live_orders.values() if now - order.created_at >= self.cfg.order_stale_secs]
        for order_id in stale:
            await self.cancel_order_if_present(order_id)

    async def place_tracked_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: Side,
        price: int,
        strategy: str,
        role: str,
        reason: str,
        group_id: Optional[str] = None,
    ) -> Optional[str]:
        if qty <= 0 or self.has_live_order(symbol=symbol, side=side):
            return None
        side_count = sum(1 for order in self.state.live_orders.values() if order.symbol == symbol and order.side == side)
        if side_count >= self.cfg.max_active_orders_per_symbol_side:
            return None
        order_id = await self.client.place_order(symbol, int(qty), side, int(price))
        if order_id is None:
            return None
        tracked = TrackedOrder(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            qty=int(qty),
            price=int(price),
            strategy=strategy,
            role=role,
            reason=reason,
            group_id=group_id,
        )
        self.state.live_orders[str(order_id)] = tracked
        self.client._trace(
            "order_submit",
            tick=self.client.current_tick,
            symbol=symbol,
            side=side.name,
            qty=qty,
            price=price,
            strategy=strategy,
            role=role,
            reason=reason,
            group_id=group_id,
            **self.client.positions_payload(),
        )
        return str(order_id)

    def sync_fill(self, order_id: str) -> Optional[TrackedOrder]:
        tracked = self.state.live_orders.get(str(order_id))
        if tracked is None:
            return None
        if order_id in self.client.open_orders:
            remaining_qty = int(self.client.open_orders[order_id][1])
            tracked.qty = remaining_qty
            if remaining_qty <= 0:
                self.state.live_orders.pop(str(order_id), None)
        else:
            self.state.live_orders.pop(str(order_id), None)
        return tracked

    def sync_rejected(self, order_id: str) -> Optional[TrackedOrder]:
        return self.state.live_orders.pop(str(order_id), None)

    def sync_cancel_response(self, order_id: str, success: bool) -> Optional[TrackedOrder]:
        if success:
            return self.state.live_orders.pop(str(order_id), None)
        return self.state.live_orders.get(str(order_id))


class RiskManager:
    def __init__(self, client: "PredictionMarketsBot7", cfg: BotConfig, state: MarketState, orders: OrderManager):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders

    def clip_qty(self, symbol: str, side: Side, desired_qty: int) -> int:
        pos = self.client.get_position(symbol)
        pending = self.orders.pending_same_side_qty(symbol, side)
        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        remaining = self.cfg.symbol_hard_position_limit - same_dir_pos - pending
        return max(0, min(int(desired_qty), remaining))


class PredictionMarketsBot7(XChangeClient):
    TRACE_ENABLED = True

    def __init__(self, host: str, username: str, password: str, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or BotConfig()
        super().__init__(host, username, password, symbols=list(self.cfg.symbols))
        self.state = MarketState()
        self.current_tick: Optional[int] = None
        self.positions_ready = False
        self._decision_lock = asyncio.Lock()
        self._trace_file = None
        self._trace_path: Optional[Path] = None
        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)

    def handle_position_snapshot(self, msg) -> None:
        super().handle_position_snapshot(msg)
        self.positions_ready = True

    def _trace_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Side):
            return value.name
        if isinstance(value, dict):
            return {str(k): self._trace_jsonable(v) for k, v in value.items()}
        if isinstance(value, set):
            return [self._trace_jsonable(v) for v in sorted(value)]
        if isinstance(value, (list, tuple)):
            return [self._trace_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return self._trace_jsonable(vars(value))
        return repr(value)

    def _trace(self, event_type: str, **kwargs) -> None:
        if not self.TRACE_ENABLED:
            return
        if self._trace_file is None:
            logs_dir = Path(__file__).resolve().parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = logs_dir / f"pred_market7_{int(time.time())}.jsonl"
            self._trace_file = self._trace_path.open("a", encoding="utf-8")
        payload = {"event_type": event_type, "timestamp": time.time(), **kwargs}
        self._trace_file.write(json.dumps(self._trace_jsonable(payload), ensure_ascii=True) + "\n")
        self._trace_file.flush()

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def positions_payload(self) -> dict[str, Any]:
        return {
            "positions": {symbol: self.get_position(symbol) for symbol in self.cfg.symbols},
            "pair_positions": dict(self.state.pair_positions),
            "package_positions": dict(self.state.package_positions),
            "cash": int(self.positions.get("cash", 0)),
        }

    def refresh_book(self, symbol: str) -> TopOfBook:
        book = self.order_books.get(symbol)
        bids = [(int(px), int(qty)) for px, qty in book.bids.items()] if book is not None else []
        asks = [(int(px), int(qty)) for px, qty in book.asks.items()] if book is not None else []
        bids = [(px, qty) for px, qty in bids if qty > 0]
        asks = [(px, qty) for px, qty in asks if qty > 0]
        best_bid = max(bids, key=lambda level: level[0]) if bids else None
        best_ask = min(asks, key=lambda level: level[0]) if asks else None
        snap = TopOfBook(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.state.books[symbol] = snap
        return snap

    def refresh_all_books(self) -> None:
        for symbol in self.cfg.symbols:
            self.refresh_book(symbol)

    def top(self, symbol: str) -> TopOfBook:
        if symbol not in self.state.books:
            return self.refresh_book(symbol)
        return self.state.books[symbol]

    def mid(self, symbol: str) -> Optional[float]:
        return self.top(symbol).mid

    def event_off_ready(self) -> bool:
        return time.time() - self.state.last_macro_event_ts >= self.cfg.event_off_secs

    def record_macro_event(self) -> None:
        self.state.last_macro_event_ts = time.time()

    def current_state(self) -> Optional[dict[str, Any]]:
        mids = {symbol: self.mid(symbol) for symbol in self.cfg.symbols}
        books = {symbol: self.top(symbol) for symbol in self.cfg.symbols}
        if any(mids[symbol] is None for symbol in self.cfg.symbols):
            return None
        ordered = sorted(mids.items(), key=lambda item: item[1], reverse=True)
        winner, winner_mid = ordered[0]
        challenger, challenger_mid = ordered[1]
        dead, dead_mid = ordered[2]
        dead_tail_regime = dead_mid <= self.cfg.dead_tail_mid or (challenger_mid - dead_mid) >= self.cfg.dead_tail_gap
        if dead_tail_regime:
            left, right = winner, challenger
        else:
            left, right = self.cfg.fed_hike, self.cfg.fed_cut
        return {
            "winner": winner,
            "challenger": challenger,
            "dead": dead,
            "mids": mids,
            "books": books,
            "dead_tail_regime": dead_tail_regime,
            "selected_pair": (left, right),
        }

    def update_pair_stats(self, left: str, right: str, spread: float) -> PairStats:
        key = (left, right)
        stats = self.state.pair_stats.setdefault(key, PairStats())
        now = time.time()
        if stats.obs == 0:
            stats.mean = spread
            stats.var = 0.0
            stats.last_ts = now
            stats.obs = 1
            return stats
        dt = max(0.001, now - stats.last_ts)
        alpha = 1.0 - math.exp(-dt / self.cfg.ewma_tau_secs)
        diff = spread - stats.mean
        stats.mean += alpha * diff
        stats.var = max(0.0, (1.0 - alpha) * (stats.var + alpha * diff * diff))
        stats.last_ts = now
        stats.obs += 1
        return stats

    def compute_pair_signal(self, state: dict[str, Any]) -> PairSignal:
        left, right = state["selected_pair"]
        left_book = self.top(left)
        right_book = self.top(right)
        spread = state["mids"][left] - state["mids"][right]
        stats = self.update_pair_stats(left, right, spread)
        sigma = math.sqrt(max(0.0, stats.var))
        zscore = None
        if stats.obs >= self.cfg.min_pair_obs and sigma >= self.cfg.min_pair_sigma:
            zscore = (spread - stats.mean) / sigma
        raw_deviation = abs(spread - stats.mean)
        eligible = (
            self.event_off_ready()
            and left_book.two_sided
            and right_book.two_sided
            and (left_book.spread or 10**9) <= self.cfg.pair_spread_cap
            and (right_book.spread or 10**9) <= self.cfg.pair_spread_cap
            and zscore is not None
            and abs(zscore) >= self.cfg.pair_entry_z
            and raw_deviation >= self.cfg.pair_entry_abs_dev
        )
        side = None
        if eligible and zscore is not None:
            side = "short_spread" if zscore >= self.cfg.pair_entry_z else "long_spread"
        return PairSignal(
            left=left,
            right=right,
            spread=spread,
            mean=stats.mean,
            sigma=sigma,
            zscore=zscore,
            raw_deviation=raw_deviation,
            eligible=eligible,
            side=side,
            winner=state["winner"],
            challenger=state["challenger"],
            dead=state["dead"],
            dead_tail_regime=state["dead_tail_regime"],
        )

    def compute_package_signal(self) -> PackageSignal:
        asks = [self.top(symbol).ask for symbol in self.cfg.symbols]
        bids = [self.top(symbol).bid for symbol in self.cfg.symbols]
        buy_edge = None if any(v is None for v in asks) else self.cfg.payout_scale - sum(int(v) for v in asks)
        sell_edge = None if any(v is None for v in bids) else sum(int(v) for v in bids) - self.cfg.payout_scale
        action = None
        if buy_edge is not None and buy_edge >= self.cfg.package_edge_threshold:
            action = "buy"
        elif sell_edge is not None and sell_edge >= self.cfg.package_edge_threshold:
            action = "sell"
        return PackageSignal(buy_edge=buy_edge, sell_edge=sell_edge, action=action)

    def inside_buy_price(self, symbol: str) -> Optional[int]:
        book = self.top(symbol)
        if book.bid is None or book.ask is None:
            return None
        if book.ask - book.bid >= 2:
            return int(book.bid + 1)
        return int(book.bid)

    def inside_sell_price(self, symbol: str) -> Optional[int]:
        book = self.top(symbol)
        if book.bid is None or book.ask is None:
            return None
        if book.ask - book.bid >= 2:
            return int(book.ask - 1)
        return int(book.ask)

    def aggressive_price(self, symbol: str, side: Side) -> Optional[int]:
        book = self.top(symbol)
        if side == Side.BUY:
            return None if book.ask is None else int(book.ask)
        return None if book.bid is None else int(book.bid)

    def active_pair_key(self) -> Optional[tuple[str, str]]:
        if self.state.pair_entry_left is None or self.state.pair_entry_right is None:
            return None
        return (self.state.pair_entry_left, self.state.pair_entry_right)

    async def maybe_cancel_for_macro_gate(self) -> bool:
        if self.event_off_ready():
            return False
        await self.order_manager.cancel_orders()
        return await self.flatten_pair_positions("macro_event")

    async def flatten_pair_positions(self, reason: str) -> bool:
        acted = False
        for symbol, pair_pos in self.state.pair_positions.items():
            if pair_pos == 0:
                continue
            side = Side.SELL if pair_pos > 0 else Side.BUY
            if self.order_manager.has_live_order(symbol=symbol, side=side):
                continue
            price = self.aggressive_price(symbol, side)
            if price is None:
                continue
            qty = self.risk_manager.clip_qty(symbol, side, min(abs(pair_pos), self.cfg.pair_quote_size))
            if qty <= 0:
                continue
            order_id = await self.order_manager.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                strategy="pair",
                role="pair_exit",
                reason=reason,
            )
            acted = acted or order_id is not None
        return acted

    async def maybe_repair_package_positions(self) -> bool:
        values = [self.state.package_positions[s] for s in self.cfg.symbols]
        nonzero = [v for v in values if v != 0]
        if not nonzero:
            return False
        if all(v > 0 for v in nonzero):
            base = min(nonzero)
        elif all(v < 0 for v in nonzero):
            base = -min(abs(v) for v in nonzero)
        else:
            base = 0
        acted = False
        for symbol in self.cfg.symbols:
            pos = self.state.package_positions[symbol]
            target = base
            delta = target - pos
            if delta == 0:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            if self.order_manager.has_live_order(symbol=symbol, side=side):
                continue
            price = self.aggressive_price(symbol, side)
            if price is None:
                continue
            qty = self.risk_manager.clip_qty(symbol, side, min(abs(delta), self.cfg.package_leg_size))
            if qty <= 0:
                continue
            order_id = await self.order_manager.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                strategy="package",
                role="package_repair",
                reason="package_repair",
            )
            acted = acted or order_id is not None
        return acted

    async def maybe_process_pair_groups(self) -> bool:
        now = time.time()
        acted = False
        finished: list[str] = []
        for group_id, group in list(self.state.pair_groups.items()):
            left_live = group.left_order_id is not None and group.left_order_id in self.state.live_orders
            right_live = group.right_order_id is not None and group.right_order_id in self.state.live_orders
            hedge_live = group.hedge_order_id is not None and group.hedge_order_id in self.state.live_orders
            if left_live or right_live:
                if group.left_filled != group.right_filled and group.hedge_deadline is None:
                    group.hedge_deadline = now + self.cfg.pair_hedge_timeout_secs
            if group.hedge_deadline is not None and now >= group.hedge_deadline:
                if left_live and group.left_order_id is not None:
                    await self.order_manager.cancel_order_if_present(group.left_order_id)
                    acted = True
                if right_live and group.right_order_id is not None:
                    await self.order_manager.cancel_order_if_present(group.right_order_id)
                    acted = True
                diff = group.left_filled - group.right_filled
                if diff > 0:
                    group.hedge_symbol = group.right
                    group.hedge_side = group.right_side
                    group.hedge_qty = diff
                elif diff < 0:
                    group.hedge_symbol = group.left
                    group.hedge_side = group.left_side
                    group.hedge_qty = -diff
                group.hedge_deadline = None
            if group.hedge_qty > 0 and group.hedge_symbol is not None and group.hedge_side is not None and not hedge_live:
                if not self.order_manager.has_live_order(symbol=group.hedge_symbol, side=group.hedge_side):
                    price = self.aggressive_price(group.hedge_symbol, group.hedge_side)
                    if price is not None:
                        qty = self.risk_manager.clip_qty(group.hedge_symbol, group.hedge_side, min(group.hedge_qty, self.cfg.pair_quote_size))
                        if qty > 0:
                            order_id = await self.order_manager.place_tracked_order(
                                symbol=group.hedge_symbol,
                                qty=qty,
                                side=group.hedge_side,
                                price=price,
                                strategy="pair",
                                role="pair_hedge",
                                reason="pair_hedge",
                                group_id=group_id,
                            )
                            if order_id is not None:
                                group.hedge_order_id = order_id
                                acted = True
            if not left_live and not right_live and not hedge_live and group.left_filled == group.right_filled:
                finished.append(group_id)
        for group_id in finished:
            self.state.pair_groups.pop(group_id, None)
        if not self.state.pair_groups and self.state.pair_flat():
            self.state.clear_pair_state()
        return acted

    async def maybe_enter_pair(self, signal: PairSignal) -> bool:
        if not signal.eligible or signal.side is None:
            return False
        if self.state.pair_groups:
            return False
        if any(self.order_manager.has_live_order(symbol=symbol, strategy="pair") for symbol in self.cfg.symbols):
            return False
        current_abs = max(abs(self.state.pair_positions[signal.left]), abs(self.state.pair_positions[signal.right]))
        if current_abs >= self.cfg.pair_max_leg:
            return False
        qty = min(self.cfg.pair_quote_size, self.cfg.pair_max_leg - current_abs)
        if qty <= 0:
            return False
        if signal.side == "short_spread":
            left_side, right_side = Side.SELL, Side.BUY
            left_price = self.inside_sell_price(signal.left)
            right_price = self.inside_buy_price(signal.right)
        else:
            left_side, right_side = Side.BUY, Side.SELL
            left_price = self.inside_buy_price(signal.left)
            right_price = self.inside_sell_price(signal.right)
        if left_price is None or right_price is None:
            return False
        left_qty = self.risk_manager.clip_qty(signal.left, left_side, qty)
        right_qty = self.risk_manager.clip_qty(signal.right, right_side, qty)
        qty = min(left_qty, right_qty)
        if qty <= 0:
            return False
        group_id = self.state.next_group_id()
        group = PairGroup(
            group_id=group_id,
            left=signal.left,
            right=signal.right,
            left_side=left_side,
            right_side=right_side,
            qty=qty,
        )
        left_order_id = await self.order_manager.place_tracked_order(
            symbol=signal.left,
            qty=qty,
            side=left_side,
            price=left_price,
            strategy="pair",
            role="pair_entry",
            reason=signal.side,
            group_id=group_id,
        )
        right_order_id = await self.order_manager.place_tracked_order(
            symbol=signal.right,
            qty=qty,
            side=right_side,
            price=right_price,
            strategy="pair",
            role="pair_entry",
            reason=signal.side,
            group_id=group_id,
        )
        if left_order_id is None and right_order_id is None:
            return False
        group.left_order_id = left_order_id
        group.right_order_id = right_order_id
        self.state.pair_groups[group_id] = group
        return True

    async def maybe_enter_package(self, signal: PackageSignal) -> bool:
        if signal.action is None:
            return False
        if not self.event_off_ready():
            return False
        if not self.state.pair_flat() or self.state.pair_groups:
            return False
        await self.maybe_repair_package_positions()
        if any(self.order_manager.has_live_order(strategy="package", symbol=symbol) for symbol in self.cfg.symbols):
            return False
        if signal.action == "buy":
            side = Side.BUY
            target_prices = {symbol: self.top(symbol).ask for symbol in self.cfg.symbols}
        else:
            side = Side.SELL
            target_prices = {symbol: self.top(symbol).bid for symbol in self.cfg.symbols}
        if any(px is None for px in target_prices.values()):
            return False
        top_qtys = {
            symbol: (self.top(symbol).ask_qty if side == Side.BUY else self.top(symbol).bid_qty)
            for symbol in self.cfg.symbols
        }
        if any(qty < self.cfg.package_leg_size for qty in top_qtys.values()):
            return False
        desired_qty = self.cfg.package_leg_size
        for symbol in self.cfg.symbols:
            overlay = self.state.package_positions[symbol]
            same_dir = max(0, overlay) if side == Side.BUY else max(0, -overlay)
            desired_qty = min(desired_qty, self.cfg.package_max_leg - same_dir)
        desired_qty = min(
            desired_qty,
            *(self.risk_manager.clip_qty(symbol, side, desired_qty) for symbol in self.cfg.symbols),
        )
        if desired_qty <= 0:
            return False
        acted = False
        for symbol in self.cfg.symbols:
            order_id = await self.order_manager.place_tracked_order(
                symbol=symbol,
                qty=desired_qty,
                side=side,
                price=int(target_prices[symbol]),
                strategy="package",
                role="package_entry",
                reason=f"package_{signal.action}",
            )
            acted = acted or order_id is not None
        return acted

    def pair_exit_due(self, signal: Optional[PairSignal]) -> tuple[bool, str]:
        if self.state.pair_flat():
            return False, "flat"
        if not self.event_off_ready():
            return True, "macro_event"
        if self.state.pair_entry_ts > 0.0 and time.time() - self.state.pair_entry_ts >= self.cfg.pair_max_hold_secs:
            return True, "time_stop"
        pair_key = self.active_pair_key()
        if pair_key is None:
            return True, "state_unknown"
        left, right = pair_key
        left_book = self.top(left)
        right_book = self.top(right)
        if not left_book.two_sided or not right_book.two_sided:
            return True, "one_sided"
        if signal is None or signal.left != left or signal.right != right:
            return True, "pair_shift"
        if signal.zscore is not None and abs(signal.zscore) <= self.cfg.pair_exit_z:
            return True, "zscore_exit"
        return False, "hold"

    def pair_position_opened(self) -> bool:
        return not self.state.pair_flat()

    def apply_overlay_fill(self, tracked: TrackedOrder, fill_qty: int) -> None:
        signed_qty = fill_qty if tracked.side == Side.BUY else -fill_qty
        if tracked.strategy == "pair":
            self.state.pair_positions[tracked.symbol] += signed_qty
            if tracked.role in {"pair_entry", "pair_hedge"} and tracked.group_id is not None:
                group = self.state.pair_groups.get(tracked.group_id)
                if group is not None:
                    if tracked.role == "pair_hedge" and group.hedge_symbol == tracked.symbol:
                        if tracked.symbol == group.left:
                            group.left_filled += fill_qty
                        elif tracked.symbol == group.right:
                            group.right_filled += fill_qty
                        group.hedge_qty = max(0, group.hedge_qty - fill_qty)
                    elif tracked.symbol == group.left:
                        group.left_filled += fill_qty
                    elif tracked.symbol == group.right:
                        group.right_filled += fill_qty
            if self.pair_position_opened() and self.state.pair_entry_left is None:
                group = self.state.pair_groups.get(tracked.group_id or "")
                if group is not None:
                    self.state.pair_entry_left = group.left
                    self.state.pair_entry_right = group.right
                    self.state.pair_entry_ts = group.created_at
            if self.state.pair_flat():
                self.state.clear_pair_state()
        elif tracked.strategy == "package":
            self.state.package_positions[tracked.symbol] += signed_qty

    def pair_group_for_order(self, tracked: Optional[TrackedOrder]) -> Optional[PairGroup]:
        if tracked is None or tracked.group_id is None:
            return None
        return self.state.pair_groups.get(tracked.group_id)

    def remove_order_from_group(self, order_id: str, tracked: Optional[TrackedOrder]) -> None:
        group = self.pair_group_for_order(tracked)
        if group is None:
            return
        if group.left_order_id == str(order_id):
            group.left_order_id = None
        if group.right_order_id == str(order_id):
            group.right_order_id = None
        if group.hedge_order_id == str(order_id):
            group.hedge_order_id = None

    def macro_headline_score(self, content: str) -> float:
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
        caution = (
            "balanced risks",
            "mixed economic indicators",
            "communication remains cautious",
            "await upcoming data",
        )
        for phrase, value in hawkish.items():
            if phrase in text:
                score += value
        for phrase, value in dovish.items():
            if phrase in text:
                score += value
        if any(phrase in text for phrase in caution):
            score *= 0.5
            if score == 0.0:
                score = 0.25
        return score

    async def evaluate(self) -> None:
        if not self.positions_ready:
            return
        async with self._decision_lock:
            self.refresh_all_books()
            await self.order_manager.cancel_stale_orders()

            state = self.current_state()
            if state is None:
                self._trace("state", tick=self.current_tick, reason="books_not_ready", event_off_ready=self.event_off_ready(), **self.positions_payload())
                return

            pair_signal = self.compute_pair_signal(state)
            package_signal = self.compute_package_signal()

            self._trace(
                "state",
                tick=self.current_tick,
                event_off_ready=self.event_off_ready(),
                last_macro_age=None if self.state.last_macro_event_ts <= 0.0 else max(0.0, time.time() - self.state.last_macro_event_ts),
                winner=state["winner"],
                challenger=state["challenger"],
                dead=state["dead"],
                dead_tail_regime=state["dead_tail_regime"],
                selected_pair=list(state["selected_pair"]),
                mids=state["mids"],
                spreads={symbol: self.top(symbol).spread for symbol in self.cfg.symbols},
                **self.positions_payload(),
            )
            self._trace("pair_signal", tick=self.current_tick, **vars(pair_signal), **self.positions_payload())
            self._trace("package_signal", tick=self.current_tick, **vars(package_signal), **self.positions_payload())

            if await self.maybe_cancel_for_macro_gate():
                return

            if await self.maybe_process_pair_groups():
                return

            should_exit, exit_reason = self.pair_exit_due(pair_signal)
            if should_exit and await self.flatten_pair_positions(exit_reason):
                return

            if await self.maybe_repair_package_positions():
                return

            if await self.maybe_enter_pair(pair_signal):
                return

            await self.maybe_enter_package(package_signal)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        tracked = self.order_manager.sync_cancel_response(order_id, success)
        self.state.pending_cancels.discard(str(order_id))
        self.remove_order_from_group(str(order_id), tracked)
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="cancel_success" if success else "cancel_fail",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            strategy=None if tracked is None else tracked.strategy,
            role=None if tracked is None else tracked.role,
            reason=error,
            **self.positions_payload(),
        )
        await self.evaluate()

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        tracked = self.order_manager.sync_fill(order_id)
        self.apply_overlay_fill(tracked, int(qty)) if tracked is not None else None
        if tracked is not None and str(order_id) not in self.state.live_orders:
            self.remove_order_from_group(str(order_id), tracked)
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="fill",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            strategy=None if tracked is None else tracked.strategy,
            role=None if tracked is None else tracked.role,
            qty=qty,
            price=price,
            **self.positions_payload(),
        )
        await self.evaluate()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.remove_order_from_group(str(order_id), tracked)
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="reject",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            strategy=None if tracked is None else tracked.strategy,
            role=None if tracked is None else tracked.role,
            reason=reason,
            **self.positions_payload(),
        )
        await self.evaluate()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.symbols:
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.symbols:
            self.refresh_book(symbol)
            await self.evaluate()

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")
        parsed_type = "other"
        parsed_values: dict[str, Any] = {}
        actionable_macro = False

        if kind == "structured":
            subtype = new_data.get("structured_subtype")
            if subtype == "cpi_print":
                parsed_type = "cpi"
                actual = float(new_data["actual"])
                forecast = float(new_data["forecast"])
                parsed_values = {
                    "actual": actual,
                    "forecast": forecast,
                    "surprise": actual - forecast,
                }
                actionable_macro = True
            elif subtype == "earnings":
                parsed_type = "earnings"
                parsed_values = {"asset": new_data.get("asset"), "value": new_data.get("value")}
        elif kind == "unstructured":
            parsed_type = "headline"
            content = str(new_data.get("content", ""))
            score = self.macro_headline_score(content)
            parsed_values = {"content": content, "headline_score": score, "message_type": new_data.get("type")}
            actionable_macro = abs(score) > 0.0

        if actionable_macro:
            self.record_macro_event()
            await self.order_manager.cancel_orders()

        self._trace(
            "news",
            tick=self.current_tick,
            raw_kind=kind,
            raw_new_data=dict(new_data),
            parsed_type=parsed_type,
            parsed_values=parsed_values,
            actionable_macro=actionable_macro,
            **self.positions_payload(),
        )
        await self.evaluate()

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        self.current_tick = tick
        self._trace("round_end", tick=tick, event="resolved", winning_symbol=winning_symbol, payout_amount=None, **self.positions_payload())

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        self.current_tick = tick
        self._trace("round_end", tick=tick, event="payout", winning_symbol=None, payout_amount=amount, **self.positions_payload())

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def start(self):
        await self.connect()


async def main():
    client = PredictionMarketsBot7(
        os.getenv("UTC_HOST", "34.197.188.76:3333"),
        os.getenv("UTC_USERNAME", "uiuc"),
        os.getenv("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
