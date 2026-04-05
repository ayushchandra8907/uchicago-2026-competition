from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any


RETURN_WINDOWS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
TRAILING_WINDOWS_SECONDS = (0.25, 0.5, 1.0, 2.0, 5.0)


@dataclass
class BookSnapshotFeatures:
    best_bid_px: int | None
    best_bid_qty: int | None
    best_ask_px: int | None
    best_ask_qty: int | None
    mid_px: float | None
    spread: float | None
    microprice: float | None
    top_of_book_imbalance: float | None
    total_bid_depth_top_k: int
    total_ask_depth_top_k: int
    bid_levels_json: str
    ask_levels_json: str


@dataclass
class MarketSnapshot:
    monotonic_ns: int
    wall_time_ns: int
    exchange_tick: int | None
    event_kind: str
    event_index: int
    mid_px: float | None
    spread: float | None
    microprice: float | None
    imbalance: float | None
    best_bid_px: int | None
    best_bid_qty: int | None
    best_ask_px: int | None
    best_ask_qty: int | None
    trade_count_delta: int = 0
    trade_volume_delta: int = 0


def _sorted_levels(book_side: dict[int, int], descending: bool, limit: int | None = None) -> list[tuple[int, int]]:
    items = [(int(px), int(qty)) for px, qty in book_side.items() if int(qty) > 0]
    items.sort(key=lambda item: item[0], reverse=descending)
    if limit is not None:
        return items[:limit]
    return items


def compute_book_features(book: Any, *, top_k_depth: int, top_n_levels: int) -> BookSnapshotFeatures:
    bids = _sorted_levels(getattr(book, "bids", {}), descending=True)
    asks = _sorted_levels(getattr(book, "asks", {}), descending=False)

    best_bid_px, best_bid_qty = bids[0] if bids else (None, None)
    best_ask_px, best_ask_qty = asks[0] if asks else (None, None)

    mid_px = None
    spread = None
    if best_bid_px is not None and best_ask_px is not None:
        mid_px = (best_bid_px + best_ask_px) / 2.0
        spread = float(best_ask_px - best_bid_px)

    microprice = compute_microprice(best_bid_px, best_bid_qty, best_ask_px, best_ask_qty)
    imbalance = compute_imbalance(best_bid_qty, best_ask_qty)

    bid_depth_top_k = sum(qty for _, qty in bids[:top_k_depth])
    ask_depth_top_k = sum(qty for _, qty in asks[:top_k_depth])

    return BookSnapshotFeatures(
        best_bid_px=best_bid_px,
        best_bid_qty=best_bid_qty,
        best_ask_px=best_ask_px,
        best_ask_qty=best_ask_qty,
        mid_px=mid_px,
        spread=spread,
        microprice=microprice,
        top_of_book_imbalance=imbalance,
        total_bid_depth_top_k=bid_depth_top_k,
        total_ask_depth_top_k=ask_depth_top_k,
        bid_levels_json=serialize_levels(bids[:top_n_levels]),
        ask_levels_json=serialize_levels(asks[:top_n_levels]),
    )


def serialize_levels(levels: list[tuple[int, int]]) -> str:
    return json.dumps([{"px": px, "qty": qty} for px, qty in levels], separators=(",", ":"))


def compute_microprice(
    best_bid_px: int | None,
    best_bid_qty: int | None,
    best_ask_px: int | None,
    best_ask_qty: int | None,
) -> float | None:
    if None in (best_bid_px, best_bid_qty, best_ask_px, best_ask_qty):
        return None
    denom = best_bid_qty + best_ask_qty
    if denom <= 0:
        return None
    return ((best_ask_px * best_bid_qty) + (best_bid_px * best_ask_qty)) / denom


def compute_imbalance(best_bid_qty: int | None, best_ask_qty: int | None) -> float | None:
    if best_bid_qty is None or best_ask_qty is None:
        return None
    denom = best_bid_qty + best_ask_qty
    if denom <= 0:
        return None
    return (best_bid_qty - best_ask_qty) / denom


def mark_post_news_window(now_ns: int, last_news_monotonic_ns: int | None, window_seconds: float) -> bool:
    if last_news_monotonic_ns is None:
        return False
    return (now_ns - last_news_monotonic_ns) <= int(window_seconds * 1_000_000_000)


def seconds_since(last_monotonic_ns: int | None, now_ns: int) -> float | None:
    if last_monotonic_ns is None:
        return None
    return (now_ns - last_monotonic_ns) / 1_000_000_000


def regime_for_post_news(seconds_since_news: float | None) -> str:
    if seconds_since_news is None:
        return "pre_earnings"
    if seconds_since_news < 1.0:
        return "immediate_post_earnings"
    if seconds_since_news < 5.0:
        return "late_post_earnings"
    return "post_earnings"


def bucket_post_news_elapsed(seconds_since_news: float | None) -> str:
    if seconds_since_news is None:
        return "none"
    if seconds_since_news < 0.1:
        return "lt_100ms"
    if seconds_since_news < 0.25:
        return "100_250ms"
    if seconds_since_news < 0.5:
        return "250_500ms"
    if seconds_since_news < 1.0:
        return "500ms_1s"
    if seconds_since_news < 2.0:
        return "1_2s"
    if seconds_since_news < 5.0:
        return "2_5s"
    return "gt_5s"


def naive_fair_price(latest_known_eps: float | None, pe_constant: float) -> float | None:
    if latest_known_eps is None:
        return None
    return pe_constant * latest_known_eps


def signed_distance_to_fair(mid_px: float | None, fair_px: float | None) -> float | None:
    if mid_px is None or fair_px is None:
        return None
    return mid_px - fair_px


def news_mentions_symbol(content: str | None, symbol: str) -> bool:
    if not content:
        return False
    pattern = rf"\b{re.escape(symbol)}\b"
    return re.search(pattern, content, flags=re.IGNORECASE) is not None


def nearest_mid_at_or_before(history: deque[MarketSnapshot], target_ns: int) -> float | None:
    candidate: float | None = None
    for snapshot in history:
        if snapshot.monotonic_ns <= target_ns and snapshot.mid_px is not None:
            candidate = snapshot.mid_px
        elif snapshot.monotonic_ns > target_ns:
            break
    return candidate


def nearest_snapshot_at_or_after(history: deque[MarketSnapshot], target_ns: int) -> MarketSnapshot | None:
    for snapshot in history:
        if snapshot.monotonic_ns >= target_ns and snapshot.mid_px is not None:
            return snapshot
    return None


def nearest_snapshot_at_or_before(history: deque[MarketSnapshot], target_ns: int) -> MarketSnapshot | None:
    candidate: MarketSnapshot | None = None
    for snapshot in history:
        if snapshot.monotonic_ns <= target_ns and snapshot.mid_px is not None:
            candidate = snapshot
        elif snapshot.monotonic_ns > target_ns:
            break
    return candidate


def compute_return(current_mid: float | None, reference_mid: float | None) -> float | None:
    if current_mid is None or reference_mid is None or reference_mid == 0:
        return None
    return (current_mid - reference_mid) / reference_mid


def compute_realized_volatility(history: deque[MarketSnapshot], now_ns: int, window_seconds: float) -> float | None:
    window_ns = int(window_seconds * 1_000_000_000)
    mids: list[float] = [row.mid_px for row in history if row.mid_px is not None and now_ns - row.monotonic_ns <= window_ns]
    if len(mids) < 2:
        return None
    returns: list[float] = []
    prev = mids[0]
    for mid in mids[1:]:
        if prev > 0 and mid > 0:
            returns.append(math.log(mid / prev))
        prev = mid
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(max(variance, 0.0))


def trailing_trade_stats(trades: deque[dict[str, Any]], now_ns: int, window_seconds: float) -> tuple[int, int]:
    window_ns = int(window_seconds * 1_000_000_000)
    count = 0
    volume = 0
    for trade in reversed(trades):
        if now_ns - int(trade["monotonic_ns"]) > window_ns:
            break
        count += 1
        volume += int(trade["trade_qty"])
    return count, volume
