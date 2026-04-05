from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import BookState, MarketEvent, NewsState, RunCatalog, SessionData, SkippedRun, TradeState


def discover_run_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def build_run_catalog(data_root: Path, symbol: str = "A") -> RunCatalog:
    sessions: list[SessionData] = []
    skipped: list[SkippedRun] = []
    for run_dir in discover_run_dirs(data_root):
        try:
            session = load_session(run_dir, symbol=symbol)
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append(SkippedRun(path=run_dir, reason=f"loader_error:{type(exc).__name__}:{exc}"))
            continue
        if session is None:
            skipped.append(SkippedRun(path=run_dir, reason="insufficient_a_data"))
            continue
        sessions.append(session)
    return RunCatalog(data_root=data_root, sessions=tuple(sessions), skipped_runs=tuple(skipped))


def load_session(run_dir: Path, symbol: str = "A") -> SessionData | None:
    news_rows = _prepare_news_dataframe(run_dir)
    layout = _detect_layout(run_dir)
    if layout is None:
        return None

    if layout == "new_raw":
        book_rows = _reconstruct_new_layout_books(run_dir, symbol, news_rows)
        trade_rows = _load_new_layout_trades(run_dir, symbol, news_rows)
    else:
        book_rows, trade_rows, news_rows = _load_legacy_layout(run_dir, symbol, news_rows)

    if book_rows.empty:
        return None

    book_rows = book_rows.dropna(subset=["time_ms"]).sort_values(["time_ms", "source_seq"]).reset_index(drop=True)
    trade_rows = trade_rows.dropna(subset=["time_ms"]).sort_values(["time_ms", "source_seq"]).reset_index(drop=True)
    news_rows = news_rows.dropna(subset=["time_ms"]).sort_values(["time_ms", "source_seq"]).reset_index(drop=True)

    events = _build_events(run_dir.name, book_rows, trade_rows, news_rows)
    if not events:
        return None

    diagnostics = {
        "layout": layout,
        "book_event_count": int(len(book_rows)),
        "trade_event_count": int(len(trade_rows)),
        "news_event_count": int(len(news_rows)),
        "earnings_event_count": int(
            len(
                news_rows[
                    (news_rows["kind"] == "structured")
                    & (news_rows["structured_subtype"] == "earnings")
                    & (news_rows["earnings_asset"] == symbol)
                ]
            )
        ),
    }
    return SessionData(
        session_id=run_dir.name,
        path=run_dir,
        source_layout=layout,
        events=tuple(events),
        book_rows=book_rows,
        trade_rows=trade_rows,
        news_rows=news_rows,
        diagnostics=diagnostics,
    )


def _detect_layout(run_dir: Path) -> str | None:
    if any((run_dir / f"raw_book_updates_{symbol}.csv").exists() for symbol in ("A", "B", "C", "ETF")):
        return "new_raw"
    if (run_dir / "raw_book_events.csv").exists():
        return "legacy_derived"
    return None


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_optional_trade_file(run_dir: Path, symbol: str) -> pd.DataFrame:
    per_symbol = run_dir / f"raw_trade_events_{symbol}.csv"
    if per_symbol.exists():
        return _load_csv(per_symbol)
    generic = run_dir / "raw_trade_events.csv"
    if generic.exists():
        return _load_csv(generic)
    return pd.DataFrame()


def _prepare_news_dataframe(run_dir: Path) -> pd.DataFrame:
    news = _load_csv(run_dir / "raw_news_events.csv")
    if news.empty:
        return pd.DataFrame(
            columns=[
                "kind",
                "symbol",
                "structured_subtype",
                "earnings_asset",
                "earnings_value",
                "content",
                "time_ms",
                "source_seq",
            ]
        )

    news = news.copy()
    if "message_index" in news.columns:
        news["source_seq"] = pd.to_numeric(news["message_index"], errors="coerce")
    elif "event_index" in news.columns:
        news["source_seq"] = pd.to_numeric(news["event_index"], errors="coerce")
    else:
        news["source_seq"] = pd.RangeIndex(start=1, stop=len(news) + 1)

    if "tick_ms" in news.columns:
        news["time_ms"] = pd.to_numeric(news["tick_ms"], errors="coerce")
    elif "exchange_tick" in news.columns:
        news["time_ms"] = pd.to_numeric(news["exchange_tick"], errors="coerce") * 200.0
    elif "tick" in news.columns:
        news["time_ms"] = pd.to_numeric(news["tick"], errors="coerce") * 200.0
    elif "monotonic_ns" in news.columns:
        base_ns = pd.to_numeric(news["monotonic_ns"], errors="coerce").dropna()
        base = float(base_ns.min()) if not base_ns.empty else 0.0
        news["time_ms"] = (pd.to_numeric(news["monotonic_ns"], errors="coerce") - base) / 1_000_000.0
    else:
        news["time_ms"] = pd.RangeIndex(start=0, stop=len(news))

    if "kind" not in news.columns:
        news["kind"] = None
    if "structured_subtype" not in news.columns:
        news["structured_subtype"] = None

    if "earnings_asset" not in news.columns:
        news["earnings_asset"] = news.get("symbol")
    news["earnings_asset"] = news["earnings_asset"].where(news["earnings_asset"].notna(), news.get("symbol"))

    if "earnings_value" in news.columns:
        news["earnings_value"] = pd.to_numeric(news["earnings_value"], errors="coerce")
    else:
        news["earnings_value"] = pd.to_numeric(news.get("new_known_eps_for_A"), errors="coerce")

    if "content" not in news.columns:
        news["content"] = news.apply(_extract_raw_content, axis=1)
    if "symbol" not in news.columns:
        news["symbol"] = None

    news = news.sort_values(["time_ms", "source_seq"], na_position="last").reset_index(drop=True)
    return news


def _extract_raw_content(row: pd.Series) -> str | None:
    raw = row.get("raw_content")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    normalized = row.get("normalized_content")
    if isinstance(normalized, str) and normalized.strip():
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            return normalized.strip()
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return normalized.strip()
    return None


def _news_time_anchors(news: pd.DataFrame) -> tuple[list[int], list[float]]:
    if news.empty or "source_seq" not in news.columns or "time_ms" not in news.columns:
        return [], []
    anchors = news[["source_seq", "time_ms"]].copy()
    anchors["source_seq"] = pd.to_numeric(anchors["source_seq"], errors="coerce")
    anchors["time_ms"] = pd.to_numeric(anchors["time_ms"], errors="coerce")
    anchors = anchors.dropna(subset=["source_seq", "time_ms"]).sort_values("source_seq")
    anchors = anchors.drop_duplicates(subset=["source_seq"], keep="last")
    if anchors.empty:
        return [], []
    return [int(value) for value in anchors["source_seq"].tolist()], [float(value) for value in anchors["time_ms"].tolist()]


def _interpolate_time_ms(source_seq: int, anchor_indices: list[int], anchor_times_ms: list[float]) -> float | None:
    if not anchor_indices:
        return None
    if len(anchor_indices) == 1:
        return anchor_times_ms[0]
    position = bisect.bisect_left(anchor_indices, source_seq)
    if position <= 0:
        left_index, right_index = 0, 1
    elif position >= len(anchor_indices):
        left_index, right_index = len(anchor_indices) - 2, len(anchor_indices) - 1
    else:
        left_index, right_index = position - 1, position
    idx0 = anchor_indices[left_index]
    idx1 = anchor_indices[right_index]
    time0 = anchor_times_ms[left_index]
    time1 = anchor_times_ms[right_index]
    if idx1 == idx0:
        return time0
    slope = (time1 - time0) / (idx1 - idx0)
    return time0 + slope * (source_seq - idx0)


def _parse_levels_json(value: Any) -> tuple[tuple[int, int], ...]:
    if isinstance(value, list):
        levels = value
    elif isinstance(value, str) and value.strip():
        try:
            levels = json.loads(value)
        except json.JSONDecodeError:
            return ()
    else:
        return ()
    parsed: list[tuple[int, int]] = []
    for item in levels:
        if not isinstance(item, dict):
            continue
        px = item.get("px")
        qty = item.get("qty")
        if px is None or qty is None:
            continue
        parsed.append((int(px), int(qty)))
    return tuple(parsed)


def _levels_from_state(side_levels: dict[int, int], *, descending: bool) -> tuple[tuple[int, int], ...]:
    items = [(int(px), int(qty)) for px, qty in side_levels.items() if int(qty) > 0]
    items.sort(key=lambda item: item[0], reverse=descending)
    return tuple(items[:10])


def _best_levels_from_state(bids: dict[int, int], asks: dict[int, int]) -> tuple[int | None, int | None, int | None, int | None]:
    bid_levels = _levels_from_state(bids, descending=True)
    ask_levels = _levels_from_state(asks, descending=False)
    best_bid_px, best_bid_qty = bid_levels[0] if bid_levels else (None, None)
    best_ask_px, best_ask_qty = ask_levels[0] if ask_levels else (None, None)
    return best_bid_px, best_bid_qty, best_ask_px, best_ask_qty


def _reconstruct_new_layout_books(run_dir: Path, symbol: str, news_rows: pd.DataFrame) -> pd.DataFrame:
    snapshots = _load_csv(run_dir / f"raw_book_snapshots_{symbol}.csv")
    updates = _load_csv(run_dir / f"raw_book_updates_{symbol}.csv")
    anchor_indices, anchor_times_ms = _news_time_anchors(news_rows)

    events: list[dict[str, Any]] = []
    if not snapshots.empty:
        snapshots["message_index"] = pd.to_numeric(snapshots["message_index"], errors="coerce")
        for _, row in snapshots.dropna(subset=["message_index"]).iterrows():
            events.append(
                {
                    "kind": "snapshot",
                    "source_seq": int(row["message_index"]),
                    "bids": {px: qty for px, qty in _parse_levels_json(row.get("bids_json"))},
                    "asks": {px: qty for px, qty in _parse_levels_json(row.get("asks_json"))},
                }
            )
    if not updates.empty:
        updates["message_index"] = pd.to_numeric(updates["message_index"], errors="coerce")
        updates["px"] = pd.to_numeric(updates["px"], errors="coerce")
        updates["dq"] = pd.to_numeric(updates["dq"], errors="coerce")
        for _, row in updates.dropna(subset=["message_index", "px", "dq"]).iterrows():
            events.append(
                {
                    "kind": "update",
                    "source_seq": int(row["message_index"]),
                    "side": row.get("side"),
                    "px": int(row["px"]),
                    "dq": int(row["dq"]),
                }
            )

    if not events:
        return pd.DataFrame()

    events.sort(key=lambda item: (item["source_seq"], 0 if item["kind"] == "snapshot" else 1))
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] == "snapshot":
            bids = dict(event["bids"])
            asks = dict(event["asks"])
        else:
            book = bids if event["side"] == "BUY" else asks
            px = event["px"]
            book[px] = book.get(px, 0) + event["dq"]
            if book[px] <= 0:
                book.pop(px, None)

        bid_levels = _levels_from_state(bids, descending=True)
        ask_levels = _levels_from_state(asks, descending=False)
        best_bid_px, best_bid_qty, best_ask_px, best_ask_qty = _best_levels_from_state(bids, asks)
        mid_px = None if best_bid_px is None or best_ask_px is None else (best_bid_px + best_ask_px) / 2.0
        spread = None if best_bid_px is None or best_ask_px is None else float(best_ask_px - best_bid_px)
        rows.append(
            {
                "symbol": symbol,
                "event_kind": event["kind"],
                "source_seq": event["source_seq"],
                "time_ms": _interpolate_time_ms(event["source_seq"], anchor_indices, anchor_times_ms),
                "best_bid_px": best_bid_px,
                "best_bid_qty": best_bid_qty,
                "best_ask_px": best_ask_px,
                "best_ask_qty": best_ask_qty,
                "mid_px": mid_px,
                "spread": spread,
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
            }
        )
    return pd.DataFrame(rows)


def _load_new_layout_trades(run_dir: Path, symbol: str, news_rows: pd.DataFrame) -> pd.DataFrame:
    trades = _load_optional_trade_file(run_dir, symbol)
    if trades.empty:
        return pd.DataFrame(columns=["source_seq", "time_ms", "price_px", "qty"])

    anchor_indices, anchor_times_ms = _news_time_anchors(news_rows)
    column = "message_index" if "message_index" in trades.columns else "event_index"
    trades[column] = pd.to_numeric(trades[column], errors="coerce")
    price_col = "price" if "price" in trades.columns else "trade_px"
    qty_col = "qty" if "qty" in trades.columns else "trade_qty"
    symbol_col = "symbol" if "symbol" in trades.columns else None
    if symbol_col is not None:
        trades = trades[trades[symbol_col] == symbol].copy()
    trades[price_col] = pd.to_numeric(trades[price_col], errors="coerce")
    trades[qty_col] = pd.to_numeric(trades[qty_col], errors="coerce")
    trades = trades.dropna(subset=[column, price_col, qty_col]).copy()
    trades["source_seq"] = trades[column].astype(int)
    trades["time_ms"] = trades["source_seq"].apply(lambda value: _interpolate_time_ms(int(value), anchor_indices, anchor_times_ms))
    trades["price_px"] = trades[price_col].astype(int)
    trades["qty"] = trades[qty_col].astype(int)
    return trades[["source_seq", "time_ms", "price_px", "qty"]]


def _load_legacy_layout(run_dir: Path, symbol: str, news_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    books = _load_csv(run_dir / "raw_book_events.csv")
    trades = _load_optional_trade_file(run_dir, symbol)

    base_candidates: list[float] = []
    for frame in (books, trades, news_rows):
        if "monotonic_ns" in frame.columns:
            values = pd.to_numeric(frame["monotonic_ns"], errors="coerce").dropna()
            if not values.empty:
                base_candidates.append(float(values.min()))
    base_ns = min(base_candidates) if base_candidates else 0.0

    books = books[books.get("symbol") == symbol].copy()
    books["source_seq"] = pd.to_numeric(books.get("event_index"), errors="coerce")
    books["time_ms"] = (pd.to_numeric(books.get("monotonic_ns"), errors="coerce") - base_ns) / 1_000_000.0
    books["best_bid_px"] = pd.to_numeric(books.get("best_bid_px"), errors="coerce")
    books["best_bid_qty"] = pd.to_numeric(books.get("best_bid_qty"), errors="coerce")
    books["best_ask_px"] = pd.to_numeric(books.get("best_ask_px"), errors="coerce")
    books["best_ask_qty"] = pd.to_numeric(books.get("best_ask_qty"), errors="coerce")
    books["mid_px"] = pd.to_numeric(books.get("mid_px"), errors="coerce")
    books["spread"] = pd.to_numeric(books.get("spread"), errors="coerce")
    books["bid_levels"] = books.get("bid_levels_json", pd.Series(dtype=object)).apply(_parse_levels_json)
    books["ask_levels"] = books.get("ask_levels_json", pd.Series(dtype=object)).apply(_parse_levels_json)
    books["event_kind"] = books.get("event_type", "book")
    book_rows = books[
        [
            "source_seq",
            "time_ms",
            "event_kind",
            "best_bid_px",
            "best_bid_qty",
            "best_ask_px",
            "best_ask_qty",
            "mid_px",
            "spread",
            "bid_levels",
            "ask_levels",
        ]
    ].dropna(subset=["source_seq"])

    price_col = "trade_px" if "trade_px" in trades.columns else "price"
    qty_col = "trade_qty" if "trade_qty" in trades.columns else "qty"
    if "symbol" in trades.columns:
        trades = trades[trades["symbol"] == symbol].copy()
    trades["source_seq"] = pd.to_numeric(trades.get("event_index"), errors="coerce")
    trades["time_ms"] = (pd.to_numeric(trades.get("monotonic_ns"), errors="coerce") - base_ns) / 1_000_000.0
    trades["price_px"] = pd.to_numeric(trades.get(price_col), errors="coerce")
    trades["qty"] = pd.to_numeric(trades.get(qty_col), errors="coerce")
    trade_rows = trades[["source_seq", "time_ms", "price_px", "qty"]].dropna(subset=["source_seq", "price_px", "qty"])
    trade_rows["source_seq"] = trade_rows["source_seq"].astype(int)
    trade_rows["price_px"] = trade_rows["price_px"].astype(int)
    trade_rows["qty"] = trade_rows["qty"].astype(int)

    if "monotonic_ns" in news_rows.columns:
        news_rows = news_rows.copy()
        news_rows["time_ms"] = (pd.to_numeric(news_rows["monotonic_ns"], errors="coerce") - base_ns) / 1_000_000.0
    return book_rows.reset_index(drop=True), trade_rows.reset_index(drop=True), news_rows.reset_index(drop=True)


def _build_events(
    session_id: str,
    book_rows: pd.DataFrame,
    trade_rows: pd.DataFrame,
    news_rows: pd.DataFrame,
) -> list[MarketEvent]:
    combined: list[tuple[float, int, EventKind, dict[str, Any]]] = []

    for _, row in book_rows.iterrows():
        book = BookState(
            best_bid_px=_to_optional_int(row.get("best_bid_px")),
            best_bid_qty=_to_optional_int(row.get("best_bid_qty")),
            best_ask_px=_to_optional_int(row.get("best_ask_px")),
            best_ask_qty=_to_optional_int(row.get("best_ask_qty")),
            bid_levels=tuple(row.get("bid_levels") or ()),
            ask_levels=tuple(row.get("ask_levels") or ()),
        )
        combined.append(
            (
                float(row["time_ms"]),
                int(row["source_seq"]),
                "book",
                {"book": book},
            )
        )

    for _, row in trade_rows.iterrows():
        trade = TradeState(price_px=int(row["price_px"]), qty=int(row["qty"]))
        combined.append((float(row["time_ms"]), int(row["source_seq"]), "trade", {"trade": trade}))

    for _, row in news_rows.iterrows():
        news = NewsState(
            kind=str(row.get("kind") or "unstructured"),
            symbol=_to_optional_str(row.get("symbol")),
            structured_subtype=_to_optional_str(row.get("structured_subtype")),
            earnings_asset=_to_optional_str(row.get("earnings_asset")),
            earnings_value=_to_optional_float(row.get("earnings_value")),
            content=_to_optional_str(row.get("content")),
            raw={key: _json_safe(value) for key, value in row.to_dict().items()},
        )
        combined.append((float(row["time_ms"]), int(row["source_seq"]), "news", {"news": news}))

    combined.sort(key=lambda item: (item[0], item[1], {"news": 0, "trade": 1, "book": 2}[item[2]]))
    events: list[MarketEvent] = []
    for seq, (time_ms, source_seq, kind, payload) in enumerate(combined, start=1):
        events.append(
            MarketEvent(
                kind=kind,
                session_id=session_id,
                seq=seq,
                time_ms=time_ms,
                source_seq=source_seq,
                **payload,
            )
        )
    return events


def _to_optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _to_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _to_optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)
