from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any


WINDOWS = [
    ("100ms", 100.0),
    ("250ms", 250.0),
    ("500ms", 500.0),
    ("1s", 1_000.0),
    ("2s", 2_000.0),
    ("5s", 5_000.0),
]


def _load_pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "market_research_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline analysis for multi-symbol market research logs.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Run directory, or any file inside a run directory created by market_research_logger.py",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir_flag",
        help="Legacy alias for the per-run log directory created by market_research_logger.py",
    )
    parser.add_argument("--output-name", default="earnings_event_summary.csv", help="Output CSV filename")
    parser.add_argument("--plot", action="store_true", help="Create per-symbol plots under the run's graphs folder")
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    raw_path = args.path or args.run_dir_flag
    if not raw_path:
        raise SystemExit(
            "Pass a run folder path, for example: "
            "`python3 data_scraping/analyze_logs.py --plot data_scraping/data/market_research_2026-04-04_21-22-29_live_round`"
        )

    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def _load_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_optional_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "session_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _plot_symbols(metadata: dict[str, Any]) -> list[str]:
    config = metadata.get("config", {})
    values = config.get("plot_symbols")
    if isinstance(values, list) and all(isinstance(item, str) for item in values):
        return values
    return ["A", "B", "C", "ETF"]


def _direct_earnings_symbols(metadata: dict[str, Any]) -> set[str]:
    config = metadata.get("config", {})
    values = config.get("direct_earnings_symbols")
    if isinstance(values, list) and all(isinstance(item, str) for item in values):
        return set(values)
    return {"A", "C"}


def _etf_news_assets(metadata: dict[str, Any]) -> set[str]:
    config = metadata.get("config", {})
    values = config.get("etf_news_assets")
    if isinstance(values, list) and all(isinstance(item, str) for item in values):
        return set(values)
    return {"A", "C"}


def _pe_constants(metadata: dict[str, Any]) -> dict[str, float]:
    config = metadata.get("config", {})
    values = config.get("pe_constants")
    if not isinstance(values, dict):
        return {}
    converted: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(key, str):
            converted[key] = float(value)
    return converted


def _is_new_raw_layout(run_dir: Path) -> bool:
    return any((run_dir / f"raw_book_updates_{symbol}.csv").exists() for symbol in ("A", "B", "C", "ETF"))


def _parse_level_json(value: Any) -> dict[int, int]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    levels: dict[int, int] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        px = item.get("px")
        qty = item.get("qty")
        if px is None or qty is None:
            continue
        levels[int(px)] = int(qty)
    return levels


def _best_bid_ask_from_state(bids: dict[int, int], asks: dict[int, int]) -> tuple[int | None, int | None, int | None, int | None]:
    valid_bids = [(px, qty) for px, qty in bids.items() if int(qty) > 0]
    valid_asks = [(px, qty) for px, qty in asks.items() if int(qty) > 0]
    best_bid = max(valid_bids, key=lambda item: item[0]) if valid_bids else (None, None)
    best_ask = min(valid_asks, key=lambda item: item[0]) if valid_asks else (None, None)
    return best_bid[0], best_bid[1], best_ask[0], best_ask[1]


def _reconstruct_symbol_books_from_raw(run_dir: Path, symbol: str, news: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    snapshots = _load_optional_csv(run_dir / f"raw_book_snapshots_{symbol}.csv")
    updates = _load_optional_csv(run_dir / f"raw_book_updates_{symbol}.csv")

    events: list[dict[str, Any]] = []

    if not snapshots.empty:
        snapshots = snapshots.copy()
        snapshots["message_index"] = pd.to_numeric(snapshots["message_index"], errors="coerce")
        for _, row in snapshots.iterrows():
            if pd.isna(row.get("message_index")):
                continue
            events.append(
                {
                    "message_index": int(row["message_index"]),
                    "event_kind": "snapshot",
                    "bids": _parse_level_json(row.get("bids_json")),
                    "asks": _parse_level_json(row.get("asks_json")),
                }
            )

    if not updates.empty:
        updates = updates.copy()
        updates["message_index"] = pd.to_numeric(updates["message_index"], errors="coerce")
        updates["px"] = pd.to_numeric(updates["px"], errors="coerce")
        updates["dq"] = pd.to_numeric(updates["dq"], errors="coerce")
        for _, row in updates.iterrows():
            if pd.isna(row.get("message_index")) or pd.isna(row.get("px")) or pd.isna(row.get("dq")):
                continue
            events.append(
                {
                    "message_index": int(row["message_index"]),
                    "event_kind": "update",
                    "side": row.get("side"),
                    "px": int(row["px"]),
                    "dq": int(row["dq"]),
                }
            )

    if not events:
        return pd.DataFrame(columns=["symbol", "message_index", "event_kind", "best_bid_px", "best_ask_px", "mid_px", "spread", "time_ms"])

    events.sort(key=lambda item: (item["message_index"], 0 if item["event_kind"] == "snapshot" else 1))

    anchor_indices, anchor_times_ms = _news_time_anchors(news)
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    rows: list[dict[str, Any]] = []

    for event in events:
        if event["event_kind"] == "snapshot":
            bids = dict(event["bids"])
            asks = dict(event["asks"])
        else:
            side = event["side"]
            target_book = bids if side == "BUY" else asks
            px = int(event["px"])
            dq = int(event["dq"])
            if px not in target_book:
                target_book[px] = dq
            else:
                target_book[px] += dq

        best_bid_px, _, best_ask_px, _ = _best_bid_ask_from_state(bids, asks)
        mid_px = None
        spread = None
        if best_bid_px is not None and best_ask_px is not None:
            mid_px = (best_bid_px + best_ask_px) / 2.0
            spread = float(best_ask_px - best_bid_px)

        message_index = int(event["message_index"])
        rows.append(
            {
                "symbol": symbol,
                "message_index": message_index,
                "event_kind": event["event_kind"],
                "best_bid_px": best_bid_px,
                "best_ask_px": best_ask_px,
                "mid_px": mid_px,
                "spread": spread,
                "time_ms": _interpolate_time_ms(message_index, anchor_indices, anchor_times_ms),
            }
        )

    return pd.DataFrame(rows)


def _reconstruct_symbol_books_legacy(run_dir: Path, symbol: str) -> pd.DataFrame:
    import pandas as pd

    path = run_dir / "raw_book_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "message_index", "event_kind", "best_bid_px", "best_ask_px", "mid_px", "spread", "time_ms"])

    books = _load_csv(path)
    if books.empty:
        return pd.DataFrame(columns=["symbol", "message_index", "event_kind", "best_bid_px", "best_ask_px", "mid_px", "spread", "time_ms"])

    symbol_books = books[books["symbol"] == symbol].copy()
    if symbol_books.empty:
        return pd.DataFrame(columns=["symbol", "message_index", "event_kind", "best_bid_px", "best_ask_px", "mid_px", "spread", "time_ms"])

    symbol_books["monotonic_ns"] = pd.to_numeric(symbol_books["monotonic_ns"], errors="coerce")
    symbol_books = symbol_books.dropna(subset=["monotonic_ns"]).sort_values("monotonic_ns").reset_index(drop=True)
    if symbol_books.empty:
        return pd.DataFrame(columns=["symbol", "message_index", "event_kind", "best_bid_px", "best_ask_px", "mid_px", "spread", "time_ms"])

    base_ns = int(symbol_books["monotonic_ns"].iloc[0])
    return pd.DataFrame(
        {
            "symbol": symbol_books["symbol"],
            "message_index": pd.to_numeric(symbol_books.get("event_index"), errors="coerce"),
            "event_kind": symbol_books.get("event_type", "book"),
            "best_bid_px": pd.to_numeric(symbol_books.get("best_bid_px"), errors="coerce"),
            "best_ask_px": pd.to_numeric(symbol_books.get("best_ask_px"), errors="coerce"),
            "mid_px": pd.to_numeric(symbol_books.get("mid_px"), errors="coerce"),
            "spread": pd.to_numeric(symbol_books.get("spread"), errors="coerce"),
            "time_ms": (pd.to_numeric(symbol_books["monotonic_ns"], errors="coerce") - base_ns) / 1_000_000.0,
        }
    )


def _news_time_anchors(news: pd.DataFrame) -> tuple[list[int], list[float]]:
    import pandas as pd

    if news.empty or "message_index" not in news.columns or "tick_ms" not in news.columns:
        return [], []

    anchors = news[["message_index", "tick_ms"]].copy()
    anchors["message_index"] = pd.to_numeric(anchors["message_index"], errors="coerce")
    anchors["tick_ms"] = pd.to_numeric(anchors["tick_ms"], errors="coerce")
    anchors = anchors.dropna(subset=["message_index", "tick_ms"]).sort_values("message_index")
    if anchors.empty:
        return [], []

    anchors = anchors.drop_duplicates(subset=["message_index"], keep="last")
    return [int(value) for value in anchors["message_index"].tolist()], [float(value) for value in anchors["tick_ms"].tolist()]


def _interpolate_time_ms(message_index: int, anchor_indices: list[int], anchor_times_ms: list[float]) -> float | None:
    if not anchor_indices:
        return None
    if len(anchor_indices) == 1:
        return anchor_times_ms[0]

    position = bisect.bisect_left(anchor_indices, message_index)
    if position <= 0:
        left_index = 0
        right_index = 1
    elif position >= len(anchor_indices):
        left_index = len(anchor_indices) - 2
        right_index = len(anchor_indices) - 1
    else:
        left_index = position - 1
        right_index = position

    idx0 = anchor_indices[left_index]
    idx1 = anchor_indices[right_index]
    time0 = anchor_times_ms[left_index]
    time1 = anchor_times_ms[right_index]
    if idx1 == idx0:
        return time0

    slope = (time1 - time0) / (idx1 - idx0)
    return time0 + slope * (message_index - idx0)


def _prepare_news_dataframe(run_dir: Path) -> pd.DataFrame:
    import pandas as pd

    news = _load_csv(run_dir / "raw_news_events.csv")
    if news.empty:
        return news

    if "message_index" in news.columns:
        news["message_index"] = pd.to_numeric(news["message_index"], errors="coerce")
    if "tick" in news.columns:
        news["tick"] = pd.to_numeric(news["tick"], errors="coerce")
    if "tick_ms" not in news.columns and "exchange_tick" in news.columns:
        news["tick"] = pd.to_numeric(news["exchange_tick"], errors="coerce")
        news["tick_ms"] = news["tick"] * 200.0
    else:
        news["tick_ms"] = pd.to_numeric(news.get("tick_ms"), errors="coerce")

    if "kind" not in news.columns:
        news["kind"] = None
    if "structured_subtype" not in news.columns:
        news["structured_subtype"] = None
    if "earnings_asset" not in news.columns and "symbol" in news.columns:
        news["earnings_asset"] = news["symbol"]
    if "earnings_value" in news.columns:
        news["earnings_value"] = pd.to_numeric(news["earnings_value"], errors="coerce")
    else:
        news["earnings_value"] = pd.NA

    if "raw_content" not in news.columns:
        news["raw_content"] = None
    if "normalized_content" not in news.columns:
        news["normalized_content"] = None

    news = news.sort_values(
        by=[column for column in ("tick_ms", "message_index") if column in news.columns],
        na_position="last",
    ).reset_index(drop=True)

    previous_eps_by_asset: dict[str, float] = {}
    old_eps_values: list[float | None] = []
    new_eps_values: list[float | None] = []
    for _, row in news.iterrows():
        asset = row.get("earnings_asset")
        if row.get("kind") == "structured" and row.get("structured_subtype") == "earnings" and isinstance(asset, str):
            old_eps = previous_eps_by_asset.get(asset)
            new_eps = row.get("earnings_value")
            old_eps_values.append(old_eps)
            new_eps_values.append(None if pd.isna(new_eps) else float(new_eps))
            if not pd.isna(new_eps):
                previous_eps_by_asset[asset] = float(new_eps)
        else:
            old_eps_values.append(None)
            new_eps_values.append(None)

    news["old_eps"] = old_eps_values
    news["new_eps"] = new_eps_values
    news["time_ms"] = news["tick_ms"]
    return news


def _content_mentions_symbol(content: str | None, symbol: str) -> bool:
    import re

    if not content:
        return False
    return re.search(rf"\b{re.escape(symbol)}\b", content, flags=re.IGNORECASE) is not None


def _raw_content(event: pd.Series) -> str | None:
    value = event.get("raw_content")
    if isinstance(value, str) and value.strip():
        return value.strip()
    normalized_content = event.get("normalized_content")
    if isinstance(normalized_content, str) and normalized_content.strip():
        try:
            parsed = json.loads(normalized_content)
        except json.JSONDecodeError:
            return normalized_content.strip()
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return normalized_content.strip()
    return None


def _is_a_specific_news(event: pd.Series) -> bool:
    symbol = event.get("symbol")
    if isinstance(symbol, str) and symbol == "A":
        return True

    earnings_asset = event.get("earnings_asset")
    if isinstance(earnings_asset, str) and earnings_asset == "A":
        return True

    petition_asset = event.get("petition_asset")
    if isinstance(petition_asset, str) and petition_asset == "A":
        return True

    return _content_mentions_symbol(_raw_content(event), "A")


def _relevant_earnings_news(news: pd.DataFrame, plot_symbol: str, metadata: dict[str, Any]) -> pd.DataFrame:
    earnings_news = news[(news["kind"] == "structured") & (news["structured_subtype"] == "earnings")].copy()
    if earnings_news.empty:
        return earnings_news

    direct_earnings_symbols = _direct_earnings_symbols(metadata)
    etf_news_assets = _etf_news_assets(metadata)

    if plot_symbol == "ETF":
        return earnings_news[earnings_news["earnings_asset"].isin(sorted(etf_news_assets))].copy()
    if plot_symbol in direct_earnings_symbols:
        return earnings_news[earnings_news["earnings_asset"] == plot_symbol].copy()
    return earnings_news.iloc[0:0].copy()


def _relevant_symbol_news(news: pd.DataFrame, plot_symbol: str) -> pd.DataFrame:
    if news.empty:
        return news.iloc[0:0].copy()

    non_earnings = news[~((news["kind"] == "structured") & (news["structured_subtype"] == "earnings"))].copy()
    if non_earnings.empty:
        return non_earnings

    if plot_symbol == "C":
        # C is driven by macro/government signals, so include all non-A non-earnings news.
        c_relevant = ~non_earnings.apply(_is_a_specific_news, axis=1)
        return non_earnings[c_relevant].copy()

    symbol_matches = non_earnings.get("symbol") == plot_symbol
    content_matches = non_earnings.apply(lambda row: _content_mentions_symbol(_raw_content(row), plot_symbol), axis=1)
    return non_earnings[symbol_matches | content_matches].copy()


def _label_text(event: pd.Series) -> str | None:
    import pandas as pd

    asset = event.get("earnings_asset") or event.get("symbol") or "?"
    old_eps = event.get("old_eps")
    new_eps = event.get("new_eps")
    if new_eps is None or pd.isna(new_eps):
        return None
    if old_eps is None or pd.isna(old_eps):
        return f"{asset} EPS START -> {float(new_eps):.2f}"
    return f"{asset} EPS {float(old_eps):.2f} -> {float(new_eps):.2f}"


def _news_label_text(event: pd.Series) -> str | None:
    return _raw_content(event)


def _place_news_lane_labels(fig, news_ax, relevant_symbol_news: pd.DataFrame) -> None:
    from matplotlib.transforms import blended_transform_factory

    if relevant_symbol_news.empty:
        return

    transform = blended_transform_factory(news_ax.transData, news_ax.transAxes)
    renderer = None
    placed_bboxes = []
    max_rows = max(4, min(8, len(relevant_symbol_news) + 1))
    row_step = 0.18
    base_row_y = 0.12

    for _, event in relevant_symbol_news.iterrows():
        label_text = _news_label_text(event)
        if label_text is None:
            continue

        event_time_s = float(event["time_s"])
        wrapped_label = textwrap.fill(label_text, width=24)
        placed = False

        for row_index in range(max_rows):
            candidate_y = min(0.94, base_row_y + row_index * row_step)
            text_artist = news_ax.text(
                event_time_s,
                candidate_y,
                wrapped_label,
                transform=transform,
                rotation=0,
                color="darkgreen",
                fontsize=8,
                ha="right",
                va="bottom",
                alpha=0.95,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "green",
                    "alpha": 0.9,
                },
            )

            fig.canvas.draw()
            if renderer is None:
                renderer = fig.canvas.get_renderer()
            bbox = text_artist.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
            if any(bbox.overlaps(existing_bbox) for existing_bbox in placed_bboxes):
                text_artist.remove()
                continue

            placed_bboxes.append(bbox)
            placed = True
            break

        if placed:
            continue

        fallback_y = min(0.94, base_row_y + (max_rows - 1) * row_step)
        fallback_artist = news_ax.text(
            event_time_s,
            fallback_y,
            wrapped_label,
            transform=transform,
            rotation=0,
            color="darkgreen",
            fontsize=8,
            ha="right",
            va="bottom",
            alpha=0.95,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "green",
                "alpha": 0.9,
            },
        )
        fig.canvas.draw()
        if renderer is None:
            renderer = fig.canvas.get_renderer()
        placed_bboxes.append(fallback_artist.get_window_extent(renderer=renderer).expanded(1.02, 1.08))


def _nearest_before(frame: pd.DataFrame, target_ms: float) -> pd.Series | None:
    subset = frame[frame["time_ms"] <= target_ms]
    if subset.empty:
        return None
    return subset.iloc[-1]


def _nearest_after(frame: pd.DataFrame, target_ms: float) -> pd.Series | None:
    subset = frame[frame["time_ms"] >= target_ms]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_earnings_summary(
    books_by_symbol: dict[str, pd.DataFrame],
    news: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    import pandas as pd

    pe_constants = _pe_constants(metadata)
    rows: list[dict[str, Any]] = []

    for plot_symbol in _plot_symbols(metadata):
        symbol_books = books_by_symbol.get(plot_symbol)
        if symbol_books is None or symbol_books.empty:
            continue

        relevant_news = _relevant_earnings_news(news, plot_symbol, metadata)
        for _, event in relevant_news.iterrows():
            event_time_ms = float(event["time_ms"])
            before = _nearest_before(symbol_books, event_time_ms)
            before_mid = None if before is None else before.get("mid_px")
            old_eps = event.get("old_eps")
            new_eps = event.get("new_eps")
            asset = event.get("earnings_asset")
            pe_constant = pe_constants.get(asset) if isinstance(asset, str) else None
            fair_jump = None
            if pe_constant is not None and old_eps is not None and new_eps is not None and not pd.isna(old_eps) and not pd.isna(new_eps):
                fair_jump = float(new_eps - old_eps) * pe_constant

            summary = {
                "plot_symbol": plot_symbol,
                "earnings_asset": asset,
                "news_message_index": event.get("message_index"),
                "exchange_tick": event.get("tick"),
                "exchange_time_ms": event_time_ms,
                "old_eps": old_eps,
                "new_eps": new_eps,
                "model_fair_value_jump": fair_jump,
                "market_mid_before_news": None if before is None else before.get("mid_px"),
                "spread_before_news": None if before is None else before.get("spread"),
            }

            excursion_candidates: list[float] = []
            for label, window_ms in WINDOWS:
                row = _nearest_after(symbol_books, event_time_ms + window_ms)
                summary[f"market_mid_after_{label}"] = None if row is None else row.get("mid_px")
                summary[f"spread_after_{label}"] = None if row is None else row.get("spread")
                if row is not None and before_mid is not None and not pd.isna(row.get("mid_px")):
                    excursion_candidates.append(float(row.get("mid_px")) - float(before_mid))

            summary["max_excursion_from_pre_news_mid"] = max(excursion_candidates, key=abs) if excursion_candidates else None
            final_row = _nearest_after(symbol_books, event_time_ms + 5_000.0)
            summary["final_settling_move_5s"] = (
                None
                if final_row is None or before_mid is None or pd.isna(final_row.get("mid_px"))
                else float(final_row.get("mid_px")) - float(before_mid)
            )
            rows.append(summary)

    return pd.DataFrame(rows)


def plot_symbol_graph(
    *,
    plot_symbol: str,
    books: pd.DataFrame,
    news: pd.DataFrame,
    metadata: dict[str, Any],
    graphs_dir: Path,
    time_axis_label: str,
) -> Path | None:
    import pandas as pd

    try:
        plt = _load_pyplot()
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return None

    symbol_books = books.sort_values("time_ms").reset_index(drop=True)
    if symbol_books.empty:
        return None

    relevant_earnings_news = _relevant_earnings_news(news, plot_symbol, metadata)
    relevant_symbol_news = _relevant_symbol_news(news, plot_symbol)
    base_time_ms = float(symbol_books["time_ms"].iloc[0])

    symbol_books = symbol_books.copy()
    relevant_earnings_news = relevant_earnings_news.copy()
    relevant_symbol_news = relevant_symbol_news.copy()
    symbol_books["time_s"] = (symbol_books["time_ms"] - base_time_ms) / 1_000.0
    relevant_earnings_news["time_s"] = (relevant_earnings_news["time_ms"] - base_time_ms) / 1_000.0
    relevant_symbol_news["time_s"] = (relevant_symbol_news["time_ms"] - base_time_ms) / 1_000.0

    has_news_lane = not relevant_symbol_news.empty
    if has_news_lane:
        fig, (ax, news_ax) = plt.subplots(
            2,
            1,
            figsize=(10, 6.1),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [4.0, 2.1], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        news_ax = None

    mid_line, = ax.plot(symbol_books["time_s"], symbol_books["mid_px"], label=f"{plot_symbol} mid")
    y_max = symbol_books["mid_px"].max()
    y_min = symbol_books["mid_px"].min()
    y_span = (y_max - y_min) if pd.notna(y_max) and pd.notna(y_min) else 0
    base_label_y = (y_max + 0.03 * y_span) if pd.notna(y_max) else None
    max_label_y = base_label_y

    for _, event in relevant_symbol_news.iterrows():
        ax.axvline(event["time_s"], color="green", linestyle="--", alpha=0.45, linewidth=1.2)

    for index, (_, event) in enumerate(relevant_earnings_news.iterrows()):
        ax.axvline(event["time_s"], color="red", linestyle="--", alpha=0.5)
        label_text = _label_text(event)
        if base_label_y is None or label_text is None:
            continue
        label_y = base_label_y + (index % 2) * max(0.05 * y_span, 1.0)
        max_label_y = label_y if max_label_y is None else max(max_label_y, label_y)
        ax.text(
            event["time_s"],
            label_y,
            label_text,
            rotation=90,
            color="black",
            fontsize=9,
            ha="left",
            va="bottom",
            alpha=0.95,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    ax.set_ylabel("mid_px")
    if max_label_y is not None:
        ax.set_ylim(top=max_label_y + max(0.12 * y_span, 2.0))

    if news_ax is not None:
        news_ax.set_ylim(0.0, 1.0)
        news_ax.set_yticks([])
        news_ax.set_ylabel("news")
        news_ax.spines["top"].set_visible(False)
        news_ax.spines["right"].set_visible(False)
        news_ax.spines["left"].set_visible(False)
        for _, event in relevant_symbol_news.iterrows():
            news_ax.axvline(event["time_s"], color="green", linestyle="--", alpha=0.45, linewidth=1.2)
        _place_news_lane_labels(fig, news_ax, relevant_symbol_news)
        news_ax.set_xlabel(time_axis_label)
    else:
        ax.set_xlabel(time_axis_label)

    legend_handles = [mid_line]
    if not relevant_earnings_news.empty:
        legend_handles.append(Line2D([0], [0], color="red", linestyle="--", label="earnings"))
    if not relevant_symbol_news.empty:
        legend_handles.append(Line2D([0], [0], color="green", linestyle="--", label=f"{plot_symbol} news"))
    ax.legend(handles=legend_handles)
    output_path = graphs_dir / f"{plot_symbol}_mid_price.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def analyze_run(run_dir: Path, *, output_name: str = "earnings_event_summary.csv", create_plots: bool = False) -> pd.DataFrame:
    news = _prepare_news_dataframe(run_dir)
    metadata = _load_metadata(run_dir)
    raw_layout = _is_new_raw_layout(run_dir)

    books_by_symbol: dict[str, pd.DataFrame] = {}
    for plot_symbol in _plot_symbols(metadata):
        if raw_layout:
            books_by_symbol[plot_symbol] = _reconstruct_symbol_books_from_raw(run_dir, plot_symbol, news)
        else:
            books_by_symbol[plot_symbol] = _reconstruct_symbol_books_legacy(run_dir, plot_symbol)

    summary = build_earnings_summary(books_by_symbol, news, metadata)
    output_path = run_dir / output_name
    summary.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    if create_plots:
        graphs_dir = run_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        time_axis_label = (
            "approx seconds since first event (inferred from exchange news ticks)"
            if raw_layout
            else "seconds since run start"
        )
        for plot_symbol in _plot_symbols(metadata):
            plot_path = plot_symbol_graph(
                plot_symbol=plot_symbol,
                books=books_by_symbol.get(plot_symbol, _reconstruct_symbol_books_legacy(run_dir, plot_symbol)),
                news=news,
                metadata=metadata,
                graphs_dir=graphs_dir,
                time_axis_label=time_axis_label,
            )
            if plot_path is not None:
                print(f"Wrote {plot_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    analyze_run(run_dir, output_name=args.output_name, create_plots=args.plot)


if __name__ == "__main__":
    main()
