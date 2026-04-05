from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any


WINDOWS = [
    ("100ms", 0.1),
    ("250ms", 0.25),
    ("500ms", 0.5),
    ("1s", 1.0),
    ("2s", 2.0),
    ("5s", 5.0),
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
    frame = pd.read_csv(path)
    if "monotonic_ns" in frame.columns:
        frame["monotonic_ns"] = pd.to_numeric(frame["monotonic_ns"], errors="coerce")
    return frame


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "session_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _plot_symbols(metadata: dict[str, Any]) -> list[str]:
    config = metadata.get("config", {})
    plot_symbols = config.get("plot_symbols")
    if isinstance(plot_symbols, list) and all(isinstance(item, str) for item in plot_symbols):
        return plot_symbols
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


def _nearest_before(frame: pd.DataFrame, target_ns: int) -> pd.Series | None:
    subset = frame[frame["monotonic_ns"] <= target_ns]
    if subset.empty:
        return None
    return subset.iloc[-1]


def _nearest_after(frame: pd.DataFrame, target_ns: int) -> pd.Series | None:
    subset = frame[frame["monotonic_ns"] >= target_ns]
    if subset.empty:
        return None
    return subset.iloc[0]


def _earnings_asset(event: pd.Series) -> str | None:
    asset = event.get("earnings_asset")
    if isinstance(asset, str) and asset:
        return asset
    fallback_symbol = event.get("symbol")
    if isinstance(fallback_symbol, str) and fallback_symbol:
        return fallback_symbol
    return None


def _old_eps(event: pd.Series) -> float | None:
    import pandas as pd

    value = event.get("previous_known_eps_for_asset")
    if value is None or pd.isna(value):
        value = event.get("previous_known_eps_for_A")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _new_eps(event: pd.Series) -> float | None:
    import pandas as pd

    value = event.get("new_known_eps_for_asset")
    if value is None or pd.isna(value):
        value = event.get("new_known_eps_for_A")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _affected_symbols(event: pd.Series) -> set[str]:
    value = event.get("affected_symbols_json")
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return {item for item in parsed if isinstance(item, str) and item}

    fallback_symbol = event.get("symbol")
    if isinstance(fallback_symbol, str) and fallback_symbol:
        return {fallback_symbol}
    return set()


def _relevant_earnings_news(news: pd.DataFrame, plot_symbol: str, metadata: dict[str, Any]) -> pd.DataFrame:
    earnings_news = news[(news["kind"] == "structured") & (news["structured_subtype"] == "earnings")].copy()
    if earnings_news.empty:
        return earnings_news

    earnings_asset_series = earnings_news["earnings_asset"] if "earnings_asset" in earnings_news.columns else earnings_news["symbol"]
    direct_earnings_symbols = _direct_earnings_symbols(metadata)
    etf_news_assets = _etf_news_assets(metadata)

    if plot_symbol == "ETF":
        return earnings_news[earnings_asset_series.isin(sorted(etf_news_assets))].copy()
    if plot_symbol in direct_earnings_symbols:
        return earnings_news[earnings_asset_series == plot_symbol].copy()
    return earnings_news.iloc[0:0].copy()


def _relevant_symbol_news(news: pd.DataFrame, plot_symbol: str) -> pd.DataFrame:
    if news.empty:
        return news.iloc[0:0].copy()

    relevant_rows = news.apply(lambda row: plot_symbol in _affected_symbols(row), axis=1)
    non_earnings_rows = ~(
        (news["kind"] == "structured")
        & (news["structured_subtype"] == "earnings")
    )
    return news[relevant_rows & non_earnings_rows].copy()


def build_earnings_summary(run_dir: Path) -> pd.DataFrame:
    import pandas as pd

    books = _load_csv(run_dir / "raw_book_events.csv").sort_values("monotonic_ns").reset_index(drop=True)
    news = _load_csv(run_dir / "raw_news_events.csv").sort_values("monotonic_ns").reset_index(drop=True)
    metadata = _load_metadata(run_dir)
    rows: list[dict[str, Any]] = []

    for plot_symbol in _plot_symbols(metadata):
        symbol_books = books[books["symbol"] == plot_symbol].copy()
        if symbol_books.empty:
            continue
        relevant_news = _relevant_earnings_news(news, plot_symbol, metadata)
        for _, event in relevant_news.iterrows():
            event_ns = int(event["monotonic_ns"])
            before = _nearest_before(symbol_books, event_ns)
            after_rows = {label: _nearest_after(symbol_books, event_ns + int(seconds * 1_000_000_000)) for label, seconds in WINDOWS}

            model_fair_before = event.get("inferred_fair_price_before")
            model_fair_after = event.get("inferred_fair_price_after")
            before_mid = None if before is None else before.get("mid_px")

            summary = {
                "plot_symbol": plot_symbol,
                "earnings_asset": _earnings_asset(event),
                "news_id": event["news_id"],
                "timestamp_iso": event["wall_time_iso"],
                "exchange_tick": event.get("exchange_tick"),
                "old_eps": _old_eps(event),
                "new_eps": _new_eps(event),
                "model_fair_value_jump": (
                    None
                    if pd.isna(model_fair_before) or pd.isna(model_fair_after)
                    else float(model_fair_after) - float(model_fair_before)
                ),
                "market_mid_before_news": None if before is None else before.get("mid_px"),
                "spread_before_news": None if before is None else before.get("spread"),
            }

            excursion_candidates = []
            for label, _ in WINDOWS:
                row = after_rows[label]
                summary[f"market_mid_after_{label}"] = None if row is None else row.get("mid_px")
                summary[f"spread_after_{label}"] = None if row is None else row.get("spread")
                if row is not None and before_mid is not None and not pd.isna(row.get("mid_px")):
                    excursion_candidates.append(float(row.get("mid_px")) - float(before_mid))

            summary["max_excursion_from_pre_news_mid"] = max(excursion_candidates, key=abs) if excursion_candidates else None
            final_row = after_rows["5s"]
            summary["final_settling_move_5s"] = (
                None
                if final_row is None or before_mid is None or pd.isna(final_row.get("mid_px"))
                else float(final_row.get("mid_px")) - float(before_mid)
            )
            rows.append(summary)

    return pd.DataFrame(rows)


def _label_text(event: pd.Series) -> str | None:
    import pandas as pd

    asset = _earnings_asset(event) or "?"
    old_eps = _old_eps(event)
    new_eps = _new_eps(event)
    if new_eps is None or pd.isna(new_eps):
        return None
    if old_eps is None or pd.isna(old_eps):
        return f"{asset} EPS START -> {float(new_eps):.2f}"
    return f"{asset} EPS {float(old_eps):.2f} -> {float(new_eps):.2f}"


def _news_label_text(event: pd.Series) -> str | None:
    import pandas as pd

    raw_content = event.get("raw_content")
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content.strip()

    normalized_content = event.get("normalized_content")
    if isinstance(normalized_content, str) and normalized_content.strip():
        try:
            parsed = json.loads(normalized_content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            structured_subtype = parsed.get("structured_subtype")
            if isinstance(structured_subtype, str) and structured_subtype:
                return structured_subtype
        elif not pd.isna(normalized_content):
            return normalized_content.strip()
    return None


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
        horizontal_alignment = "right"
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
                ha=horizontal_alignment,
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
            ha=horizontal_alignment,
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


def plot_symbol_graph(
    run_dir: Path,
    *,
    plot_symbol: str,
    books: pd.DataFrame,
    news: pd.DataFrame,
    metadata: dict[str, Any],
    graphs_dir: Path,
) -> Path | None:
    import pandas as pd

    try:
        plt = _load_pyplot()
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return None

    symbol_books = books[books["symbol"] == plot_symbol].sort_values("monotonic_ns").reset_index(drop=True)
    if symbol_books.empty:
        return None

    relevant_earnings_news = _relevant_earnings_news(news, plot_symbol, metadata)
    relevant_symbol_news = _relevant_symbol_news(news, plot_symbol)
    base_monotonic_ns = symbol_books["monotonic_ns"].iloc[0]
    start_time_label = None
    if "wall_time_iso" in symbol_books.columns and pd.notna(symbol_books["wall_time_iso"].iloc[0]):
        start_time_label = str(symbol_books["wall_time_iso"].iloc[0]).replace("T", " ")
        if "." in start_time_label:
            start_time_label = start_time_label.split(".", 1)[0]

    symbol_books = symbol_books.copy()
    relevant_earnings_news = relevant_earnings_news.copy()
    relevant_symbol_news = relevant_symbol_news.copy()
    symbol_books["time_s"] = (symbol_books["monotonic_ns"] - base_monotonic_ns) / 1_000_000_000
    relevant_earnings_news["time_s"] = (relevant_earnings_news["monotonic_ns"] - base_monotonic_ns) / 1_000_000_000
    relevant_symbol_news["time_s"] = (relevant_symbol_news["monotonic_ns"] - base_monotonic_ns) / 1_000_000_000

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

    if start_time_label is not None:
        x_label = f"seconds since run start ({start_time_label})"
    else:
        x_label = "seconds since run start"
    if news_ax is not None:
        news_ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label)

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
    books = _load_csv(run_dir / "raw_book_events.csv").sort_values("monotonic_ns").reset_index(drop=True)
    news = _load_csv(run_dir / "raw_news_events.csv").sort_values("monotonic_ns").reset_index(drop=True)
    metadata = _load_metadata(run_dir)
    summary = build_earnings_summary(run_dir)
    output_path = run_dir / output_name
    summary.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    if create_plots:
        graphs_dir = run_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        for plot_symbol in _plot_symbols(metadata):
            plot_path = plot_symbol_graph(
                run_dir,
                plot_symbol=plot_symbol,
                books=books,
                news=news,
                metadata=metadata,
                graphs_dir=graphs_dir,
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
