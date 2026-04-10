from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any

from .logger import load_trace_events


def _load_pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "market_a_v3_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _content_text(event: dict[str, Any]) -> str | None:
    raw = event.get("raw_payload") or {}
    if isinstance(raw, dict):
        new_data = raw.get("new_data") or {}
        content = new_data.get("content")
        if content is None:
            content = raw.get("content")
        if content is not None:
            return str(content)
    content = event.get("content")
    return None if content is None else str(content)


def _extract_book_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") not in {"book_update", "decision_evaluated", "fill_state"}:
            continue
        mid = event.get("mid")
        if mid is None:
            continue
        rows.append(
            {
                "time_ms": int(event.get("monotonic_ms", 0)),
                "mid_px": float(mid),
            }
        )
    rows.sort(key=lambda item: item["time_ms"])
    return rows


def _extract_news_rows(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    earnings_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []
    last_eps: float | None = None
    for event in sorted(events, key=lambda item: int(item.get("monotonic_ms", 0))):
        if event.get("event_type") != "news_received":
            continue
        raw = event.get("raw_payload") or {}
        if not isinstance(raw, dict):
            raw = {}
        new_data = raw.get("new_data") or {}
        if raw.get("kind") == "structured" and new_data.get("structured_subtype") == "earnings" and (new_data.get("asset") or raw.get("symbol")) == "A":
            try:
                new_eps = float(new_data.get("value"))
            except (TypeError, ValueError):
                continue
            if last_eps is None:
                label = f"A EPS START -> {new_eps:.2f}"
            else:
                label = f"A EPS {last_eps:.2f} -> {new_eps:.2f}"
            earnings_rows.append(
                {
                    "time_ms": int(event.get("monotonic_ms", 0)),
                    "label": label,
                }
            )
            last_eps = new_eps
        elif raw.get("kind") == "unstructured":
            label = _content_text(event)
            if label is None or not label.strip():
                label = "A news"
            news_rows.append(
                {
                    "time_ms": int(event.get("monotonic_ms", 0)),
                    "label": label,
                }
            )
    return earnings_rows, news_rows


def _extract_pnl_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("mtm_pnl_estimate") is None:
            continue
        rows.append(
            {
                "time_ms": int(event.get("monotonic_ms", 0)),
                "total_pnl": float(event.get("mtm_pnl_estimate") or 0.0),
                "shock_pnl": float(event.get("shock_pnl") or event.get("mtm_pnl_estimate") or 0.0),
                "mm_pnl": float(event.get("mm_pnl") or 0.0),
            }
        )
    rows.sort(key=lambda item: item["time_ms"])
    return rows


def _draw_event_lines(ax, earnings_rows: list[dict[str, Any]], news_rows: list[dict[str, Any]]) -> None:
    for row in news_rows:
        ax.axvline(row["time_s"], color="green", linestyle="--", alpha=0.45, linewidth=1.0)
    for row in earnings_rows:
        ax.axvline(row["time_s"], color="red", linestyle="--", alpha=0.5, linewidth=1.0)


def _place_news_lane_labels(fig, news_ax, news_rows: list[dict[str, Any]]) -> None:
    from matplotlib.transforms import blended_transform_factory

    if not news_rows:
        return

    transform = blended_transform_factory(news_ax.transData, news_ax.transAxes)
    renderer = None
    placed_bboxes = []
    max_rows = max(4, min(8, len(news_rows) + 1))
    row_step = 0.18
    base_row_y = 0.12

    for row in news_rows:
        wrapped_label = textwrap.fill(str(row["label"]), width=24)
        event_time_s = float(row["time_s"])
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


def generate_pnl_graph(run_dir: str | Path, *, output_dir: str | Path | None = None, events: list[dict[str, Any]] | None = None) -> Path | None:
    plt = _load_pyplot()
    from matplotlib.lines import Line2D

    run_path = Path(run_dir).expanduser().resolve()
    loaded_events = load_trace_events(run_path) if events is None else list(events)
    if not loaded_events:
        return None

    book_rows = _extract_book_rows(loaded_events)
    pnl_rows = _extract_pnl_rows(loaded_events)
    earnings_rows, news_rows = _extract_news_rows(loaded_events)
    if not book_rows or not pnl_rows:
        return None

    base_time_ms = min(float(book_rows[0]["time_ms"]), float(pnl_rows[0]["time_ms"]))
    for row in book_rows:
        row["time_s"] = (row["time_ms"] - base_time_ms) / 1_000.0
    for row in pnl_rows:
        row["time_s"] = (row["time_ms"] - base_time_ms) / 1_000.0
    for row in earnings_rows:
        row["time_s"] = (row["time_ms"] - base_time_ms) / 1_000.0
    for row in news_rows:
        row["time_s"] = (row["time_ms"] - base_time_ms) / 1_000.0

    target_root = run_path if output_dir is None else Path(output_dir).expanduser().resolve()
    graphs_dir = target_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output_path = graphs_dir / "pnl_A.png"

    fig, (mid_ax, event_ax, pnl_ax) = plt.subplots(
        3,
        1,
        figsize=(11.5, 8.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.7, 2.6, 2.8], "hspace": 0.05},
    )

    mid_times = [row["time_s"] for row in book_rows]
    mid_prices = [row["mid_px"] for row in book_rows]
    pnl_times = [row["time_s"] for row in pnl_rows]

    mid_line, = mid_ax.plot(mid_times, mid_prices, color="black", linewidth=1.2, label="A mid")
    total_line, = pnl_ax.plot(pnl_times, [row["total_pnl"] for row in pnl_rows], color="#1f77b4", linewidth=1.3, label="total pnl")
    shock_line, = pnl_ax.plot(pnl_times, [row["shock_pnl"] for row in pnl_rows], color="#d62728", linewidth=1.1, label="shock pnl")
    mm_line, = pnl_ax.plot(pnl_times, [row["mm_pnl"] for row in pnl_rows], color="#2ca02c", linewidth=1.1, label="mm pnl")
    pnl_ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)

    _draw_event_lines(mid_ax, earnings_rows, news_rows)
    _draw_event_lines(event_ax, earnings_rows, news_rows)
    _draw_event_lines(pnl_ax, earnings_rows, news_rows)

    y_max = max(mid_prices)
    y_min = min(mid_prices)
    y_span = y_max - y_min
    base_label_y = y_max + 0.03 * y_span
    max_label_y = base_label_y

    for index, row in enumerate(earnings_rows):
        label_y = base_label_y + (index % 2) * max(0.05 * y_span, 1.0)
        max_label_y = max(max_label_y, label_y)
        mid_ax.text(
            row["time_s"],
            label_y,
            row["label"],
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

    mid_ax.set_ylabel("A mid_px")
    mid_ax.set_ylim(top=max_label_y + max(0.12 * y_span, 2.0))
    pnl_ax.set_ylabel("PnL")
    pnl_ax.set_xlabel("seconds since run start")

    event_ax.set_ylim(0.0, 1.0)
    event_ax.set_yticks([])
    event_ax.set_ylabel("news")
    event_ax.spines["top"].set_visible(False)
    event_ax.spines["right"].set_visible(False)
    event_ax.spines["left"].set_visible(False)
    _place_news_lane_labels(fig, event_ax, news_rows)

    mid_ax.legend(handles=[mid_line], loc="best")
    pnl_legend_handles = [total_line, shock_line, mm_line]
    if earnings_rows:
        pnl_legend_handles.append(Line2D([0], [0], color="red", linestyle="--", label="earnings"))
    if news_rows:
        pnl_legend_handles.append(Line2D([0], [0], color="green", linestyle="--", label="A news"))
    pnl_ax.legend(handles=pnl_legend_handles, loc="best")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_run_graphs(run_dir: str | Path, *, output_dir: str | Path | None = None, events: list[dict[str, Any]] | None = None) -> list[Path]:
    output = generate_pnl_graph(run_dir, output_dir=output_dir, events=events)
    return [] if output is None else [output]
