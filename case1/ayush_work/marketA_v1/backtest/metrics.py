from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..models import BacktestResult


def summarize_results(results: list[BacktestResult]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_session_rows: list[dict[str, float | int | str]] = []
    fill_rows: list[dict[str, float | int | str | bool]] = []
    inventory_rows: list[dict[str, float | int | str]] = []
    mode_pnl_totals: defaultdict[str, float] = defaultdict(float)

    for result in results:
        row = {
            "session_id": result.session_id,
            "total_pnl": result.total_pnl,
            "final_inventory": result.final_inventory,
            "final_cash": result.final_cash,
            "mark_px": result.mark_px,
            "max_drawdown": result.max_drawdown,
            "passive_fill_count": result.passive_fill_count,
            "aggressive_fill_count": result.aggressive_fill_count,
        }
        for mode, pnl in sorted(result.pnl_by_mode.items()):
            row[f"pnl_mode_{mode.lower()}"] = pnl
            mode_pnl_totals[mode] += pnl
        per_session_rows.append(row)

        for time_ms, inventory in result.inventory_path:
            inventory_rows.append({"session_id": result.session_id, "time_ms": time_ms, "inventory": inventory})
        for fill in result.fills:
            fill_rows.append(
                {
                    "session_id": result.session_id,
                    "time_ms": fill.time_ms,
                    "side": fill.side,
                    "px": fill.px,
                    "qty": fill.qty,
                    "aggressive": fill.aggressive,
                    "mode": fill.mode,
                    "edge_px": fill.edge_px,
                }
            )

    per_session = pd.DataFrame(per_session_rows)
    fill_df = pd.DataFrame(fill_rows)
    inventory_df = pd.DataFrame(inventory_rows)

    summary = pd.DataFrame(
        [
            {
                "session_count": int(len(results)),
                "avg_pnl": float(per_session["total_pnl"].mean()) if not per_session.empty else 0.0,
                "median_pnl": float(per_session["total_pnl"].median()) if not per_session.empty else 0.0,
                "stdev_pnl": float(per_session["total_pnl"].std(ddof=0)) if len(per_session) > 1 else 0.0,
                "min_pnl": float(per_session["total_pnl"].min()) if not per_session.empty else 0.0,
                "max_pnl": float(per_session["total_pnl"].max()) if not per_session.empty else 0.0,
                "avg_max_drawdown": float(per_session["max_drawdown"].mean()) if not per_session.empty else 0.0,
                "passive_fill_count": int(per_session["passive_fill_count"].sum()) if not per_session.empty else 0,
                "aggressive_fill_count": int(per_session["aggressive_fill_count"].sum()) if not per_session.empty else 0,
                "avg_abs_final_inventory": float(per_session["final_inventory"].abs().mean()) if not per_session.empty else 0.0,
                **{f"total_pnl_mode_{mode.lower()}": pnl for mode, pnl in sorted(mode_pnl_totals.items())},
            }
        ]
    )
    return summary, per_session, fill_df, inventory_df


def write_metrics_outputs(results: list[BacktestResult], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary, per_session, fill_df, inventory_df = summarize_results(results)
    summary.to_csv(output_root / "metrics_summary.csv", index=False)
    per_session.to_csv(output_root / "per_session_metrics.csv", index=False)
    fill_df.to_csv(output_root / "fill_decomposition.csv", index=False)
    inventory_df.to_csv(output_root / "inventory_path.csv", index=False)
