from __future__ import annotations

import argparse

from ..config import load_app_config
from ..data_loader import build_run_catalog
from ..fair_value import fit_pe_ratio, write_pe_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the A PE ratio from historical earnings reactions.")
    parser.add_argument("--config", help="Optional JSON config override path.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap on sessions processed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    catalog = build_run_catalog(config.paths.data_root)
    sessions = catalog.sessions[: args.max_sessions] if args.max_sessions else catalog.sessions
    result = fit_pe_ratio(
        sessions,
        price_scale=config.strategy.price_scale,
        default_pe_ratio=config.strategy.initial_pe_ratio,
    )
    write_pe_outputs(result, config.paths.output_root)


if __name__ == "__main__":
    main()
