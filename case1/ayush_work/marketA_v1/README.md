# A-Only Market-Making Project

This package builds an end-to-end A-only research, replay, and live trading workflow under `case1/ayush_work`.

The design goal is shared logic:
- one loader for the historical data in `data_scraping/data`
- one feature and fair-value stack
- one strategy engine reused by replay and live
- one output directory for diagnostics and iteration artifacts

## What Was Built

- `data_loader.py`
  Supports all observed A-related log layouts in this repo:
  - `A_research_*` runs with `raw_book_events.csv`, `raw_trade_events.csv`, `raw_news_events.csv`
  - older `market_research_*` derived runs with `raw_book_events.csv`
  - newer `market_research_*` raw runs with `raw_book_snapshots_A.csv` plus `raw_book_updates_A.csv`

- `features.py`
  Maintains rolling market state for A and computes:
  - best bid / ask
  - mid
  - spread
  - microprice
  - top-of-book imbalance
  - realized volatility
  - recent trade intensity
  - signed trade-pressure proxy
  - short-horizon own-fill pressure

- `fair_value.py`
  Tracks the latest A earnings fair value using:

  `fair_px = round(pe_ratio * price_scale * earnings)`

  `pe_ratio` is stored in natural units near `10`, while `price_scale` defaults to `100` to match exchange integer prices.

- `market_maker.py`, `regimes.py`, `risk.py`, `execution.py`
  Shared strategy core with:
  - normal market making
  - post-earnings shock mode
  - overshoot fade mode
  - inventory unwind mode
  - hard risk-off mode

- `backtest/`
  Conservative event-driven replay with passive and aggressive fill simulation.

- `research/`
  Scripts for PE fitting, parameter sweeps, diagnostics, and offline earnings-transition analysis.

- `live/a_bot.py`
  A live A-only bot that subclasses the repo’s checked-in `XChangeClient`.

## Strategy Summary

### Baseline market making

When no fresh A earnings shock is active, the strategy:
- uses the latest earnings fair if available, otherwise the current mid
- nudges fair with microprice and recent trade pressure
- applies inventory-aware reservation pricing
- widens quotes with spread, short-horizon volatility, toxicity, and recent one-sided fill pressure
- crosses the spread only when estimated edge is meaningfully large

### Structured earnings response

On structured A earnings:
- the fair value is updated immediately
- the bot enters `EARNINGS_SHOCK` mode for `2.5s`
- aggressive crossing is allowed with a lower edge threshold
- passive quotes become directionally biased toward the new fair

After the initial shock window:
- the bot can enter `OVERSHOOT_FADE` mode for up to `7s`
- this only activates when price is clearly through fair and recent trade pressure is no longer strongly reinforcing the move

### Deferred work

Unstructured news is intentionally deferred in v1.

The earnings-transition model is offline-only and does not affect live trading.

## Replay Assumptions

The replay is designed for consistency across variants, not perfect matching-engine realism.

Passive fills are conservative:
- fills are assumed when trades clearly hit our price
- or when book evolution clearly removes the visible queue ahead of us at our level

Aggressive fills are also conservative:
- they only fill when the current top of book is actually marketable versus our crossing limit
- they fill at the visible best level, capped by visible top-of-book size

Mark-to-market defaults to:
1. current mid
2. last trade
3. model fair

## Outputs

Scripts write to `case1/ayush_work/marketA_v1/outputs`.

Expected files:
- `metrics_summary.csv`
- `per_session_metrics.csv`
- `fill_decomposition.csv`
- `inventory_path.csv`
- `earnings_event_analysis.csv`
- `pe_fit_summary.csv`
- `parameter_sweep_results.csv`
- `session_diagnostics.csv`
- `earnings_transition_summary.csv`
- `best_params.json`
- `best_params_summary.json`

## How To Run

From repo root, use the repo virtualenv:

```bash
./.venv/bin/python -m case1.ayush_work.marketA_v1.research.diagnostics
./.venv/bin/python -m case1.ayush_work.marketA_v1.research.fit_pe_model
./.venv/bin/python -m case1.ayush_work.marketA_v1.backtest.simulator
./.venv/bin/python -m case1.ayush_work.marketA_v1.research.fit_mm_params
./.venv/bin/python -m case1.ayush_work.marketA_v1.research.fit_earnings_transition
```

`fit_mm_params` sweeps on a representative subset by default, then validates the winning configuration on the full selected session set.

Run tests:

```bash
./.venv/bin/python -m unittest discover case1/ayush_work/marketA_v1/tests
```

Run the live bot:

```bash
./.venv/bin/python -m case1.ayush_work.marketA_v1.live.a_bot
```

Live credentials are loaded from:
- `case1/ayush_work/marketA_v1/local_config.json`

Optional config overrides can be supplied with `A_BOT_CONFIG_PATH` pointing to a JSON file shaped like:

```json
{
  "risk": {
    "open_volume_limit": 80
  },
  "strategy": {
    "initial_pe_ratio": 10.0,
    "base_half_spread_px": 3,
    "inventory_penalty": 0.45
  }
}
```

## Known Limitations

- News-time interpolation for new raw layouts is approximate because only news callbacks carry exchange tick timing.
- Passive fills remain a heuristic because the logs are not full queue-position ground truth.
- Live aggressive orders are implemented as marketable limits at the current touch and may need tighter cancel policies after real trading feedback.
- Parameter sweeps are discrete and dependency-light by design; there is no heavy optimizer in v1.
