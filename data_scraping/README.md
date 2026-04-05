# Market Research Logger

This folder contains a passive research logger for the UChicago Trading Competition exchange. It listens through `XChangeClient`, never trades, logs market data for `A`, `B`, `C`, and `ETF`, and automatically creates plots when the run ends.

## Files

- `market_research_logger.py`: live passive logger for the full data pipeline
- `config.py`: reads settings from `local_config.json`
- `local_config.json`: one-time local credentials and logger settings
- `feature_extractor.py`: order-book and short-horizon feature helpers
- `csv_writer.py`: append-safe CSV writer
- `analyze_logs.py`: multi-symbol offline analysis and plot generation

## Normal Workflow

On macOS in VSCode:

1. Open `data_scraping/local_config.json`.
2. Edit `host`, `username`, and `password` if needed.
3. Go to Run and Debug.
4. Select `Market Research Logger`.
5. Click Run at the start of the round.
6. Stop it when the round is over.

That is the full workflow. You do not need environment variables or command-line flags.

If you want terminal fallback once in a while, it is still just:

```bash
python3 data_scraping/market_research_logger.py
```

## What Happens On Each Run

The logger writes one run folder under:

```text
data_scraping/data/market_research_YYYY-MM-DD_HH-MM-SS/
```

When you stop the logger, it automatically runs:

```text
python3 data_scraping/analyze_logs.py --run-dir <that_run_dir> --plot
```

So every run ends with:

- raw CSV logs
- a summary CSV
- a `graphs/` folder with plots

## Symbols And Earnings Markers

The logger records market data for:

- `A`
- `B`
- `C`
- `ETF`

The plots use earnings markers like this:

- `A` graph: only `A` earnings
- `C` graph: only `C` earnings
- `ETF` graph: both `A` and `C` earnings
- `B` graph: no earnings markers by default

Those defaults come from `local_config.json`.

## Output Files

### `raw_book_events.csv`

One row per monitored-symbol book update, plus heartbeat snapshots.

This combined file contains all monitored symbols together. You also get separate per-symbol book files:

- `raw_book_events_A.csv`
- `raw_book_events_B.csv`
- `raw_book_events_C.csv`
- `raw_book_events_ETF.csv`

Important columns:

- `symbol`
- `event_type`
- `wall_time_iso`, `wall_time_ns`, `monotonic_ns`, `exchange_tick`
- `best_bid_px`, `best_bid_qty`, `best_ask_px`, `best_ask_qty`
- `mid_px`, `spread`, `microprice`, `top_of_book_imbalance`
- `bid_levels_json`, `ask_levels_json`
- `current_position_symbol`, `current_cash`
- `last_trade_px_for_symbol`, `last_trade_qty_for_symbol`
- `most_recent_news_id_affecting_symbol`
- `seconds_since_last_news_for_symbol`

### `raw_trade_events.csv`

One row per trade print for any monitored symbol.

This combined file contains all monitored symbols together. You also get separate per-symbol trade files:

- `raw_trade_events_A.csv`
- `raw_trade_events_B.csv`
- `raw_trade_events_C.csv`
- `raw_trade_events_ETF.csv`

Important columns:

- `symbol`, `trade_px`, `trade_qty`
- `mid_at_trade`, `spread_at_trade`
- `time_since_last_symbol_news`
- `latest_known_eps_for_symbol`

### `raw_news_events.csv`

Logged whenever news affects at least one monitored symbol.

Important columns:

- `kind`
- `structured_subtype`
- `earnings_asset`
- `earnings_value`
- `affected_symbols_json`
- `previous_known_eps_for_asset`
- `new_known_eps_for_asset`
- `inferred_fair_price_before`
- `inferred_fair_price_after`
- `raw_content`, `normalized_content`

### `derived_feature_rows.csv`

A research-friendly aligned feature table for all monitored symbols. Each row is tied to one symbol and is built from either a book event or a news event.

Important columns:

- `symbol`
- best bid/ask, mid, spread, microprice, imbalance
- trailing trade counts and volumes
- trailing realized volatility
- past returns and future returns
- `latest_known_eps_for_symbol`
- `naive_fair_price_for_symbol`
- `seconds_since_last_news_for_symbol`
- pre/post news context columns for news-triggered rows

By default, `naive_fair_price_for_symbol` is only populated for `A`. `C` is intentionally left without a constant-PE fair-value model because its valuation is not constant in this case.

### `earnings_event_summary.csv`

Created automatically when the run ends. This is a combined summary across the plotted symbols.

Important columns:

- `plot_symbol`
- `earnings_asset`
- `old_eps`
- `new_eps`
- `model_fair_value_jump`
- pre-news mid/spread
- post-news mid/spread at `100ms`, `250ms`, `500ms`, `1s`, `2s`, `5s`
- `max_excursion_from_pre_news_mid`
- `final_settling_move_5s`

For `C` and any other symbol without a configured constant PE model, `model_fair_value_jump` will be blank.

### `graphs/`

Created automatically when the run ends.

Expected files:

- `graphs/A_mid_price.png`
- `graphs/B_mid_price.png`
- `graphs/C_mid_price.png`
- `graphs/ETF_mid_price.png`

Each plot shows:

- the symbol mid-price over time
- earnings markers where applicable
- EPS change labels such as `A EPS 0.95 -> 1.03` or `C EPS START -> 0.88`

### `session_metadata.json`

Run metadata including the config used, positions, latest known EPS values, and notes about the exchange API.

## Config Notes

The most important settings in `local_config.json` are:

- `monitored_symbols`
- `plot_symbols`
- `direct_earnings_symbols`
- `etf_news_assets`
- `pe_constants`
- `heartbeat_interval_seconds`
- `log_root`

The default setup already matches the workflow you asked for, so in practice you usually only need to touch:

- `host`
- `username`
- `password`

`pe_constants` is optional. The default config only includes `A`. That means the pipeline will not pretend that `C` has a constant PE ratio.

## Manual Re-Analysis

If you ever want to regenerate summaries or plots for an old run:

```bash
python3 data_scraping/analyze_logs.py --run-dir data_scraping/data/market_research_YYYY-MM-DD_HH-MM-SS --plot
```

## Notes

- The logger never places trades.
- Round/day are left blank because the current user API does not expose them directly.
- CSVs flush on every write to reduce data loss if the process stops unexpectedly.
