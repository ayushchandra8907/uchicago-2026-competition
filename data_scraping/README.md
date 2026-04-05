# Market Research Logger

This folder now uses a two-stage pipeline:

1. `market_research_logger.py` is a lightweight live collector that only writes compact raw exchange data.
2. `analyze_logs.py` runs afterward and reconstructs book state, computes mid/spread offline, and creates plots.

The live logger does not compute features, does not create heartbeats, and does not auto-run analysis when the round ends.

## Files

- `market_research_logger.py`: live passive raw-data collector
- `analyze_logs.py`: offline reconstruction, summary generation, and plotting
- `config.py`: reads settings from `local_config.json`
- `local_config.json`: one-time local credentials and logger settings
- `csv_writer.py`: buffered append-safe CSV writer

## Normal Workflow

On macOS in VSCode:

1. Open `data_scraping/local_config.json`.
2. Edit `host`, `username`, and `password` if needed.
3. Go to Run and Debug.
4. Select `Market Research Logger`.
5. Click Run at the start of the round.
6. Stop it when the round is over.
7. After the run, generate plots manually:

```bash
python3 data_scraping/analyze_logs.py --plot data_scraping/data/market_research_YYYY-MM-DD_HH-MM-SS_live_round
```

That is the intended workflow. The live logger only collects data. The analyzer does all reconstruction and plotting afterward.

## What Happens On Each Run

The live logger writes one run folder under:

```text
data_scraping/data/market_research_YYYY-MM-DD_HH-MM-SS/
```

That folder contains compact raw CSVs plus `session_metadata.json`.

No mid, spread, microprice, imbalance, volatility, or other derived fields are written during the live run.

## Raw Output Files

### `raw_book_snapshots_<SYMBOL>.csv`

One row per full book snapshot for that symbol.

Examples:

- `raw_book_snapshots_A.csv`
- `raw_book_snapshots_B.csv`
- `raw_book_snapshots_C.csv`
- `raw_book_snapshots_ETF.csv`

Columns:

- `message_index`
- `symbol`
- `bids_json`
- `asks_json`

These are compact serializations of the exact snapshot levels received from the exchange.

### `raw_book_updates_<SYMBOL>.csv`

One row per incremental book update for that symbol.

Examples:

- `raw_book_updates_A.csv`
- `raw_book_updates_B.csv`
- `raw_book_updates_C.csv`
- `raw_book_updates_ETF.csv`

Columns:

- `message_index`
- `symbol`
- `side`
- `px`
- `dq`

This is the raw exchange delta stream. No reconstructed top-of-book fields are added here.

### `raw_trade_events_<SYMBOL>.csv`

One row per trade message for that symbol.

Examples:

- `raw_trade_events_A.csv`
- `raw_trade_events_B.csv`
- `raw_trade_events_C.csv`
- `raw_trade_events_ETF.csv`

Columns:

- `message_index`
- `symbol`
- `price`
- `qty`

### `raw_news_events.csv`

One row per inbound news callback from the exchange. This file is no longer filtered.

Columns:

- `message_index`
- `tick`
- `tick_ms`
- `kind`
- `symbol`
- `message_type`
- `structured_subtype`
- `earnings_asset`
- `earnings_value`
- `petition_asset`
- `petition_new_signatures`
- `petition_cumulative`
- `cpi_forecast`
- `cpi_actual`
- `raw_content`
- `normalized_content`

Important behavior:

- structured earnings arrive as numeric values
- unstructured stock news arrives as literal text in `raw_content`
- `tick_ms` is just `tick * 200`

## Why `message_index` Exists

The exchange envelope includes a global message index. We log that directly because:

- book updates do not carry exchange tick
- trades do not carry exchange tick
- news does carry exchange tick

Offline, `analyze_logs.py` uses news ticks plus message indices to infer an approximate exchange-time axis for book updates and trades. That keeps the live logger raw while still allowing price-vs-signal plots later.

## Offline Outputs

When you run:

```bash
python3 data_scraping/analyze_logs.py --plot data_scraping/data/market_research_YYYY-MM-DD_HH-MM-SS_live_round
```

the analyzer reconstructs each symbol’s book from snapshots plus updates, then writes:

### `earnings_event_summary.csv`

A compact summary of earnings reactions for the plotted symbols.

Important columns:

- `plot_symbol`
- `earnings_asset`
- `news_message_index`
- `exchange_tick`
- `exchange_time_ms`
- `old_eps`
- `new_eps`
- `model_fair_value_jump`
- `market_mid_before_news`
- `spread_before_news`
- `market_mid_after_100ms`
- `market_mid_after_250ms`
- `market_mid_after_500ms`
- `market_mid_after_1s`
- `market_mid_after_2s`
- `market_mid_after_5s`
- `max_excursion_from_pre_news_mid`
- `final_settling_move_5s`

### `graphs/`

Expected files:

- `graphs/A_mid_price.png`
- `graphs/B_mid_price.png`
- `graphs/C_mid_price.png`
- `graphs/ETF_mid_price.png`

Each plot shows:

- reconstructed mid-price over time
- red dashed earnings markers where applicable
- a lower news lane with green dashed non-earnings markers
- green labels using the literal logged headline/content text

Current earnings/news plotting rules:

- `A` graph: only `A` earnings
- `C` graph: only `C` earnings, plus non-earnings news that is not directly about `A`
- `ETF` graph: both `A` and `C` earnings
- `B` graph: no earnings markers by default

For unstructured news markers, a graph only gets green lines when that same symbol is tagged or explicitly mentioned in the news text.

## Config Notes

The settings that still matter in `local_config.json` are:

- `host`
- `username`
- `password`
- `monitored_symbols`
- `plot_symbols`
- `direct_earnings_symbols`
- `etf_news_assets`
- `pe_constants`
- `log_root`
- `run_label`

`pe_constants` is only used offline for the optional `model_fair_value_jump` calculation in the summary. It does not affect raw collection.

## Notes

- The logger never places trades.
- The live collector is intentionally small: no derived features, no heartbeat snapshots, no automatic plotting.
- Buffered CSV writes are used for the high-frequency streams to reduce I/O overhead during the round.
- Because book/trade messages do not include exchange tick, the analyzer’s time axis is approximate and inferred from nearby news ticks.
