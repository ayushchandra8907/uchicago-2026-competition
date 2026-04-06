# MarketA_v1 Postmortem

## Scope

This directory was an A-only market-making research and trading project built under:

`case1/ayush_work/marketA_v1`

The goal was to:

- load historical A-related logs from `data_scraping/data`
- fit a fair-value model for asset A using structured earnings
- build a replay/backtest environment
- build a live A-only bot using `utcxchangelib`
- iterate from live run traces and improve the bot over multiple versions

This postmortem exists because the project was ultimately not working well enough in live trading and the directory was intentionally deleted.

## What The Project Contained

At its largest, the directory contained:

- `data_loader.py`
  - schema-aware loader for multiple historical log layouts
- `features.py`
  - book and trade features such as mid, spread, microprice, imbalance, volatility, and trade pressure
- `fair_value.py`
  - PE-based earnings fair value model for A
- `market_maker.py`
  - the strategy core
- `regimes.py`
  - mode selection logic
- `risk.py`
  - position and exposure controls
- `execution.py`
  - quote synchronization logic
- `backtest/`
  - replay engine, simulator, metrics
- `research/`
  - PE fitting, diagnostics, parameter sweep, offline earnings transition analysis
- `live/a_bot.py`
  - live exchange bot
- `live_trace.py`
  - structured trace logging for live runs
- `analyze_live_run.py`
  - post-run summarizer
- `tests/`
  - unit tests for loader, features, strategy, replay, live adapter, and trace logic
- `outputs/`
  - replay metrics and research artifacts
- `analysis_runs/`
  - timestamped live trace folders

## High-Level Design

The architecture tried to share one strategy core across replay and live:

- one `StrategyEngine`
- one feature stack
- one fair value model
- one execution model
- one risk layer

The idea was sound in theory:

- avoid live/replay divergence
- keep one set of parameters
- learn from backtests and live traces iteratively

In practice, the project failed because the replay model and live behavior diverged in the places that mattered most:

- passive fill realism
- short-term adverse selection
- order churn
- execution timing
- stale/faulty fair value anchoring

## Strategy Iterations That Were Tried

The project went through several strategy phases.

### 1. Fuller multi-mode strategy

Early versions had:

- `NORMAL_MM`
- `EARNINGS_SHOCK`
- `OVERSHOOT_FADE`
- `INVENTORY_UNWIND`
- `RISK_OFF`

It used:

- earnings fair value
- microprice and trade-pressure adjustments
- adaptive spreads
- inventory-aware reservation pricing
- optional aggressive crossing
- overshoot/reversion logic

This became too complex too early.

Main failure:

- the fair-value model was often wrong or stale in live conditions
- the strategy still trusted it enough to post bad quotes
- mode logic made debugging harder

### 2. Conservative no-crossing baseline

Later versions disabled:

- normal aggressive crossing
- shock aggressive crossing
- overshoot aggression

This reduced risk, but it still performed poorly live because:

- the bot kept getting picked off passively
- it still relied on fair-value nudges that were not reliably predictive
- replay still overstated quality

### 3. Very simple passive bot

The final version before deletion was simplified down to:

- two-sided passive quoting only when flat
- capped earnings response in a short shock window
- one-sided passive flattening when inventory was nonzero
- no aggressive orders
- slower requoting
- reduced cancel-replace sensitivity

This was deliberately much simpler than the earlier versions, but it still was not validated as a good live strategy before deletion.

## Key Technical Mistakes

### 1. The backtester was initially too optimistic

This was one of the biggest problems in the entire project.

Earlier replay assumptions credited passive fills too easily:

- fills were inferred from book movement in ways that were too generous
- cancel/replace behavior was too idealized
- queue position realism was weak

This created misleading replay results, including earlier outputs that suggested the bot could make positive average PnL while live performance stayed consistently negative.

That meant replay was not a trustworthy filter for strategy quality.

This was eventually corrected by:

- disabling book-move passive fills by default
- adding minimum quote age
- deferring replacement to the next event

After those corrections, replay results became much weaker and much more believable.

That was the correct direction, but it also made clear that the strategy itself had little proven edge.

### 2. Live passive fills were toxic

Across multiple live runs, the bot was getting bad passive fills:

- buys often had negative short-horizon markouts
- sells often had negative short-horizon markouts
- in some runs the average buy price exceeded the average sell price by a lot

That is structurally fatal for a market maker.

It means the bot was not capturing spread; it was being adversely selected.

### 3. Earnings fair was too influential relative to the live market

The project assumed A fair value should be tied to:

`fair = PE * earnings`

This is directionally correct from the case description, but in live trading:

- the market did not always instantly reflect that fair
- the fitted PE and reaction windows were noisy
- stale earnings fair could remain misaligned with the live mid

When the strategy treated this model fair too seriously, it posted quotes far from what the market was actually trading.

That caused:

- one-sided fills
- inventory drift
- bad shock-mode selling/buying

Even after later caps and clamps were added, this remained a recurring source of bad pricing.

### 4. Too much order churn

One of the later critical findings from live traces was that the bot spent too much time canceling and replacing orders.

In one bad run:

- `2,641` orders were submitted
- `2,274` cancel requests were sent
- only `511` fills occurred

Average fill lifetime was only about `315 ms`.

That means the bot was acting too twitchy rather than calmly making markets.

This matched the observed problem:

- it would get into a position
- then burn time re-quoting instead of working an orderly exit

Later changes reduced this by:

- increasing requote cooldown
- introducing a replacement threshold in ticks
- simplifying the regime logic

But the project never got far enough after those changes to prove a live turnaround.

### 5. Complexity outran evidence

The project added complexity faster than it proved value.

Examples:

- multi-regime logic
- overshoot fade behavior
- toxicity filters
- fair-value clamps
- inventory unwind variations
- replay heuristics

Each change was individually defensible, but together they made the system hard to reason about.

The result was that the strategy could be “locally patched” after each failure without ever demonstrating a stable core edge.

## Important Live Run Findings

### Run: `a_live_20260405_131952_c829092b`

Observed:

- estimated MTM PnL around `-13,803`
- toxic markouts on both sides
- average buy price materially above average sell price
- `EARNINGS_SHOCK` and `NORMAL_MM` both losing via adverse selection

Meaning:

- this was not a pure inventory-blowout issue
- pricing and fill quality were the main problem

### Run: `a_live_20260405_202833_0d91c016`

Observed:

- estimated MTM PnL around `-5,700`
- `380` passive fills
- `0` aggressive fills
- average inventory only around `2.1`
- `4,445` cancel requests
- buys and sells both had negative short-horizon markouts

Meaning:

- the bot was not losing because it took huge positions
- it was losing because it kept getting bad passive fills and churning

### Run: `a_live_20260405_211141_0906eb5d`

Observed:

- estimated MTM PnL around `-24,666`
- `511` passive fills
- `0` aggressive fills
- largest long inventory `+7`
- largest short inventory `-7`
- average buy price about `1057.16`
- average sell price about `1034.96`
- `2,274` cancel requests
- buy-side 1-second markout about `-9.01`

Meaning:

- again, not a position-limit problem
- execution quality was poor
- the bot was too active in quote maintenance and too weak in selecting where to rest

## Why The Backtester Became Distrusted

At one point the replay outputs claimed positive average PnL on historical data while live runs were consistently bad.

That discrepancy damaged trust in the framework.

The postmortem conclusion is:

- the replay was useful for structural debugging
- it was not good enough to certify live profitability

The biggest replay gap was passive execution realism.

If this work is ever restarted, replay should be treated as:

- a falsification tool
- a regression detector
- a sanity check

It should not be treated as proof that a strategy has edge.

## What Was Learned

### Things that were useful

- unified loader for multiple data layouts
- structured live trace capture
- post-run summaries with:
  - fill markouts
  - buy/sell imbalance
  - cancel counts
  - average fill prices
  - inventory extremes
- discovering that inventory size was often not the real issue
- discovering that execution quality and churn were worse than initially assumed

### Things that did not work well enough

- PE-based earnings fair as a central live anchor
- complex multi-mode strategy before proving baseline edge
- optimistic passive fill replay
- repeatedly patching behavior after losses without proving a stable positive core

## Likely Root-Cause Summary

The project failed because it never established a strong simple edge in passive market making.

More specifically:

1. the fair-value signal was too weak or too noisy for how strongly it influenced quoting
2. passive fills were often adverse-selected
3. the bot churned quotes too much
4. replay overstated performance for too long
5. complexity accumulated faster than confidence

In plain terms:

- the bot was usually not making money from spread capture
- it was mostly donating edge through bad passive execution

## If Rebuilt Later

If a future A-only project is started from scratch, it should probably begin with:

1. a very small, very calm passive market maker
2. no complex regime stack at first
3. no strong reliance on model fair outside a very narrow event window
4. stronger live-only diagnostics from day one
5. replay assumptions that are intentionally harsher than comfortable

And the standard for adding complexity should be:

- only add one new behavior at a time
- only keep it if live evidence is clearly positive

## Final State Before Deletion

Immediately before deletion, the last direction of the project was:

- simplify strategy
- reduce quote churn
- quote both sides only when flat
- flatten immediately when inventory became nonzero
- keep earnings response passive and capped

Tests were still passing, but the project had not earned enough confidence to justify keeping the codebase around.

That is why this directory was intentionally reduced to this single summary file.
