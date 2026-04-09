# Market A Strategy Codex Handoff

This document is the handoff package for the live `A` strategy in `marketA_v3`.

Use it when another teammate wants their own Codex thread to plug the exact same `A` trading behavior into a different runner without reverse-engineering the codebase.

## Source Of Truth

These are the files that define live `A` behavior:

- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/market_A_strategy/A_strategy.py`
- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/market_A_strategy/a_news_sentiment.py`
- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/config.py`
- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/core/types.py`
- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/botrunner.py`
- `/Users/ayushchandra/Programming/Competitions/UchicagoTrading/uchicago-2026-competition/case1/ayush_work/marketA_v3/market_A_strategy/a_news_tracker.py`

If this doc and code ever disagree, trust the code.

## Ready-To-Paste Prompt For Partner Codex

Paste this into the partner thread:

```text
You are integrating the existing live A trading strategy from marketA_v3 into a separate runner.

Do not redesign the strategy. Reuse its exact live behavior.

Source files:
- market_A_strategy/A_strategy.py
- market_A_strategy/a_news_sentiment.py
- config.py (StrategyConfig)
- core/types.py

Your job:
1. Import and run AStrategy exactly as written.
2. Feed it StrategySnapshot objects on book / trade / fill / timer / news.
3. Convert exchange news into NewsEvent exactly like marketA_v3 does:
   - structured A earnings:
     - kind="structured"
     - structured_subtype="earnings"
     - symbol or asset = "A"
     - value = earnings/EPS
   - unstructured A headline:
     - kind="unstructured"
     - symbol or asset = "A"
     - content = headline text
4. Honor Decision semantics:
   - desired_order is the order to work
   - cancel_all means cancel live A orders before replacing
   - observe_only means no new order should be sent
5. Enforce execution constraints:
   - max exchange order qty = 40
   - larger targets must be built through smaller slices
   - current slice defaults are 7..15 with target 12
6. Keep runner-managed inventory and order state accurate and feed fills back through on_fill().
7. Preserve logging if practical, but do not change A strategy logic unless explicitly asked.

Important:
- AStrategy is a shock/unwind strategy, not a market-maker.
- It trades only:
  - structured A earnings
  - A-specific unstructured headlines
- Most tuning should happen in config knobs or a_news_sentiment.py, not by rewriting core control flow.

If something behaves badly:
- first inspect StrategyConfig values
- then inspect a_news_sentiment term weights
- only change A_strategy.py if the problem is clearly structural
```

## What A Is

`A` is a single-name event-driven shock strategy.

Tradable event classes:

- structured A earnings
- A-specific unstructured headlines

Main design:

- convert event into fair value shift
- size a basket from confidence
- enter `SHOCK`
- monetize the move
- trim on overshoot
- unwind back to zero on equilibrium, reversal, timeout, or forced flatten

It is not a passive holder. Outside an active shock, target inventory is zero.

## File Responsibilities

### `/market_A_strategy/A_strategy.py`

Owns:

- signal handling
- fair value logic
- target sizing
- shock lifecycle
- overshoot logic
- equilibrium detection
- decay logic
- unwind logic
- flatten-first takeover for new signals

### `/market_A_strategy/a_news_sentiment.py`

Owns:

- normalized headline parsing
- weighted unigram and bigram libraries
- amplifiers and dampeners
- contextual outcome overrides
- score bucket assignment
- unknown-term harvesting

### `/config.py`

Owns:

- `StrategyConfig`
- all A live knobs
- env var overrides for every A live knob

### `/core/types.py`

Owns the runner contract:

- `NewsEvent`
- `DesiredOrder`
- `Decision`
- `StrategySnapshot`

### `/botrunner.py`

Reference for:

- snapshot building
- news mapping
- order slicing
- replace/cancel cadence
- fill handling

### `/market_A_strategy/a_news_tracker.py`

Offline review and dictionary-learning loop.

## Runner Integration Contract

### Instantiate

```python
from case1.ayush_work.marketA_v3.config import StrategyConfig
from case1.ayush_work.marketA_v3.market_A_strategy import AStrategy

strategy = AStrategy(StrategyConfig())
```

### Feed `StrategySnapshot`

Fields A actually relies on:

- `now_ms`
- `exchange_tick`
- `book`
- `inventory`
- `cash`
- `fair_value`
- `trusted_multiplier`
- `latest_earnings`
- `mode`
- `open_orders`
- `last_trade_px`
- `message_index`

Multi-symbol fields exist on the dataclass, but A mostly uses the single-symbol main fields.

### Call The Hooks

- `on_book(snapshot)`
- `on_trade(snapshot)`
- `on_fill(snapshot)`
- `on_timer(snapshot)`
- `on_news(snapshot, news_event)`

### Honor `Decision`

Interpretation:

- `desired_order is None`: no new order to send
- `observe_only = True`: wait unless `cancel_all` is also true
- `cancel_all = True`: cancel current managed A orders before replacing
- `target_inventory`: strategy’s intended total A inventory
- `desired_order`: next aggressive order to send

### Execution Rail

The strategy does not directly enforce exchange slicing. The runner must enforce:

- max exchange order qty `40`
- larger baskets worked in many smaller orders

Current live slice defaults:

- target slice `12`
- min slice `7`
- max slice `15`

### Fill Handling

After each fill:

- update managed inventory
- update open-order state
- call `on_fill(snapshot)`

This matters because A uses live inventory to:

- ratchet targets inward
- unwind correctly
- avoid rebuilding away from zero after manual intervention or partial reductions

## Exact News Mapping Requirements

### Structured A Earnings

Must map to:

- `kind="structured"`
- `structured_subtype="earnings"`
- `symbol="A"` or `asset="A"`
- `value=<earnings or eps float>`

Tradable property:

- `NewsEvent.is_structured_a_earnings`

### Unstructured A News

Must map to:

- `kind="unstructured"`
- `symbol="A"` or `asset="A"`
- `content=<headline string>`

Tradable property:

- `NewsEvent.is_a_specific_unstructured`

All other news is ignored by A.

## Strategy State Machine

Modes:

- `IDLE`
- `SHOCK`
- `UNWIND`

High-level behavior:

1. signal arrives
2. compute fair shift and target inventory
3. enter `SHOCK`
4. aggressively work target
5. while in shock:
   - emergency dump on sharp wrong-way move
   - hard time stop
   - overshoot trims
   - equilibrium flatten
   - slow decay if move stalls
6. `UNWIND` always targets zero

## Structured Earnings Path

When structured earnings arrives:

1. Build pre-event baseline from median mids in `first_earnings_baseline_window_ms`.
2. Require at least `first_earnings_min_mid_samples`.
3. If `trusted_multiplier` is unset:
   - `trusted_multiplier = baseline_mid / first_earnings_anchor`
4. Set:
   - `latest_earnings`
   - `base_fair_value = round(trusted_multiplier * earnings)`
   - `fair_value = base_fair_value`
5. Measure initial shock edge from current reference mid.
6. Size via `_scaled_target(edge, fair_change_ticks=...)`.
7. Enter `SHOCK` if edge exceeds `shock_min_edge_ticks`.

Important:

- `fair_change_ticks` is an extra confidence term, not a hard cap.
- if the edge is too small, strategy returns to `IDLE`.

## Unstructured A News Path

When A-specific unstructured news arrives:

1. Normalize and score headline with `score_a_unstructured_headline()`.
2. Determine base fair in this order:
   - `base_fair_value`
   - `fair_value`
   - `trusted_multiplier * latest_earnings`
   - baseline mid
3. Convert score bucket to fair offset.
4. Build `news_fair_value = base_fair + signed_offset_ticks`.
5. Convert to target inventory with `_news_target_inventory(...)`.
6. Behavior by confidence:
   - `strong` / `extreme` or `abs(score) >= 5`: immediate
   - `medium`: wait for confirmation
   - `light`: still tradable if position survives sizing threshold
   - `none`: no trade
7. If already holding inventory or live orders:
   - flatten first
   - then let latest signal take over

Important:

- A-news can be tradable even without exact whole-headline matches.
- scorer is built around 1-word and 2-word reusable terms.

## A-News Sentiment Model

The sentiment engine is dictionary-based and fast.

Core groups:

- `positive_bigrams`
- `negative_bigrams`
- `positive_unigrams`
- `negative_unigrams`
- `amplifiers`
- `dampeners`

Scoring pipeline:

1. normalize text
2. match bigrams first
3. match unmatched unigrams second
4. sum weights
5. apply contextual outcome overrides
6. apply at most one max amplifier
7. apply at most one min dampener
8. clamp to `[-6.0, 6.0]`

Bucket thresholds:

- `none`: `abs(score) <= 0.0`
- `light`: `< 1.75`
- `medium`: `< 3.0`
- `strong`: `< 4.25`
- `extreme`: `>= 4.25`

Returned fields in `SentimentResult`:

- `score`
- `bucket`
- `direction`
- `matched_phrases`
- `matched_unigrams`
- `matched_bigrams`
- `unknown_candidate_phrases`
- `unknown_candidate_unigrams`
- `unknown_candidate_bigrams`

Important scorer details:

- bigrams are preferred over unigrams
- outcome words like `delayed`, `stalling`, `termination`, `unsuccessful`, `violations`, `fines` override generic positive nouns
- `none` means effectively untradable
- unknown candidates are harvested for post-run dictionary tuning

## Pending News / Takeover Logic

If a fresh signal arrives while loaded:

- strategy queues latest signal
- enters flatten-first takeover
- flattens current inventory and/or cancels live orders
- once flat, restarts with the newest pending signal

This prevents stale baskets from fighting new information.

## Shock Lifecycle Details

### Sizing

Structured sizing uses `_scaled_target(...)`.

It combines:

- current edge
- fair-value change magnitude
- global position cap

News sizing uses:

- bucket-specific fair offsets
- bucket-specific position caps
- same confidence machinery underneath

### Ratchet Toward Zero

While in `SHOCK`, if actual inventory has already moved closer to zero than the old target:

- strategy ratchets target inward
- it does not rebuild away from zero

If inventory crosses sign against the old target:

- target is zeroed

### Emergency Dump

If loaded inventory moves sharply the wrong way from shock reference:

- strategy immediately enters `UNWIND`
- full flatten order is sent

### Hard Max Hold

After `shock_max_hold_ms`:

- any remaining nonzero shock inventory is forced into `UNWIND`

### Overshoot Trims

If price overshoots through fair and then starts reverting:

- strategy trims staged portions of inventory
- large baskets can trim much more aggressively than small baskets
- overshoot keeps a runner floor instead of flattening everything immediately

### Equilibrium Detection

If post-event mids stabilize enough:

- strategy treats that as equilibrium
- immediately switches to `UNWIND`

Equilibrium checks:

- enough elapsed time
- enough samples
- mids stay inside a band
- residual edge sufficiently small or enough of original move captured

### Slow Decay

If still in `SHOCK` and move has stalled:

- inventory decays in timed steps
- decay only starts after a delay
- decay is blocked if recent price action is still moving meaningfully
- decay never fully beats proper unwind logic

### Unwind

`UNWIND` always means:

- target inventory `0`
- use aggressive crossing orders to flatten
- once near zero, clear all active signal state and return to `IDLE`

## Current Live Defaults

These are the current `StrategyConfig` defaults and their env var overrides.

| Field | Default | Env Var | Meaning |
|---|---:|---|---|
| `symbol` | `A` | none | Traded symbol |
| `position_cap` | `200` | `A_V3_POSITION_CAP` | Global target cap |
| `max_exchange_order_qty` | `40` | `A_V3_MAX_EXCHANGE_ORDER_QTY` | Legal per-order max |
| `max_open_orders` | `50` | `A_V3_MAX_OPEN_ORDERS` | Runner rail |
| `max_outstanding_volume` | `120` | `A_V3_MAX_OUTSTANDING_VOLUME` | Runner rail |
| `max_absolute_position` | `200` | `A_V3_MAX_ABSOLUTE_POSITION` | Absolute inventory cap |
| `first_earnings_anchor` | `1.0` | `A_V3_FIRST_EARNINGS_ANCHOR` | First EPS anchor for multiplier |
| `first_earnings_baseline_window_ms` | `12000` | `A_V3_FIRST_EARNINGS_BASELINE_WINDOW_MS` | Pre-report baseline lookback |
| `first_earnings_min_mid_samples` | `8` | `A_V3_FIRST_EARNINGS_MIN_MID_SAMPLES` | Minimum baseline samples |
| `shock_min_edge_ticks` | `10` | `A_V3_SHOCK_MIN_EDGE_TICKS` | Minimum structured edge to trade |
| `shock_position_scale` | `1.20` | `A_V3_SHOCK_POSITION_SCALE` | Edge-to-size multiplier |
| `shock_full_confidence_edge_ticks` | `80` | `A_V3_SHOCK_FULL_CONFIDENCE_EDGE_TICKS` | Edge for max confidence |
| `shock_change_position_scale` | `0.75` | `A_V3_SHOCK_CHANGE_POSITION_SCALE` | Fair-change size multiplier |
| `shock_full_confidence_change_ticks` | `40` | `A_V3_SHOCK_FULL_CONFIDENCE_CHANGE_TICKS` | Fair-change for max confidence |
| `shock_min_position` | `4` | `A_V3_SHOCK_MIN_POSITION` | Smallest structured basket |
| `shock_initial_clip` | `200` | `A_V3_SHOCK_INITIAL_CLIP` | Strategy-side initial clip before runner slicing |
| `shock_reinforce_clip` | `80` | `A_V3_SHOCK_REINFORCE_CLIP` | Reinforcement clip |
| `shock_emergency_dump_min_elapsed_ms` | `250` | `A_V3_SHOCK_EMERGENCY_DUMP_MIN_ELAPSED_MS` | Minimum age before emergency dump |
| `shock_emergency_dump_ticks` | `40` | `A_V3_SHOCK_EMERGENCY_DUMP_TICKS` | Minimum wrong-way move to dump |
| `shock_emergency_dump_fraction` | `0.20` | `A_V3_SHOCK_EMERGENCY_DUMP_FRACTION` | Fraction of original edge for dump threshold |
| `shock_emergency_dump_min_inventory` | `12` | `A_V3_SHOCK_EMERGENCY_DUMP_MIN_INVENTORY` | Minimum inventory for dump logic |
| `shock_max_hold_ms` | `12500` | `A_V3_SHOCK_MAX_HOLD_MS` | Hard max shock hold time |
| `shock_decay_start_ms` | `5000` | `A_V3_SHOCK_DECAY_START_MS` | When stall decay can begin |
| `shock_decay_interval_ms` | `500` | `A_V3_SHOCK_DECAY_INTERVAL_MS` | Time between decay trims |
| `shock_decay_fraction` | `0.08` | `A_V3_SHOCK_DECAY_FRACTION` | Decay step as fraction of original target |
| `shock_decay_min_qty` | `6` | `A_V3_SHOCK_DECAY_MIN_QTY` | Minimum decay trim |
| `shock_decay_max_qty` | `10` | `A_V3_SHOCK_DECAY_MAX_QTY` | Maximum decay trim |
| `shock_decay_min_inventory` | `40` | `A_V3_SHOCK_DECAY_MIN_INVENTORY` | Only decay larger baskets |
| `shock_decay_min_residual_fraction` | `0.10` | `A_V3_SHOCK_DECAY_MIN_RESIDUAL_FRACTION` | Decay runner floor |
| `shock_decay_stall_window_ms` | `1200` | `A_V3_SHOCK_DECAY_STALL_WINDOW_MS` | Window to decide stall |
| `shock_decay_stall_threshold_ticks` | `12` | `A_V3_SHOCK_DECAY_STALL_THRESHOLD_TICKS` | Max recent range to count as stall |
| `overshoot_hold_ms` | `225` | `A_V3_OVERSHOOT_HOLD_MS` | Overshoot stability hold |
| `overshoot_max_wait_ms` | `600` | `A_V3_OVERSHOOT_MAX_WAIT_MS` | Max wait after crossing fair |
| `overshoot_band_ticks` | `10` | `A_V3_OVERSHOOT_BAND_TICKS` | Overshoot stability band |
| `overshoot_reversal_ticks` | `2` | `A_V3_OVERSHOOT_REVERSAL_TICKS` | Reversal needed to trim |
| `overshoot_stage1_fraction` | `0.30` | `A_V3_OVERSHOOT_STAGE1_FRACTION` | Stage-1 trim fraction |
| `overshoot_stage2_fraction` | `0.25` | `A_V3_OVERSHOOT_STAGE2_FRACTION` | Stage-2 trim fraction |
| `overshoot_stage3_fraction` | `0.20` | `A_V3_OVERSHOOT_STAGE3_FRACTION` | Stage-3 trim fraction |
| `overshoot_stage_min_qty` | `4` | `A_V3_OVERSHOOT_STAGE_MIN_QTY` | Minimum trim size |
| `overshoot_stage_max_qty` | `16` | `A_V3_OVERSHOOT_STAGE_MAX_QTY` | Maximum trim size |
| `overshoot_min_residual_fraction` | `0.30` | `A_V3_OVERSHOOT_MIN_RESIDUAL_FRACTION` | Default overshoot runner floor |
| `overshoot_large_position_threshold` | `100` | `A_V3_OVERSHOOT_LARGE_POSITION_THRESHOLD` | Large-basket threshold |
| `overshoot_large_position_stage1_fraction` | `0.50` | `A_V3_OVERSHOOT_LARGE_POSITION_STAGE1_FRACTION` | Large-basket stage-1 trim |
| `overshoot_large_position_residual_fraction` | `0.50` | `A_V3_OVERSHOOT_LARGE_POSITION_RESIDUAL_FRACTION` | Large-basket runner floor |
| `news_overshoot_hold_ms` | `200` | `A_V3_NEWS_OVERSHOOT_HOLD_MS` | News-specific overshoot hold |
| `news_overshoot_band_ticks` | `10` | `A_V3_NEWS_OVERSHOOT_BAND_TICKS` | News-specific overshoot band |
| `news_overshoot_reversal_ticks` | `2` | `A_V3_NEWS_OVERSHOOT_REVERSAL_TICKS` | News-specific overshoot reversal |
| `equilibrium_band_ticks` | `8` | `A_V3_EQUILIBRIUM_BAND_TICKS` | Structured equilibrium band |
| `equilibrium_hold_ms` | `1000` | `A_V3_EQUILIBRIUM_HOLD_MS` | Structured equilibrium hold window |
| `equilibrium_min_samples` | `6` | `A_V3_EQUILIBRIUM_MIN_SAMPLES` | Minimum equilibrium samples |
| `equilibrium_min_elapsed_ms` | `1000` | `A_V3_EQUILIBRIUM_MIN_ELAPSED_MS` | Min elapsed before equilibrium |
| `equilibrium_residual_edge_ticks` | `40` | `A_V3_EQUILIBRIUM_RESIDUAL_EDGE_TICKS` | Residual edge tolerance |
| `equilibrium_min_capture_fraction` | `0.55` | `A_V3_EQUILIBRIUM_MIN_CAPTURE_FRACTION` | Min move capture fraction |
| `news_light_offset_ticks` | `12` | `A_V3_NEWS_LIGHT_OFFSET_TICKS` | Light news fair offset |
| `news_medium_offset_ticks` | `24` | `A_V3_NEWS_MEDIUM_OFFSET_TICKS` | Medium news fair offset |
| `news_strong_offset_ticks` | `48` | `A_V3_NEWS_STRONG_OFFSET_TICKS` | Strong news fair offset |
| `news_extreme_offset_ticks` | `80` | `A_V3_NEWS_EXTREME_OFFSET_TICKS` | Extreme news fair offset |
| `news_very_extreme_offset_ticks` | `120` | `A_V3_NEWS_VERY_EXTREME_OFFSET_TICKS` | Near-max news fair offset |
| `news_light_position` | `8` | `A_V3_NEWS_LIGHT_POSITION` | Light news position cap |
| `news_medium_position` | `36` | `A_V3_NEWS_MEDIUM_POSITION` | Medium news position cap |
| `news_strong_position` | `90` | `A_V3_NEWS_STRONG_POSITION` | Strong news position cap |
| `news_extreme_position` | `130` | `A_V3_NEWS_EXTREME_POSITION` | Extreme news position cap |
| `news_very_extreme_position` | `200` | `A_V3_NEWS_VERY_EXTREME_POSITION` | Max news position cap |
| `news_zero_position_threshold` | `3` | `A_V3_NEWS_ZERO_POSITION_THRESHOLD` | Zero-out tiny news baskets |
| `news_confirmation_timeout_ms` | `900` | `A_V3_NEWS_CONFIRMATION_TIMEOUT_MS` | Medium-news confirmation timeout |
| `news_confirmation_move_ticks` | `3` | `A_V3_NEWS_CONFIRMATION_MOVE_TICKS` | Move needed to confirm medium news |
| `news_takeover_flatten_ms` | `1200` | `A_V3_NEWS_TAKEOVER_FLATTEN_MS` | Reserved timing knob for takeover flatten |
| `news_takeover_near_flat_threshold` | `4` | `A_V3_NEWS_TAKEOVER_NEAR_FLAT_THRESHOLD` | Near-flat threshold for takeover logic |
| `news_equilibrium_hold_ms` | `1400` | `A_V3_NEWS_EQUILIBRIUM_HOLD_MS` | News-specific equilibrium hold |
| `news_equilibrium_min_elapsed_ms` | `1200` | `A_V3_NEWS_EQUILIBRIUM_MIN_ELAPSED_MS` | News-specific equilibrium minimum age |
| `news_equilibrium_residual_edge_ticks` | `40` | `A_V3_NEWS_EQUILIBRIUM_RESIDUAL_EDGE_TICKS` | News residual edge tolerance |
| `news_equilibrium_min_capture_fraction` | `0.55` | `A_V3_NEWS_EQUILIBRIUM_MIN_CAPTURE_FRACTION` | News capture fraction |
| `news_overshoot_max_wait_ms` | `700` | `A_V3_NEWS_OVERSHOOT_MAX_WAIT_MS` | News max wait after crossing fair |
| `flatten_deadline_ms` | `2400` | `A_V3_FLATTEN_DEADLINE_MS` | Unwind flatten deadline |
| `flatten_force_cross_ms` | `700` | `A_V3_FLATTEN_FORCE_CROSS_MS` | Force-cross timing for runner |
| `flatten_near_zero_threshold` | `1` | `A_V3_FLATTEN_NEAR_ZERO_THRESHOLD` | Near-flat threshold |
| `order_slice_target_qty` | `12` | `A_V3_ORDER_SLICE_TARGET_QTY` | Runner target slice |
| `order_slice_min_qty` | `7` | `A_V3_ORDER_SLICE_MIN_QTY` | Runner min slice |
| `order_slice_max_qty` | `15` | `A_V3_ORDER_SLICE_MAX_QTY` | Runner max slice |
| `multiplier_update_alpha` | `0.35` | `A_V3_MULTIPLIER_UPDATE_ALPHA` | Reserved multiplier blend knob |
| `multiplier_update_clamp_fraction` | `0.18` | `A_V3_MULTIPLIER_UPDATE_CLAMP_FRACTION` | Multiplier clamp |
| `multiplier_clean_sample_limit` | `5` | `A_V3_MULTIPLIER_CLEAN_SAMPLE_LIMIT` | Clean sample count |
| `multiplier_sample_clamp_fraction` | `0.15` | `A_V3_MULTIPLIER_SAMPLE_CLAMP_FRACTION` | Sample clamp |
| `timer_interval_ms` | `60` | `A_V3_TIMER_INTERVAL_MS` | Timer cadence |
| `min_order_live_ms` | `75` | `A_V3_MIN_ORDER_LIVE_MS` | Minimum order live time |
| `replace_qty_tolerance` | `1` | `A_V3_REPLACE_QTY_TOLERANCE` | Replacement qty tolerance |
| `replace_price_tolerance_ticks` | `0` | `A_V3_REPLACE_PRICE_TOLERANCE_TICKS` | Replacement price tolerance |

## Logger Knobs

| Field | Default | Env Var | Meaning |
|---|---:|---|---|
| `enabled` | `True` | `A_V3_LOGGER_ENABLED` | Enable run logging |
| `queue_max_events` | `2000` | `A_V3_LOGGER_QUEUE_MAX_EVENTS` | Logger queue rail |
| `write_decision_snapshots` | `True` | `A_V3_WRITE_DECISION_SNAPSHOTS` | Write decision CSV |
| `midrun_checkpoint_enabled` | `True` | `A_V3_MIDRUN_CHECKPOINT_ENABLED` | Halfway artifacts |
| `midrun_checkpoint_ms` | `450000` | `A_V3_MIDRUN_CHECKPOINT_MS` | Halfway checkpoint time |

## Tuning Playbook

### Missed A-news because wording was unseen

Change:

- `/market_A_strategy/a_news_sentiment.py`

Usually edit:

- `positive_bigrams`
- `negative_bigrams`
- `positive_unigrams`
- `negative_unigrams`

Then inspect:

- `a_news_tracker.json`
- `unknown_a_news_terms.json`

### A-news was directionally right but too small

First change:

- `news_light_position`
- `news_medium_position`
- `news_strong_position`
- `news_extreme_position`
- `news_very_extreme_position`

Then, if the fair shift itself is too small:

- `news_light_offset_ticks`
- `news_medium_offset_ticks`
- `news_strong_offset_ticks`
- `news_extreme_offset_ticks`
- `news_very_extreme_offset_ticks`

### Medium news is waiting too long

Change:

- `news_confirmation_timeout_ms`
- `news_confirmation_move_ticks`

Or promote wording in `a_news_sentiment.py` so its score lands in `strong`/`extreme`.

### Structured earnings basket is too small

Change:

- `shock_position_scale`
- `shock_change_position_scale`
- `shock_full_confidence_edge_ticks`
- `shock_full_confidence_change_ticks`
- `position_cap`
- `max_absolute_position`

### We are holding too long after the move is over

Change:

- `equilibrium_hold_ms`
- `equilibrium_min_elapsed_ms`
- `equilibrium_band_ticks`
- `equilibrium_residual_edge_ticks`
- `equilibrium_min_capture_fraction`
- `shock_max_hold_ms`

### We are letting go too early

Change:

- `shock_decay_start_ms`
- `shock_decay_interval_ms`
- `shock_decay_fraction`
- `shock_decay_min_residual_fraction`
- `shock_decay_stall_window_ms`
- `shock_decay_stall_threshold_ticks`

### We miss overshoots

Change:

- `overshoot_hold_ms`
- `overshoot_band_ticks`
- `overshoot_reversal_ticks`
- `overshoot_stage1_fraction`
- `overshoot_stage2_fraction`
- `overshoot_stage3_fraction`
- `overshoot_stage_max_qty`
- `overshoot_large_position_stage1_fraction`
- `overshoot_large_position_residual_fraction`

### We trim too much on overshoots

Change:

- `overshoot_stage1_fraction`
- `overshoot_stage2_fraction`
- `overshoot_stage3_fraction`
- `overshoot_min_residual_fraction`
- `overshoot_large_position_residual_fraction`

### We get killed on wrong-way post-event moves

Change:

- `shock_emergency_dump_min_elapsed_ms`
- `shock_emergency_dump_ticks`
- `shock_emergency_dump_fraction`
- `shock_emergency_dump_min_inventory`

### Reversals take too long to flatten and flip

Runner-side first:

- `order_slice_target_qty`
- `order_slice_min_qty`
- `order_slice_max_qty`
- `timer_interval_ms`
- `min_order_live_ms`

Then strategy-side if needed:

- `shock_initial_clip`
- `shock_reinforce_clip`

### Bot keeps rebuilding away from zero after manual intervention

This should already be handled by `_ratchet_shock_target_toward_zero()`.

If it fails, the issue is structural in `A_strategy.py`, not a simple knob.

## Current Practical Guidance

If partner Codex needs to make A stronger or safer, the order of operations should be:

1. tune `a_news_sentiment.py`
2. tune `StrategyConfig`
3. only then edit `A_strategy.py`

In practice:

- most missed A PnL is dictionary coverage or sizing
- most execution lag is runner slicing/cadence
- `A_strategy.py` should be treated as stable unless there is a clear structural bug

## Reference Artifacts To Review After Each Run

Most useful A outputs:

- `analysis_runs/<run>/session_summary.json`
- `analysis_runs/<run>/session_summary.md`
- `analysis_runs/<run>/a_news_tracker.json`
- `analysis_runs/<run>/a_news_tracker.md`
- `analysis_runs/<run>/unknown_a_news_terms.json`
- `analysis_runs/<run>/decision_snapshots.csv`
- `analysis_runs/<run>/trace_events.jsonl`
- `analysis_runs/<run>/checkpoints/halfway/a_news_tracker.json`

Use these to answer:

- did we miss the headline entirely
- was the sign wrong
- was the basket too small
- did confirmation delay kill the move
- did unwind or decay cut the move too early

## Minimal Implementation Checklist For Another Runner

- import `AStrategy`
- create `StrategyConfig`
- build `StrategySnapshot` accurately
- map structured A earnings into `NewsEvent`
- map A-specific unstructured headlines into `NewsEvent`
- route book/trade/fill/timer/news events to the right strategy hooks
- honor `Decision.cancel_all`
- honor `Decision.observe_only`
- slice larger targets into legal child orders
- feed fills back through `on_fill`
- keep inventory state exact

If all of that is done, the other runner should behave like the live `marketA_v3` A strategy.
