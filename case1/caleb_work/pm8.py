"""
merged_pm.py — Merged Prediction Markets Bot
Ayush's strategy intelligence (logit posteriors, 3-phase system, macro leg state machine)
on top of new_PM1's execution layer (XChangeClient, OrderManager, tracing).

Competition-day bot for R_HIKE, R_HOLD, R_CUT.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from utcxchangelib import Side, XChangeClient


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
def _env(name: str, default):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    t = type(default)
    if t is bool:
        return v.strip().lower() not in {"0", "false", "no", "off"}
    return t(v.strip())


@dataclass(frozen=True)
class Cfg:
    hike: str = "R_HIKE"
    hold: str = "R_HOLD"
    cut: str = "R_CUT"
    scale: float = 1000.0  # payout scale

    # ── CPI ──
    cpi_small_surprise: float = 0.00003   # below this = no trade
    cpi_medium_surprise: float = 0.00020
    cpi_large_surprise: float = 0.00050
    cpi_small_logit: float = 0.25
    cpi_medium_logit: float = 0.55
    cpi_large_logit: float = 0.90

    # ── Headlines ──
    headline_relevance_min: float = 0.8
    headline_light_logit: float = 0.15
    headline_medium_logit: float = 0.35
    headline_strong_logit: float = 0.60
    headline_extreme_logit: float = 0.85
    headline_score_divisor: float = 3.0

    # ── Posterior ──
    posterior_floor: float = 0.02

    # ── Phases (ms from session start to round end) ──
    round_duration_ms: int = 900_000       # 15 min
    endgame_countdown_ms: int = 60_000     # last 60s
    probe_countdown_ms: int = 180_000      # last 3 min

    # ── Macro signal window ──
    macro_signal_timeout_ms: int = 12_000

    # ── Macro leg state machine ──
    macro_move_light: int = 8
    macro_move_medium: int = 16
    macro_move_strong: int = 28
    macro_move_extreme: int = 45
    macro_equilibrium_hold_ms: int = 3000
    macro_equilibrium_min_elapsed_ms: int = 2000
    macro_equilibrium_min_samples: int = 6
    macro_equilibrium_band: float = 8.0
    macro_equilibrium_residual: float = 12.0
    macro_reversal_min_progress: float = 6.0
    macro_reversal_exit: float = 8.0
    macro_overshoot_trigger_frac: float = 0.6
    macro_overshoot_min_trigger: float = 8.0
    macro_overshoot_trim_frac: float = 0.40
    macro_overshoot_min_residual: int = 10

    # ── Position sizing ──
    light_position: int = 40
    medium_position: int = 80
    strong_position: int = 120
    extreme_position: int = 160
    macro_target_cap: int = 160
    max_abs_position: int = 200

    # ── Baseline contrarian targets ──
    baseline_neutral_low: float = 250.0
    baseline_neutral_high: float = 750.0
    baseline_target_cap: int = 40
    baseline_full_distance: float = 200.0

    # ── Endgame ──
    endgame_long_target: int = 200
    endgame_short_target: int = -100
    endgame_dead_price: float = 50.0

    # ── Probe ──
    probe_base_target: int = 60
    probe_confident_target: int = 120
    probe_confident_price: float = 700.0
    probe_confidence_gap: float = 150.0

    # ── Execution ──
    max_order_size: int = 40
    hard_position_limit: int = 200
    order_stale_secs: float = 0.40
    urgent_stale_secs: float = 0.08
    aggressive_ticks: int = 2
    reentry_block_secs: float = 0.8
    add_cooldown_secs: float = 0.25
    startup_flatten_chunk: int = 40
    loop_sleep_secs: float = 0.15

    # ── Tracing ──
    trace_enabled: bool = True
    trace_dir: str = str(Path(__file__).resolve().parent / "logs")

    @property
    def symbols(self) -> tuple[str, str, str]:
        return (self.hike, self.hold, self.cut)


def load_cfg() -> Cfg:
    return Cfg(
        hike=_env("MP_HIKE", "R_HIKE"),
        hold=_env("MP_HOLD", "R_HOLD"),
        cut=_env("MP_CUT", "R_CUT"),
        scale=_env("MP_SCALE", 1000.0),
        cpi_small_surprise=_env("MP_CPI_SMALL", 0.00003),
        cpi_medium_surprise=_env("MP_CPI_MED", 0.00020),
        cpi_large_surprise=_env("MP_CPI_LARGE", 0.00050),
        aggressive_ticks=_env("MP_AGG_TICKS", 2),
        max_order_size=_env("MP_MAX_ORDER", 40),
        hard_position_limit=_env("MP_HARD_LIMIT", 200),
        trace_enabled=_env("MP_TRACE", True),
        trace_dir=_env("MP_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
    )


# ═══════════════════════════════════════════════════════════════════
# SENTIMENT SCORING (from Ayush's c_news_sentiment.py, 3-bucket)
# ═══════════════════════════════════════════════════════════════════
HIKE_BI = {
    "inflation elevated": 1.8, "upside inflation": 1.8, "inflation sticky": 2.0,
    "sticky inflation": 2.0, "persistent inflation": 2.0, "market tight": 1.8,
    "labor strong": 1.6, "higher longer": 2.2, "strong demand": 1.8,
    "restrictive policy": 1.8, "price pressures": 1.8, "resilient demand": 1.7,
    "upside risks": 1.4, "inflation risks": 1.5, "overheating economy": 2.0,
    "sticky prices": 2.0, "wage growth": 1.4, "reaccelerating inflation": 2.2,
    "stay restrictive": 1.7, "keep pressure": 1.4, "reassess path": 1.6,
    "path cuts": 1.8, "signal policy": 1.0, "looks concerned": 1.0,
    "concerned about": 0.8, "emphasizes inflation": 1.1,
}
HOLD_BI = {
    "balanced risks": 2.2, "mixed indicators": 1.8, "data dependent": 2.0,
    "data dependence": 2.2, "more evidence": 2.0, "uncertain outlook": 1.6,
    "incoming data": 1.2, "upcoming data": 1.5, "no rush": 1.8,
    "no clear": 1.4, "clear signal": 1.7, "next move": 1.5,
    "hold steady": 2.2, "on hold": 2.0, "stay hold": 1.8,
    "policy positioned": 1.6, "well positioned": 1.8, "options open": 2.0,
    "signals conflict": 2.0, "remains cautious": 2.0, "wait evidence": 1.8,
    "monitor data": 1.2, "policy pause": 1.8, "steady policy": 1.6,
    "await upcoming": 1.6, "communication remains": 1.6, "keeps options": 1.7,
    "mixed economic": 1.4, "economic indicators": 1.5, "growth signals": 1.4,
    "markets await": 1.1, "chair reiterates": 0.9, "reiterates data": 1.0,
}
CUT_BI = {
    "disinflation progress": 2.2, "inflation cooling": 1.8, "growth risks": 1.5,
    "downside growth": 1.8, "growth slowing": 1.8, "downside risks": 1.6,
    "labor softening": 1.8, "cooling labor": 1.6, "easing inflation": 1.9,
    "inflation pressures": 0.6, "weaker demand": 1.8, "restrictive enough": 1.8,
    "room ease": 1.8, "cuts appropriate": 2.2, "slowdown risks": 1.6,
    "recession risks": 2.0, "lower inflation": 1.6, "room cut": 1.8,
    "growth weakness": 1.6, "toward cuts": 2.0, "policy easing": 1.5,
    "moving back": 1.6, "back target": 1.8, "confidence inflation": 1.4,
    "softening data": 2.0, "increasing confidence": 1.4, "signals increasing": 0.9,
    "note downside": 1.0, "markets lean": 1.1,
}
HIKE_UNI = {
    "inflation": 0.7, "sticky": 1.2, "tight": 1.0, "restrictive": 0.9,
    "elevated": 1.0, "hawkish": 1.2, "overheating": 1.4, "resilient": 0.8,
    "premature": 0.8, "pressures": 0.8, "pressure": 0.9, "risks": 0.5,
    "concerned": 0.8, "wage": 0.5,
}
HOLD_UNI = {
    "balanced": 1.0, "mixed": 1.0, "dependent": 0.9, "dependence": 1.1,
    "patience": 1.4, "pause": 1.5, "steady": 1.2, "wait": 1.0,
    "monitor": 0.8, "positioned": 0.8, "uncertain": 0.9, "cautious": 1.2,
    "conflict": 1.2, "await": 1.0, "upcoming": 0.8, "reiterates": 0.8,
    "signal": 0.4, "indicators": 0.7, "evidence": 0.8,
}
CUT_UNI = {
    "disinflation": 1.4, "cooling": 1.0, "slowing": 1.0, "softening": 1.2,
    "downside": 1.1, "weaker": 1.2, "easing": 1.2, "slowdown": 1.4,
    "recession": 1.5, "dovish": 1.2, "ease": 0.9, "cuts": 1.0,
    "confidence": 0.7, "lean": 0.8,
}
CTX_OVERRIDES = {
    "higher for longer": (1.4, 0.6, -1.2),
    "policy may stay restrictive": (1.2, 0.5, -1.0),
    "premature to cut": (0.6, 0.9, -1.6),
    "balanced risks": (-0.4, 1.8, -0.4),
    "mixed economic indicators": (-0.4, 2.2, -0.4),
    "await upcoming data": (-0.2, 2.0, -0.2),
    "data dependence": (-0.4, 2.3, -0.4),
    "reiterates data dependence": (-0.3, 1.7, -0.3),
    "no clear signal": (-0.5, 2.3, -0.5),
    "hold steady": (-0.4, 2.0, -0.4),
    "on hold": (-0.3, 1.8, -0.3),
    "lean toward cuts": (-1.2, -0.2, 1.9),
    "cooling labor market": (-1.2, 0.1, 1.8),
    "easing inflation pressures": (-1.4, 0.1, 2.0),
    "moving back to target": (-1.2, 0.1, 1.9),
    "options open": (-0.3, 1.8, -0.3),
    "signals conflict": (-0.3, 1.8, -0.3),
    "communication remains cautious": (-0.3, 1.9, -0.3),
    "increasing confidence inflation is moving back to target": (-1.6, 0.0, 2.2),
    "softening data raises expectations of policy easing": (-1.7, 2.4, 0.9),
    "reassess path of cuts": (1.8, 0.3, -1.9),
    "stay restrictive for longer": (1.5, 0.6, -1.2),
    "strong demand and sticky prices": (1.8, 0.2, -1.0),
    "concerned about wage growth": (1.2, 0.1, -0.4),
    "emphasizes inflation risks": (1.3, 0.1, -0.6),
}
AMPLIFIERS = {"clearly": 1.15, "significantly": 1.20, "sharply": 1.20, "decisively": 1.20, "materially": 1.15}
DAMPENERS = {"somewhat": 0.85, "modestly": 0.85, "slightly": 0.85, "gradually": 0.90}
_STOP = {"the","and","that","with","into","from","amid","this","will","they","their","while","have","has","been","over","more","than","would","could","should","officials","federal","reserve"}


@dataclass(frozen=True)
class Sentiment:
    hike: float; hold: float; cut: float
    relevance: float; bucket: str
    phrases: tuple[str, ...]; unknowns: tuple[str, ...]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("'s", ""))).strip()


def score_headline(text: str) -> Sentiment:
    n = _norm(text)
    if not n:
        return Sentiment(0,0,0,0,"none",(),())
    tokens = n.split()
    used = set()
    h, ho, c = 0.0, 0.0, 0.0
    rel = 0.0
    phrases = []

    # Context overrides first
    for phrase, (dh, dho, dc) in CTX_OVERRIDES.items():
        if phrase in n:
            h += dh; ho += dho; c += dc
            phrases.append(phrase)
            for i, tok in enumerate(tokens):
                if phrase.startswith(tok):
                    for j in range(i, min(i + len(phrase.split()), len(tokens))):
                        used.add(j)

    # Bigrams
    for i in range(len(tokens) - 1):
        if i in used or i+1 in used:
            continue
        bi = f"{tokens[i]} {tokens[i+1]}"
        matched = False
        if bi in HIKE_BI: h += HIKE_BI[bi]; matched = True
        if bi in HOLD_BI: ho += HOLD_BI[bi]; matched = True
        if bi in CUT_BI: c += CUT_BI[bi]; matched = True
        if matched:
            phrases.append(bi); used.add(i); used.add(i+1)

    # Unigrams
    for i, tok in enumerate(tokens):
        if i in used: continue
        matched = False
        if tok in HIKE_UNI: h += HIKE_UNI[tok]; matched = True
        if tok in HOLD_UNI: ho += HOLD_UNI[tok]; matched = True
        if tok in CUT_UNI: c += CUT_UNI[tok]; matched = True
        if matched: phrases.append(tok); used.add(i)

    # Amplifiers/dampeners
    mult = 1.0
    for a, v in AMPLIFIERS.items():
        if a in tokens: mult = max(mult, v)
    for d, v in DAMPENERS.items():
        if d in tokens: mult = min(mult, v)
    h = max(-6, min(6, h * mult))
    ho = max(-6, min(6, ho * mult))
    c = max(-6, min(6, c * mult))

    mag = max(abs(h), abs(ho), abs(c))
    rel = max(rel, mag)
    bucket = "none" if mag <= 0 else "light" if mag < 1.0 else "medium" if mag < 2.25 else "strong" if mag < 3.75 else "extreme"

    # Unknown candidates
    unknowns = []
    for i in range(len(tokens)-1):
        if i in used or i+1 in used: continue
        if tokens[i] in _STOP or tokens[i+1] in _STOP: continue
        if len(tokens[i]) < 4 or len(tokens[i+1]) < 4: continue
        unknowns.append(f"{tokens[i]} {tokens[i+1]}")

    return Sentiment(h, ho, c, rel, bucket, tuple(phrases), tuple(unknowns))


# CPI text fallback parser
_CPI_RE = re.compile(
    r"cpi[^0-9\-+]*actual[^0-9\-+]*(?P<actual>[-+]?\d*\.?\d+)"
    r"[^0-9a-zA-Z]+vs[^0-9a-zA-Z]+forecast[^0-9\-+]*(?P<forecast>[-+]?\d*\.?\d+)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TopOfBook:
    bid: Optional[int] = None
    ask: Optional[int] = None
    bid_qty: int = 0
    ask_qty: int = 0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.bid or self.ask


@dataclass
class TrackedOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: int
    role: str
    reason: str
    thesis: Optional[str]
    event_id: int
    created_at: float = field(default_factory=time.time)


@dataclass
class MacroLeg:
    symbol: str
    direction: int  # +1 or -1
    target: int
    original_target: int
    ref_mid: float
    fair: float
    bucket: str
    started_ms: int
    best_mid: float
    trimmed: bool = False
    mids: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class MacroSignal:
    event_id: int
    source: str
    headline: Optional[str]
    posterior: dict[str, float]
    fair_values: dict[str, int]
    pos_symbol: Optional[str]
    neg_symbol: Optional[str]
    pair_size: int
    bucket: str
    deltas: dict[str, float]


@dataclass
class State:
    books: dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)

    # Probabilities
    prior: Optional[dict[str, float]] = None
    posterior: Optional[dict[str, float]] = None
    fair_values: Optional[dict[str, int]] = None

    # Macro signal
    active_signal: Optional[MacroSignal] = None
    pending_signal: Optional[MacroSignal] = None
    legs: dict[str, Optional[MacroLeg]] = field(default_factory=dict)
    macro_targets: dict[str, int] = field(default_factory=dict)
    baseline_targets: dict[str, int] = field(default_factory=dict)
    combined_targets: dict[str, int] = field(default_factory=dict)

    last_tradeable_ms: Optional[int] = None
    event_counter: int = 0
    session_start_ms: Optional[int] = None
    news_urgent_until: float = 0.0

    # Execution state
    startup_done: bool = False
    session_cash: Optional[float] = None
    session_mtm: Optional[float] = None
    blocked_direction: Optional[str] = None
    blocked_until: float = 0.0
    last_add_ts: float = 0.0
    last_status_ts: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# STRATEGY ENGINE (Ayush-inspired)
# ═══════════════════════════════════════════════════════════════════
class StrategyEngine:
    def __init__(self, cfg: Cfg, state: State):
        self.cfg = cfg
        self.s = state

    def _equal_probs(self) -> dict[str, float]:
        return {s: 1/3 for s in self.cfg.symbols}

    def _zero_targets(self) -> dict[str, int]:
        return {s: 0 for s in self.cfg.symbols}

    def prior_from_market(self, books: dict[str, TopOfBook]) -> dict[str, float]:
        """Extract logit-based prior from market mids."""
        mids = {}
        for s in self.cfg.symbols:
            b = books.get(s)
            if b is None or b.mid is None:
                return self._equal_probs()
            mids[s] = max(1.0, min(self.cfg.scale - 1, b.mid))

        floor = self.cfg.posterior_floor
        logits = {}
        for s, m in mids.items():
            p = max(floor, min(1 - floor, m / self.cfg.scale))
            logits[s] = math.log(p / (1 - p))

        mx = max(logits.values())
        exps = {s: math.exp(l - mx) for s, l in logits.items()}
        total = sum(exps.values())
        return {s: exps[s] / total for s in self.cfg.symbols}

    def posterior_from_deltas(self, prior: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
        """Bayesian logit update."""
        floor = self.cfg.posterior_floor
        logits = []
        for s in self.cfg.symbols:
            p = max(floor, min(1 - floor, prior.get(s, 1/3)))
            logits.append(math.log(p) + deltas.get(s, 0))
        mx = max(logits)
        exps = [math.exp(l - mx) for l in logits]
        total = sum(exps)
        return {s: exps[i] / total for i, s in enumerate(self.cfg.symbols)}

    def fair_from_probs(self, probs: dict[str, float]) -> dict[str, int]:
        return {s: int(round(self.cfg.scale * probs.get(s, 0))) for s in self.cfg.symbols}

    def signal_from_cpi(self, actual: float, forecast: float, books: dict[str, TopOfBook]) -> Optional[MacroSignal]:
        surprise = actual - forecast
        abss = abs(surprise)
        if abss < self.cfg.cpi_small_surprise:
            return None
        if abss < self.cfg.cpi_medium_surprise:
            shift, bucket = self.cfg.cpi_small_logit, "light"
        elif abss < self.cfg.cpi_large_surprise:
            shift, bucket = self.cfg.cpi_medium_logit, "medium"
        elif abss < self.cfg.cpi_large_surprise * 1.75:
            shift, bucket = self.cfg.cpi_large_logit, "strong"
        else:
            shift, bucket = self.cfg.cpi_large_logit * 1.35, "extreme"

        sign = 1 if surprise > 0 else -1
        deltas = {
            self.cfg.hike: sign * shift,
            self.cfg.hold: -0.45 * shift,  # hold always loses a bit
            self.cfg.cut: -sign * shift,
        }
        return self._build_signal(f"CPI {actual:.6f} vs {forecast:.6f}", "cpi", deltas, bucket, books)

    def signal_from_headline(self, text: str, books: dict[str, TopOfBook]) -> Optional[MacroSignal]:
        sent = score_headline(text)
        if sent.relevance < self.cfg.headline_relevance_min or sent.bucket == "none":
            return None
        logit_scale = self._bucket_logit(sent.bucket) / max(self.cfg.headline_score_divisor, 0.01)
        deltas = {
            self.cfg.hike: logit_scale * sent.hike,
            self.cfg.hold: logit_scale * sent.hold,
            self.cfg.cut: logit_scale * sent.cut,
        }
        return self._build_signal(text, "headline", deltas, sent.bucket, books)

    def _bucket_logit(self, bucket: str) -> float:
        return {"light": self.cfg.headline_light_logit, "medium": self.cfg.headline_medium_logit,
                "strong": self.cfg.headline_strong_logit, "extreme": self.cfg.headline_extreme_logit}.get(bucket, 0)

    def _build_signal(self, headline: str, source: str, deltas: dict[str, float],
                      bucket: str, books: dict[str, TopOfBook]) -> Optional[MacroSignal]:
        prior = self.prior_from_market(books)
        posterior = self.posterior_from_deltas(prior, deltas)
        fairs = self.fair_from_probs(posterior)

        # Select pair: most positive delta vs most negative
        pos_sym = max(self.cfg.symbols, key=lambda s: deltas[s])
        neg_sym = min(self.cfg.symbols, key=lambda s: deltas[s])
        if deltas[pos_sym] < 0.05 or deltas[neg_sym] > -0.05:
            # No meaningful directional signal
            return None
        if pos_sym == neg_sym:
            return None

        size = min(self.cfg.macro_target_cap, self._size_for_bucket(bucket))
        self.s.event_counter += 1

        self.s.prior = prior
        self.s.posterior = posterior
        self.s.fair_values = fairs

        return MacroSignal(
            event_id=self.s.event_counter, source=source, headline=headline,
            posterior=posterior, fair_values=fairs,
            pos_symbol=pos_sym, neg_symbol=neg_sym,
            pair_size=size, bucket=bucket, deltas=deltas,
        )

    def _size_for_bucket(self, b: str) -> int:
        return {"light": self.cfg.light_position, "medium": self.cfg.medium_position,
                "strong": self.cfg.strong_position, "extreme": self.cfg.extreme_position}.get(b, 0)

    def _move_for_bucket(self, b: str) -> int:
        return {"light": self.cfg.macro_move_light, "medium": self.cfg.macro_move_medium,
                "strong": self.cfg.macro_move_strong, "extreme": self.cfg.macro_move_extreme}.get(b, 8)

    def activate_signal(self, sig: MacroSignal, books: dict[str, TopOfBook], now_ms: int):
        self.s.active_signal = sig
        self.s.macro_targets = self._zero_targets()
        self.s.legs = {s: None for s in self.cfg.symbols}
        for sym, direction in ((sig.pos_symbol, 1), (sig.neg_symbol, -1)):
            if sym is None: continue
            b = books.get(sym)
            if b is None or b.mid is None: continue
            mark = b.mid
            target = sig.pair_size * direction
            fair = max(0, min(int(self.cfg.scale), int(mark + direction * self._move_for_bucket(sig.bucket))))
            leg = MacroLeg(
                symbol=sym, direction=direction, target=target,
                original_target=target, ref_mid=mark, fair=float(fair),
                bucket=sig.bucket, started_ms=now_ms, best_mid=mark,
                mids=[(now_ms, mark)],
            )
            self.s.legs[sym] = leg
            self.s.macro_targets[sym] = target

    def update_legs(self, books: dict[str, TopOfBook], now_ms: int):
        """Update macro leg state machine: overshoot, equilibrium, reversal."""
        if self.s.active_signal is None:
            self.s.macro_targets = self._zero_targets()
            return

        any_live = False
        for sym, leg in list(self.s.legs.items()):
            if leg is None or leg.target == 0: continue
            b = books.get(sym)
            if b is None or b.mid is None: continue
            mark = b.mid

            # Track mids
            leg.mids.append((now_ms, mark))
            cutoff = now_ms - max(self.cfg.macro_equilibrium_hold_ms * 3, 20000)
            leg.mids = [(t, m) for t, m in leg.mids if t >= cutoff]

            # Update best
            if leg.direction > 0:
                leg.best_mid = max(leg.best_mid, mark)
            else:
                leg.best_mid = min(leg.best_mid, mark)

            # Check reversal
            best_progress = leg.direction * (leg.best_mid - leg.ref_mid)
            curr_progress = leg.direction * (mark - leg.ref_mid)
            if best_progress >= self.cfg.macro_reversal_min_progress and \
               (best_progress - curr_progress) >= self.cfg.macro_reversal_exit:
                leg.target = 0
                self.s.macro_targets[sym] = 0
                continue

            # Check overshoot trim
            if not leg.trimmed:
                trigger = max(self.cfg.macro_overshoot_min_trigger,
                              self._move_for_bucket(leg.bucket) * self.cfg.macro_overshoot_trigger_frac)
                if leg.direction * (mark - leg.fair) >= trigger:
                    trimmed = max(self.cfg.macro_overshoot_min_residual,
                                  int(abs(leg.original_target) * (1 - self.cfg.macro_overshoot_trim_frac)))
                    leg.target = min(abs(leg.target), trimmed) * leg.direction
                    leg.trimmed = True
                    self.s.macro_targets[sym] = leg.target

            # Check equilibrium
            elapsed = now_ms - leg.started_ms
            if elapsed >= self.cfg.macro_equilibrium_min_elapsed_ms:
                recent = [m for t, m in leg.mids if t >= now_ms - self.cfg.macro_equilibrium_hold_ms]
                if len(recent) >= self.cfg.macro_equilibrium_min_samples:
                    if (max(recent) - min(recent)) <= self.cfg.macro_equilibrium_band:
                        if abs(recent[-1] - leg.fair) <= self.cfg.macro_equilibrium_residual:
                            leg.target = 0
                            self.s.macro_targets[sym] = 0
                            continue

            if leg.target != 0:
                any_live = True

        if not any_live:
            self.s.active_signal = None

    def refresh_baseline(self, books: dict[str, TopOfBook]):
        """Contrarian baseline targets for extreme prices."""
        targets = self._zero_targets()
        lo, hi = self.cfg.baseline_neutral_low, self.cfg.baseline_neutral_high
        cap = self.cfg.baseline_target_cap
        dist = max(1.0, self.cfg.baseline_full_distance)
        for s in self.cfg.symbols:
            b = books.get(s)
            if b is None or b.mid is None: continue
            m = b.mid
            if lo <= m <= hi:
                targets[s] = 0
            elif m > hi:
                magnitude = min(cap, cap * min(1.0, (m - hi) / dist))
                targets[s] = -int(round(magnitude))
            else:
                magnitude = min(cap, cap * min(1.0, (lo - m) / dist))
                targets[s] = int(round(magnitude))
        self.s.baseline_targets = targets

    def refresh_endgame(self, books: dict[str, TopOfBook]):
        """Winner-take-all endgame positioning."""
        marks = {}
        for s in self.cfg.symbols:
            b = books.get(s)
            marks[s] = b.mid if b and b.mid else 0
        winner = max(self.cfg.symbols, key=lambda s: marks[s])
        targets = {}
        for s in self.cfg.symbols:
            if s == winner:
                targets[s] = self.cfg.endgame_long_target
            elif marks[s] < self.cfg.endgame_dead_price:
                targets[s] = self.cfg.endgame_short_target
            else:
                targets[s] = self.cfg.endgame_short_target
        self.s.combined_targets = targets

    def refresh_probe(self, books: dict[str, TopOfBook]):
        """Pre-endgame probe positioning."""
        marks = {}
        for s in self.cfg.symbols:
            b = books.get(s)
            marks[s] = b.mid if b and b.mid else 0
        ranked = sorted(self.cfg.symbols, key=lambda s: marks[s], reverse=True)
        leader = ranked[0]
        second = marks[ranked[1]]
        confident = marks[leader] >= self.cfg.probe_confident_price or \
                    (marks[leader] - second) >= self.cfg.probe_confidence_gap
        size = self.cfg.probe_confident_target if confident else self.cfg.probe_base_target
        targets = {s: -size for s in self.cfg.symbols}
        targets[leader] = size
        self.s.combined_targets = targets

    def compute_combined(self, books: dict[str, TopOfBook], now_ms: int):
        """Compute final combined targets based on current phase."""
        remaining = self._remaining_ms(now_ms)

        if remaining <= self.cfg.endgame_countdown_ms:
            self.refresh_endgame(books)
            return

        if remaining <= self.cfg.probe_countdown_ms:
            self.refresh_probe(books)
            return

        # Trading phase
        self.refresh_baseline(books)
        self.update_legs(books, now_ms)

        combined = self._zero_targets()
        for s in self.cfg.symbols:
            raw = self.s.baseline_targets.get(s, 0) + self.s.macro_targets.get(s, 0)
            combined[s] = max(-self.cfg.max_abs_position, min(self.cfg.max_abs_position, raw))
        self.s.combined_targets = combined

    def _remaining_ms(self, now_ms: int) -> int:
        if self.s.session_start_ms is None:
            return self.cfg.round_duration_ms
        return max(0, self.cfg.round_duration_ms - (now_ms - self.s.session_start_ms))

    def macro_window_active(self, now_ms: int) -> bool:
        if self.s.last_tradeable_ms is None:
            return False
        return (now_ms - self.s.last_tradeable_ms) <= self.cfg.macro_signal_timeout_ms

    def in_trading_phase(self, now_ms: int) -> bool:
        return self._remaining_ms(now_ms) > self.cfg.probe_countdown_ms


# ═══════════════════════════════════════════════════════════════════
# ORDER MANAGER (from new_PM1)
# ═══════════════════════════════════════════════════════════════════
class OrderMgr:
    def __init__(self, client: "MergedPMClient", cfg: Cfg, state: State):
        self.c = client
        self.cfg = cfg
        self.s = state

    def has_live(self, *, symbol: str = None, role: str = None) -> bool:
        for o in self.s.live_orders.values():
            if symbol and o.symbol != symbol: continue
            if role and o.role != role: continue
            return True
        return False

    async def cancel_if_present(self, oid: str):
        key = str(oid)
        if key in self.s.pending_cancels: return
        self.s.pending_cancels.add(key)
        try:
            await self.c.cancel_order(oid)
        except Exception as e:
            if "No such order" in str(e):
                self.s.live_orders.pop(key, None)
        finally:
            self.s.pending_cancels.discard(key)

    async def cancel_stale(self):
        now = time.time()
        stale_secs = self.cfg.urgent_stale_secs if now < self.s.news_urgent_until else self.cfg.order_stale_secs
        stale = [oid for oid, o in self.s.live_orders.items() if now - o.created_at >= stale_secs]
        for oid in stale:
            await self.cancel_if_present(oid)

    async def place(self, *, symbol: str, qty: int, side: Side, price: int,
                    role: str, reason: str, thesis: str = None, event_id: int = 0) -> bool:
        if qty <= 0 or self.has_live(symbol=symbol):
            return False
        # Clip to limits
        pos = self.c.get_position(symbol)
        if side == Side.BUY:
            room = self.cfg.hard_position_limit - max(0, pos)
        else:
            room = self.cfg.hard_position_limit - max(0, -pos)
        qty = min(qty, room, self.cfg.max_order_size)
        if qty <= 0:
            return False

        oid = await self.c.place_order(symbol, qty, side, price)
        if oid is None: return False
        self.s.live_orders[str(oid)] = TrackedOrder(
            order_id=str(oid), symbol=symbol, side=side, qty=qty,
            price=price, role=role, reason=reason, thesis=thesis, event_id=event_id,
        )
        self.c._trace("order_submit", symbol=symbol, side=side.name, qty=qty,
                       price=price, role=role, reason=reason, thesis=thesis)
        return True

    def sync_fill(self, oid: str) -> Optional[TrackedOrder]:
        key = str(oid)
        tracked = self.s.live_orders.get(key)
        if tracked is None: return None
        if key in self.c.open_orders:
            rem = int(self.c.open_orders[key][1])
            tracked.qty = rem
            if rem <= 0:
                self.s.live_orders.pop(key, None)
        else:
            self.s.live_orders.pop(key, None)
        return tracked

    def sync_reject(self, oid: str) -> Optional[TrackedOrder]:
        return self.s.live_orders.pop(str(oid), None)

    def sync_cancel(self, oid: str, success: bool) -> Optional[TrackedOrder]:
        key = str(oid)
        if success:
            return self.s.live_orders.pop(key, None)
        # Failed cancel — check if order even exists
        if key not in self.c.open_orders:
            return self.s.live_orders.pop(key, None)
        return self.s.live_orders.get(key)


# ═══════════════════════════════════════════════════════════════════
# MAIN CLIENT
# ═══════════════════════════════════════════════════════════════════
class MergedPMClient(XChangeClient):
    def __init__(self, host: str, user: str, pw: str):
        self.cfg = load_cfg()
        super().__init__(host, user, pw, silent=False, symbols=list(self.cfg.symbols))
        self.state = State(
            legs={s: None for s in self.cfg.symbols},
            macro_targets={s: 0 for s in self.cfg.symbols},
            baseline_targets={s: 0 for s in self.cfg.symbols},
            combined_targets={s: 0 for s in self.cfg.symbols},
        )
        self.strategy = StrategyEngine(self.cfg, self.state)
        self.orders = OrderMgr(self, self.cfg, self.state)
        self._lock = asyncio.Lock()
        self._trace_file = None
        self._now_ms = 0  # monotonic ms approximation

    def _trace(self, event_type: str, **kw):
        if not self.cfg.trace_enabled: return
        if self._trace_file is None:
            d = Path(self.cfg.trace_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._trace_file = (d / f"merged_pm_{int(time.time())}.jsonl").open("a")
        payload = {"event_type": event_type, "ts": time.time(), "now_ms": self._now_ms, **kw}
        try:
            self._trace_file.write(json.dumps(payload, default=str) + "\n")
            self._trace_file.flush()
        except Exception:
            pass

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def refresh_book(self, symbol: str) -> TopOfBook:
        book = self.order_books.get(symbol)
        bids, asks = [], []
        if book:
            bids = [(int(p), int(q)) for p, q in book.bids.items() if int(q) > 0]
            asks = [(int(p), int(q)) for p, q in book.asks.items() if int(q) > 0]
        bb = max(bids, key=lambda x: x[0]) if bids else None
        ba = min(asks, key=lambda x: x[0]) if asks else None
        t = TopOfBook(
            bid=bb[0] if bb else None, ask=ba[0] if ba else None,
            bid_qty=bb[1] if bb else 0, ask_qty=ba[1] if ba else 0,
        )
        self.state.books[symbol] = t
        return t

    def refresh_all_books(self):
        for s in self.cfg.symbols:
            self.refresh_book(s)

    def _marketable_price(self, book: TopOfBook, side: Side) -> Optional[int]:
        if side == Side.BUY:
            return int(book.ask + self.cfg.aggressive_ticks) if book.ask is not None else None
        return int(max(0, book.bid - self.cfg.aggressive_ticks)) if book.bid is not None else None

    async def _startup_flatten(self) -> bool:
        # Cancel inherited orders
        for oid in [str(o) for o in self.open_orders if str(o) not in self.state.live_orders]:
            await self.orders.cancel_if_present(oid)
        # Flatten positions
        for s in self.cfg.symbols:
            pos = self.get_position(s)
            if pos == 0: continue
            b = self.state.books.get(s)
            if not b or b.bid is None or b.ask is None: continue
            if self.orders.has_live(symbol=s): continue
            side = Side.SELL if pos > 0 else Side.BUY
            price = self._marketable_price(b, side)
            if price is None: continue
            await self.orders.place(
                symbol=s, qty=min(abs(pos), self.cfg.startup_flatten_chunk),
                side=side, price=price, role="flatten", reason="startup",
            )
        flat = all(self.get_position(s) == 0 for s in self.cfg.symbols)
        no_orders = not self.open_orders and not self.state.live_orders
        if flat and no_orders:
            self.state.startup_done = True
            self._trace("startup_done")
        return self.state.startup_done

    async def _execute_targets(self):
        """Execute toward combined targets, one order per symbol at a time."""
        targets = self.state.combined_targets
        now = time.time()

        for s in self.cfg.symbols:
            target = targets.get(s, 0)
            pos = self.get_position(s)
            delta = target - pos
            if delta == 0 or self.orders.has_live(symbol=s):
                continue
            if now - self.state.last_add_ts < self.cfg.add_cooldown_secs:
                continue

            b = self.state.books.get(s)
            if not b: continue

            side = Side.BUY if delta > 0 else Side.SELL
            price = self._marketable_price(b, side)
            if price is None: continue

            qty = min(abs(delta), self.cfg.max_order_size)
            placed = await self.orders.place(
                symbol=s, qty=qty, side=side, price=price,
                role="entry", reason="target_chase",
                thesis=self.state.active_signal.source if self.state.active_signal else "phase",
                event_id=self.state.event_counter,
            )
            if placed:
                self.state.last_add_ts = now

    async def _flatten_exposure(self):
        """Flatten any remaining positions when targets are all zero."""
        for s in self.cfg.symbols:
            pos = self.get_position(s)
            if pos == 0 or self.orders.has_live(symbol=s): continue
            b = self.state.books.get(s)
            if not b: continue
            side = Side.SELL if pos > 0 else Side.BUY
            price = self._marketable_price(b, side)
            if price is None: continue
            await self.orders.place(
                symbol=s, qty=min(abs(pos), self.cfg.max_order_size),
                side=side, price=price, role="exit", reason="flatten",
            )

    async def evaluate(self):
        async with self._lock:
            self.refresh_all_books()
            await self.orders.cancel_stale()

            # Clean up orders that vanished from exchange
            for oid in list(self.state.live_orders.keys()):
                if oid not in self.open_orders:
                    self.state.live_orders.pop(oid, None)

            if not self.state.startup_done:
                await self._startup_flatten()
                return

            if self.state.session_start_ms is None:
                self.state.session_start_ms = self._now_ms

            books = self.state.books
            self.strategy.compute_combined(books, self._now_ms)

            has_targets = any(t != 0 for t in self.state.combined_targets.values())
            has_exposure = any(self.get_position(s) != 0 for s in self.cfg.symbols)

            if has_targets:
                await self._execute_targets()
            elif has_exposure:
                await self._flatten_exposure()

    # ── Callbacks ──
    async def bot_handle_book_update(self, symbol: str):
        if symbol in self.cfg.symbols:
            self._now_ms = int(time.time() * 1000)
            await self.evaluate()

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.symbols:
            await self.evaluate()

    async def bot_handle_order_fill(self, oid: str, qty: int, price: int):
        tracked = self.orders.sync_fill(oid)
        # Clear regimes if flat
        if all(self.get_position(s) == 0 for s in self.cfg.symbols):
            if not any(o.symbol in self.cfg.symbols for o in self.state.live_orders.values()):
                pass  # regime clears naturally via strategy

        self._trace("fill", order_id=str(oid),
                     symbol=tracked.symbol if tracked else "?",
                     side=tracked.side.name if tracked else "?",
                     qty=qty, price=price,
                     positions={s: self.get_position(s) for s in self.cfg.symbols})
        await self.evaluate()

    async def bot_handle_order_rejected(self, oid: str, reason: str):
        tracked = self.orders.sync_reject(oid)
        if tracked and tracked.role == "entry":
            self.state.blocked_until = time.time() + self.cfg.reentry_block_secs
        self._trace("reject", order_id=str(oid), reason=reason)

    async def bot_handle_cancel_response(self, oid: str, success: bool, error: str = None):
        self.orders.sync_cancel(oid, success)
        self.state.pending_cancels.discard(str(oid))
        if not success:
            self._trace("cancel_fail", order_id=str(oid), error=error)

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        data = news_release.get("new_data", {}) or {}
        tick = news_release.get("tick")
        self._now_ms = int(time.time() * 1000)
        if self.state.session_start_ms is None:
            self.state.session_start_ms = self._now_ms

        signal = None

        if kind == "structured":
            subtype = data.get("structured_subtype", "")
            if subtype == "cpi_print" and "actual" in data and "forecast" in data:
                actual, forecast = float(data["actual"]), float(data["forecast"])
                self.refresh_all_books()
                signal = self.strategy.signal_from_cpi(actual, forecast, self.state.books)
                self.state.news_urgent_until = time.time() + 3.0
                self._trace("news_cpi", tick=tick, actual=actual, forecast=forecast,
                             surprise=actual-forecast, signal_bucket=signal.bucket if signal else "none")

        elif kind == "unstructured":
            content = str(data.get("content", ""))
            msg_type = str(data.get("type", ""))

            # Check for CPI in text
            m = _CPI_RE.search(content)
            if m:
                actual, forecast = float(m.group("actual")), float(m.group("forecast"))
                self.refresh_all_books()
                signal = self.strategy.signal_from_cpi(actual, forecast, self.state.books)
                self.state.news_urgent_until = time.time() + 3.0
                self._trace("news_cpi_text", tick=tick, actual=actual, forecast=forecast)
            elif msg_type.lower() == "fedspeak" or any(kw in content.lower() for kw in
                    ["fed", "inflation", "policy", "rate", "rates", "cpi", "easing", "restrictive"]):
                self.refresh_all_books()
                signal = self.strategy.signal_from_headline(content, self.state.books)
                if signal:
                    self.state.news_urgent_until = time.time() + 2.0
                self._trace("news_headline", tick=tick, content=content,
                             bucket=signal.bucket if signal else "none")

        if signal is not None and self.strategy.in_trading_phase(self._now_ms):
            self.state.last_tradeable_ms = self._now_ms
            # If we already have an active signal, queue takeover
            if self.state.active_signal is not None:
                self.state.pending_signal = signal
                # Clear current macro targets to flatten, then activate new
                self.state.macro_targets = {s: 0 for s in self.cfg.symbols}
                self.state.active_signal = None
            else:
                self.strategy.activate_signal(signal, self.state.books, self._now_ms)

            # Check if pending can activate (previous position flat)
            if self.state.pending_signal is not None:
                if all(self.get_position(s) == 0 for s in self.cfg.symbols):
                    sig = self.state.pending_signal
                    self.state.pending_signal = None
                    self.strategy.activate_signal(sig, self.state.books, self._now_ms)

        await self.evaluate()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_market_resolved(self, market_id: str, winning: str, tick: int):
        self._trace("resolved", winner=winning, tick=tick,
                     positions={s: self.get_position(s) for s in self.cfg.symbols})

    async def bot_handle_settlement_payout(self, user: str, mid: str, amount: int, tick: int):
        self._trace("payout", amount=amount, tick=tick)

    async def trade(self):
        await asyncio.sleep(1.5)
        while True:
            try:
                self._now_ms = int(time.time() * 1000)
                await self.evaluate()
            except Exception as e:
                self._trace("loop_error", error=repr(e))
            await asyncio.sleep(self.cfg.loop_sleep_secs)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    client = MergedPMClient(
        _env("UTC_HOST", "34.197.188.76:3333"),
        _env("UTC_USERNAME", "uiuc"),
        _env("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())