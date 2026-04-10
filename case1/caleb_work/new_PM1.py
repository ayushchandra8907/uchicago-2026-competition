from __future__ import annotations

import asyncio
import json
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


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class BotConfig:
    fed_hike: str
    fed_hold: str
    fed_cut: str

    payout_scale: float
    cpi_to_rate_bp: float
    max_temp_rate_bias_bp: float
    cpi_bias_ttl_secs: float
    headline_bias_ttl_secs: float
    headline_apply_min_abs_bias_bp: float

    dead_tail_mid_frac: float
    dead_tail_gap_frac: float

    rate_hard_position_limit: int
    rate_max_order_size: int
    rate_normal_size: int
    rate_strong_size: int
    rate_extreme_size: int
    rate_cpi_entry_edge_bp: float
    rate_headline_entry_edge_bp: float
    rate_headline_strong_edge_bp: float
    rate_exit_edge_bp: float
    rate_add_edge_step_bp: float
    rate_add_edge_frac: float
    rate_reentry_block_secs: float
    add_cooldown_secs: float

    max_active_orders_per_symbol: int
    order_stale_secs: float
    urgent_order_stale_secs: float
    hedge_followup_secs: float
    entry_pair_grace_secs: float
    repair_aggressive_ticks: int
    cleanup_aggressive_ticks: int
    loop_sleep_secs: float
    status_log_interval_secs: float
    startup_flatten_chunk_rate: int

    trace_enabled: bool
    trace_dir: str

    @property
    def rate_symbols(self) -> tuple[str, str, str]:
        return (self.fed_hike, self.fed_hold, self.fed_cut)


def load_config() -> BotConfig:
    return BotConfig(
        fed_hike=env_str("PM1_FED_HIKE_SYMBOL", "R_HIKE"),
        fed_hold=env_str("PM1_FED_HOLD_SYMBOL", "R_HOLD"),
        fed_cut=env_str("PM1_FED_CUT_SYMBOL", "R_CUT"),
        payout_scale=env_float("PM1_PAYOUT_SCALE", 1000.0),
        cpi_to_rate_bp=env_float("PM1_CPI_TO_RATE_BP", 4243.66),
        max_temp_rate_bias_bp=env_float("PM1_MAX_TEMP_RATE_BIAS_BP", 8.0),
        cpi_bias_ttl_secs=env_float("PM1_CPI_BIAS_TTL_SECS", 2.5),
        headline_bias_ttl_secs=env_float("PM1_HEADLINE_BIAS_TTL_SECS", 2.25),
        headline_apply_min_abs_bias_bp=env_float("PM1_HEADLINE_MIN_ABS_BIAS_BP", 0.75),
        dead_tail_mid_frac=env_float("PM1_DEAD_TAIL_MID_FRAC", 0.12),
        dead_tail_gap_frac=env_float("PM1_DEAD_TAIL_GAP_FRAC", 0.18),
        rate_hard_position_limit=env_int("PM1_RATE_HARD_POSITION_LIMIT", 200),
        rate_max_order_size=env_int("PM1_RATE_MAX_ORDER_SIZE", 40),
        rate_normal_size=env_int("PM1_RATE_NORMAL_SIZE", 80),
        rate_strong_size=env_int("PM1_RATE_STRONG_SIZE", 120),
        rate_extreme_size=env_int("PM1_RATE_EXTREME_SIZE", 160),
        rate_cpi_entry_edge_bp=env_float("PM1_RATE_CPI_ENTRY_EDGE_BP", 2.0),
        rate_headline_entry_edge_bp=env_float("PM1_RATE_HEADLINE_ENTRY_EDGE_BP", 1.25),
        rate_headline_strong_edge_bp=env_float("PM1_RATE_HEADLINE_STRONG_EDGE_BP", 1.75),
        rate_exit_edge_bp=env_float("PM1_RATE_EXIT_EDGE_BP", 1.0),
        rate_add_edge_step_bp=env_float("PM1_RATE_ADD_EDGE_STEP_BP", 1.5),
        rate_add_edge_frac=env_float("PM1_RATE_ADD_EDGE_FRAC", 0.65),
        rate_reentry_block_secs=env_float("PM1_RATE_REENTRY_BLOCK_SECS", 1.0),
        add_cooldown_secs=env_float("PM1_ADD_COOLDOWN_SECS", 0.35),
        max_active_orders_per_symbol=env_int("PM1_MAX_ACTIVE_ORDERS_PER_SYMBOL", 1),
        order_stale_secs=env_float("PM1_ORDER_STALE_SECS", 0.50),
        urgent_order_stale_secs=env_float("PM1_URGENT_ORDER_STALE_SECS", 0.10),
        hedge_followup_secs=env_float("PM1_HEDGE_FOLLOWUP_SECS", 0.10),
        entry_pair_grace_secs=env_float("PM1_ENTRY_PAIR_GRACE_SECS", 0.25),
        repair_aggressive_ticks=env_int("PM1_REPAIR_AGGRESSIVE_TICKS", 1),
        cleanup_aggressive_ticks=env_int("PM1_CLEANUP_AGGRESSIVE_TICKS", 1),
        loop_sleep_secs=env_float("PM1_LOOP_SLEEP_SECS", 0.20),
        status_log_interval_secs=env_float("PM1_STATUS_LOG_INTERVAL_SECS", 2.0),
        startup_flatten_chunk_rate=env_int("PM1_STARTUP_FLATTEN_CHUNK_RATE", 40),
        trace_enabled=env_bool("PM1_TRACE_ENABLED", True),
        trace_dir=env_str("PM1_TRACE_DIR", str(Path(__file__).resolve().parent / "logs")),
    )


HAWKISH_BIGRAMS: dict[str, float] = {
    "inflation risks": 2.6,
    "persistent inflation": 2.4,
    "sticky prices": 2.6,
    "higher for": 1.8,
    "for longer": 1.8,
    "higher for longer": 3.2,
    "stay restrictive": 2.8,
    "restrictive for longer": 3.0,
    "policy restrictive": 2.0,
    "strong demand": 1.8,
    "tight labor": 2.0,
    "labor tight": 1.8,
    "wage growth": 1.6,
    "upside inflation": 2.8,
    "reaccelerating inflation": 3.0,
    "hot inflation": 3.0,
    "hot cpi": 3.2,
    "above forecast": 2.8,
    "above forecasts": 2.8,
    "above target": 2.4,
    "path of": 1.0,
    "of cuts": 0.8,
    "path of cuts": 2.0,
    "reassess path": 1.4,
    "pressure on": 0.8,
    "on the": 0.2,
    "the fed": 0.2,
    "pressure on the fed": 2.2,
    "inflation surprise": 2.4,
    "hawkish surprise": 2.8,
}

DOVISH_BIGRAMS: dict[str, float] = {
    "cooling inflation": -2.0,
    "softer inflation": -2.0,
    "cooling labor": -2.0,
    "labor market": -0.3,
    "cooling labor market": -2.4,
    "moving back": -1.2,
    "back to": -0.8,
    "to target": -1.2,
    "back to target": -2.4,
    "easing inflation": -2.0,
    "inflation pressures": -1.0,
    "easing inflation pressures": -2.6,
    "softening data": -2.4,
    "policy easing": -2.4,
    "expectations of": -1.0,
    "of policy": -0.4,
    "policy easing": -2.4,
    "expectations of policy easing": -3.0,
    "downside growth": -1.8,
    "growth risks": -1.2,
    "downside growth risks": -2.2,
    "lean toward": -1.2,
    "toward cuts": -1.2,
    "lean toward cuts": -2.4,
    "below forecast": -2.8,
    "below forecasts": -2.8,
    "below expectations": -2.6,
    "disinflation trend": -2.2,
    "dovish surprise": -2.8,
}

HAWKISH_UNIGRAMS: dict[str, float] = {
    "inflation": 0.4,
    "sticky": 1.2,
    "persistent": 1.2,
    "restrictive": 1.0,
    "strong": 0.6,
    "tight": 0.8,
    "wage": 0.6,
    "growth": 0.2,
    "reassess": 0.8,
    "hawkish": 1.6,
    "upside": 0.8,
    "hot": 1.2,
    "elevated": 0.8,
    "risks": 0.3,
}

DOVISH_UNIGRAMS: dict[str, float] = {
    "cooling": -1.0,
    "softening": -1.2,
    "easing": -1.0,
    "disinflation": -1.2,
    "downside": -0.8,
    "cuts": -0.6,
    "cut": -0.4,
    "weaker": -0.8,
    "slowdown": -1.0,
    "dovish": -1.6,
    "below": -0.8,
    "target": -0.3,
}

AMPLIFIERS: dict[str, float] = {
    "significantly": 1.25,
    "materially": 1.25,
    "sharply": 1.35,
    "unexpectedly": 1.25,
    "much": 1.15,
    "far": 1.15,
    "clearly": 1.10,
}

DAMPENERS: dict[str, float] = {
    "balanced risks": 0.35,
    "mixed economic indicators": 0.45,
    "mixed indicators": 0.45,
    "data dependence": 0.55,
    "data dependent": 0.55,
    "no clear signal": 0.40,
    "await upcoming data": 0.40,
    "remains cautious": 0.50,
    "communication remains cautious": 0.50,
    "cautious": 0.70,
}

MACRO_RELEVANCE_KEYWORDS = {
    "fed",
    "fomc",
    "cpi",
    "inflation",
    "disinflation",
    "policy",
    "rate",
    "rates",
    "cut",
    "cuts",
    "hike",
    "hikes",
    "labor market",
    "wage",
    "prices",
    "sticky",
    "restrictive",
    "target",
    "easing",
}

STOPWORDS = {
    "about",
    "amid",
    "after",
    "ahead",
    "again",
    "against",
    "also",
    "because",
    "between",
    "chair",
    "clear",
    "could",
    "data",
    "expected",
    "highlights",
    "into",
    "looks",
    "market",
    "markets",
    "move",
    "next",
    "note",
    "officials",
    "raises",
    "signal",
    "than",
    "that",
    "their",
    "they",
    "toward",
    "upcoming",
    "with",
}

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
CPI_TEXT_PATTERNS = (
    re.compile(
        r"cpi[^0-9\-+]*actual[^0-9\-+]*(?P<actual>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[^0-9a-zA-Z]+vs[^0-9a-zA-Z]+forecast[^0-9\-+]*(?P<forecast>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"cpi[^0-9\-+]*forecast[^0-9\-+]*(?P<forecast>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[^0-9a-zA-Z]+actual[^0-9\-+]*(?P<actual>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    lowered = lowered.replace("'s", "")
    lowered = re.sub(r"[^a-z0-9.+\-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_term(term: str) -> str:
    return normalize_text(term)


_HAWKISH_BIGRAMS = {normalize_term(k): v for k, v in HAWKISH_BIGRAMS.items()}
_DOVISH_BIGRAMS = {normalize_term(k): v for k, v in DOVISH_BIGRAMS.items()}
_HAWKISH_UNIGRAMS = {normalize_term(k): v for k, v in HAWKISH_UNIGRAMS.items()}
_DOVISH_UNIGRAMS = {normalize_term(k): v for k, v in DOVISH_UNIGRAMS.items()}
_AMPLIFIERS = {normalize_term(k): v for k, v in AMPLIFIERS.items()}
_DAMPENERS = {normalize_term(k): v for k, v in DAMPENERS.items()}
_PHRASE_WEIGHTS = {**_HAWKISH_BIGRAMS, **_DOVISH_BIGRAMS}
_UNIGRAM_WEIGHTS = {**_HAWKISH_UNIGRAMS, **_DOVISH_UNIGRAMS}
_MAX_PHRASE_TOKENS = max((len(term.split()) for term in _PHRASE_WEIGHTS), default=2)


@dataclass(frozen=True)
class MacroSentimentResult:
    score: float
    bucket: str
    direction: int
    implied_bias_bp: float
    matched_phrases: tuple[str, ...]
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]


@dataclass(frozen=True)
class ParsedMacroEvent:
    parsed_type: str
    source: str
    bias_bp: float
    ttl_secs: float
    label: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    surprise: Optional[float] = None
    sentiment: Optional[MacroSentimentResult] = None
    raw_content: Optional[str] = None
    message_type: Optional[str] = None


def contains_phrase(tokens: list[str], phrase: str) -> bool:
    parts = phrase.split()
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in tokens
    for start in range(0, len(tokens) - len(parts) + 1):
        if tokens[start : start + len(parts)] == parts:
            return True
    return False


def bucket_for_score(score: float) -> str:
    absolute = abs(score)
    if absolute <= 0.0:
        return "none"
    if absolute < 1.75:
        return "light"
    if absolute < 3.0:
        return "medium"
    if absolute < 4.25:
        return "strong"
    return "extreme"


def bias_for_bucket(score: float, bucket: str) -> float:
    direction = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
    if bucket == "none":
        return 0.0
    if bucket == "light":
        return 0.75 * direction
    if bucket == "medium":
        return 1.50 * direction
    if bucket == "strong":
        return 2.25 * direction
    if abs(score) >= 5.25:
        return 4.00 * direction
    return 3.25 * direction


def extract_unknown_candidates(tokens: list[str], used: list[bool]) -> tuple[str, ...]:
    unknowns: list[str] = []
    seen: set[str] = set()
    for start in range(0, len(tokens) - 1):
        if used[start] or used[start + 1]:
            continue
        left = tokens[start]
        right = tokens[start + 1]
        if len(left) < 4 or len(right) < 4:
            continue
        if left in STOPWORDS or right in STOPWORDS:
            continue
        phrase = f"{left} {right}"
        if phrase in _PHRASE_WEIGHTS or phrase in _AMPLIFIERS or phrase in _DAMPENERS or phrase in seen:
            continue
        seen.add(phrase)
        unknowns.append(phrase)
    return tuple(unknowns)


def score_macro_headline(text: str | None) -> MacroSentimentResult:
    normalized = normalize_text(text)
    if not normalized:
        return MacroSentimentResult(
            score=0.0,
            bucket="none",
            direction=0,
            implied_bias_bp=0.0,
            matched_phrases=(),
            matched_unigrams=(),
            matched_bigrams=(),
            unknown_candidate_phrases=(),
        )

    tokens = normalized.split()
    used = [False] * len(tokens)
    matched_bigrams: list[str] = []
    matched_unigrams: list[str] = []
    score = 0.0

    for phrase_len in range(_MAX_PHRASE_TOKENS, 1, -1):
        index = 0
        while index <= len(tokens) - phrase_len:
            if any(used[index : index + phrase_len]):
                index += 1
                continue
            phrase = " ".join(tokens[index : index + phrase_len])
            weight = _PHRASE_WEIGHTS.get(phrase)
            if weight is None:
                index += 1
                continue
            matched_bigrams.append(phrase)
            score += weight
            for used_index in range(index, index + phrase_len):
                used[used_index] = True
            index += phrase_len

    for idx, token in enumerate(tokens):
        if used[idx]:
            continue
        weight = _UNIGRAM_WEIGHTS.get(token)
        if weight is None:
            continue
        matched_unigrams.append(token)
        score += weight
        used[idx] = True

    amplifier = max((weight for phrase, weight in _AMPLIFIERS.items() if contains_phrase(tokens, phrase)), default=1.0)
    dampener = min((weight for phrase, weight in _DAMPENERS.items() if contains_phrase(tokens, phrase)), default=1.0)
    score *= amplifier
    score *= dampener
    score = max(-6.0, min(6.0, score))

    bucket = bucket_for_score(score)
    direction = 1 if score > 0 else -1 if score < 0 else 0
    implied_bias_bp = bias_for_bucket(score, bucket)
    matched_phrases = tuple(dict.fromkeys([*matched_bigrams, *matched_unigrams]))

    return MacroSentimentResult(
        score=score,
        bucket=bucket,
        direction=direction,
        implied_bias_bp=implied_bias_bp,
        matched_phrases=matched_phrases,
        matched_unigrams=tuple(dict.fromkeys(matched_unigrams)),
        matched_bigrams=tuple(dict.fromkeys(matched_bigrams)),
        unknown_candidate_phrases=extract_unknown_candidates(tokens, used),
    )


def parse_float_token(text: str) -> Optional[float]:
    if not text:
        return None
    match = NUMBER_RE.search(text.replace("%", ""))
    if not match:
        return None
    return float(match.group(0))


def parse_cpi_text_event(content: str) -> tuple[Optional[float], Optional[float]]:
    for pattern in CPI_TEXT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        actual = parse_float_token(match.group("actual"))
        forecast = parse_float_token(match.group("forecast"))
        if actual is not None and forecast is not None:
            return actual, forecast
    return None, None


@dataclass
class TopOfBook:
    bid: Optional[int] = None
    bid_qty: int = 0
    ask: Optional[int] = None
    ask_qty: int = 0
    updated_ts: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        if self.bid is not None:
            return float(self.bid)
        if self.ask is not None:
            return float(self.ask)
        return None


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
    signal_strength: float
    event_id: int
    created_at: float = field(default_factory=time.time)


@dataclass
class RateSnapshot:
    q_hike: float
    q_hold: float
    q_cut: float
    market_expected_rate_bp: float
    effective_expected_rate_bp: float
    delta_market_bp: float
    delta_effective_bp: float
    bias_bp: float
    urgent: bool
    fresh_macro_event: bool
    macro_source: Optional[str]
    macro_bucket: Optional[str]
    macro_score: float
    macro_label: Optional[str]


@dataclass
class RatesEntryDecision:
    direction: str
    edge_bp: float
    target_size: int
    buy_symbol: str
    sell_symbol: str


@dataclass
class MarketState:
    books: dict[str, TopOfBook] = field(default_factory=dict)
    live_orders: dict[str, TrackedOrder] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)

    temp_rate_bias_bp: float = 0.0
    temp_rate_bias_started_at: float = 0.0
    temp_rate_bias_expires_at: float = 0.0
    last_macro_event_ts: float = 0.0
    last_macro_source: Optional[str] = None
    last_macro_bias_bp: float = 0.0
    last_macro_event_id: int = 0
    last_macro_bucket: Optional[str] = None
    last_macro_score: float = 0.0
    last_macro_label: Optional[str] = None
    news_urgency_until: float = 0.0

    last_market_expected_rate_bp: Optional[float] = None
    last_effective_expected_rate_bp: Optional[float] = None

    startup_flatten_complete: bool = False
    session_start_cash: Optional[float] = None
    session_start_mtm: Optional[float] = None
    last_status_log_ts: float = 0.0

    rates_regime_direction: Optional[str] = None
    rates_buy_symbol: Optional[str] = None
    rates_sell_symbol: Optional[str] = None
    rates_active_event_id: int = 0
    rates_last_closed_event_id: int = 0
    rates_entry_stage: int = 0
    rates_last_entry_edge: float = 0.0
    rates_last_add_ts: float = 0.0
    rates_blocked_direction: Optional[str] = None
    rates_blocked_until: float = 0.0
    rates_unwind_active: bool = False
    rates_pairing_until: float = 0.0

    def clear_rates_regime(self) -> None:
        self.rates_regime_direction = None
        self.rates_buy_symbol = None
        self.rates_sell_symbol = None
        self.rates_active_event_id = 0
        self.rates_entry_stage = 0
        self.rates_last_entry_edge = 0.0
        self.rates_last_add_ts = 0.0
        self.rates_unwind_active = False
        self.rates_pairing_until = 0.0


class OrderManager:
    def __init__(self, client: "NewPM1Client", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def has_live_order(
        self,
        *,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        side: Optional[Side] = None,
    ) -> bool:
        for order in self.state.live_orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if side is not None and order.side != side:
                continue
            return True
        return False

    def pending_qty(self, symbol: str, side: Side) -> int:
        return sum(order.qty for order in self.state.live_orders.values() if order.symbol == symbol and order.side == side)

    async def cancel_order_if_present(self, order_id: str) -> None:
        order_key = str(order_id)
        if order_key in self.state.pending_cancels:
            return
        self.state.pending_cancels.add(order_key)
        try:
            await self.client.cancel_order(order_id)
        finally:
            self.state.pending_cancels.discard(order_key)

    async def cancel_stale_orders(self) -> None:
        now = time.time()
        stale_secs = self.cfg.urgent_order_stale_secs if now < self.state.news_urgency_until else self.cfg.order_stale_secs
        stale = [order_id for order_id, order in self.state.live_orders.items() if now - order.created_at >= stale_secs]
        for order_id in stale:
            await self.cancel_order_if_present(order_id)

    async def cancel_counterpart_after_fill(self, tracked: Optional[TrackedOrder]) -> None:
        if tracked is None or tracked.role not in {"entry", "exit"}:
            return
        now = time.time()
        for other in list(self.state.live_orders.values()):
            if other.order_id == tracked.order_id:
                continue
            if other.role != tracked.role or other.event_id != tracked.event_id:
                continue
            if tracked.role == "entry" and other.thesis != tracked.thesis:
                continue
            if now - other.created_at < self.cfg.hedge_followup_secs:
                continue
            await self.cancel_order_if_present(other.order_id)

    async def place_tracked_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: Side,
        price: int,
        role: str,
        reason: str,
        thesis: Optional[str],
        signal_strength: float,
        event_id: int,
    ) -> bool:
        if qty <= 0 or self.has_live_order(symbol=symbol):
            return False
        same_symbol_orders = [order for order in self.state.live_orders.values() if order.symbol == symbol]
        if len(same_symbol_orders) >= self.cfg.max_active_orders_per_symbol:
            await self.cancel_order_if_present(same_symbol_orders[0].order_id)
            return False
        order_id = await self.client.place_order(symbol, int(qty), side, int(price))
        if order_id is None:
            return False
        tracked = TrackedOrder(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            qty=int(qty),
            price=int(price),
            role=role,
            reason=reason,
            thesis=thesis,
            signal_strength=float(signal_strength),
            event_id=int(event_id),
        )
        self.state.live_orders[str(order_id)] = tracked
        self.client._trace(
            "order_submit",
            tick=self.client.current_tick,
            symbol=symbol,
            side=side.name,
            qty=qty,
            price=price,
            role=role,
            reason=reason,
            thesis=thesis,
            signal_strength=signal_strength,
            event_id=event_id,
            **self.client.positions_payload(),
        )
        return True

    def sync_fill(self, order_id: str) -> Optional[TrackedOrder]:
        order_key = str(order_id)
        tracked = self.state.live_orders.get(order_key)
        if tracked is None:
            return None
        if order_key in self.client.open_orders:
            remaining_qty = int(self.client.open_orders[order_key][1])
            tracked.qty = remaining_qty
            if remaining_qty <= 0:
                self.state.live_orders.pop(order_key, None)
        else:
            self.state.live_orders.pop(order_key, None)
        return tracked

    def sync_rejected(self, order_id: str) -> Optional[TrackedOrder]:
        return self.state.live_orders.pop(str(order_id), None)

    def sync_cancel_response(self, order_id: str, success: bool) -> Optional[TrackedOrder]:
        if success:
            return self.state.live_orders.pop(str(order_id), None)
        return self.state.live_orders.get(str(order_id))


class RiskManager:
    def __init__(self, client: "NewPM1Client", cfg: BotConfig, state: MarketState, orders: OrderManager):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders

    def clip_rate_qty(self, symbol: str, side: Side, desired_qty: int, thesis_cap: int) -> int:
        pos = self.client.get_position(symbol)
        pending = self.orders.pending_qty(symbol, side)
        same_dir_pos = max(0, pos) if side == Side.BUY else max(0, -pos)
        hard_remaining = self.cfg.rate_hard_position_limit - same_dir_pos - pending
        soft_remaining = thesis_cap - same_dir_pos - pending
        return max(0, min(int(desired_qty), hard_remaining, soft_remaining, self.cfg.rate_max_order_size))

    def arm_session_baseline_if_ready(self) -> bool:
        if self.state.session_start_cash is not None and self.state.session_start_mtm is not None:
            return False
        for symbol in self.cfg.rate_symbols:
            if self.client.get_position(symbol) != 0:
                return False
        if self.client.open_orders or self.state.live_orders:
            return False
        cash, mtm = self.client.cash_and_total_mtm()
        self.state.session_start_cash = cash
        self.state.session_start_mtm = mtm
        self.client._trace("session_baseline", tick=self.client.current_tick, mtm=mtm, **self.client.positions_payload())
        return True

    async def startup_flatten_step(self) -> bool:
        inherited_order_ids = [str(order_id) for order_id in self.client.open_orders.keys() if str(order_id) not in self.state.live_orders]
        for order_id in inherited_order_ids:
            await self.orders.cancel_order_if_present(order_id)
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=symbol):
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.startup_flatten_chunk_rate, self.cfg.rate_max_order_size)
            price = int(book.bid if side == Side.SELL else book.ask)
            await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="flatten",
                reason="startup_flatten",
                thesis=None,
                signal_strength=float(abs(pos)),
                event_id=0,
            )
        all_flat = all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols)
        no_orders = not self.client.open_orders and not self.state.live_orders
        if all_flat and no_orders:
            self.state.startup_flatten_complete = True
            self.arm_session_baseline_if_ready()
            self.client._trace("startup_flatten_complete", tick=self.client.current_tick, **self.client.positions_payload())
            return True
        return False


class RatesSignalEngine:
    def __init__(self, client: "NewPM1Client", cfg: BotConfig, state: MarketState):
        self.client = client
        self.cfg = cfg
        self.state = state

    def source_ttl(self, source: Optional[str]) -> float:
        if source and source.startswith("cpi"):
            return self.cfg.cpi_bias_ttl_secs
        return self.cfg.headline_bias_ttl_secs

    def current_temp_bias(self) -> float:
        now = time.time()
        if now >= self.state.temp_rate_bias_expires_at or self.state.temp_rate_bias_expires_at <= self.state.temp_rate_bias_started_at:
            self.state.temp_rate_bias_bp = 0.0
            return 0.0
        duration = self.state.temp_rate_bias_expires_at - self.state.temp_rate_bias_started_at
        remaining = max(0.0, self.state.temp_rate_bias_expires_at - now)
        if duration <= 0:
            return 0.0
        return self.state.temp_rate_bias_bp * (remaining / duration)

    def mark_news_urgent(self, ttl_secs: float) -> None:
        self.state.news_urgency_until = max(self.state.news_urgency_until, time.time() + ttl_secs)

    def is_news_urgent(self) -> bool:
        return time.time() < self.state.news_urgency_until

    def apply_temp_rate_bias(
        self,
        delta_bp: float,
        ttl_secs: float,
        source: str,
        *,
        bucket: Optional[str] = None,
        score: float = 0.0,
        label: Optional[str] = None,
    ) -> None:
        current_bias = self.current_temp_bias()
        next_bias = self.client.clip(current_bias + delta_bp, -self.cfg.max_temp_rate_bias_bp, self.cfg.max_temp_rate_bias_bp)
        now = time.time()
        self.state.temp_rate_bias_bp = next_bias
        self.state.temp_rate_bias_started_at = now
        self.state.temp_rate_bias_expires_at = now + ttl_secs
        self.state.last_macro_event_ts = now
        self.state.last_macro_source = source
        self.state.last_macro_bias_bp = next_bias
        self.state.last_macro_bucket = bucket
        self.state.last_macro_score = score
        self.state.last_macro_label = label
        self.state.last_macro_event_id += 1
        self.mark_news_urgent(ttl_secs)

    def is_macro_relevant(self, content: str, message_type: Optional[str]) -> bool:
        if message_type and message_type.lower() in {"fedspeak", "macro", "rates", "cpi"}:
            return True
        lowered = content.lower()
        return any(keyword in lowered for keyword in MACRO_RELEVANCE_KEYWORDS)

    def classify_macro_event(self, kind: str | None, new_data: dict[str, Any]) -> Optional[ParsedMacroEvent]:
        if kind == "structured":
            subtype = str(new_data.get("structured_subtype") or "")
            if subtype == "cpi_print" and "actual" in new_data and "forecast" in new_data:
                actual = float(new_data["actual"])
                forecast = float(new_data["forecast"])
                surprise = actual - forecast
                bias_bp = self.client.clip(
                    surprise * self.cfg.cpi_to_rate_bp,
                    -self.cfg.max_temp_rate_bias_bp,
                    self.cfg.max_temp_rate_bias_bp,
                )
                return ParsedMacroEvent(
                    parsed_type="cpi",
                    source="cpi_print",
                    bias_bp=bias_bp,
                    ttl_secs=self.cfg.cpi_bias_ttl_secs,
                    label=f"cpi actual {actual:.6f} vs forecast {forecast:.6f}",
                    actual=actual,
                    forecast=forecast,
                    surprise=surprise,
                )
            return None

        if kind != "unstructured":
            return None

        content = str(new_data.get("content", "") or "")
        message_type = str(new_data.get("type", "") or "")
        if not content:
            return None

        actual, forecast = parse_cpi_text_event(content)
        if actual is not None and forecast is not None:
            surprise = actual - forecast
            bias_bp = self.client.clip(
                surprise * self.cfg.cpi_to_rate_bp,
                -self.cfg.max_temp_rate_bias_bp,
                self.cfg.max_temp_rate_bias_bp,
            )
            return ParsedMacroEvent(
                parsed_type="cpi_text",
                source="cpi_text",
                bias_bp=bias_bp,
                ttl_secs=self.cfg.cpi_bias_ttl_secs,
                label=f"cpi actual {actual:.6f} vs forecast {forecast:.6f}",
                actual=actual,
                forecast=forecast,
                surprise=surprise,
                raw_content=content,
                message_type=message_type,
            )

        if not self.is_macro_relevant(content, message_type):
            return None

        sentiment = score_macro_headline(content)
        if abs(sentiment.implied_bias_bp) < self.cfg.headline_apply_min_abs_bias_bp:
            return ParsedMacroEvent(
                parsed_type="headline",
                source="headline",
                bias_bp=0.0,
                ttl_secs=self.cfg.headline_bias_ttl_secs,
                label=content,
                sentiment=sentiment,
                raw_content=content,
                message_type=message_type,
            )
        return ParsedMacroEvent(
            parsed_type="headline",
            source="headline",
            bias_bp=sentiment.implied_bias_bp,
            ttl_secs=self.cfg.headline_bias_ttl_secs,
            label=content,
            sentiment=sentiment,
            raw_content=content,
            message_type=message_type,
        )

    def fed_probs(self) -> Optional[tuple[float, float, float]]:
        mid_hike = self.client.mid(self.cfg.fed_hike)
        mid_hold = self.client.mid(self.cfg.fed_hold)
        mid_cut = self.client.mid(self.cfg.fed_cut)
        if mid_hike is None or mid_hold is None or mid_cut is None:
            return None
        q_hike = mid_hike / self.cfg.payout_scale
        q_hold = mid_hold / self.cfg.payout_scale
        q_cut = mid_cut / self.cfg.payout_scale
        total = q_hike + q_hold + q_cut
        if total <= 1e-9:
            return None
        return q_hike / total, q_hold / total, q_cut / total

    def expected_rate_bp(self) -> Optional[float]:
        probs = self.fed_probs()
        if probs is None:
            return None
        q_hike, _, q_cut = probs
        return 25.0 * q_hike - 25.0 * q_cut

    def snapshot(self) -> Optional[RateSnapshot]:
        probs = self.fed_probs()
        if probs is None:
            return None
        market_expected_rate_bp = self.expected_rate_bp()
        if market_expected_rate_bp is None:
            return None
        bias_bp = self.current_temp_bias()
        effective_expected_rate_bp = market_expected_rate_bp + bias_bp
        delta_market_bp = 0.0 if self.state.last_market_expected_rate_bp is None else market_expected_rate_bp - self.state.last_market_expected_rate_bp
        delta_effective_bp = 0.0 if self.state.last_effective_expected_rate_bp is None else effective_expected_rate_bp - self.state.last_effective_expected_rate_bp
        fresh_macro = False
        if self.state.last_macro_event_ts > 0.0:
            fresh_macro = time.time() - self.state.last_macro_event_ts <= self.source_ttl(self.state.last_macro_source)
        return RateSnapshot(
            q_hike=probs[0],
            q_hold=probs[1],
            q_cut=probs[2],
            market_expected_rate_bp=market_expected_rate_bp,
            effective_expected_rate_bp=effective_expected_rate_bp,
            delta_market_bp=delta_market_bp,
            delta_effective_bp=delta_effective_bp,
            bias_bp=bias_bp,
            urgent=self.is_news_urgent(),
            fresh_macro_event=fresh_macro,
            macro_source=self.state.last_macro_source,
            macro_bucket=self.state.last_macro_bucket,
            macro_score=self.state.last_macro_score,
            macro_label=self.state.last_macro_label,
        )


class RatesTradingEngine:
    def __init__(
        self,
        client: "NewPM1Client",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk

    def current_pair_direction(self) -> Optional[str]:
        if self.state.rates_buy_symbol is None or self.state.rates_sell_symbol is None:
            return None
        buy_pos = self.client.get_position(self.state.rates_buy_symbol)
        sell_pos = self.client.get_position(self.state.rates_sell_symbol)
        if buy_pos > 0 and sell_pos < 0:
            return self.state.rates_regime_direction
        return None

    def any_rate_inventory(self) -> bool:
        return any(self.client.get_position(symbol) != 0 for symbol in self.cfg.rate_symbols)

    def has_orphaned_inventory(self) -> bool:
        if not self.any_rate_inventory():
            return False
        if self.state.rates_buy_symbol is None or self.state.rates_sell_symbol is None:
            return True
        buy_pos = self.client.get_position(self.state.rates_buy_symbol)
        sell_pos = self.client.get_position(self.state.rates_sell_symbol)
        other_symbols = set(self.cfg.rate_symbols) - {self.state.rates_buy_symbol, self.state.rates_sell_symbol}
        if any(self.client.get_position(symbol) != 0 for symbol in other_symbols):
            return True
        if buy_pos == 0 and sell_pos == 0:
            return False
        if buy_pos <= 0 or sell_pos >= 0:
            return True
        return abs(buy_pos) != abs(sell_pos)

    def marketable_price(self, book: TopOfBook, side: Side, aggressive_ticks: int = 0) -> Optional[int]:
        if side == Side.BUY:
            if book.ask is None:
                return None
            return int(book.ask + max(0, aggressive_ticks))
        if book.bid is None:
            return None
        return int(max(0, book.bid - max(0, aggressive_ticks)))

    def max_entry_stages(self, target_size: int) -> int:
        return max(1, (max(0, target_size) + max(1, self.cfg.rate_max_order_size) - 1) // max(1, self.cfg.rate_max_order_size))

    def select_pair(self, direction: str) -> tuple[str, str]:
        mids = {}
        for symbol in self.cfg.rate_symbols:
            mid = self.client.mid(symbol)
            mids[symbol] = -1.0 if mid is None else float(mid)
        ranked = sorted(mids.items(), key=lambda item: item[1], reverse=True)
        winner, challenger, dead = ranked[0][0], ranked[1][0], ranked[2][0]
        dead_mid = ranked[2][1]
        challenger_mid = ranked[1][1]
        dead_tail = dead_mid <= self.cfg.payout_scale * self.cfg.dead_tail_mid_frac or (challenger_mid - dead_mid) >= self.cfg.payout_scale * self.cfg.dead_tail_gap_frac
        if direction == "hawkish":
            if dead_tail and dead == self.cfg.fed_cut:
                return self.cfg.fed_hike, self.cfg.fed_hold
            return self.cfg.fed_hike, self.cfg.fed_cut
        if dead_tail and dead == self.cfg.fed_hike:
            return self.cfg.fed_cut, self.cfg.fed_hold
        return self.cfg.fed_cut, self.cfg.fed_hike

    async def flatten_all_rates(self, reason: str) -> bool:
        acted = False
        for symbol in self.cfg.rate_symbols:
            pos = self.client.get_position(symbol)
            if pos == 0:
                continue
            book = self.client.top(symbol)
            if book.bid is None or book.ask is None or self.orders.has_live_order(symbol=symbol):
                continue
            side = Side.SELL if pos > 0 else Side.BUY
            qty = min(abs(pos), self.cfg.rate_max_order_size)
            price = self.marketable_price(book, side, self.cfg.cleanup_aggressive_ticks)
            if price is None:
                continue
            placed = await self.orders.place_tracked_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price,
                role="exit",
                reason=reason,
                thesis=self.state.rates_regime_direction,
                signal_strength=float(abs(pos)),
                event_id=self.state.rates_active_event_id,
            )
            acted = acted or placed
        if acted:
            self.state.rates_blocked_direction = self.state.rates_regime_direction
            self.state.rates_blocked_until = max(
                self.state.temp_rate_bias_expires_at if reason == "orphan_cleanup" else 0.0,
                time.time() + self.cfg.rate_reentry_block_secs,
            )
            self.state.rates_unwind_active = True
            self.state.rates_pairing_until = 0.0
        return acted

    async def handle_missing_signal_exit(self) -> bool:
        if self.any_rate_inventory():
            return await self.flatten_all_rates("signal_lost")
        return False

    async def maybe_exit(self, snapshot: RateSnapshot) -> bool:
        direction = self.current_pair_direction()
        if direction is None:
            if self.any_rate_inventory():
                if time.time() < self.state.rates_pairing_until or self.orders.has_live_order(role="entry"):
                    return False
                return await self.flatten_all_rates("orphan_cleanup")
            return False
        edge_bp = abs(snapshot.bias_bp)
        compressed = edge_bp <= max(self.cfg.rate_exit_edge_bp, self.state.rates_last_entry_edge * 0.30)
        opposite = (direction == "hawkish" and snapshot.bias_bp <= -self.cfg.rate_exit_edge_bp) or (direction == "dovish" and snapshot.bias_bp >= self.cfg.rate_exit_edge_bp)
        stale = edge_bp < self.cfg.rate_exit_edge_bp
        if stale or opposite or compressed:
            return await self.flatten_all_rates("bias_decay" if stale or compressed else "macro_reversal")
        return False

    def compute_entry_decision(self, snapshot: RateSnapshot) -> Optional[RatesEntryDecision]:
        if not snapshot.fresh_macro_event:
            return None
        source = snapshot.macro_source or ""
        source_is_cpi = source.startswith("cpi")
        entry_edge = self.cfg.rate_cpi_entry_edge_bp if source_is_cpi else self.cfg.rate_headline_entry_edge_bp
        if snapshot.bias_bp >= entry_edge:
            direction = "hawkish"
        elif snapshot.bias_bp <= -entry_edge:
            direction = "dovish"
        else:
            return None
        edge_bp = abs(snapshot.bias_bp)
        if source_is_cpi and edge_bp >= 5.0:
            target_size = self.cfg.rate_extreme_size
        elif source_is_cpi and edge_bp >= 3.5:
            target_size = self.cfg.rate_strong_size
        elif not source_is_cpi and edge_bp >= self.cfg.rate_headline_strong_edge_bp:
            target_size = self.cfg.rate_strong_size
        else:
            target_size = self.cfg.rate_normal_size
        buy_symbol, sell_symbol = self.select_pair(direction)
        return RatesEntryDecision(
            direction=direction,
            edge_bp=edge_bp,
            target_size=target_size,
            buy_symbol=buy_symbol,
            sell_symbol=sell_symbol,
        )

    async def maybe_enter(self, snapshot: RateSnapshot) -> bool:
        decision = self.compute_entry_decision(snapshot)
        if decision is None:
            return False
        now = time.time()
        if self.state.rates_unwind_active:
            return False
        if self.state.rates_last_closed_event_id == self.state.last_macro_event_id and not self.any_rate_inventory():
            return False
        if self.state.rates_blocked_direction == decision.direction and now < self.state.rates_blocked_until:
            return False
        if self.orders.has_live_order(role="exit"):
            return False

        current_direction = self.current_pair_direction()
        same_regime = (
            self.state.rates_regime_direction == decision.direction
            and self.state.rates_buy_symbol == decision.buy_symbol
            and self.state.rates_sell_symbol == decision.sell_symbol
            and self.any_rate_inventory()
        )

        if self.has_orphaned_inventory() and not same_regime:
            if now < self.state.rates_pairing_until or self.orders.has_live_order(role="entry"):
                return False
            return False

        if current_direction is not None and not same_regime:
            return False

        if same_regime:
            if self.state.rates_entry_stage >= self.max_entry_stages(decision.target_size):
                return False
            if decision.edge_bp < self.state.rates_last_entry_edge + self.cfg.rate_add_edge_step_bp:
                base_edge = self.cfg.rate_cpi_entry_edge_bp if (snapshot.macro_source or "").startswith("cpi") else self.cfg.rate_headline_entry_edge_bp
                required_edge = max(base_edge, self.state.rates_last_entry_edge * self.cfg.rate_add_edge_frac)
                if decision.edge_bp < required_edge:
                    return False
            if now - self.state.rates_last_add_ts < self.cfg.add_cooldown_secs:
                return False

        buy_book = self.client.top(decision.buy_symbol)
        sell_book = self.client.top(decision.sell_symbol)
        if buy_book.ask is None or sell_book.bid is None:
            return False

        buy_pos = self.client.get_position(decision.buy_symbol)
        sell_pos = self.client.get_position(decision.sell_symbol)
        buy_filled_abs = max(0, buy_pos)
        sell_filled_abs = max(0, -sell_pos)
        if buy_filled_abs >= decision.target_size and sell_filled_abs >= decision.target_size:
            return False

        buy_needed = max(0, decision.target_size - buy_filled_abs)
        sell_needed = max(0, decision.target_size - sell_filled_abs)
        if same_regime and buy_filled_abs != sell_filled_abs:
            if buy_filled_abs < sell_filled_abs:
                repair_qty = min(buy_needed, sell_filled_abs - buy_filled_abs)
                buy_qty = self.risk.clip_rate_qty(decision.buy_symbol, Side.BUY, repair_qty, decision.target_size)
                sell_qty = 0
            else:
                repair_qty = min(sell_needed, buy_filled_abs - sell_filled_abs)
                buy_qty = 0
                sell_qty = self.risk.clip_rate_qty(decision.sell_symbol, Side.SELL, repair_qty, decision.target_size)
        else:
            buy_qty = self.risk.clip_rate_qty(decision.buy_symbol, Side.BUY, buy_needed, decision.target_size)
            sell_qty = self.risk.clip_rate_qty(decision.sell_symbol, Side.SELL, sell_needed, decision.target_size)
            paired_qty = min(buy_qty, sell_qty)
            buy_qty = paired_qty
            sell_qty = paired_qty

        if buy_qty <= 0 and sell_qty <= 0:
            return False
        if (buy_qty > 0 and self.orders.has_live_order(symbol=decision.buy_symbol)) or (sell_qty > 0 and self.orders.has_live_order(symbol=decision.sell_symbol)):
            return False

        buy_price = self.marketable_price(
            buy_book,
            Side.BUY,
            self.cfg.repair_aggressive_ticks if same_regime and buy_qty > 0 and sell_qty == 0 else 0,
        )
        sell_price = self.marketable_price(
            sell_book,
            Side.SELL,
            self.cfg.repair_aggressive_ticks if same_regime and sell_qty > 0 and buy_qty == 0 else 0,
        )
        if (buy_qty > 0 and buy_price is None) or (sell_qty > 0 and sell_price is None):
            return False

        placed_buy = False
        placed_sell = False
        if buy_qty > 0:
            placed_buy = await self.orders.place_tracked_order(
                symbol=decision.buy_symbol,
                qty=buy_qty,
                side=Side.BUY,
                price=buy_price,
                role="entry",
                reason="rates_entry",
                thesis=decision.direction,
                signal_strength=decision.edge_bp,
                event_id=self.state.last_macro_event_id,
            )
        if sell_qty > 0:
            placed_sell = await self.orders.place_tracked_order(
                symbol=decision.sell_symbol,
                qty=sell_qty,
                side=Side.SELL,
                price=sell_price,
                role="entry",
                reason="rates_entry",
                thesis=decision.direction,
                signal_strength=decision.edge_bp,
                event_id=self.state.last_macro_event_id,
            )

        placed = placed_buy or placed_sell
        if placed:
            self.state.rates_regime_direction = decision.direction
            self.state.rates_buy_symbol = decision.buy_symbol
            self.state.rates_sell_symbol = decision.sell_symbol
            self.state.rates_active_event_id = self.state.last_macro_event_id
            self.state.rates_entry_stage = 1 if not same_regime else self.state.rates_entry_stage + 1
            self.state.rates_last_entry_edge = decision.edge_bp
            self.state.rates_last_add_ts = now
            self.state.rates_pairing_until = now + self.cfg.entry_pair_grace_secs
        return placed


class Coordinator:
    def __init__(
        self,
        client: "NewPM1Client",
        cfg: BotConfig,
        state: MarketState,
        orders: OrderManager,
        risk: RiskManager,
        rates_signals: RatesSignalEngine,
        rates_trading: RatesTradingEngine,
    ):
        self.client = client
        self.cfg = cfg
        self.state = state
        self.orders = orders
        self.risk = risk
        self.rates_signals = rates_signals
        self.rates_trading = rates_trading

    def sync_regimes_to_positions(self) -> None:
        if all(self.client.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
                self.state.clear_rates_regime()

    async def evaluate(self) -> None:
        self.client.refresh_all_books()
        await self.orders.cancel_stale_orders()
        self.sync_regimes_to_positions()

        if not self.state.startup_flatten_complete:
            await self.risk.startup_flatten_step()
            self.log_status("startup_flatten")
            return

        self.risk.arm_session_baseline_if_ready()

        snapshot = self.rates_signals.snapshot()
        if snapshot is None:
            await self.rates_trading.handle_missing_signal_exit()
            self.client._trace("decision", tick=self.client.current_tick, reason="signal_not_ready", snapshot=None, books={s: vars(self.client.top(s)) for s in self.cfg.rate_symbols}, **self.client.positions_payload())
            self.log_status("signal_not_ready")
            return

        if await self.rates_trading.maybe_exit(snapshot):
            self.update_last_signals(snapshot)
            self.client._trace("decision", tick=self.client.current_tick, reason="exit", snapshot=self.client.current_snapshot_payload(snapshot), books={s: vars(self.client.top(s)) for s in self.cfg.rate_symbols}, **self.client.positions_payload())
            return

        if await self.rates_trading.maybe_enter(snapshot):
            self.update_last_signals(snapshot)
            self.client._trace("decision", tick=self.client.current_tick, reason="entry", snapshot=self.client.current_snapshot_payload(snapshot), books={s: vars(self.client.top(s)) for s in self.cfg.rate_symbols}, **self.client.positions_payload())
            return

        self.client._trace("decision", tick=self.client.current_tick, reason="no_trade", snapshot=self.client.current_snapshot_payload(snapshot), books={s: vars(self.client.top(s)) for s in self.cfg.rate_symbols}, **self.client.positions_payload())
        self.log_status("no_trade", snapshot)
        self.update_last_signals(snapshot)

    def update_last_signals(self, snapshot: RateSnapshot) -> None:
        self.state.last_market_expected_rate_bp = snapshot.market_expected_rate_bp
        self.state.last_effective_expected_rate_bp = snapshot.effective_expected_rate_bp

    def log_status(self, reason: str, snapshot: Optional[RateSnapshot] = None) -> None:
        now = time.time()
        if now - self.state.last_status_log_ts < self.cfg.status_log_interval_secs:
            return
        self.state.last_status_log_ts = now
        cash, mtm = self.client.cash_and_total_mtm()
        session_cash, session_mtm = self.client.session_pnl_snapshot(cash, mtm)
        payload = {
            "reason": reason,
            "cash": cash,
            "mtm": mtm,
            "session_cash": session_cash,
            "session_mtm": session_mtm,
            **self.client.positions_payload(),
        }
        if snapshot is not None:
            payload["snapshot"] = self.client.current_snapshot_payload(snapshot)
        self.client._trace("status", tick=self.client.current_tick, **payload)


class NewPM1Client(XChangeClient):
    def __init__(self, host: str, username: str, password: str, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or load_config()
        super().__init__(host, username, password, silent=True, symbols=list(self.cfg.rate_symbols))
        self.state = MarketState()
        self.current_tick: Optional[int] = None
        self._decision_lock = asyncio.Lock()
        self.order_manager = OrderManager(self, self.cfg, self.state)
        self.risk_manager = RiskManager(self, self.cfg, self.state, self.order_manager)
        self.rates_signal_engine = RatesSignalEngine(self, self.cfg, self.state)
        self.rates_trading_engine = RatesTradingEngine(self, self.cfg, self.state, self.order_manager, self.risk_manager)
        self.coordinator = Coordinator(
            self,
            self.cfg,
            self.state,
            self.order_manager,
            self.risk_manager,
            self.rates_signal_engine,
            self.rates_trading_engine,
        )
        self._trace_file = None
        self._trace_path: Optional[Path] = None

    def _trace_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Side):
            return value.name
        if isinstance(value, dict):
            return {str(k): self._trace_jsonable(v) for k, v in value.items()}
        if isinstance(value, set):
            return [self._trace_jsonable(v) for v in sorted(value)]
        if isinstance(value, (list, tuple)):
            return [self._trace_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return self._trace_jsonable(vars(value))
        return repr(value)

    def _trace(self, event_type: str, **kwargs) -> None:
        if not self.cfg.trace_enabled:
            return
        if self._trace_file is None:
            trace_dir = Path(self.cfg.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = trace_dir / f"new_PM1_{int(time.time())}.jsonl"
            self._trace_file = self._trace_path.open("a", encoding="utf-8")
        payload = {"event_type": event_type, "timestamp": time.time(), **kwargs}
        self._trace_file.write(json.dumps(self._trace_jsonable(payload), ensure_ascii=True) + "\n")
        self._trace_file.flush()

    @staticmethod
    def clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def get_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def positions_payload(self) -> dict[str, Any]:
        return {
            "positions": {symbol: self.get_position(symbol) for symbol in self.cfg.rate_symbols},
            "cash": float(self.positions.get("cash", 0)),
        }

    def refresh_book(self, symbol: str) -> TopOfBook:
        book = self.order_books.get(symbol)
        bids = []
        asks = []
        if book is not None:
            bids = [(int(px), int(qty)) for px, qty in book.bids.items() if int(qty) > 0]
            asks = [(int(px), int(qty)) for px, qty in book.asks.items() if int(qty) > 0]
        best_bid = max(bids, key=lambda level: level[0]) if bids else None
        best_ask = min(asks, key=lambda level: level[0]) if asks else None
        snapshot = TopOfBook(
            bid=None if best_bid is None else best_bid[0],
            bid_qty=0 if best_bid is None else best_bid[1],
            ask=None if best_ask is None else best_ask[0],
            ask_qty=0 if best_ask is None else best_ask[1],
            updated_ts=time.time(),
        )
        self.state.books[symbol] = snapshot
        return snapshot

    def refresh_all_books(self) -> None:
        for symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)

    def top(self, symbol: str) -> TopOfBook:
        if symbol not in self.state.books:
            return self.refresh_book(symbol)
        return self.state.books[symbol]

    def mid(self, symbol: str) -> Optional[float]:
        return self.top(symbol).mid

    def cash_and_total_mtm(self) -> tuple[float, float]:
        cash = float(self.positions.get("cash", 0))
        mtm = cash
        for symbol in self.cfg.rate_symbols:
            pos = self.get_position(symbol)
            if pos == 0:
                continue
            mark = self.mid(symbol)
            if mark is not None:
                mtm += pos * mark
        return cash, mtm

    def session_pnl_snapshot(self, cash: float, mtm: float) -> tuple[float, float]:
        if self.state.session_start_cash is None or self.state.session_start_mtm is None:
            return 0.0, 0.0
        return cash - self.state.session_start_cash, mtm - self.state.session_start_mtm

    def current_snapshot_payload(self, snapshot: Optional[RateSnapshot]) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None
        return {
            "q_hike": snapshot.q_hike,
            "q_hold": snapshot.q_hold,
            "q_cut": snapshot.q_cut,
            "market_expected_rate_bp": snapshot.market_expected_rate_bp,
            "effective_expected_rate_bp": snapshot.effective_expected_rate_bp,
            "bias_bp": snapshot.bias_bp,
            "urgent": snapshot.urgent,
            "fresh_macro_event": snapshot.fresh_macro_event,
            "macro_source": snapshot.macro_source,
            "macro_bucket": snapshot.macro_bucket,
            "macro_score": snapshot.macro_score,
            "macro_label": snapshot.macro_label,
        }

    async def evaluate(self) -> None:
        async with self._decision_lock:
            await self.coordinator.evaluate()

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str] = None) -> None:
        tracked = self.order_manager.sync_cancel_response(order_id, success)
        self.state.pending_cancels.discard(str(order_id))
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="cancel_success" if success else "cancel_fail",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            reason=error,
            **self.positions_payload(),
        )

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        tracked = self.order_manager.sync_fill(order_id)
        if all(self.get_position(symbol) == 0 for symbol in self.cfg.rate_symbols):
            if not any(order.symbol in self.cfg.rate_symbols for order in self.state.live_orders.values()):
                if self.state.rates_unwind_active and self.state.rates_active_event_id:
                    self.state.rates_last_closed_event_id = self.state.rates_active_event_id
                self.state.clear_rates_regime()
        cash, mtm = self.cash_and_total_mtm()
        session_cash, session_mtm = self.session_pnl_snapshot(cash, mtm)
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="fill",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            qty=qty,
            price=price,
            mtm=mtm,
            session_cash=session_cash,
            session_mtm=session_mtm,
            **self.positions_payload(),
        )
        await self.order_manager.cancel_counterpart_after_fill(tracked)
        await self.evaluate()

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        tracked = self.order_manager.sync_rejected(order_id)
        self.state.pending_cancels.discard(str(order_id))
        reason_text = (reason or "").lower()
        limit_rejection = "exceeds limits" in reason_text or "limit" in reason_text
        if tracked is not None and tracked.role == "entry":
            self.state.rates_blocked_direction = tracked.thesis
            cooldown = self.cfg.rate_reentry_block_secs * (2.0 if limit_rejection else 1.0)
            self.state.rates_blocked_until = time.time() + cooldown
        self._trace(
            "order_event",
            tick=self.current_tick,
            event_type_detail="reject",
            order_id=str(order_id),
            symbol=None if tracked is None else tracked.symbol,
            side=None if tracked is None else tracked.side.name,
            reason=reason,
            **self.positions_payload(),
        )

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        if symbol in self.cfg.rate_symbols:
            await self.evaluate()

    async def bot_handle_book_update(self, symbol: str) -> None:
        if symbol in self.cfg.rate_symbols:
            self.refresh_book(symbol)
            await self.evaluate()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        kind = news_release.get("kind")
        new_data = news_release.get("new_data", {}) or {}
        self.current_tick = news_release.get("tick")

        event = self.rates_signal_engine.classify_macro_event(kind, new_data)
        applied_bias_bp = 0.0
        parsed_values: dict[str, Any] = {}
        parsed_type = "ignored"

        if event is not None:
            parsed_type = event.parsed_type
            parsed_values = {
                "source": event.source,
                "bias_bp": event.bias_bp,
                "label": event.label,
                "actual": event.actual,
                "forecast": event.forecast,
                "surprise": event.surprise,
                "message_type": event.message_type,
            }
            if event.sentiment is not None:
                parsed_values.update(
                    {
                        "score": event.sentiment.score,
                        "bucket": event.sentiment.bucket,
                        "matched_phrases": event.sentiment.matched_phrases,
                        "unknown_candidate_phrases": event.sentiment.unknown_candidate_phrases,
                    }
                )
            if abs(event.bias_bp) >= 0.25:
                applied_bias_bp = event.bias_bp
                self.rates_signal_engine.apply_temp_rate_bias(
                    event.bias_bp,
                    event.ttl_secs,
                    event.source,
                    bucket=None if event.sentiment is None else event.sentiment.bucket,
                    score=0.0 if event.sentiment is None else event.sentiment.score,
                    label=event.label,
                )

        self._trace(
            "news",
            tick=self.current_tick,
            raw_kind=kind,
            raw_new_data=dict(new_data),
            parsed_type=parsed_type,
            parsed_values=parsed_values,
            applied_bias_bp=applied_bias_bp,
            **self.positions_payload(),
        )
        await self.evaluate()

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        self.current_tick = tick
        self._trace("round_end", tick=tick, event="resolved", winning_symbol=winning_symbol, payout_amount=None, **self.positions_payload())

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        self.current_tick = tick
        self._trace("round_end", tick=tick, event="payout", winning_symbol=None, payout_amount=amount, **self.positions_payload())

    async def trade(self):
        await asyncio.sleep(2.0)
        while True:
            try:
                await self.evaluate()
            except Exception as exc:
                self._trace("loop_error", tick=self.current_tick, error=repr(exc), **self.positions_payload())
            await asyncio.sleep(self.cfg.loop_sleep_secs)

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    client = NewPM1Client(
        env_str("UTC_HOST", "34.197.188.76:3333"),
        env_str("UTC_USERNAME", "uiuc"),
        env_str("UTC_PASSWORD", "mesa-lynx-octopus"),
    )
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
