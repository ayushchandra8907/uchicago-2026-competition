from __future__ import annotations

from dataclasses import dataclass
import math
import re


hike_bigrams: dict[str, float] = {
    "inflation elevated": 1.8,
    "upside inflation": 1.8,
    "inflation sticky": 2.0,
    "sticky inflation": 2.0,
    "persistent inflation": 2.0,
    "labor market": 0.5,
    "market tight": 1.8,
    "labor strong": 1.6,
    "higher longer": 2.2,
    "strong demand": 1.8,
    "restrictive policy": 1.8,
    "price pressures": 1.8,
    "resilient demand": 1.7,
    "upside risks": 1.4,
    "inflation risks": 1.5,
    "overheating economy": 2.0,
    "sticky prices": 2.0,
    "wage growth": 1.4,
    "reaccelerating inflation": 2.2,
    "stay restrictive": 1.7,
    "keep pressure": 1.4,
    "reassess path": 1.6,
    "path cuts": 1.8,
    "signal policy": 1.0,
    "looks concerned": 1.0,
    "concerned about": 0.8,
    "emphasizes inflation": 1.1,
}

hold_bigrams: dict[str, float] = {
    "balanced risks": 2.2,
    "mixed indicators": 1.8,
    "data dependent": 2.0,
    "data dependence": 2.2,
    "more evidence": 2.0,
    "uncertain outlook": 1.6,
    "incoming data": 1.2,
    "upcoming data": 1.5,
    "not in a hurry": 2.0,
    "remain patient": 2.0,
    "no urgency": 2.0,
    "wait for": 1.5,
    "more evidence": 2.0,
    "no rush": 1.8,
    "no clear": 1.4,
    "clear signal": 1.7,
    "next move": 1.5,
    "hold steady": 2.2,
    "on hold": 2.0,
    "stay hold": 1.8,
    "policy positioned": 1.6,
    "well positioned": 1.8,
    "options open": 2.0,
    "signals conflict": 2.0,
    "remains cautious": 2.0,
    "wait evidence": 1.8,
    "monitor data": 1.2,
    "policy pause": 1.8,
    "steady policy": 1.6,
    "await upcoming": 1.6,
    "communication remains": 1.6,
    "keeps options": 1.7,
    "mixed economic": 1.4,
    "economic indicators": 1.5,
    "wait for more evidence": 2.4,
    "growth signals": 1.4,
    "markets await": 1.1,
    "chair reiterates": 0.9,
    "reiterates data": 1.0,
    "raises expectations": 1.0,
    "policy easing": 0.9,
    "softening data": 0.8,
}

cut_bigrams: dict[str, float] = {
    "disinflation progress": 2.2,
    "inflation cooling": 1.8,
    "growth risks": 1.5,
    "downside growth": 1.8,
    "growth slowing": 1.8,
    "downside risks": 1.6,
    "labor softening": 1.8,
    "cooling labor": 1.6,
    "easing inflation": 1.9,
    "inflation pressures": 0.6,
    "weaker demand": 1.8,
    "restrictive enough": 1.8,
    "room ease": 1.8,
    "cuts appropriate": 2.2,
    "slowdown risks": 1.6,
    "recession risks": 2.0,
    "lower inflation": 1.6,
    "room cut": 1.8,
    "growth weakness": 1.6,
    "toward cuts": 2.0,
    "policy easing": 1.5,
    "moving back": 1.6,
    "back target": 1.8,
    "confidence inflation": 1.4,
    "softening data": 2.0,
    "raises expectations": 0.7,
    "increasing confidence": 1.4,
    "signals increasing": 0.9,
    "note downside": 1.0,
    "markets lean": 1.1,
}

hike_unigrams: dict[str, float] = {
    "inflation": 0.7,
    "sticky": 1.2,
    "tight": 1.0,
    "restrictive": 0.9,
    "elevated": 1.0,
    "hawkish": 1.2,
    "overheating": 1.4,
    "resilient": 0.8,
    "premature": 0.8,
    "pressures": 0.8,
    "pressure": 0.9,
    "risks": 0.5,
    "concerned": 0.8,
    "wage": 0.5,
}

hold_unigrams: dict[str, float] = {
    "balanced": 1.0,
    "mixed": 1.0,
    "dependent": 0.9,
    "dependence": 1.1,
    "patience": 1.4,
    "pause": 1.5,
    "steady": 1.2,
    "wait": 1.0,
    "monitor": 0.8,
    "positioned": 0.8,
    "uncertain": 0.9,
    "cautious": 1.2,
    "conflict": 1.2,
    "await": 1.0,
    "upcoming": 0.8,
    "reiterates": 0.8,
    "signal": 0.4,
    "indicators": 0.7,
    "evidence": 0.8,
}

cut_unigrams: dict[str, float] = {
    "disinflation": 1.4,
    "cooling": 1.0,
    "slowing": 1.0,
    "softening": 1.2,
    "downside": 1.1,
    "weaker": 1.2,
    "easing": 1.2,
    "slowdown": 1.4,
    "recession": 1.5,
    "dovish": 1.2,
    "ease": 0.9,
    "cuts": 1.0,
    "confidence": 0.7,
    "lean": 0.8,
}

relevance_bigrams: dict[str, float] = {
    "fed officials": 1.8,
    "central bank": 1.6,
    "policy rate": 1.8,
    "interest rates": 1.8,
    "labor market": 1.2,
    "price pressures": 1.2,
    "inflation outlook": 1.4,
    "economic indicators": 1.0,
    "wage growth": 1.0,
    "upcoming data": 1.0,
}

relevance_unigrams: dict[str, float] = {
    "fed": 1.5,
    "policy": 1.0,
    "rates": 1.2,
    "inflation": 1.0,
    "labor": 0.8,
    "growth": 0.6,
    "officials": 0.8,
    "chair": 0.5,
    "communication": 0.6,
}

amplifiers: dict[str, float] = {
    "clearly": 1.15,
    "materially": 1.15,
    "strongly": 1.20,
    "significantly": 1.20,
    "sharply": 1.20,
    "decisively": 1.20,
}

dampeners: dict[str, float] = {
    "somewhat": 0.85,
    "modestly": 0.85,
    "slightly": 0.85,
    "gradually": 0.90,
}

context_overrides: dict[str, tuple[float, float, float]] = {
    "higher for longer": (1.4, 0.6, -1.2),
    "policy may stay restrictive": (1.2, 0.5, -1.0),
    "premature to cut": (0.6, 0.9, -1.6),
    "balanced risks": (-0.4, 1.8, -0.4),
    "mixed economic indicators": (-0.4, 2.2, -0.4),
    "await upcoming data": (-0.2, 2.0, -0.2),
    "markets await upcoming data": (-0.2, 2.1, -0.2),
    "data dependence": (-0.4, 2.3, -0.4),
    "reiterates data dependence": (-0.3, 1.7, -0.3),
    "not in a hurry": (-0.4, 2.2, -0.4),
    "remain patient": (-0.4, 2.1, -0.4),
    "no urgency": (-0.4, 2.1, -0.4),
    "no clear signal": (-0.5, 2.3, -0.5),
    "hold steady": (-0.4, 2.0, -0.4),
    "on hold": (-0.3, 1.8, -0.3),
    "cuts may be appropriate": (-1.3, -0.2, 1.8),
    "lean toward cuts": (-1.2, -0.2, 1.9),
    "cooling labor market": (-1.2, 0.1, 1.8),
    "easing inflation pressures": (-1.4, 0.1, 2.0),
    "moving back to target": (-1.2, 0.1, 1.9),
    "options open": (-0.3, 1.8, -0.3),
    "growth signals conflict": (-0.3, 1.8, -0.3),
    "signals conflict": (-0.3, 1.8, -0.3),
    "communication remains cautious": (-0.3, 1.9, -0.3),
    "increasing confidence inflation is moving back to target": (-1.6, 0.0, 2.2),
    "inflation is moving back to target": (-1.3, 0.1, 1.9),
    "increasing confidence": (-0.8, 0.0, 1.2),
    "softening data": (-1.0, 1.0, 1.1),
    "raises expectations": (-0.4, 0.8, 0.5),
    "policy easing": (-0.8, 1.0, 1.0),
    "softening data raises expectations of policy easing": (-1.7, 2.4, 0.9),
    "path of cuts": (1.4, 0.3, -1.5),
    "reassess path of cuts": (1.8, 0.3, -1.9),
    "stay restrictive for longer": (1.5, 0.6, -1.2),
    "keep pressure on the fed": (1.4, 0.2, -1.0),
    "concerned about wage growth": (1.2, 0.1, -0.4),
    "emphasizes inflation risks": (1.3, 0.1, -0.6),
    "room to ease": (-1.0, 0.2, 1.4),
    "restrictive enough": (-0.8, 0.4, 1.2),
    "no rush to cut": (0.4, 1.0, -1.4),
    "wait for more evidence": (-0.4, 1.8, -0.4),
}

_STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "into",
    "onto",
    "from",
    "amid",
    "this",
    "will",
    "they",
    "their",
    "while",
    "have",
    "has",
    "been",
    "over",
    "more",
    "than",
    "would",
    "could",
    "should",
    "officials",
    "federal",
    "reserve",
}


@dataclass(frozen=True)
class FedSentimentResult:
    normalized_text: str
    relevance_score: float
    bucket: str
    delta_hike: float
    delta_hold: float
    delta_cut: float
    matched_phrases: tuple[str, ...]
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    matched_hike_terms: tuple[str, ...]
    matched_hold_terms: tuple[str, ...]
    matched_cut_terms: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]
    unknown_candidate_unigrams: tuple[str, ...]
    unknown_candidate_bigrams: tuple[str, ...]
    amplifier_terms: tuple[str, ...]
    dampener_terms: tuple[str, ...]

    @property
    def max_abs_score(self) -> float:
        return max(abs(self.delta_hike), abs(self.delta_hold), abs(self.delta_cut))

    @property
    def hike_score(self) -> float:
        return self.delta_hike

    @property
    def hold_score(self) -> float:
        return self.delta_hold

    @property
    def cut_score(self) -> float:
        return self.delta_cut


def normalize_text(text: str) -> str:
    cleaned = text.lower().replace("’", "'").replace("`", "'")
    cleaned = re.sub(r"[^a-z0-9']+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def score_fed_speak_headline(text: str) -> FedSentimentResult:
    normalized = normalize_text(text)
    if not normalized:
        return FedSentimentResult(
            normalized_text="",
            relevance_score=0.0,
            bucket="none",
            delta_hike=0.0,
            delta_hold=0.0,
            delta_cut=0.0,
            matched_phrases=(),
            matched_unigrams=(),
            matched_bigrams=(),
            matched_hike_terms=(),
            matched_hold_terms=(),
            matched_cut_terms=(),
            unknown_candidate_phrases=(),
            unknown_candidate_unigrams=(),
            unknown_candidate_bigrams=(),
            amplifier_terms=(),
            dampener_terms=(),
        )

    tokens = normalized.split()
    matched_bigrams: list[str] = []
    matched_unigrams: list[str] = []
    matched_hike_terms: list[str] = []
    matched_hold_terms: list[str] = []
    matched_cut_terms: list[str] = []
    used_positions: set[int] = set()

    hike_score = 0.0
    hold_score = 0.0
    cut_score = 0.0
    relevance = 0.0

    for index in range(len(tokens) - 1):
        phrase = f"{tokens[index]} {tokens[index + 1]}"
        phrase_matched = False
        if phrase in hike_bigrams:
            hike_score += hike_bigrams[phrase]
            matched_hike_terms.append(phrase)
            phrase_matched = True
        if phrase in hold_bigrams:
            hold_score += hold_bigrams[phrase]
            matched_hold_terms.append(phrase)
            phrase_matched = True
        if phrase in cut_bigrams:
            cut_score += cut_bigrams[phrase]
            matched_cut_terms.append(phrase)
            phrase_matched = True
        if phrase in relevance_bigrams:
            relevance += relevance_bigrams[phrase]
            phrase_matched = True
        if phrase_matched:
            matched_bigrams.append(phrase)
            used_positions.update({index, index + 1})

    for index, token in enumerate(tokens):
        if token in relevance_unigrams:
            relevance += relevance_unigrams[token]
        if index in used_positions:
            continue
        token_matched = False
        if token in hike_unigrams:
            hike_score += hike_unigrams[token]
            matched_hike_terms.append(token)
            token_matched = True
        if token in hold_unigrams:
            hold_score += hold_unigrams[token]
            matched_hold_terms.append(token)
            token_matched = True
        if token in cut_unigrams:
            cut_score += cut_unigrams[token]
            matched_cut_terms.append(token)
            token_matched = True
        if token_matched:
            matched_unigrams.append(token)

    for phrase, deltas in context_overrides.items():
        if phrase in normalized:
            hike_delta, hold_delta, cut_delta = deltas
            hike_score += hike_delta
            hold_score += hold_delta
            cut_score += cut_delta
            matched_bigrams.append(phrase)
            if hike_delta != 0:
                matched_hike_terms.append(phrase)
            if hold_delta != 0:
                matched_hold_terms.append(phrase)
            if cut_delta != 0:
                matched_cut_terms.append(phrase)

    amplifier_terms = tuple(term for term in amplifiers if term in tokens)
    dampener_terms = tuple(term for term in dampeners if term in tokens)
    score_multiplier = 1.0
    if amplifier_terms:
        score_multiplier *= max(amplifiers[term] for term in amplifier_terms)
    if dampener_terms:
        score_multiplier *= min(dampeners[term] for term in dampener_terms)

    hike_score = max(-6.0, min(6.0, hike_score * score_multiplier))
    hold_score = max(-6.0, min(6.0, hold_score * score_multiplier))
    cut_score = max(-6.0, min(6.0, cut_score * score_multiplier))
    relevance_score = max(relevance, max(abs(hike_score), abs(hold_score), abs(cut_score)))
    bucket = _bucket_for_scores(hike_score, hold_score, cut_score)

    unknown_unigrams, unknown_bigrams = _extract_unknown_candidates(tokens, matched_unigrams, matched_bigrams)
    unknown_phrases = tuple(dict.fromkeys([*unknown_bigrams, *unknown_unigrams]))

    matched_phrases = tuple(dict.fromkeys([*matched_bigrams, *matched_unigrams]))
    return FedSentimentResult(
        normalized_text=normalized,
        relevance_score=round(relevance_score, 3),
        bucket=bucket,
        delta_hike=round(hike_score, 3),
        delta_hold=round(hold_score, 3),
        delta_cut=round(cut_score, 3),
        matched_phrases=matched_phrases,
        matched_unigrams=tuple(dict.fromkeys(matched_unigrams)),
        matched_bigrams=tuple(dict.fromkeys(matched_bigrams)),
        matched_hike_terms=tuple(dict.fromkeys(matched_hike_terms)),
        matched_hold_terms=tuple(dict.fromkeys(matched_hold_terms)),
        matched_cut_terms=tuple(dict.fromkeys(matched_cut_terms)),
        unknown_candidate_phrases=unknown_phrases,
        unknown_candidate_unigrams=unknown_unigrams,
        unknown_candidate_bigrams=unknown_bigrams,
        amplifier_terms=tuple(dict.fromkeys(amplifier_terms)),
        dampener_terms=tuple(dict.fromkeys(dampener_terms)),
    )


def get_c_term_polarity(term: str) -> str:
    term = normalize_text(term)
    weights = {
        "hike": _lookup_term_weight(term, hike_unigrams, hike_bigrams),
        "hold": _lookup_term_weight(term, hold_unigrams, hold_bigrams),
        "cut": _lookup_term_weight(term, cut_unigrams, cut_bigrams),
    }
    polarity, weight = max(weights.items(), key=lambda item: abs(item[1]))
    return "unknown" if weight == 0 else polarity


def get_c_term_weight(term: str) -> float | None:
    term = normalize_text(term)
    weights = [
        _lookup_term_weight(term, hike_unigrams, hike_bigrams),
        _lookup_term_weight(term, hold_unigrams, hold_bigrams),
        _lookup_term_weight(term, cut_unigrams, cut_bigrams),
    ]
    best = max(weights, key=lambda value: abs(value))
    return None if best == 0 else best


def _lookup_term_weight(term: str, unigrams: dict[str, float], bigrams: dict[str, float]) -> float:
    if term in bigrams:
        return float(bigrams[term])
    if term in unigrams:
        return float(unigrams[term])
    return 0.0


def _bucket_for_scores(hike_score: float, hold_score: float, cut_score: float) -> str:
    magnitude = max(abs(hike_score), abs(hold_score), abs(cut_score))
    if magnitude <= 0:
        return "none"
    if magnitude < 1.0:
        return "light"
    if magnitude < 2.25:
        return "medium"
    if magnitude < 3.75:
        return "strong"
    return "extreme"


def _extract_unknown_candidates(tokens: list[str], matched_unigrams: list[str], matched_bigrams: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    matched_unigram_set = set(matched_unigrams)
    matched_bigram_set = set(matched_bigrams)
    matched_bigram_tokens = {
        token
        for phrase in matched_bigram_set
        for token in phrase.split()
    }
    unknown_unigrams: list[str] = []
    for token in tokens:
        if (
            token in matched_unigram_set
            or token in matched_bigram_tokens
            or token in _STOPWORDS
            or len(token) < 4
        ):
            continue
        unknown_unigrams.append(token)

    unknown_bigrams: list[str] = []
    for index in range(len(tokens) - 1):
        first = tokens[index]
        second = tokens[index + 1]
        phrase = f"{first} {second}"
        if phrase in matched_bigram_set:
            continue
        if first in _STOPWORDS or second in _STOPWORDS:
            continue
        if len(first) < 4 or len(second) < 4:
            continue
        unknown_bigrams.append(phrase)

    return tuple(dict.fromkeys(unknown_unigrams)), tuple(dict.fromkeys(unknown_bigrams))


def _softmax(scores: tuple[float, float, float]) -> tuple[float, float, float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(value / total for value in exps)  # type: ignore[return-value]


__all__ = [
    "FedSentimentResult",
    "get_c_term_polarity",
    "get_c_term_weight",
    "normalize_text",
    "score_fed_speak_headline",
]
