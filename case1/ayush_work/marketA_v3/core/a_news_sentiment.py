from __future__ import annotations

from dataclasses import dataclass
import re


positive_bigrams: dict[str, float] = {
    "earnings beat": 2.8,
    "beats estimates": 2.8,
    "beats expectations": 2.8,
    "beats forecasts": 2.8,
    "beats forecast": 2.6,
    "exceeds expectations": 2.4,
    "above forecasts": 2.4,
    "subscriber growth": 1.8,
    "growth forecasts": 1.2,
    "record demand": 3.0,
    "major partnership": 3.0,
    "new partnership": 2.2,
    "regulatory approval": 3.0,
    "corporate governance": 2.2,
    "market share": 1.6,
    "surpasses earnings": 3.0,
    "earnings expectations": 1.6,
    "consecutive quarter": 1.2,
    "revenue growth": 1.8,
    "sales growth": 1.8,
    "profit growth": 2.0,
    "innovative product": 1.8,
    "innovative service": 1.2,
    "product line": 1.0,
    "new contract": 2.2,
    "contract award": 2.6,
    "federal contract": 2.8,
    "strong demand": 2.2,
    "higher demand": 2.2,
    "demand rebound": 2.4,
    "faster adoption": 2.2,
    "successful launch": 2.6,
    "new customers": 1.8,
    "customer growth": 1.8,
    "improves retention": 2.2,
    "customer retention": 2.0,
    "expands margins": 2.2,
    "margin expansion": 2.4,
    "expanding margins": 2.4,
    "cost savings": 2.0,
    "cost discipline": 1.8,
    "pricing power": 2.0,
    "share buyback": 2.4,
    "buyback program": 2.2,
    "dividend hike": 2.2,
    "cash generation": 2.0,
    "strong bookings": 2.4,
    "strategic alliance": 2.6,
    "high value": 1.8,
    "industry leaders": 1.8,
    "leading position": 2.0,
    "growing niche": 1.8,
    "niche market": 2.0,
    "promising commercial": 1.6,
    "commercial applications": 1.8,
    "holiday season": 1.2,
    "sales surge": 2.4,
    "strategic supplier": 2.2,
    "international corporation": 1.2,
    "selected for": 0.8,
    "high profile": 1.0,
    "federal technology": 1.8,
    "technology initiative": 1.8,
    "international presence": 1.8,
    "outstanding contributions": 1.8,
    "prestigious award": 2.0,
    "impressive customer": 1.6,
    "rigorous industry": 1.6,
    "industry testing": 1.8,
    "testing success": 1.4,
    "product clears": 1.4,
    "subscriber forecasts": 1.2,
    "federal initiative": 1.2,
    "all time": 1.8,
    "time high": 2.4,
    "all time high": 3.2,
    "satisfaction ratings": 1.8,
    "employee benefits": 1.8,
    "receives recognition": 1.8,
    "payment system": 2.0,
    "system adopted": 2.2,
    "financial institutions": 2.6,
    "leading financial": 1.4,
}

negative_bigrams: dict[str, float] = {
    "earnings miss": -2.8,
    "misses estimates": -2.8,
    "misses forecasts": -2.8,
    "misses forecast": -2.6,
    "cuts guidance": -3.0,
    "guidance cut": -3.0,
    "cuts outlook": -2.8,
    "weak demand": -2.6,
    "demand slowdown": -2.4,
    "subscriber losses": -2.6,
    "customer losses": -2.4,
    "margin pressure": -2.4,
    "pricing pressure": -2.4,
    "profit decline": -2.4,
    "sales decline": -2.2,
    "cash burn": -2.6,
    "default risk": -2.8,
    "execution risk": -2.4,
    "supply issues": -2.4,
    "supply shortage": -2.6,
    "production halt": -3.0,
    "analyst downgrade": -2.8,
    "regulatory scrutiny": -2.6,
    "legal action": -2.6,
    "fraud allegations": -3.0,
    "customer churn": -2.4,
    "contract loss": -2.6,
    "contract slips": -2.6,
    "loses out": -2.4,
    "exclusive partnership": -2.0,
    "slips through": -2.8,
    "order cancellations": -2.6,
    "rival firm": -2.4,
    "supplier disruption": -2.6,
    "delays rollout": -2.6,
    "insider selling": -3.2,
    "investor sentiment": -1.8,
    "into decline": -1.8,
    "stock decline": -2.0,
    "brand image": -1.4,
    "talent acquisition": -1.2,
    "quality control": -2.2,
    "safety concerns": -3.0,
    "data breach": -3.2,
    "regulatory probe": -3.0,
    "antitrust investigation": -3.2,
    "damning investigative": -3.0,
    "investigative report": -2.0,
    "serious allegations": -3.0,
    "insolvency risks": -3.4,
    "insolvency risk": -3.0,
    "express doubts": -3.0,
    "growth strategy": -1.8,
    "falling demand": -2.8,
    "consumer behavior": -1.0,
    "demand pressure": -1.8,
    "supply chain": -1.8,
    "chain disruptions": -2.0,
    "critical overseas": -1.2,
    "overseas markets": -1.2,
    "misleading marketing": -2.6,
    "marketing claims": -1.6,
    "reputational damage": -2.8,
    "operational inefficiencies": -3.0,
    "remain unaddressed": -2.4,
    "regulatory bans": -3.2,
    "flagship product": -1.4,
    "analysts express": -1.2,
    "subscriber losses": -2.6,
    "drop subscriptions": -2.0,
    "profit warning": -3.2,
    "environmental violations": -3.0,
    "potential fines": -2.8,
    "fines looming": -2.6,
    "rising costs": -2.8,
    "stalling progress": -2.8,
    "shrink significantly": -2.6,
    "take hit": -2.0,
    "takes hit": -2.0,
    "termination confirmed": -2.8,
    "termination major": -2.2,
    "major strategic": -0.8,
    "strategic alliance": 2.0,
    "delayed stalling": -2.6,
    "proves unsuccessful": -3.4,
    "diluted shareholder": -2.8,
    "shareholder value": -2.2,
}

positive_unigrams: dict[str, float] = {
    "exceeds": 1.4,
    "beats": 1.6,
    "surpasses": 1.8,
    "strong": 1.0,
    "record": 1.2,
    "major": 0.2,
    "partnership": 1.4,
    "approval": 1.6,
    "growth": 0.4,
    "expectations": 0.8,
    "forecasts": 0.4,
    "innovative": 1.0,
    "improves": 1.0,
    "retention": 1.0,
    "faster": 0.8,
    "adoption": 1.0,
    "successful": 0.8,
    "customer": 0.2,
    "subscriber": 0.4,
    "expansion": 0.2,
    "expands": 0.8,
    "expanding": 0.8,
    "margins": 0.1,
    "savings": 1.2,
    "buyback": 1.4,
    "dividend": 1.0,
    "hike": 1.0,
    "bookings": 1.2,
    "bolsters": 1.4,
    "prestigious": 1.4,
    "award": 1.4,
    "sustainability": 1.2,
    "promising": 1.2,
    "commercial": 0.8,
    "applications": 0.8,
    "holiday": 0.6,
    "surge": 1.6,
    "strategic": 0.2,
    "supplier": 0.8,
    "selected": 0.8,
    "federal": 0.6,
    "technology": 0.1,
    "initiative": 0.2,
    "impressive": 1.2,
    "clears": 1.2,
    "rigorous": 0.8,
    "success": 0.8,
    "outstanding": 1.2,
    "international": 0.8,
    "presence": 0.8,
    "earns": 0.8,
    "recognition": 1.2,
    "satisfaction": 1.0,
    "benefits": 1.0,
    "adopted": 1.8,
    "financial": 1.0,
    "institutions": 1.0,
    "leading": 1.0,
    "payment": 0.8,
}

negative_unigrams: dict[str, float] = {
    "insider": -1.8,
    "selling": -1.2,
    "sours": -1.6,
    "decline": -1.6,
    "declining": -1.6,
    "hinders": -1.8,
    "disruption": -1.8,
    "disruptions": -2.0,
    "recall": -2.0,
    "recalls": -2.0,
    "probe": -1.8,
    "breach": -2.0,
    "delay": -1.6,
    "delays": -1.6,
    "downgrade": -2.0,
    "lawsuit": -2.0,
    "scrutiny": -1.8,
    "fraud": -2.2,
    "burn": -1.8,
    "default": -2.2,
    "losses": -1.8,
    "churn": -1.8,
    "cancellations": -1.8,
    "pressure": -1.4,
    "misses": -1.8,
    "cuts": -1.8,
    "warning": -1.6,
    "grim": -2.0,
    "headwinds": -1.6,
    "problems": -1.8,
    "concerns": -1.2,
    "banned": -2.2,
    "allegations": -1.8,
    "damning": -2.4,
    "investigative": -1.2,
    "safety": -1.2,
    "insolvency": -2.2,
    "warned": -1.0,
    "warns": -1.4,
    "risk": -1.0,
    "risks": -1.0,
    "loses": -1.6,
    "doubt": -1.8,
    "doubts": -1.8,
    "falling": -1.6,
    "demand": -1.0,
    "strategy": -0.8,
    "termination": -3.2,
    "terminated": -3.0,
    "delayed": -2.2,
    "stalling": -2.2,
    "shrink": -2.4,
    "shrinks": -2.4,
    "rising": -1.8,
    "costs": -1.8,
    "violations": -2.4,
    "violation": -2.2,
    "accused": -2.0,
    "unsuccessful": -3.2,
    "diluted": -2.4,
    "tumbles": -2.4,
    "fines": -2.0,
    "fine": -1.8,
    "looming": -1.6,
    "confirmed": -0.6,
    "hit": -0.2,
    "drop": -1.6,
    "subscriptions": -1.6,
    "misleading": -1.8,
    "damage": -1.6,
    "reputational": -1.6,
    "suffers": -1.6,
    "setbacks": -1.8,
    "struggles": -1.6,
    "inefficiencies": -2.0,
    "unaddressed": -1.8,
    "bans": -2.2,
    "threaten": -2.0,
    "flagship": -1.0,
    "weak": -1.2,
    "consumer": 0.0,
    "behavior": -0.4,
    "slowdown": -1.4,
}

amplifiers: dict[str, float] = {
    "significant": 1.15,
    "major": 1.15,
    "record": 1.20,
    "exceptional": 1.20,
    "grim": 1.15,
    "massive": 1.20,
    "prestigious": 1.10,
    "exclusive": 1.10,
    "fifth consecutive": 1.15,
}

dampeners: dict[str, float] = {
    "rumor": 0.70,
    "modest": 0.80,
    "limited": 0.85,
    "slight": 0.85,
    "slightly": 0.85,
    "possible": 0.90,
}

_STOPWORDS = {
    "a",
    "about",
    "after",
    "against",
    "all",
    "amid",
    "among",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "company",
    "for",
    "from",
    "following",
    "global",
    "has",
    "have",
    "in",
    "into",
    "investors",
    "its",
    "itself",
    "keeps",
    "more",
    "news",
    "not",
    "of",
    "on",
    "or",
    "quarter",
    "quarters",
    "shares",
    "stock",
    "than",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "under",
    "with",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("'s", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_term(term: str) -> str:
    return normalize_text(term)

_POSITIVE_BIGRAMS = {normalize_term(term): weight for term, weight in positive_bigrams.items()}
_NEGATIVE_BIGRAMS = {normalize_term(term): weight for term, weight in negative_bigrams.items()}
_POSITIVE_UNIGRAMS = {normalize_term(term): weight for term, weight in positive_unigrams.items()}
_NEGATIVE_UNIGRAMS = {normalize_term(term): weight for term, weight in negative_unigrams.items()}
_AMPLIFIERS = {normalize_term(term): weight for term, weight in amplifiers.items()}
_DAMPENERS = {normalize_term(term): weight for term, weight in dampeners.items()}


def _merge_weighted(*groups: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for group in groups:
        merged.update(group)
    return merged


_BIGRAM_WEIGHTS = _merge_weighted(_POSITIVE_BIGRAMS, _NEGATIVE_BIGRAMS)
_UNIGRAM_WEIGHTS = _merge_weighted(_POSITIVE_UNIGRAMS, _NEGATIVE_UNIGRAMS)

_NEGATIVE_OUTCOME_BIGRAMS = {
    "environmental violations",
    "potential fines",
    "fines looming",
    "rising costs",
    "stalling progress",
    "shrink significantly",
    "termination confirmed",
    "delayed stalling",
    "proves unsuccessful",
    "diluted shareholder",
    "shareholder value",
}
_NEGATIVE_OUTCOME_UNIGRAMS = {
    "termination",
    "terminated",
    "delayed",
    "stalling",
    "shrink",
    "shrinks",
    "rising",
    "costs",
    "violations",
    "violation",
    "accused",
    "fines",
    "fine",
    "looming",
    "abandoned",
    "doom",
    "doomed",
    "frustration",
    "frustrated",
    "damage",
    "loss",
    "losses",
    "cut",
    "cuts",
    "downturn",
}
_POSITIVE_OUTCOME_BIGRAMS = {
    "all time",
    "time high",
    "all time high",
    "sales surge",
    "record demand",
    "new partnership",
    "major partnership",
    "receives recognition",
    "payment system",
    "system adopted",
    "financial institutions",
}
_POSITIVE_OUTCOME_UNIGRAMS = {
    "high",
    "surge",
    "approval",
    "recognition",
    "award",
    "successful",
    "success",
    "improves",
    "bolsters",
    "earns",
    "exceeds",
    "beats",
    "surpasses",
    "selected",
}
_CONTEXTUAL_POSITIVE_BIGRAMS = {
    "strategic alliance",
    "high value",
    "industry leaders",
    "international presence",
    "technology initiative",
    "federal technology",
    "leading position",
    "niche market",
}
_CONTEXTUAL_POSITIVE_UNIGRAMS = {
    "major",
    "strategic",
    "technology",
    "initiative",
    "expansion",
    "growth",
    "margins",
    "customer",
    "subscriber",
    "international",
    "presence",
    "partnership",
    "leading",
    "financial",
    "institutions",
    "payment",
}
_NEGATIVE_CONTEXT_RETAIN_FRACTION = 0.15


@dataclass(frozen=True)
class SentimentResult:
    score: float
    bucket: str
    direction: int
    matched_phrases: tuple[str, ...]
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]
    unknown_candidate_unigrams: tuple[str, ...]
    unknown_candidate_bigrams: tuple[str, ...]


def _bucket_for_score(score: float) -> str:
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


def score_a_unstructured_headline(text: str | None) -> SentimentResult:
    normalized = normalize_text(text)
    if not normalized:
        return SentimentResult(
            score=0.0,
            bucket="none",
            direction=0,
            matched_phrases=(),
            matched_unigrams=(),
            matched_bigrams=(),
            unknown_candidate_phrases=(),
            unknown_candidate_unigrams=(),
            unknown_candidate_bigrams=(),
        )

    tokens = normalized.split()
    used = [False] * len(tokens)
    matched_bigrams: list[str] = []
    matched_unigrams: list[str] = []
    matched_bigram_weights: list[tuple[str, float]] = []
    matched_unigram_weights: list[tuple[str, float]] = []
    score = 0.0

    index = 0
    while index < len(tokens) - 1:
        if used[index] or used[index + 1]:
            index += 1
            continue
        bigram = f"{tokens[index]} {tokens[index + 1]}"
        weight = _BIGRAM_WEIGHTS.get(bigram)
        if weight is None:
            index += 1
            continue
        matched_bigrams.append(bigram)
        matched_bigram_weights.append((bigram, weight))
        score += weight
        used[index] = True
        used[index + 1] = True
        index += 2

    for index, token in enumerate(tokens):
        if used[index]:
            continue
        weight = _UNIGRAM_WEIGHTS.get(token)
        if weight is None:
            continue
        matched_unigrams.append(token)
        matched_unigram_weights.append((token, weight))
        score += weight
        used[index] = True

    score = _apply_outcome_context_overrides(
        tokens=tokens,
        score=score,
        matched_bigram_weights=matched_bigram_weights,
        matched_unigram_weights=matched_unigram_weights,
    )

    amplifier = max((weight for phrase, weight in _AMPLIFIERS.items() if _contains_phrase(tokens, phrase)), default=1.0)
    dampener = min((weight for phrase, weight in _DAMPENERS.items() if _contains_phrase(tokens, phrase)), default=1.0)
    score *= amplifier
    score *= dampener
    score = max(-6.0, min(6.0, score))
    bucket = _bucket_for_score(score)
    direction = 1 if score > 0 else -1 if score < 0 else 0

    unknown_unigrams, unknown_bigrams = _extract_unknown_candidates(tokens, used)
    matched_phrases = list(dict.fromkeys([*matched_bigrams, *matched_unigrams]))
    unknown_phrases = list(dict.fromkeys([*unknown_bigrams, *unknown_unigrams]))

    return SentimentResult(
        score=score,
        bucket=bucket,
        direction=direction,
        matched_phrases=tuple(matched_phrases),
        matched_unigrams=tuple(dict.fromkeys(matched_unigrams)),
        matched_bigrams=tuple(dict.fromkeys(matched_bigrams)),
        unknown_candidate_phrases=tuple(unknown_phrases),
        unknown_candidate_unigrams=tuple(unknown_unigrams),
        unknown_candidate_bigrams=tuple(unknown_bigrams),
    )


def get_a_news_term_weight(term: str) -> float | None:
    normalized = normalize_term(term)
    if " " in normalized:
        return _BIGRAM_WEIGHTS.get(normalized)
    return _UNIGRAM_WEIGHTS.get(normalized)


def get_a_news_term_polarity(term: str) -> str | None:
    weight = get_a_news_term_weight(term)
    if weight is None or weight == 0:
        return None
    return "positive" if weight > 0 else "negative"


def _contains_phrase(tokens: list[str], phrase: str) -> bool:
    parts = phrase.split()
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in tokens
    for start in range(0, len(tokens) - len(parts) + 1):
        if tokens[start : start + len(parts)] == parts:
            return True
    return False


def _apply_outcome_context_overrides(
    *,
    tokens: list[str],
    score: float,
    matched_bigram_weights: list[tuple[str, float]],
    matched_unigram_weights: list[tuple[str, float]],
) -> float:
    has_negative_outcome = _contains_any_phrase(tokens, _NEGATIVE_OUTCOME_BIGRAMS) or any(
        token in _NEGATIVE_OUTCOME_UNIGRAMS for token in tokens
    )
    has_positive_outcome = _contains_any_phrase(tokens, _POSITIVE_OUTCOME_BIGRAMS) or any(
        token in _POSITIVE_OUTCOME_UNIGRAMS for token in tokens
    )

    adjusted = score
    if has_negative_outcome:
        for term, weight in matched_bigram_weights:
            if weight > 0 and term in _CONTEXTUAL_POSITIVE_BIGRAMS:
                adjusted -= weight * (1.0 - _NEGATIVE_CONTEXT_RETAIN_FRACTION)
        for term, weight in matched_unigram_weights:
            if weight > 0 and term in _CONTEXTUAL_POSITIVE_UNIGRAMS:
                adjusted -= weight * (1.0 - _NEGATIVE_CONTEXT_RETAIN_FRACTION)

    if has_positive_outcome:
        for term, weight in matched_unigram_weights:
            if weight < 0 and term in {"consumer", "hit"}:
                adjusted -= weight

    return adjusted


def _contains_any_phrase(tokens: list[str], phrases: set[str]) -> bool:
    return any(_contains_phrase(tokens, phrase) for phrase in phrases)


def _extract_unknown_candidates(tokens: list[str], used: list[bool]) -> tuple[list[str], list[str]]:
    unknown_unigrams: list[str] = []
    unknown_bigrams: list[str] = []
    seen_unigrams: set[str] = set()
    seen_bigrams: set[str] = set()

    for index, token in enumerate(tokens):
        if used[index] or not _is_candidate_token(token):
            continue
        if token not in seen_unigrams:
            seen_unigrams.add(token)
            unknown_unigrams.append(token)

    for start in range(0, len(tokens) - 1):
        if used[start] or used[start + 1]:
            continue
        first = tokens[start]
        second = tokens[start + 1]
        if not _is_candidate_token(first) or not _is_candidate_token(second):
            continue
        phrase = f"{first} {second}"
        if phrase in _BIGRAM_WEIGHTS or phrase in _AMPLIFIERS or phrase in _DAMPENERS:
            continue
        if phrase in seen_bigrams:
            continue
        seen_bigrams.add(phrase)
        unknown_bigrams.append(phrase)

    return unknown_unigrams, unknown_bigrams


def _is_candidate_token(token: str) -> bool:
    return len(token) >= 4 and token not in _STOPWORDS
