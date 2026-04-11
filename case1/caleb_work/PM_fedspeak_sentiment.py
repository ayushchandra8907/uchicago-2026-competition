from __future__ import annotations

from dataclasses import dataclass
import re


# FedSpeak lexicon tuned for UTC PM language.
# Positive score => hawkish (R_HIKE up), negative => dovish (R_HOLD/R_CUT up).
HAWKISH_BIGRAMS: dict[str, float] = {
    "additional tightening": 3.2,
    "open to additional": 2.2,
    "persistent inflation": 2.8,
    "sticky inflation": 2.8,
    "price pressures": 2.4,
    "reignite price": 3.0,
    "premature easing": 2.8,
    "stay firm": 2.2,
    "stay restrictive": 2.8,
    "higher for longer": 3.2,
    "not in a hurry": 2.0,
    "tight labor": 2.2,
    "rising wages": 2.2,
    "strong payrolls": 2.2,
    "hawkish stance": 2.8,
    "inflation risks": 2.6,
    "concerned about": 1.6,
    "flags lingering": 1.4,
    "acknowledges progress": 0.6,
}

DOVISH_BIGRAMS: dict[str, float] = {
    "cooling inflation": -2.6,
    "easing inflation": -2.2,
    "easing inflation pressures": -3.0,
    "cooling labor market": -2.8,
    "softening data": -2.6,
    "downside growth risks": -2.8,
    "rate relief": -2.6,
    "policy easing": -2.4,
    "preemptive cut": -3.2,
    "near term cuts": -2.8,
    "lean toward cuts": -2.8,
    "moving back to target": -2.6,
    "no longer be needed": -2.4,
    "overtightening": -2.4,
    "recession": -2.0,
    "weaker retail": -2.2,
    "declining job openings": -2.2,
}

HAWKISH_UNIGRAMS: dict[str, float] = {
    "inflation": 0.5,
    "sticky": 1.2,
    "persistent": 1.0,
    "tightening": 1.2,
    "tight": 0.8,
    "restrictive": 1.0,
    "strong": 0.7,
    "wages": 0.7,
    "hawkish": 1.6,
    "firm": 0.7,
    "higher": 0.5,
}

DOVISH_UNIGRAMS: dict[str, float] = {
    "cooling": -1.0,
    "easing": -0.9,
    "softening": -1.1,
    "downside": -0.8,
    "cuts": -0.8,
    "cut": -0.5,
    "recession": -1.0,
    "weaker": -0.8,
    "declining": -0.8,
    "relief": -0.9,
    "dovish": -1.6,
}

AMPLIFIERS: dict[str, float] = {
    "significantly": 1.25,
    "materially": 1.25,
    "sharply": 1.30,
    "strongly": 1.20,
}

NEUTRALIZERS: dict[str, float] = {
    "lingering uncertainties": 0.85,
    "uncertain outlook": 0.72,
    "mixed signals": 0.70,
    "wide range of views": 0.62,
    "divided": 0.74,
    "both hawks and doves": 0.60,
    "balanced risks": 0.60,
}


STOPWORDS = {
    "about",
    "amid",
    "also",
    "after",
    "ahead",
    "again",
    "against",
    "committee",
    "federal",
    "fed",
    "fomc",
    "market",
    "markets",
    "officials",
    "policy",
    "rate",
    "rates",
    "statement",
    "their",
    "they",
    "this",
    "with",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower().replace("'s", "")
    lowered = re.sub(r"[^a-z0-9.+\-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_term(term: str) -> str:
    return normalize_text(term)


_HAWK_BIGRAMS = {normalize_term(k): v for k, v in HAWKISH_BIGRAMS.items()}
_DOVE_BIGRAMS = {normalize_term(k): v for k, v in DOVISH_BIGRAMS.items()}
_HAWK_UNIGRAMS = {normalize_term(k): v for k, v in HAWKISH_UNIGRAMS.items()}
_DOVE_UNIGRAMS = {normalize_term(k): v for k, v in DOVISH_UNIGRAMS.items()}
_AMPLIFIERS = {normalize_term(k): v for k, v in AMPLIFIERS.items()}
_NEUTRALIZERS = {normalize_term(k): v for k, v in NEUTRALIZERS.items()}

_PHRASE_WEIGHTS = {**_HAWK_BIGRAMS, **_DOVE_BIGRAMS}
_UNIGRAM_WEIGHTS = {**_HAWK_UNIGRAMS, **_DOVE_UNIGRAMS}
_MAX_PHRASE_TOKENS = max((len(term.split()) for term in _PHRASE_WEIGHTS), default=2)


@dataclass(frozen=True)
class PMFedSpeakSentiment:
    score: float
    bucket: str
    direction: int
    implied_bias_bp: float
    matched_phrases: tuple[str, ...]
    matched_unigrams: tuple[str, ...]
    matched_bigrams: tuple[str, ...]
    unknown_candidate_phrases: tuple[str, ...]


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


def _bucket_for_score(score: float) -> str:
    absolute = abs(score)
    if absolute < 1.0:
        return "none"
    if absolute < 2.0:
        return "light"
    if absolute < 3.25:
        return "medium"
    if absolute < 4.75:
        return "strong"
    return "extreme"


def _bias_for_bucket(score: float, bucket: str) -> float:
    direction = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
    if bucket == "none":
        return 0.0
    if bucket == "light":
        return 0.75 * direction
    if bucket == "medium":
        return 1.50 * direction
    if bucket == "strong":
        return 2.25 * direction
    return 3.25 * direction


def _extract_unknown_candidates(tokens: list[str], used: list[bool]) -> tuple[str, ...]:
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
        if phrase in _PHRASE_WEIGHTS or phrase in _AMPLIFIERS or phrase in _NEUTRALIZERS:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        unknowns.append(phrase)
    return tuple(unknowns)


def score_pm_fedspeak(text: str | None) -> PMFedSpeakSentiment:
    normalized = normalize_text(text)
    if not normalized:
        return PMFedSpeakSentiment(
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

    amplifier = max((weight for phrase, weight in _AMPLIFIERS.items() if _contains_phrase(tokens, phrase)), default=1.0)
    neutralizer = min((weight for phrase, weight in _NEUTRALIZERS.items() if _contains_phrase(tokens, phrase)), default=1.0)
    score *= amplifier
    score *= neutralizer
    score = max(-6.0, min(6.0, score))

    bucket = _bucket_for_score(score)
    direction = 1 if score > 0 else -1 if score < 0 else 0
    implied_bias_bp = _bias_for_bucket(score, bucket)
    matched_phrases = tuple(dict.fromkeys([*matched_bigrams, *matched_unigrams]))

    return PMFedSpeakSentiment(
        score=float(score),
        bucket=bucket,
        direction=direction,
        implied_bias_bp=float(implied_bias_bp),
        matched_phrases=matched_phrases,
        matched_unigrams=tuple(dict.fromkeys(matched_unigrams)),
        matched_bigrams=tuple(dict.fromkeys(matched_bigrams)),
        unknown_candidate_phrases=_extract_unknown_candidates(tokens, used),
    )


def get_pm_fedspeak_term_weight(term: str) -> float | None:
    normalized = normalize_term(term)
    if normalized in _HAWK_BIGRAMS:
        return _HAWK_BIGRAMS[normalized]
    if normalized in _DOVE_BIGRAMS:
        return _DOVE_BIGRAMS[normalized]
    if normalized in _HAWK_UNIGRAMS:
        return _HAWK_UNIGRAMS[normalized]
    if normalized in _DOVE_UNIGRAMS:
        return _DOVE_UNIGRAMS[normalized]
    return None


def get_pm_fedspeak_term_polarity(term: str) -> str | None:
    weight = get_pm_fedspeak_term_weight(term)
    if weight is None:
        return None
    if weight > 0:
        return "hawkish"
    if weight < 0:
        return "dovish"
    return "neutral"
