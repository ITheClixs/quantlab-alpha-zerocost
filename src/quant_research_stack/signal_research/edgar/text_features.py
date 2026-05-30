"""Pure classical text-feature functions for 10-K sections (no external models).

Deterministic, dependency-light (regex + Counter). All features are computed from
the filing text alone — no labels, no prices, no future information.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from quant_research_stack.signal_research.edgar.lm_lexicon import CATEGORIES

_WORD_RE = re.compile(r"[a-z]+(?:['-][a-z]+)*")
_NUM_RE = re.compile(r"\b\d[\d,.]*\b")
# Sentence terminator: .!? followed by whitespace or end-of-text — so decimals
# like "12.5" (period followed by a digit) do NOT split a sentence.
_SENT_RE = re.compile(r"[.!?]+(?=\s|$)")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return max(1, len(_SENT_RE.findall(text)))


def numeric_density(text: str) -> float:
    """Numeric tokens / (numeric + word tokens). 0 when empty."""
    words = len(tokenize(text))
    nums = len(_NUM_RE.findall(text))
    total = words + nums
    return nums / total if total else 0.0


def readability(text: str) -> dict[str, float]:
    """Lightweight readability proxies (full Fog needs syllables; we approximate)."""
    toks = tokenize(text)
    sents = sentence_count(text)
    if not toks:
        return {"avg_words_per_sentence": 0.0, "avg_word_len": 0.0, "pct_long_words": 0.0}
    long_words = sum(1 for t in toks if len(t) >= 10)
    return {
        "avg_words_per_sentence": len(toks) / sents,
        "avg_word_len": sum(len(t) for t in toks) / len(toks),
        "pct_long_words": long_words / len(toks),
    }


def lm_counts(text: str) -> dict[str, int]:
    toks = tokenize(text)
    counts = Counter(toks)
    return {cat: int(sum(counts[w] for w in words)) for cat, words in CATEGORIES.items()}


def lm_tone_ratios(text: str) -> dict[str, float]:
    toks = tokenize(text)
    n = len(toks) or 1
    counts = lm_counts(text)
    ratios = {f"{cat}_ratio": counts[cat] / n for cat in CATEGORIES}
    pos, neg = counts["positive"], counts["negative"]
    ratios["net_tone"] = (pos - neg) / n
    ratios["polarity"] = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    return ratios


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity in [0, 1]. 0 if either side is empty."""
    a, b = _tf(tokenize(text_a)), _tf(tokenize(text_b))
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return min(1.0, dot / (na * nb)) if na and nb else 0.0


def jaccard_new_deleted(text_now: str, text_prev: str) -> dict[str, float]:
    """Fraction of vocabulary newly added / deleted year-over-year."""
    now, prev = set(tokenize(text_now)), set(tokenize(text_prev))
    if not now and not prev:
        return {"new_word_frac": 0.0, "deleted_word_frac": 0.0}
    union = now | prev
    new = now - prev
    deleted = prev - now
    return {
        "new_word_frac": len(new) / len(union) if union else 0.0,
        "deleted_word_frac": len(deleted) / len(union) if union else 0.0,
    }


def section_features(text: str, *, prefix: str) -> dict[str, float]:
    """All single-section classical features (length/tone/readability/numeric)."""
    text = text or ""
    toks = tokenize(text)
    feats: dict[str, float] = {
        f"{prefix}_char_len": float(len(text)),
        f"{prefix}_word_count": float(len(toks)),
        f"{prefix}_sentence_count": float(sentence_count(text)),
        f"{prefix}_numeric_density": numeric_density(text),
    }
    for k, v in readability(text).items():
        feats[f"{prefix}_{k}"] = v
    for k, v in lm_tone_ratios(text).items():
        feats[f"{prefix}_{k}"] = v
    return feats
