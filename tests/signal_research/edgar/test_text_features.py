from __future__ import annotations

from quant_research_stack.signal_research.edgar.text_features import (
    cosine_similarity,
    jaccard_new_deleted,
    lm_tone_ratios,
    numeric_density,
    section_features,
    sentence_count,
    tokenize,
    word_count,
)


def test_tokenize_and_counts() -> None:
    t = "The Company reported a 12.5% loss. Risks may increase!"
    assert word_count(t) >= 8
    assert sentence_count(t) == 2
    assert "loss" in tokenize(t)


def test_numeric_density_detects_numbers() -> None:
    assert numeric_density("revenue was 1,234.5 million in 2021") > 0.0
    assert numeric_density("no digits here at all") == 0.0
    assert numeric_density("") == 0.0


def test_lm_tone_negative_vs_positive() -> None:
    neg = lm_tone_ratios("loss decline litigation adverse weakness default")
    pos = lm_tone_ratios("growth profit strong improvement success favorable")
    assert neg["negative_ratio"] > 0.0 and neg["net_tone"] < 0.0
    assert pos["positive_ratio"] > 0.0 and pos["net_tone"] > 0.0


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity("a b c", "a b c") == 1.0
    assert cosine_similarity("a b c", "x y z") == 0.0
    assert cosine_similarity("", "a b") == 0.0
    mid = cosine_similarity("risk risk factor", "risk factor new")
    assert 0.0 < mid < 1.0


def test_jaccard_new_deleted() -> None:
    out = jaccard_new_deleted("alpha beta gamma", "beta gamma delta")
    assert out["new_word_frac"] > 0.0   # alpha is new
    assert out["deleted_word_frac"] > 0.0  # delta deleted
    same = jaccard_new_deleted("a b", "a b")
    assert same["new_word_frac"] == 0.0 and same["deleted_word_frac"] == 0.0


def test_section_features_prefixed_and_no_label_leak() -> None:
    feats = section_features("loss decline 2021 revenue 1,234", prefix="rf")
    assert all(k.startswith("rf_") for k in feats)
    assert "rf_word_count" in feats and "rf_negative_ratio" in feats
    # never emits anything return/label-like
    assert not any("return" in k or "label" in k or "fwd" in k for k in feats)
