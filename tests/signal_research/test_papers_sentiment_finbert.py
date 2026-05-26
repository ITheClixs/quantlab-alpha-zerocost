"""FinBERT placeholder tests (spec §3.3 #8)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.papers.sentiment_finbert import (
    FinBERTGatedError,
    FinBERTSentimentFeature,
)


def test_finbert_refuses_without_audit_token() -> None:
    with pytest.raises(FinBERTGatedError):
        FinBERTSentimentFeature(audit_token=None)


def test_finbert_constructs_with_audit_token() -> None:
    fb = FinBERTSentimentFeature(audit_token="dummy-audit-token-v1")
    assert fb.audit_token == "dummy-audit-token-v1"
