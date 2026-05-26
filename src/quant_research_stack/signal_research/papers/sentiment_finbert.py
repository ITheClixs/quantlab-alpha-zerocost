"""FinBERT sentiment (spec §3.3 #8).

In v1: research_only_default. The pipeline below is a placeholder that
refuses to operate unless an audit_token from the 10-criterion FinBERT
ladder is provided.
"""

from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator


class FinBERTGatedError(RuntimeError):
    pass


class FinBERTSentimentFeature(FeatureGenerator):
    def __init__(self, *, audit_token: str | None = None) -> None:
        if audit_token is None:
            raise FinBERTGatedError(
                "FinBERT is research_only_default in v1 (spec §3.3 #8). "
                "Provide a validated audit_token after passing the 10-criterion "
                "sentiment timestamp/leakage audit."
            )
        self.audit_token = audit_token

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel
