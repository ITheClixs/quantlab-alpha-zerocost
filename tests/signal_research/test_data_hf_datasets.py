"""HuggingFace datasets loader — gated by default (spec §2.6)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.data.hf_datasets import (
    HFDatasetGatedError,
    load_hf_dataset_gated,
)


def test_loader_blocks_when_research_only_default_and_not_audited() -> None:
    with pytest.raises(HFDatasetGatedError):
        load_hf_dataset_gated(
            dataset_id="Lettria/financial-news-sentiment",
            audit_token=None,
        )


def test_loader_rejects_audit_token_without_passing_audit() -> None:
    with pytest.raises(HFDatasetGatedError):
        load_hf_dataset_gated(
            dataset_id="Lettria/financial-news-sentiment",
            audit_token="not-a-real-audit-token",
        )
