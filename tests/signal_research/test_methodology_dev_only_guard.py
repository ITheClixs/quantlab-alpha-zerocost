"""Dev-only invariant guard (spec §4.9, §0 non-negotiable #2)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.methodology.dev_only_guard import (
    HoldoutAccessError,
    enforce_dev_only,
)


def test_methodology_caller_cannot_access_holdout() -> None:
    with pytest.raises(HoldoutAccessError):
        enforce_dev_only(
            caller="methodology.cpcv",
            holdout_indices=[10, 11, 12],
            accessed_indices=[11],
        )


def test_inference_evaluate_holdout_caller_allowed() -> None:
    enforce_dev_only(
        caller="inference.evaluate_holdout",
        holdout_indices=[10, 11, 12],
        accessed_indices=[11],
    )
