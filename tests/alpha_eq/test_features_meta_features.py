"""Meta-features default-disabled gate (spec §3.3-7)."""

from __future__ import annotations

import pytest

from quant_research_stack.alpha_eq.features.meta_features import (
    MetaFeaturesDisabledError,
    build_meta_features,
)


def test_meta_features_disabled_by_default_raises() -> None:
    with pytest.raises(MetaFeaturesDisabledError):
        build_meta_features(panel=None, enable=False)


def test_meta_features_audited_gate_required() -> None:
    with pytest.raises(MetaFeaturesDisabledError):
        build_meta_features(panel=None, enable=True, audit_pass_token=None)
