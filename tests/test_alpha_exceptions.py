from __future__ import annotations

import pytest

from quant_research_stack.alpha.exceptions import (
    ArtifactCorruptError,
    ArtifactsMissingError,
    FeatureSchemaError,
)


def test_exceptions_are_runtimeerrors():
    assert issubclass(FeatureSchemaError, RuntimeError)
    assert issubclass(ArtifactsMissingError, RuntimeError)
    assert issubclass(ArtifactCorruptError, RuntimeError)


def test_feature_schema_error_carries_message():
    with pytest.raises(FeatureSchemaError, match="sha256 mismatch"):
        raise FeatureSchemaError("sha256 mismatch")


def test_artifacts_missing_error_carries_path():
    err = ArtifactsMissingError("missing models/ridge.joblib")
    assert "ridge.joblib" in str(err)
