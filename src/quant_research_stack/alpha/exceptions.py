"""Exceptions raised by the alpha persistence + serving layer."""

from __future__ import annotations


class FeatureSchemaError(RuntimeError):
    """Raised when a feature-schema invariant is violated.

    Cases:
    - feature_cols.json sha256 mismatch (file edited by hand)
    - caller passes a DataFrame whose columns don't cover the trained feature set
    """


class ArtifactsMissingError(RuntimeError):
    """Raised when a run directory lacks one or more required S0 artifacts.

    Typically encountered when loading a pre-S0 run (only stacker.joblib exists).
    """


class ArtifactCorruptError(RuntimeError):
    """Raised when an on-disk artifact exists but fails to load.

    Wraps the underlying library exception (joblib / torch / lgb / xgb / catboost)
    so callers don't have to catch six different types.
    """
