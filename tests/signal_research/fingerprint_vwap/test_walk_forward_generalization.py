from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
)
from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    fingerprint_columns,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_default_config_has_new_optional_fields() -> None:
    cfg = MetaLabelWalkForwardConfig()
    assert cfg.primary_position_col is None
    assert cfg.extra_feature_columns == ()


def test_caller_primary_and_extra_features_are_used(panel: pl.DataFrame) -> None:
    fp_cols = fingerprint_columns((20, 60))
    prepared = vwap_primary_position(
        build_fingerprint_features(daily_vwap_proxy(panel, window=5), windows=(20, 60)),
        band=0.0,
    )
    cfg = MetaLabelWalkForwardConfig(
        train_window_days=120, test_window_days=30, step_days=30, min_train_events=20,
        primary_position_col="primary_position", extra_feature_columns=fp_cols,
    )
    result = train_meta_label_walk_forward(panel=prepared, config=cfg)
    assert result.summary["fold_count"] >= 1
