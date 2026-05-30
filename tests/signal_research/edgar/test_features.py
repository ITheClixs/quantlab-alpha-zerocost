from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.signal_research.edgar.features import (
    _assert_no_label_leak,
    feature_columns,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame({
        "cik": [1, 2], "company": ["A", "B"], "sic": ["3841", "7372"], "sector2": ["38", "73"],
        "filing_date": ["2015-02-10", "2016-02-11"], "filing_year": [2015, 2016],
        "has_prior_filing": [0.0, 1.0],
        "rf_word_count": [100.0, 120.0], "rf_negative_ratio": [0.01, 0.02],
        "mda_net_tone": [-0.001, 0.002], "size_log_mktcap": [15.0, 16.0],
        "fwd_ret_21": [0.01, -0.02], "fwd_ret_63": [0.03, -0.01], "fwd_ret_252": [0.1, 0.2],
    })


def test_feature_columns_exclude_labels_and_meta() -> None:
    cols = feature_columns(_frame())
    assert "rf_word_count" in cols and "mda_net_tone" in cols and "size_log_mktcap" in cols
    for forbidden in ("fwd_ret_21", "fwd_ret_63", "fwd_ret_252", "filing_year", "cik", "has_prior_filing"):
        assert forbidden not in cols


def test_label_leak_guard_passes_clean_frame() -> None:
    _assert_no_label_leak(_frame())  # should not raise


def test_label_leak_guard_catches_injected_return_feature() -> None:
    bad = _frame().with_columns(pl.lit(0.5).alias("sneaky_fwd_ret_feature"))
    # a feature whose name contains 'fwd_ret' must be caught
    with pytest.raises(AssertionError):
        _assert_no_label_leak(bad)
