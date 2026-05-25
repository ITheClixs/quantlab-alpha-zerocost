"""Seeded Gaussian noise sentinel (spec §3.3-9)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.noise_sentinel import attach_noise_sentinel


def test_noise_sentinel_is_deterministic_per_date_symbol() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 2)],
            "symbol": ["A", "A", "B"],
        }
    )
    out1 = attach_noise_sentinel(df, seed=42)
    out2 = attach_noise_sentinel(df, seed=42)
    assert out1["gaussian_noise_seed42"].to_list() == out2["gaussian_noise_seed42"].to_list()


def test_noise_sentinel_different_seeds_differ() -> None:
    df = pl.DataFrame(
        {"date": [date(2020, 1, 2)], "symbol": ["A"]}
    )
    a = attach_noise_sentinel(df, seed=42)["gaussian_noise_seed42"][0]
    b = attach_noise_sentinel(df, seed=43)["gaussian_noise_seed43"][0]
    assert a != b
