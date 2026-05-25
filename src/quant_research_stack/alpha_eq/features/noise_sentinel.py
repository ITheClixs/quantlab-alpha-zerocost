"""Seeded Gaussian noise sentinel (spec §3.3-9, §3.4)."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl


def _seeded_value(*, seed: int, date_iso: str, symbol: str) -> float:
    payload = f"{seed}|{date_iso}|{symbol}".encode()
    h = hashlib.sha256(payload).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big", signed=False))
    return float(rng.standard_normal())


def attach_noise_sentinel(df: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    col = f"gaussian_noise_seed{seed}"
    values = [
        _seeded_value(seed=seed, date_iso=str(d), symbol=s)
        for d, s in zip(df["date"].to_list(), df["symbol"].to_list(), strict=True)
    ]
    return df.with_columns(pl.Series(col, values, dtype=pl.Float64))
