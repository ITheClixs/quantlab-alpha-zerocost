"""Shared pytest fixtures for the alpha test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def synthetic_js() -> pl.DataFrame:
    """Deterministic JS-shaped 10k-row Polars DataFrame.

    Layout:
      - date_id     : monotonic int64 in [0, 1000)
      - feature_00  : informative (linear with target)
      - feature_01  : informative (linear with target, negative weight)
      - feature_02..49 : Gaussian noise
      - weight      : float, in (0, 2]
      - responder_6 : target — linear combo of informative features + noise
    """
    rng = np.random.default_rng(seed=20260521)
    n_rows = 10_000
    n_features = 50

    feature_matrix = rng.standard_normal((n_rows, n_features)).astype(np.float64)
    # Two informative features.
    informative_w = np.array([0.4, -0.3], dtype=np.float64)
    signal = feature_matrix[:, :2] @ informative_w
    noise = 0.5 * rng.standard_normal(n_rows)
    target = (signal + noise).astype(np.float64)

    date_id = (np.arange(n_rows) // 10).astype(np.int64)  # 10 rows per date_id
    weight = (0.5 + rng.uniform(size=n_rows)).astype(np.float64)

    cols: dict[str, np.ndarray] = {f"feature_{i:02d}": feature_matrix[:, i] for i in range(n_features)}
    cols["date_id"] = date_id
    cols["responder_6"] = target
    cols["weight"] = weight

    return pl.DataFrame(cols)
