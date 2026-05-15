from __future__ import annotations

import time

import numpy as np
import polars as pl

from quant_research_stack.alpha.inference import S1Predictor, build_predictor_from_stack


def test_predict_returns_float_and_confidence() -> None:
    feature_cols = ["a", "b", "c"]

    def base_a(row: np.ndarray) -> float:
        return float(row[0] + 0.5 * row[1])

    def base_b(row: np.ndarray) -> float:
        return float(row[2])

    stacker_weights = np.array([0.7, 0.3])
    predictor: S1Predictor = build_predictor_from_stack([base_a, base_b], stacker_weights, feature_cols)
    df = pl.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]})
    pred, conf = predictor.predict(df)
    expected = 0.7 * (0.1 + 0.5 * 0.2) + 0.3 * 0.3
    assert abs(pred - expected) < 1e-9
    assert 0.0 <= conf <= 1.0


def test_predict_sub_millisecond() -> None:
    feature_cols = ["a", "b", "c"]
    predictor = build_predictor_from_stack(
        [lambda row: float(row[0]), lambda row: float(row[1])],
        np.array([0.5, 0.5]),
        feature_cols,
    )
    df = pl.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]})
    # warm up
    for _ in range(10):
        predictor.predict(df)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        predictor.predict(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_call_ms = elapsed_ms / n
    assert per_call_ms < 1.0, f"expected <1 ms per call; got {per_call_ms:.3f} ms"
