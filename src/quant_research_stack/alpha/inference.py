from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray


class S1Predictor(Protocol):
    def predict(self, row: pl.DataFrame) -> tuple[float, float]: ...


@dataclass
class _StackPredictor:
    base_funcs: list[Callable[[np.ndarray], float]]
    weights: NDArray[np.float64]
    feature_cols: list[str]

    def predict(self, row: pl.DataFrame) -> tuple[float, float]:
        if row.height != 1:
            raise ValueError(f"S1 predicts one row at a time; got height={row.height}")
        x = row.select(self.feature_cols).to_numpy()[0]
        base_outs = np.fromiter((f(x) for f in self.base_funcs), dtype=np.float64, count=len(self.base_funcs))
        pred = float(np.dot(self.weights, base_outs))
        # confidence: normalized agreement among base models (1.0 = unanimous sign, 0.0 = split)
        signs = np.sign(base_outs)
        if signs.size == 0 or float(np.std(base_outs)) == 0.0:
            conf = 1.0
        else:
            same_sign = float(np.mean(signs == np.sign(np.mean(signs))))
            conf = float(np.clip(same_sign, 0.0, 1.0))
        return pred, conf


def build_predictor_from_stack(
    base_funcs: list[Callable[[np.ndarray], float]],
    stacker_weights: NDArray[np.float64],
    feature_cols: list[str],
) -> S1Predictor:
    if len(base_funcs) != stacker_weights.size:
        raise ValueError("base_funcs and stacker_weights length mismatch")
    return _StackPredictor(base_funcs=base_funcs, weights=stacker_weights, feature_cols=feature_cols)
