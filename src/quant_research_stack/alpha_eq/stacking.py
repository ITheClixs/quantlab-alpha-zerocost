"""L2-regularized linear stacker with optional non-negativity + signed
diagnostic + large-negative-weight flag (spec §4.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class LinearStackerEq:
    def __init__(
        self,
        *,
        alpha: float,
        prefer_non_negative: bool,
        feature_order: Sequence[str],
    ) -> None:
        self.alpha = float(alpha)
        self.prefer_non_negative = bool(prefer_non_negative)
        self.feature_order: tuple[str, ...] = tuple(feature_order)
        self._estimator: Ridge | None = None

    def fit(self, *, oof_predictions: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if oof_predictions.shape[1] != len(self.feature_order):
            raise ValueError("oof_predictions cols != len(feature_order)")
        est = Ridge(
            alpha=self.alpha,
            positive=self.prefer_non_negative,
            fit_intercept=False,
        )
        est.fit(oof_predictions, y)
        self._estimator = est

    def predict(self, oof: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._estimator is None:
            raise RuntimeError("stacker not fit")
        return np.asarray(self._estimator.predict(oof), dtype=np.float64)

    @property
    def weights(self) -> NDArray[np.float64]:
        if self._estimator is None:
            raise RuntimeError("stacker not fit")
        return np.asarray(self._estimator.coef_, dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "alpha": self.alpha,
                "prefer_non_negative": self.prefer_non_negative,
                "feature_order": list(self.feature_order),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LinearStackerEq:
        payload = joblib.load(path)
        m = cls(
            alpha=payload["alpha"],
            prefer_non_negative=payload["prefer_non_negative"],
            feature_order=tuple(payload["feature_order"]),
        )
        m._estimator = payload["estimator"]
        return m


@dataclass(frozen=True)
class StackerArtifact:
    feature_order: tuple[str, ...]
    weights: tuple[float, ...]
    flagged_negatives: tuple[str, ...]

    @classmethod
    def from_model(
        cls, model: LinearStackerEq, *, threshold: float = -0.25
    ) -> StackerArtifact:
        w = model.weights
        flagged = flag_large_negative_weights(
            weights=w, names=model.feature_order, threshold=threshold
        )
        return cls(
            feature_order=model.feature_order,
            weights=tuple(float(x) for x in w),
            flagged_negatives=tuple(flagged),
        )


def flag_large_negative_weights(
    *, weights: NDArray[np.float64], names: Sequence[str], threshold: float
) -> list[str]:
    return [n for w, n in zip(weights, names, strict=True) if float(w) < float(threshold)]
