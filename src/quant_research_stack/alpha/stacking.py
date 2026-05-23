from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class LinearStacker:
    def __init__(self, alpha: float = 1e-3, feature_order: Sequence[str] | None = None) -> None:
        self._alpha = float(alpha)
        self._feature_order: list[str] = list(feature_order) if feature_order is not None else []
        self._estimator = Ridge(alpha=alpha, positive=True, fit_intercept=False)

    @property
    def feature_order(self) -> list[str]:
        return list(self._feature_order)

    def fit(self, oof_predictions: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]) -> None:
        self._estimator.fit(oof_predictions, y, sample_weight=weights)

    def predict(self, base_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(base_predictions), dtype=np.float64)

    def weights(self) -> NDArray[np.float64]:
        return np.asarray(self._estimator.coef_, dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        if not hasattr(self._estimator, "coef_"):
            raise RuntimeError("cannot save un-fitted LinearStacker")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "feature_order": list(self._feature_order),
                "alpha": float(self._alpha),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LinearStacker:
        path = Path(path)
        payload = joblib.load(path)
        inst = cls(alpha=payload["alpha"], feature_order=payload["feature_order"])
        inst._estimator = payload["estimator"]
        return inst
