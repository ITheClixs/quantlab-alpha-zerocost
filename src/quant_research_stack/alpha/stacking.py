from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class LinearStacker:
    def __init__(self, alpha: float = 1e-3) -> None:
        self._estimator = Ridge(alpha=alpha, positive=True, fit_intercept=False)

    def fit(self, oof_predictions: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]) -> None:
        self._estimator.fit(oof_predictions, y, sample_weight=weights)

    def predict(self, base_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(base_predictions), dtype=np.float64)

    def weights(self) -> NDArray[np.float64]:
        return np.asarray(self._estimator.coef_, dtype=np.float64)
