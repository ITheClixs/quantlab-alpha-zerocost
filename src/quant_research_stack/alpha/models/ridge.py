from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0


class RidgeAlphaModel:
    def __init__(self, config: RidgeConfig) -> None:
        self.config = config
        self._estimator = Ridge(alpha=config.alpha, fit_intercept=True)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]) -> None:
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("ridge requires positive total weights")
        self._estimator.fit(x, y, sample_weight=weights)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"sklearn_estimator": self._estimator, "config": asdict(self.config)},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> RidgeAlphaModel:
        path = Path(path)
        payload = joblib.load(path)
        inst = cls(RidgeConfig(**payload["config"]))
        inst._estimator = payload["sklearn_estimator"]
        return inst
