"""Ridge S1-EQ base learner (target = y_xs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeEqConfig:
    alpha: float = 1.0


class RidgeEqModel:
    def __init__(self, config: RidgeEqConfig) -> None:
        self.config = config
        self._estimator = Ridge(alpha=config.alpha, fit_intercept=True)

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self._estimator.fit(x, y)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self._estimator, "config": asdict(self.config)}, path)

    @classmethod
    def load(cls, path: Path) -> RidgeEqModel:
        payload = joblib.load(path)
        inst = cls(RidgeEqConfig(**payload["config"]))
        inst._estimator = payload["estimator"]
        return inst
