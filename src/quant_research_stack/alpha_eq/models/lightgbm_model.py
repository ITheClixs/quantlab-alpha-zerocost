"""LightGBM S1-EQ base learner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LightGBMEqConfig:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    seed: int = 42


class LightGBMEqModel:
    def __init__(self, config: LightGBMEqConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "seed": self.config.seed,
            "verbose": -1,
        }
        train_set = lgb.Dataset(x, label=y)
        valid_sets: list[lgb.Dataset] = [train_set]
        valid_names: list[str] = ["train"]
        callbacks: list[Callable[..., Any]] = []
        if x_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(x_val, label=y_val))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))
        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._booster.predict(x), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> LightGBMEqModel:
        cfg = LightGBMEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._booster = lgb.Booster(model_file=str(path))
        return m
