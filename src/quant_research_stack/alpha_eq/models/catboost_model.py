"""CatBoost S1-EQ base learner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from numpy.typing import NDArray


@dataclass(frozen=True)
class CatBoostEqConfig:
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.05
    early_stopping_rounds: int = 100
    seed: int = 42


class CatBoostEqModel:
    def __init__(self, config: CatBoostEqConfig) -> None:
        self.config = config
        self._model: CatBoostRegressor | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        self._model = CatBoostRegressor(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.seed,
            verbose=False,
            allow_writing_files=False,
        )
        eval_set = (x_val, y_val) if x_val is not None and y_val is not None else None
        self._model.fit(
            x, y,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds if eval_set else None,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._model.predict(x), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._model is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path), format="cbm")
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> CatBoostEqModel:
        cfg = CatBoostEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._model = CatBoostRegressor()
        m._model.load_model(str(path), format="cbm")
        return m
