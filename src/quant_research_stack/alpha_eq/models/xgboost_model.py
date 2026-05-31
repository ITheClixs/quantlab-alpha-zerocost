"""XGBoost S1-EQ base learner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray


@dataclass(frozen=True)
class XGBoostEqConfig:
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    seed: int = 42


class XGBoostEqModel:
    def __init__(self, config: XGBoostEqConfig) -> None:
        self.config = config
        self._booster: xgb.Booster | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        dtrain = xgb.DMatrix(x, label=y)
        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        if x_val is not None and y_val is not None:
            evals.append((xgb.DMatrix(x_val, label=y_val), "valid"))
        params = {
            "objective": "reg:squarederror",
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "tree_method": self.config.tree_method,
            "seed": self.config.seed,
            "verbosity": 0,
        }
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._booster.predict(xgb.DMatrix(x)), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> XGBoostEqModel:
        cfg = XGBoostEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._booster = xgb.Booster()
        m._booster.load_model(str(path))
        return m
