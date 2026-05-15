from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    random_state: int = 42


class XGBoostAlphaModel:
    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self._booster: xgb.Booster | None = None

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        params: dict[str, object] = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "tree_method": self.config.tree_method,
            "seed": self.config.random_state,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.predict(xgb.DMatrix(x)), dtype=np.float64)
