from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LightGBMConfig:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    random_state: int = 42


class LightGBMAlphaModel:
    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

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
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "verbosity": -1,
            "seed": self.config.random_state,
        }
        train_set = lgb.Dataset(x_train, y_train, weight=w_train)
        val_set = lgb.Dataset(x_val, y_val, weight=w_val, reference=train_set)
        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.predict(x), dtype=np.float64)

    def feature_importance(self) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.feature_importance(importance_type="gain"), dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._booster is None:
            raise RuntimeError("cannot save un-fitted LightGBMAlphaModel")
        self._booster.save_model(str(path))
        sidecar = path.parent / f"{path.stem}.config.json"
        sidecar.write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> LightGBMAlphaModel:
        path = Path(path)
        sidecar = path.parent / f"{path.stem}.config.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar config: {sidecar}")
        cfg_dict = json.loads(sidecar.read_text())
        inst = cls(LightGBMConfig(**cfg_dict))
        inst._booster = lgb.Booster(model_file=str(path))
        return inst
