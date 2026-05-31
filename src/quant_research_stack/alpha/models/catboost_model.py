from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool
from numpy.typing import NDArray


@dataclass(frozen=True)
class CatBoostConfig:
    depth: int = 12
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    random_state: int = 42


class CatBoostAlphaModel:
    def __init__(self, config: CatBoostConfig) -> None:
        self.config = config
        self._estimator = CatBoostRegressor(
            depth=config.depth,
            learning_rate=config.learning_rate,
            iterations=config.n_estimators,
            random_seed=config.random_state,
            allow_writing_files=False,
            verbose=False,
        )

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        train_pool = Pool(x_train, y_train, weight=w_train)
        val_pool = Pool(x_val, y_val, weight=w_val)
        self._estimator.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self.config.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._estimator.save_model(str(path))
        sidecar = path.parent / "catboost.config.json"
        sidecar.write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> CatBoostAlphaModel:
        path = Path(path)
        sidecar = path.parent / "catboost.config.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar config: {sidecar}")
        cfg_dict = json.loads(sidecar.read_text())
        inst = cls(CatBoostConfig(**cfg_dict))
        inst._estimator = CatBoostRegressor()
        inst._estimator.load_model(str(path))
        return inst
