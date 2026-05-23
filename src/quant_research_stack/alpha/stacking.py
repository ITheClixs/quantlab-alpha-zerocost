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
        self._active_feature_order: list[str] = list(self._feature_order)
        self._residual_scale: float = 1.0
        self._signal_scale: float = 0.0
        self._estimator = Ridge(alpha=alpha, positive=True, fit_intercept=False)

    @property
    def feature_order(self) -> list[str]:
        return list(self._feature_order)

    @property
    def active_feature_order(self) -> list[str]:
        return list(self._active_feature_order)

    @property
    def residual_scale(self) -> float:
        return float(self._residual_scale)

    def fit(
        self,
        oof_predictions: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        active_feature_order: Sequence[str] | None = None,
    ) -> None:
        x = np.asarray(oof_predictions, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected 2D OOF predictions, got shape={x.shape}")
        if not self._feature_order:
            self._feature_order = [f"model_{i}" for i in range(x.shape[1])]
        if len(self._feature_order) != x.shape[1]:
            raise ValueError(
                f"feature_order has {len(self._feature_order)} names for {x.shape[1]} columns"
            )

        active = list(active_feature_order) if active_feature_order is not None else list(self._feature_order)
        unknown = sorted(set(active) - set(self._feature_order))
        if unknown:
            raise ValueError(f"active stack models are not in feature_order: {unknown}")
        if not active:
            raise ValueError("at least one active stack model is required")
        self._active_feature_order = active

        active_idx = [self._feature_order.index(name) for name in active]
        inactive_idx = [i for i in range(x.shape[1]) if i not in set(active_idx)]
        x_fit = x.copy()
        if inactive_idx:
            x_fit[:, inactive_idx] = 0.0

        self._estimator.fit(x_fit, y, sample_weight=weights)
        coef = np.maximum(np.asarray(self._estimator.coef_, dtype=np.float64), 0.0)
        if inactive_idx:
            coef[inactive_idx] = 0.0
        coef_sum = float(coef.sum())
        if coef_sum > 0.0:
            coef = coef / coef_sum
        else:
            coef = np.zeros(x.shape[1], dtype=np.float64)
            coef[active_idx] = 1.0 / len(active_idx)
        self._estimator.coef_ = coef

        pred = self.predict(x)
        residual = np.asarray(y, dtype=np.float64) - pred
        safe_weights = np.asarray(weights, dtype=np.float64)
        if float(safe_weights.sum()) <= 0.0:
            safe_weights = np.ones_like(residual)
        self._residual_scale = float(np.sqrt(np.average(residual**2, weights=safe_weights)))
        pred_mean = float(np.average(pred, weights=safe_weights))
        self._signal_scale = float(np.sqrt(np.average((pred - pred_mean) ** 2, weights=safe_weights)))

    def predict(self, base_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(base_predictions), dtype=np.float64)

    def weights(self) -> NDArray[np.float64]:
        return np.asarray(self._estimator.coef_, dtype=np.float64)

    def confidence(self, base_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(base_predictions, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred_abs = np.abs(self.predict(x))
        active_idx = [self._feature_order.index(name) for name in self._active_feature_order]
        active = x[:, active_idx]
        if active.shape[1] <= 1:
            agreement = np.ones(x.shape[0], dtype=np.float64)
        else:
            active_mean = np.mean(active, axis=1)
            active_std = np.std(active, axis=1)
            agreement = 1.0 - (active_std / (np.abs(active_mean) + active_std + 1e-12))
        scale = max(self._residual_scale, 1e-12)
        magnitude = pred_abs / (pred_abs + scale)
        return np.clip(magnitude * agreement, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path = Path(path)
        if not hasattr(self._estimator, "coef_"):
            raise RuntimeError("cannot save un-fitted LinearStacker")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "feature_order": list(self._feature_order),
                "active_feature_order": list(self._active_feature_order),
                "alpha": float(self._alpha),
                "residual_scale": float(self._residual_scale),
                "signal_scale": float(self._signal_scale),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LinearStacker:
        path = Path(path)
        payload = joblib.load(path)
        inst = cls(alpha=payload["alpha"], feature_order=payload["feature_order"])
        inst._estimator = payload["estimator"]
        inst._active_feature_order = list(payload.get("active_feature_order", payload["feature_order"]))
        inst._residual_scale = float(payload.get("residual_scale", 1.0))
        inst._signal_scale = float(payload.get("signal_scale", 0.0))
        return inst
