"""Triple-Barrier + Meta-Labeling wrapper (López de Prado 2018).

Spec §3.3 #3, §4.2:
- vertical barrier ∈ {5, 10, 20, 40} predeclared
- profit-stop barriers ±k·σ_20 with k ∈ {1.0, 1.5, 2.0} predeclared
- side from primary; meta-labeler predicts trade-vs-flat (size)
- secondary classifier: RandomForestClassifier
- survivor-only — pre-filter via methodology.meta_labeling.check_eligibility
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from quant_research_stack.signal_research.methodology.meta_labeling import (
    MetaLabelingEligibility,
)
from quant_research_stack.signal_research.papers.base import Wrapper


@dataclass(frozen=True)
class TripleBarrierConfig:
    vertical_barrier_days: int = 20
    profit_take_multiplier: float = 1.5
    stop_loss_multiplier: float = 1.5
    vol_estimator_window: int = 20
    seed: int = 42


def label_triple_barrier(
    *,
    close: NDArray[np.float64],
    positions: NDArray[np.float64],
    cfg: TripleBarrierConfig,
) -> NDArray[np.float64]:
    """Per-event label ∈ {0, 1, nan}:
    1 = primary trade profitable before barrier hit,
    0 = stop-loss or vertical-barrier no-edge,
    nan = no position / insufficient vol estimator.
    """
    T = close.size
    log_ret = np.zeros(T, dtype=np.float64)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    vol = np.full(T, np.nan, dtype=np.float64)
    for t in range(cfg.vol_estimator_window, T):
        vol[t] = float(np.std(log_ret[t - cfg.vol_estimator_window : t], ddof=1))
    labels = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        if positions[t] == 0 or np.isnan(vol[t]):
            continue
        side = float(np.sign(positions[t]))
        pt = cfg.profit_take_multiplier * vol[t]
        sl = -cfg.stop_loss_multiplier * vol[t]
        cum = 0.0
        hit: float = 0.0
        for h in range(1, cfg.vertical_barrier_days + 1):
            if t + h >= T:
                break
            cum += log_ret[t + h] * side
            if cum >= pt:
                hit = 1.0
                break
            if cum <= sl:
                hit = 0.0
                break
        labels[t] = hit
    return labels


class TripleBarrierWrapper(Wrapper):
    def __init__(
        self, config: TripleBarrierConfig, eligibility: MetaLabelingEligibility
    ) -> None:
        if not eligibility.eligible:
            raise RuntimeError(
                f"primary signal not eligible for meta-labeling: "
                f"{eligibility.rejection_reason}"
            )
        self.config = config
        self._model: RandomForestClassifier | None = None

    def train_secondary(
        self,
        *,
        primary_positions: NDArray[np.float64],
        closes: NDArray[np.float64],
        features_at_event: NDArray[np.float64],
    ) -> None:
        labels = label_triple_barrier(
            close=closes, positions=primary_positions, cfg=self.config
        )
        self.fit_labeled_events(features_at_event=features_at_event, labels=labels)

    def fit_labeled_events(
        self,
        *,
        features_at_event: NDArray[np.float64],
        labels: NDArray[np.float64],
        n_estimators: int = 200,
    ) -> None:
        if features_at_event.shape[0] != labels.shape[0]:
            raise ValueError(
                f"feature rows ({features_at_event.shape[0]}) != label rows ({labels.shape[0]})"
            )
        mask = ~np.isnan(labels)
        if int(mask.sum()) < 2:
            raise ValueError("at least two labeled events are required")
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=self.config.seed, n_jobs=-1
        )
        self._model.fit(features_at_event[mask], labels[mask].astype(int))

    def predict_trade_probability(
        self,
        features_at_event: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("secondary model is not trained")
        proba = self._model.predict_proba(features_at_event)
        classes = self._model.classes_.astype(int)
        if 1 in classes:
            one_idx = int(np.where(classes == 1)[0][0])
            return proba[:, one_idx].astype(np.float64)
        return np.zeros(features_at_event.shape[0], dtype=np.float64)

    def filter_positions(
        self,
        *,
        primary_positions: NDArray[np.float64],
        features_at_event: NDArray[np.float64],
        probability_threshold: float,
    ) -> NDArray[np.float64]:
        probabilities = self.predict_trade_probability(features_at_event)
        return np.where(probabilities >= probability_threshold, primary_positions, 0.0).astype(np.float64)

    def apply(self, positions: pl.Series) -> pl.Series:
        return positions
