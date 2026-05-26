"""Correlation deduplication (spec §4.3).

- Operates on NET OOS returns.
- Uses absolute correlation by default (inverse strategies = sign-flip = duplicate).
- Reports both signed and absolute correlation matrices.
- Three representative-selection rules reported, no single rule hiding a better candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DedupConfig:
    absolute_correlation_threshold: float = 0.90


@dataclass(frozen=True)
class DedupResult:
    cluster_ids: NDArray[np.int64]
    n_clusters: int
    signed_correlation: NDArray[np.float64]
    absolute_correlation: NDArray[np.float64]
    representatives: dict[str, list[int]] = field(default_factory=dict)


def _cluster_from_abs_corr(
    abs_corr: NDArray[np.float64], threshold: float
) -> NDArray[np.int64]:
    n = abs_corr.shape[0]
    cluster = np.full(n, -1, dtype=np.int64)
    current = 0
    for i in range(n):
        if cluster[i] != -1:
            continue
        stack = [i]
        while stack:
            k = stack.pop()
            if cluster[k] != -1:
                continue
            cluster[k] = current
            for j in range(n):
                if cluster[j] == -1 and abs_corr[k, j] >= threshold:
                    stack.append(j)
        current += 1
    return cluster


def deduplicate(
    *,
    net_returns: NDArray[np.float64],
    sharpe: NDArray[np.float64],
    turnover: NDArray[np.float64],
    dsr: NDArray[np.float64],
    drawdown: NDArray[np.float64],
    config: DedupConfig,
) -> DedupResult:
    signed = np.corrcoef(net_returns.T)
    absolute = np.abs(signed)
    clusters = _cluster_from_abs_corr(absolute, config.absolute_correlation_threshold)
    n_clusters = int(clusters.max()) + 1 if clusters.size else 0

    representatives: dict[str, list[int]] = {
        "by_sharpe_per_sqrt_turnover": [],
        "by_dsr": [],
        "by_lowest_drawdown": [],
    }
    for c in range(n_clusters):
        members = np.where(clusters == c)[0]
        sps = sharpe[members] / np.sqrt(np.maximum(turnover[members], 1e-6))
        representatives["by_sharpe_per_sqrt_turnover"].append(int(members[int(np.argmax(sps))]))
        representatives["by_dsr"].append(int(members[int(np.argmax(dsr[members]))]))
        representatives["by_lowest_drawdown"].append(int(members[int(np.argmin(drawdown[members]))]))

    return DedupResult(
        cluster_ids=clusters,
        n_clusters=n_clusters,
        signed_correlation=signed,
        absolute_correlation=absolute,
        representatives=representatives,
    )
