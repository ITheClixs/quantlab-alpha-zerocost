"""Three-tier PBO reporting (spec §4.7).

Reuses the existing strategy_benchmark.pbo for the core algorithm, then
slices by profile and by family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.pbo import compute_pbo


@dataclass(frozen=True)
class PBOMultiResult:
    raw_global: float
    per_profile: dict[str, float]
    per_family: dict[str, float]
    n_partitions: int
    n_strategies: int


def compute_three_tier_pbo(
    *,
    returns: NDArray[np.float64],
    profile: NDArray[np.str_],
    family: NDArray[np.str_],
    n_partitions: int = 16,
) -> PBOMultiResult:
    raw = compute_pbo(returns=returns, n_partitions=n_partitions)

    per_profile: dict[str, float] = {}
    for p in sorted(set(profile.tolist())):
        cols = np.where(profile == p)[0]
        if len(cols) < 3:
            continue
        per_profile[p] = compute_pbo(
            returns=returns[:, cols], n_partitions=n_partitions
        ).pbo_probability

    per_family: dict[str, float] = {}
    for f in sorted(set(family.tolist())):
        cols = np.where(family == f)[0]
        if len(cols) < 3:
            continue
        per_family[f] = compute_pbo(
            returns=returns[:, cols], n_partitions=n_partitions
        ).pbo_probability

    return PBOMultiResult(
        raw_global=raw.pbo_probability,
        per_profile=per_profile,
        per_family=per_family,
        n_partitions=n_partitions,
        n_strategies=returns.shape[1],
    )
