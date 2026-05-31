from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PBOReport:
    pbo: float
    strategy_count: int
    block_count: int
    split_count: int
    oos_rank_percentiles: list[float]
    logit_ranks: list[float]
    selected_strategy_ids: list[str]
    pbo_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bucket(pbo: float) -> str:
    if pbo < 0.10:
        return "strong"
    if pbo < 0.25:
        return "acceptable_but_cautious"
    if pbo < 0.50:
        return "fragile"
    return "reject"


def _logit(value: float) -> float:
    eps = 1e-9
    clipped = min(max(value, eps), 1.0 - eps)
    return float(math.log(clipped / (1.0 - clipped)))


def estimate_pbo(scores: pl.DataFrame, *, score_column: str, min_blocks: int = 6) -> PBOReport:
    required = {"strategy_id", "block", score_column}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"missing PBO score columns: {sorted(missing)}")
    strategies = sorted(str(value) for value in scores.get_column("strategy_id").unique().to_list())
    blocks = sorted(int(value) for value in scores.get_column("block").unique().to_list())
    if len(strategies) < 2:
        raise ValueError("PBO requires at least two strategies")
    if len(blocks) < min_blocks:
        raise ValueError(f"PBO requires at least {min_blocks} chronological blocks")
    table = {
        (str(row["strategy_id"]), int(row["block"])): float(row[score_column])
        for row in scores.iter_rows(named=True)
    }
    half = len(blocks) // 2
    rank_percentiles: list[float] = []
    logit_ranks: list[float] = []
    selected_ids: list[str] = []
    for train_blocks_tuple in itertools.combinations(blocks, half):
        train_blocks = set(train_blocks_tuple)
        test_blocks = [block for block in blocks if block not in train_blocks]
        train_scores: dict[str, float] = {}
        test_scores: dict[str, float] = {}
        for strategy in strategies:
            train_values = [table.get((strategy, block), 0.0) for block in train_blocks]
            test_values = [table.get((strategy, block), 0.0) for block in test_blocks]
            train_scores[strategy] = float(np.mean(train_values))
            test_scores[strategy] = float(np.mean(test_values))
        selected = max(strategies, key=lambda strategy: train_scores[strategy])
        ordered_oos = sorted(strategies, key=lambda strategy: test_scores[strategy], reverse=True)
        rank = ordered_oos.index(selected) + 1
        percentile = 1.0 - ((rank - 1) / max(len(strategies) - 1, 1))
        rank_percentiles.append(percentile)
        logit_ranks.append(_logit(percentile))
        selected_ids.append(selected)
    pbo = float(np.mean(np.asarray(logit_ranks) < 0.0)) if logit_ranks else 1.0
    return PBOReport(
        pbo=pbo,
        strategy_count=len(strategies),
        block_count=len(blocks),
        split_count=len(logit_ranks),
        oos_rank_percentiles=rank_percentiles,
        logit_ranks=logit_ranks,
        selected_strategy_ids=selected_ids,
        pbo_bucket=_bucket(pbo),
    )


def approximate_multiple_testing_payload(
    best_sharpe: float,
    *,
    trial_count: int,
    observations: int,
) -> dict[str, float | int]:
    if observations <= 1:
        raw_p = 1.0
    else:
        raw_p = 1.0 - NormalDist().cdf(best_sharpe * math.sqrt(observations / 365.0))
    bonferroni = min(1.0, raw_p * max(trial_count, 1))
    expected_max_noise_sharpe = NormalDist().inv_cdf(1.0 - (1.0 / max(trial_count, 2)))
    return {
        "trial_count": trial_count,
        "observations": observations,
        "raw_one_sided_p_value": raw_p,
        "bonferroni_p_value": bonferroni,
        "expected_max_noise_z": expected_max_noise_sharpe,
        "deflated_sharpe_proxy": best_sharpe - expected_max_noise_sharpe,
    }


def write_pbo_report(path: Path, report: PBOReport, *, extra: dict[str, Any] | None = None) -> None:
    payload = report.to_dict()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
