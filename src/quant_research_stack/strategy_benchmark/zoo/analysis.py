"""Multiple-testing analysis: theoretical-vs-empirical best Sharpe across tiers,
and Deflated-Sharpe of the in-sample winner."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.dsr import compute_dsr, expected_max_sharpe


def expected_vs_empirical(*, sharpe_estimates: NDArray[np.float64],
                          tiers: tuple[int, ...]) -> list[dict[str, Any]]:
    var = float(np.var(sharpe_estimates, ddof=1))
    out: list[dict[str, Any]] = []
    for n in tiers:
        n = min(n, sharpe_estimates.size)
        emp = float(np.max(sharpe_estimates[:n]))
        theo = expected_max_sharpe(n_trials=n, sharpe_variance=var)
        out.append({"n_trials": n, "empirical_max": emp, "theoretical_max": theo})
    return out


def deflate_best(*, is_returns: NDArray[np.float64]) -> dict[str, Any]:
    r = is_returns.astype(np.float64)
    mu = np.mean(r, axis=0)
    sd = np.std(r, axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    sr = np.nan_to_num(mu / sd * np.sqrt(252.0), nan=0.0, posinf=0.0, neginf=0.0)
    best = int(np.argmax(sr))
    dsr_res = compute_dsr(returns=r[:, best], sharpe_estimates=sr, selected_idx=best)
    return {"selected_idx": best, "observed_sharpe": float(sr[best]),
            "expected_max_under_null": float(dsr_res.expected_max_sharpe_under_null),
            "psr_zero": float(dsr_res.psr_zero), "dsr": float(dsr_res.dsr),
            "n_trials": int(sr.size)}
