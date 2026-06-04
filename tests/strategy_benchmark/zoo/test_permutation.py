import numpy as np
from quant_research_stack.strategy_benchmark.zoo.permutation import permutation_control


def test_permutation_best_matches_real_for_zero_skill_pool() -> None:
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 0.01, (500, 4_000))
    out = permutation_control(is_returns=mat, seed=7)
    assert abs(out["real_best_sharpe"] - out["permuted_best_sharpe"]) < 0.7
    assert out["seed"] == 7
