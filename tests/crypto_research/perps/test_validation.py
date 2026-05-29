from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    classify_perp_candidate,
    concentration_payload,
    deflated_sharpe_payload,
    estimate_registry_pbo,
    write_perp_reports,
)


def test_candidate_cannot_promote_without_pbo_and_bootstrap() -> None:
    status = classify_perp_candidate(
        {
            "net_daily_sharpe": 3.0,
            "net_total_return": 0.5,
            "pbo_probability": None,
            "bootstrap_ci_lower_95": -1.0,
            "cost_2x_net_total_return": 0.1,
            "delay_1_event_net_total_return": 0.1,
        }
    )

    assert status["promotion_eligible"] is False
    assert status["production_candidate"] is False
    assert "missing_or_high_pbo" in status["blockers"]
    assert "bootstrap_ci_not_positive" in status["blockers"]
    assert "free_data_research_only" in status["blockers"]


def test_registry_pbo_returns_probability_for_variant_matrix() -> None:
    returns = pl.DataFrame(
        {
            "event_index": list(range(40)),
            "strategy_a": [0.001] * 40,
            "strategy_b": [0.001 if i % 2 == 0 else -0.001 for i in range(40)],
            "strategy_c": [-0.001] * 40,
        }
    )

    pbo = estimate_registry_pbo(
        returns,
        strategy_columns=["strategy_a", "strategy_b", "strategy_c"],
        n_partitions=4,
    )

    assert pbo["status"] == "computed"
    assert 0.0 <= pbo["pbo_probability"] <= 1.0
    assert pbo["split_count"] > 0
    assert len(pbo["oos_rank_percentiles"]) == pbo["split_count"]
    assert len(pbo["logit_ranks"]) == pbo["split_count"]
    assert pbo["block_count"] == 4


def test_registry_pbo_handles_small_inputs_without_raising() -> None:
    pbo = estimate_registry_pbo(
        pl.DataFrame({"event_index": [0, 1], "strategy_a": [0.0, 0.0]}),
        strategy_columns=["strategy_a"],
        n_partitions=4,
    )

    assert pbo["status"] == "not_estimated"
    assert pbo["pbo_probability"] is None
    assert pbo["split_count"] == 0


def test_validation_payloads_are_deterministic_and_safe() -> None:
    returns = [0.001, 0.002, -0.0005, 0.0015, -0.0002, 0.0008, 0.0012, -0.0001]

    first = bootstrap_sharpe_payload(returns, resamples=200, seed=7)
    second = bootstrap_sharpe_payload(returns, resamples=200, seed=7)
    dsr = deflated_sharpe_payload(returns, trials=5)

    assert first == second
    assert first["status"] == "computed"
    assert first["ci_lower_95"] <= first["point_sharpe"] <= first["ci_upper_95"]
    assert dsr["status"] == "computed_approximation"
    assert 0.0 <= dsr["probability"] <= 1.0


def test_concentration_payload_flags_best_day_and_symbol_dominance() -> None:
    trades = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
            "event_time": [1, 1, 2, 3],
            "net_return": [0.08, 0.02, 0.01, -0.01],
        }
    )

    payload = concentration_payload(trades)

    assert payload["best_day_positive_pnl_share"] > 0.8
    assert payload["best_symbol_positive_pnl_share"] > 0.8
    assert payload["best_symbol"] == "BTCUSDT"
    assert payload["concentration_blocker"] is True


def test_write_perp_reports_writes_markdown_and_json(tmp_path: Path) -> None:
    payload = {
        "candidate_statuses": [
            {
                "strategy_id": "strategy_a",
                "research_candidate": False,
                "production_candidate": False,
                "promotion_eligible": False,
                "blockers": ["missing_or_high_pbo", "free_data_research_only"],
            }
        ],
        "pbo": {"status": "computed", "pbo_probability": 0.5, "split_count": 6},
    }

    outputs = write_perp_reports(tmp_path, payload)

    assert outputs["summary_json"].exists()
    assert outputs["summary_markdown"].exists()
    assert outputs["failure_report"].exists()
    saved = json.loads(outputs["summary_json"].read_text())
    assert saved["pbo"]["pbo_probability"] == 0.5
    assert "strategy_a" in outputs["summary_markdown"].read_text()
