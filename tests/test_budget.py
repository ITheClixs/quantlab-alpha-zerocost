from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.budget import (
    BudgetExceededError,
    ensure_budget_available,
    load_artifact_budget,
    path_size_report,
)


def test_load_artifact_budget_counts_configured_paths(tmp_path: Path) -> None:
    first = tmp_path / "data" / "raw"
    second = tmp_path / "models"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "a.bin").write_bytes(b"a" * 10)
    (second / "b.bin").write_bytes(b"b" * 15)
    config = {
        "artifact_budget": {
            "max_total_gb": 1,
            "hard_ceiling_gb": 2,
            "counted_paths": ["data/raw", "models", "missing"],
        }
    }
    budget = load_artifact_budget(config, tmp_path)
    assert budget.used_bytes == 25
    assert budget.max_total_gb == 1
    assert len(budget.counted_paths) == 3


def test_path_size_report_marks_missing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "data"
    existing.mkdir()
    (existing / "sample.txt").write_text("abc", encoding="utf-8")
    rows = path_size_report({"artifact_budget": {"counted_paths": ["data", "missing"]}}, tmp_path)
    assert rows[0]["exists"] is True
    assert rows[0]["size_bytes"] == 3
    assert rows[1]["exists"] is False
    assert rows[1]["size_bytes"] == 0


def test_ensure_budget_available_rejects_over_budget(tmp_path: Path) -> None:
    config = {"artifact_budget": {"max_total_gb": 0.000000001, "hard_ceiling_gb": 1, "counted_paths": ["data"]}}
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.bin").write_bytes(b"x" * 100)
    budget = load_artifact_budget(config, tmp_path)
    with pytest.raises(BudgetExceededError, match="planned"):
        ensure_budget_available(budget, 1000, label="test download")
