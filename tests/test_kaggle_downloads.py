from __future__ import annotations

from pathlib import Path

from quant_research_stack.budget import ArtifactBudget
from quant_research_stack.kaggle_artifacts import KaggleItem
from quant_research_stack.kaggle_downloads import build_kaggle_plan, kaggle_download_command


def make_item(item_id: str, priority: int, expected_max_gb: float = 1.0, resource_type: str = "competition") -> KaggleItem:
    return KaggleItem(
        id=item_id,
        resource_type=resource_type,
        group="test",
        priority=priority,
        topics=("equity",),
        purpose="fixture",
        license_hint="kaggle",
        expected_max_gb=expected_max_gb,
        enabled=True,
    )


def test_build_kaggle_plan_respects_remaining_budget(tmp_path: Path) -> None:
    budget = ArtifactBudget(max_total_gb=1.5, hard_ceiling_gb=2, counted_paths=(), used_bytes=0)
    plan = build_kaggle_plan([make_item("first", 1, 1.0), make_item("second", 2, 1.0)], tmp_path, budget)
    assert [row.decision for row in plan] == ["download", "skip_budget"]


def test_build_kaggle_plan_skips_present_without_force(tmp_path: Path) -> None:
    item = make_item("jane-street-real-time-market-data-forecasting", 1)
    local = tmp_path / "competitions" / item.id
    local.mkdir(parents=True)
    (local / "train.parquet").write_bytes(b"data")
    budget = ArtifactBudget(max_total_gb=10, hard_ceiling_gb=11, counted_paths=(), used_bytes=0)
    plan = build_kaggle_plan([item], tmp_path, budget)
    assert plan[0].decision == "skip_present"
    assert plan[0].local_size_bytes == 4


def test_kaggle_download_command_for_competition(tmp_path: Path) -> None:
    item = make_item("jane-street-real-time-market-data-forecasting", 1)
    plan = build_kaggle_plan([item], tmp_path, ArtifactBudget(10, 11, (), 0))
    cmd = kaggle_download_command(plan[0], unzip=True)
    assert cmd[:4] == ["kaggle", "competitions", "download", "-c"]
    assert item.id in cmd
    assert "--unzip" in cmd


def test_kaggle_download_command_for_dataset(tmp_path: Path) -> None:
    item = make_item("owner/dataset", 1, resource_type="dataset")
    plan = build_kaggle_plan([item], tmp_path, ArtifactBudget(10, 11, (), 0))
    cmd = kaggle_download_command(plan[0])
    assert cmd[:4] == ["kaggle", "datasets", "download", "-d"]
    assert "owner/dataset" in cmd
