"""Unified S1 training CLI.

Replaces both alpha_train_s1.py and alpha_train_s1_streaming.py.

Usage:
    PYTHONPATH=src uv run python scripts/train_s1.py \
        --config configs/alpha.yaml \
        [--streaming] \
        [--max-rows N] \
        [--experiments-root experiments/alpha_s1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from quant_research_stack.alpha.registry import RunRegistry
from quant_research_stack.alpha.training import TrainConfig, train_s1

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantLab S1 training (unified, post-S0)")
    parser.add_argument("--config", default="configs/alpha.yaml")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="memory-limited mode (M4 24 GB target)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="override max_rows_streaming when --streaming is set",
    )
    parser.add_argument("--experiments-root", default="experiments/alpha_s1")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in config file: {path}")
    return payload


def main() -> int:
    args = parse_args()
    cfg_dict = _load_config(Path(args.config))

    if args.streaming:
        cfg_dict["streaming"] = True
    if args.max_rows is not None:
        cfg_dict["max_rows_streaming"] = args.max_rows

    config = TrainConfig.from_dict(cfg_dict)
    registry = RunRegistry(root=Path(args.experiments_root))
    result = train_s1(config, registry)

    console.print(f"[bold green]Run complete:[/bold green] {result.run_dir}")
    console.print(
        "  holdout weighted zero-mean R2: "
        f"{result.holdout_weighted_zero_mean_r2:.6f}"
    )
    console.print(f"  base models persisted: {result.base_models_persisted}")
    console.print(f"  stacker: {result.stacker_path}")
    console.print(f"  feature_cols: {result.feature_cols_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
