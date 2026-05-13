from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from quant_research_stack.artifacts import read_yaml
from quant_research_stack.local_training import SignalTrainingTask, train_task, write_training_summary

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local sampled quant signal heads on processed parquet data.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--output-root", default="experiments/local_signal_training")
    parser.add_argument("--report", default="reports/local_signal_training.json")
    parser.add_argument("--rows-per-file", type=int, default=None)
    parser.add_argument("--max-files-per-task", type=int, default=None)
    parser.add_argument("--task", action="append", choices=["market", "orderbook"], default=[])
    return parser.parse_args()


def build_tasks(args: argparse.Namespace, config: dict) -> list[SignalTrainingTask]:
    training_cfg = config["local_training"]
    processed_root = Path(args.processed_root or "data/processed")
    output_root = Path(args.output_root)
    rows_per_file = int(args.rows_per_file or training_cfg["rows_per_file"])
    max_files = args.max_files_per_task if args.max_files_per_task is not None else int(training_cfg["max_files_per_task"])
    selected = set(args.task or ["market", "orderbook"])
    specs = {
        "market": (processed_root / "market", "future_return_1"),
        "orderbook": (processed_root / "orderbook", "future_mid_return_1"),
    }
    tasks = []
    for name, (input_root, target_column) in specs.items():
        if name not in selected:
            continue
        tasks.append(
            SignalTrainingTask(
                name=name,
                input_root=input_root,
                target_column=target_column,
                rows_per_file=rows_per_file,
                max_files=max_files,
                validation_fraction=float(training_cfg["validation_fraction"]),
                ridge_alpha=float(training_cfg["ridge_alpha"]),
                hist_gradient_max_iter=int(training_cfg["hist_gradient_max_iter"]),
                output_root=output_root,
            )
        )
    return tasks


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    reports = []
    for task in build_tasks(args, config):
        console.print(f"[bold]Training[/bold] {task.name} from {task.input_root}")
        report = train_task(task)
        reports.append(report)
        console.print({k: report[k] for k in ["task", "rows", "feature_count", "best_model"]})
    write_training_summary(reports, args.report)
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
