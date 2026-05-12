from __future__ import annotations

import argparse
import sys

from rich.console import Console

from quant_research_stack.artifacts import read_yaml
from quant_research_stack.jane_street import run_local_baseline, write_benchmark_report

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Jane Street 2024 validation benchmark.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--report", default="reports/jane_street_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    input_root = args.input_root or f"{config['paths']['raw_kaggle_root']}/competitions/jane-street-real-time-market-data-forecasting"
    result = run_local_baseline(input_root, sample_rows=args.sample_rows, validation_fraction=args.validation_fraction)
    write_benchmark_report(result, args.report)
    console.print(result.as_dict())
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
