from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from quant_research_stack.artifacts import read_yaml, write_json
from quant_research_stack.budget import load_artifact_budget, path_size_report

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report local artifact usage against the configured cap.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--report", default="reports/artifact_budget.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    budget = load_artifact_budget(config)
    rows = path_size_report(config)
    table = Table(title="Artifact Budget")
    table.add_column("Path")
    table.add_column("Exists")
    table.add_column("GB", justify="right")
    for row in rows:
        table.add_row(row["path"], str(row["exists"]), f"{row['size_gb']:.4f}")
    console.print(table)
    console.print(budget.as_dict())
    write_json(args.report, {"summary": budget.as_dict(), "paths": rows})
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
