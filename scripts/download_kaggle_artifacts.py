from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from quant_research_stack.artifacts import bytes_to_gb, read_yaml, write_json
from quant_research_stack.budget import load_artifact_budget
from quant_research_stack.kaggle_artifacts import load_kaggle_items
from quant_research_stack.kaggle_downloads import build_kaggle_plan, run_kaggle_download

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="150GB-capped Kaggle artifact downloader.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--manifest", default="manifests/kaggle.yaml")
    parser.add_argument("--item", action="append", default=[], help="Download only specific Kaggle ids.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--unzip", action="store_true")
    parser.add_argument("--report", default="reports/kaggle_download_plan.json")
    return parser.parse_args()


def print_plan(plan: list) -> None:
    table = Table(title="Kaggle Artifact Plan")
    table.add_column("Decision")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Group")
    table.add_column("Expected GB", justify="right")
    table.add_column("Local GB", justify="right")
    for row in plan:
        table.add_row(
            row.decision,
            row.item.resource_type,
            row.item.id,
            row.item.group,
            f"{bytes_to_gb(row.estimated_size_bytes):.4f}",
            f"{bytes_to_gb(row.local_size_bytes):.4f}",
        )
    console.print(table)


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    items = load_kaggle_items(args.manifest)
    if args.item:
        requested = set(args.item)
        items = [item for item in items if item.id in requested]
    budget = load_artifact_budget(config)
    raw_root = config["paths"]["raw_kaggle_root"]
    plan = build_kaggle_plan(items, raw_root, budget, force=args.force)
    payload = {"budget": budget.as_dict(), "items": [row.as_dict() for row in plan]}
    write_json(args.report, payload)
    print_plan(plan)
    console.print(f"Wrote {args.report}")
    if args.dry_run:
        return 0
    results: list[dict] = []
    for row in plan:
        record = row.as_dict()
        if row.decision == "download":
            console.print(f"[bold]Downloading[/bold] {row.item.id} -> {row.local_dir}")
            outcome = run_kaggle_download(row, unzip=args.unzip)
            record.update(outcome)
            if outcome.get("status") == "skip_rules_not_accepted":
                console.print(f"[yellow]Skipped {row.item.id}: rules must be accepted at https://www.kaggle.com/competitions/{row.item.id}/rules[/yellow]")
            elif outcome.get("status") == "error":
                console.print(f"[red]Error on {row.item.id}: {outcome.get('stderr','')[:200]}[/red]")
        results.append(record)
    payload["items"] = results
    write_json(args.report, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
