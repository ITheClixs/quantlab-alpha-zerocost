from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from rich.console import Console
from rich.table import Table

from quant_research_stack.artifacts import (
    GB,
    bytes_to_gb,
    folder_size,
    load_manifest_items,
    read_yaml,
    should_include,
    write_json,
)


console = Console()


def estimate_repo_size(api: HfApi, item: Any) -> tuple[int | None, list[dict[str, Any]], str | None]:
    try:
        files = list(api.list_repo_tree(item.id, repo_type=item.repo_type, recursive=True))
    except HfHubHTTPError as exc:
        return None, [], f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {exc}"

    selected: list[dict[str, Any]] = []
    total = 0
    unknown = False
    for file_info in files:
        path = getattr(file_info, "path", "")
        size = getattr(file_info, "size", None)
        if not path or not should_include(path, item.allow_patterns, item.ignore_patterns):
            continue
        selected.append({"path": path, "size_bytes": size})
        if size is None:
            unknown = True
        else:
            total += int(size)
    return (None if unknown and total == 0 else total), selected, None


def target_root(config: dict[str, Any], repo_type: str) -> Path:
    paths = config["paths"]
    if repo_type == "dataset":
        return Path(paths["raw_hf_root"])
    if repo_type == "model":
        return Path(paths["model_root"])
    raise ValueError(f"Unsupported repo_type: {repo_type}")


def load_all_items(types: set[str]) -> list[Any]:
    items = []
    if "dataset" in types:
        items.extend(load_manifest_items("manifests/datasets.yaml", "datasets"))
    if "model" in types:
        items.extend(load_manifest_items("manifests/models.yaml", "models"))
    return [item for item in items if item.enabled and item.repo_type in types]


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = read_yaml(args.config)
    api = HfApi()
    items = load_all_items(set(args.types))
    planned: list[dict[str, Any]] = []

    for item in items:
        base = target_root(config, item.repo_type)
        local_dir = base / item.local_name
        local_size = folder_size(local_dir)
        remote_size, selected_files, error = estimate_repo_size(api, item)
        planned.append(
            {
                "id": item.id,
                "repo_type": item.repo_type,
                "group": item.group,
                "priority": item.priority,
                "purpose": item.metadata.get("purpose"),
                "license_hint": item.metadata.get("license_hint"),
                "allow_patterns": list(item.allow_patterns),
                "ignore_patterns": list(item.ignore_patterns),
                "estimated_size_bytes": remote_size,
                "estimated_size_gb": bytes_to_gb(remote_size),
                "selected_file_count": len(selected_files),
                "selected_files_sample": selected_files[:12],
                "local_dir": str(local_dir),
                "local_size_bytes": local_size,
                "local_size_gb": bytes_to_gb(local_size),
                "estimate_error": error,
            }
        )

    if args.sort == "size":
        planned.sort(key=lambda row: (row["estimated_size_bytes"] is None, row["estimated_size_bytes"] or 0, row["priority"]))
    elif args.sort == "priority":
        planned.sort(key=lambda row: (row["priority"], row["estimated_size_bytes"] is None, row["estimated_size_bytes"] or 0))

    budget_gb = float(args.max_gb or config["artifact_budget"]["max_total_gb"])
    reserve_gb = float(config["artifact_budget"].get("reserve_gb_for_processed_outputs", 0))
    download_budget = max(0.0, budget_gb - reserve_gb)
    remaining = int(download_budget * GB)

    for row in planned:
        estimated = row["estimated_size_bytes"]
        already_present = row["local_size_bytes"] > 0
        if already_present and not args.force:
            row["decision"] = "skip_present"
            continue
        if estimated is None:
            row["decision"] = "skip_unknown_size"
            continue
        if estimated > remaining:
            row["decision"] = "skip_budget"
            continue
        row["decision"] = "download"
        remaining -= estimated

    summary = {
        "budget_gb": budget_gb,
        "reserve_gb_for_processed_outputs": reserve_gb,
        "download_budget_gb": download_budget,
        "planned_download_gb": bytes_to_gb(sum(row["estimated_size_bytes"] or 0 for row in planned if row["decision"] == "download")),
        "remaining_download_budget_gb": bytes_to_gb(remaining),
        "sort": args.sort,
        "types": args.types,
    }
    return summary, planned


def print_plan(summary: dict[str, Any], planned: list[dict[str, Any]]) -> None:
    table = Table(title="HF Artifact Plan")
    table.add_column("Decision")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Group")
    table.add_column("GB", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("License")
    for row in planned:
        table.add_row(
            row["decision"],
            row["repo_type"],
            row["id"],
            row["group"],
            "?" if row["estimated_size_gb"] is None else f"{row['estimated_size_gb']:.4f}",
            str(row["selected_file_count"]),
            str(row.get("license_hint") or ""),
        )
    console.print(table)
    console.print(summary)


def download(planned: list[dict[str, Any]]) -> None:
    for row in planned:
        if row["decision"] != "download":
            continue
        console.print(f"[bold]Downloading {row['repo_type']}[/bold] {row['id']} -> {row['local_dir']}")
        Path(row["local_dir"]).parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=row["id"],
            repo_type=row["repo_type"],
            local_dir=row["local_dir"],
            allow_patterns=row["allow_patterns"] or None,
            ignore_patterns=row["ignore_patterns"] or None,
            resume_download=True,
            max_workers=4,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Size-capped Hugging Face dataset/model downloader.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--types", nargs="+", choices=["dataset", "model"], default=["dataset", "model"])
    parser.add_argument("--sort", choices=["size", "priority"], default="size")
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default="reports/hf_download_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, planned = build_plan(args)
    write_json(args.report, {"summary": summary, "items": planned})
    print_plan(summary, planned)
    if args.dry_run:
        console.print(f"Dry run only. Wrote {args.report}")
        return 0
    download(planned)
    return 0


if __name__ == "__main__":
    sys.exit(main())

