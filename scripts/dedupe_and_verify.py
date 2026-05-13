from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from quant_research_stack.artifacts import bytes_to_gb, read_yaml, write_json

console = Console()


SHA_CHUNK = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def classify_bucket(path: Path, roots: dict[str, Path]) -> str:
    resolved = path.resolve()
    for bucket, root in roots.items():
        try:
            resolved.relative_to(root.resolve())
            return bucket
        except ValueError:
            continue
    return "other"


def group_duplicates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(item["sha256"], []).append(item["path"])
    return [{"sha256": sha, "paths": sorted(paths)} for sha, paths in groups.items() if len(paths) > 1]


def build_inventory(roots: dict[str, Path]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    by_bucket: dict[str, int] = dict.fromkeys(roots, 0)
    by_bucket["other"] = 0
    for bucket, root in roots.items():
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            size = file_path.stat().st_size
            sha = file_sha256(file_path)
            items.append(
                {
                    "path": str(file_path),
                    "size_bytes": size,
                    "sha256": sha,
                    "bucket": bucket,
                }
            )
            by_bucket[bucket] += size
    total = sum(by_bucket.values())
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_size_bytes": total,
        "total_size_gb": bytes_to_gb(total),
        "by_bucket": by_bucket,
        "by_bucket_gb": {bucket: bytes_to_gb(size) for bucket, size in by_bucket.items()},
        "items": items,
        "duplicates": group_duplicates(items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk corpus roots, compute SHA256, emit inventory + duplicates.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--report", default="reports/corpus_inventory.json")
    parser.add_argument("--duplicates-report", default="reports/duplicates_to_remove.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    roots = {
        "hf_datasets": Path(paths["raw_hf_root"]),
        "kaggle": Path(paths["raw_kaggle_root"]),
        "models": Path(paths["model_root"]),
        "papers_and_derived": Path(paths["raw_paper_root"]),
    }
    inventory = build_inventory(roots)
    write_json(args.report, inventory)
    write_json(args.duplicates_report, {"duplicates": inventory["duplicates"]})
    console.print(f"Wrote {args.report} ({inventory['total_size_gb']} GB across {len(inventory['items'])} files).")
    console.print(f"Duplicate groups: {len(inventory['duplicates'])}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
