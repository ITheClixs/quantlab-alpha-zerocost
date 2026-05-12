from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from rich.console import Console
from rich.table import Table

from quant_research_stack.artifacts import read_yaml, safe_repo_id, write_json


console = Console()


def paper_filename(paper: dict) -> str:
    if paper.get("arxiv_id"):
        return f"arxiv_{paper['arxiv_id'].replace('/', '_')}.pdf"
    if paper.get("doi"):
        return f"doi_{safe_repo_id(paper['doi'])}.pdf"
    return f"{safe_repo_id(paper['title'])}.pdf"


def download_arxiv(arxiv_id: str, output_path: Path) -> str:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output_path)
    return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open paper PDFs from manifests/papers.yaml.")
    parser.add_argument("--manifest", default="manifests/papers.yaml")
    parser.add_argument("--output-root", default="data/raw/papers")
    parser.add_argument("--report", default="reports/paper_downloads.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_yaml(args.manifest)
    output_root = Path(args.output_root)
    rows = []
    for paper in sorted(manifest.get("papers", []), key=lambda item: int(item.get("priority", 9999))):
        output_path = output_root / paper.get("training_category", "uncategorized") / paper_filename(paper)
        row = {
            "title": paper.get("title"),
            "source": paper.get("source"),
            "arxiv_id": paper.get("arxiv_id"),
            "doi": paper.get("doi"),
            "training_category": paper.get("training_category"),
            "open_text": bool(paper.get("open_text")),
            "output_path": str(output_path),
            "status": "pending",
        }
        if not paper.get("open_text") or not paper.get("arxiv_id"):
            row["status"] = "metadata_only"
        elif output_path.exists() and not args.force:
            row["status"] = "present"
        elif args.dry_run:
            row["status"] = "would_download"
        else:
            try:
                row["download_url"] = download_arxiv(str(paper["arxiv_id"]), output_path)
                row["status"] = "downloaded"
            except (HTTPError, URLError, OSError) as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    table = Table(title="Paper Download Plan")
    table.add_column("Status")
    table.add_column("Category")
    table.add_column("Source")
    table.add_column("ID")
    table.add_column("Title")
    for row in rows:
        table.add_row(
            row["status"],
            str(row["training_category"]),
            str(row["source"]),
            str(row.get("arxiv_id") or row.get("doi") or ""),
            str(row["title"])[:70],
        )
    console.print(table)
    write_json(args.report, {"papers": rows})
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

