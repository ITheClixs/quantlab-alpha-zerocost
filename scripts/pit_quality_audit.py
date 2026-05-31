"""Audit an existing equity-processed root and print its data-quality label.

Usage:
    PYTHONPATH=src uv run python scripts/pit_quality_audit.py \
        --equity-root data/processed/equities
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--equity-root", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.equity_root) / "_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]No manifest at {manifest_path}[/red]")
        return 2
    m = json.loads(manifest_path.read_text())
    console.print(f"[bold]Equity root:[/bold] {args.equity_root}")
    console.print(f"  data_quality_label        = [bold]{m['data_quality_label']}[/bold]")
    console.print(f"  corporate_action_quality  = {m['corporate_action_quality']}")
    console.print(f"  borrow_source_quality     = {m['borrow_source_quality']}")
    console.print(f"  pit_membership_source     = {m['pit_membership_source']}")
    console.print(f"  delisting_audit_quality   = {m['delisting_audit_quality']}")
    print(m["data_quality_label"])  # stdout-plain for grep/test
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
