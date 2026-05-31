"""Ingest Sharadar tables from local files and write the data manifest.

INFRASTRUCTURE ONLY — no purchase logic, no strategy code. If no Sharadar files are
present under --data-dir, writes a TEMPLATE manifest documenting the expected shape.

Usage:
    PYTHONPATH=src uv run python scripts/ingest_sharadar.py \\
        --data-dir data/raw/sharadar [--license-local-research true]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from quant_research_stack.data.sharadar.loaders import load_all
from quant_research_stack.data.sharadar.manifest import build_manifest, template_manifest, write_manifest
from quant_research_stack.data.sharadar.schema import SchemaError

console = Console()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest Sharadar tables -> manifest (no purchase/strategy code)")
    p.add_argument("--data-dir", default="data/raw/sharadar", type=Path)
    p.add_argument("--manifest-out", default="manifests/sharadar/sharadar_data_manifest.json", type=Path)
    p.add_argument("--license-local-research", default=None,
                   help="operator-confirmed: does the license permit local research use? true/false")
    args = p.parse_args(argv)

    lic = {"true": True, "false": False}.get(str(args.license_local_research).lower(), None)
    try:
        loaded = load_all(args.data_dir)
    except SchemaError as exc:
        console.print(f"[red]schema validation failed[/red]: {exc}")
        return 2

    if not loaded:
        manifest = template_manifest()
        manifest["license_local_research_use"] = lic
        write_manifest(manifest, args.manifest_out)
        console.print(f"[yellow]No Sharadar files under {args.data_dir} — wrote TEMPLATE manifest[/yellow] "
                      f"-> {args.manifest_out}")
        return 0

    manifest = build_manifest(loaded, license_local_research=lic)
    write_manifest(manifest, args.manifest_out)
    for name, lt in loaded.items():
        m = lt.metadata
        console.print(f"  [green]{name}[/green] rows={m['rows']:,} symbols={m.get('symbol_count')} "
                      f"range={m.get('date_min')}..{m.get('date_max')} sha={m['sha256'][:12]}")
    if manifest["warnings"]:
        console.print(f"[yellow]warnings:[/yellow] {manifest['warnings']}")
    console.print(f"[bold]status={manifest['status']}[/bold] -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
