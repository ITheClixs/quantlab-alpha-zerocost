"""End-to-end equity data preparation (spec §2).

Builds the processed-equities root from a daily-bars panel + dividend feed.

Usage:
    PYTHONPATH=src uv run python scripts/prepare_equity_data.py \
        --panel data/raw/.../panel.parquet \
        --dividends data/raw/.../dividends.parquet \
        --equity-root data/processed/equities \
        [--membership-path data/raw/.../membership.parquet] \
        [--membership-source hf_primary|kaggle|wikipedia_fallback|absent_prototype_only] \
        [--exits-path data/raw/.../exits.parquet] \
        [--source-is-total-return]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.alpha_eq.data.adv import build_adv_20d_dollar
from quant_research_stack.alpha_eq.data.borrow_proxy import build_borrow_proxy
from quant_research_stack.alpha_eq.data.corporate_actions import build_three_series
from quant_research_stack.alpha_eq.data.delisting_audit import audit_delistings
from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource
from quant_research_stack.alpha_eq.data.pit_quality import (
    PITQualityInputs,
    classify_pit_quality,
)

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _read_panel(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort(["symbol", "date"])


def _read_dividends(path: Path | None) -> pl.DataFrame:
    if path is None or not Path(path).exists():
        return pl.DataFrame(
            schema={"ex_date": pl.Date, "symbol": pl.Utf8, "dividend_per_share": pl.Float64}
        )
    return pl.read_parquet(path)


def _read_exits(path: Path | None) -> pl.DataFrame:
    if path is None or not Path(path).exists():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "exit_date": pl.Date,
                "exit_reason": pl.Utf8,
                "terminal_return_known": pl.Boolean,
                "terminal_return_value": pl.Float64,
            }
        )
    return pl.read_parquet(path)


def _schema_fingerprint(df: pl.DataFrame) -> str:
    return "cols:" + ",".join(df.columns)


def _write_parquet_with_artifact(
    df: pl.DataFrame, *, root: Path, name: str
) -> tuple[str, ManifestArtifact]:
    p = root / f"{name}.parquet"
    df.write_parquet(p)
    art = ManifestArtifact(
        path=p.name,
        sha256=sha256_of_file(p),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()) if "symbol" in df.columns else 0,
        date_range_start=str(df["date"].min()) if "date" in df.columns else "",
        date_range_end=str(df["date"].max()) if "date" in df.columns else "",
        schema_fingerprint=_schema_fingerprint(df),
    )
    return name, art


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--dividends", default=None)
    p.add_argument("--equity-root", required=True)
    p.add_argument("--membership-path", default=None)
    p.add_argument(
        "--membership-source",
        default="absent_prototype_only",
        choices=[m.value for m in MembershipSource],
    )
    p.add_argument("--exits-path", default=None)
    p.add_argument("--source-is-total-return", action="store_true")
    p.add_argument("--config", default=None, help="optional unused config for Make-target compat")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.equity_root)
    root.mkdir(parents=True, exist_ok=True)

    panel = _read_panel(Path(args.panel))
    dividends = _read_dividends(Path(args.dividends) if args.dividends else None)
    exits = _read_exits(Path(args.exits_path) if args.exits_path else None)

    bundle = build_three_series(
        panel=panel, dividends=dividends, source_is_total_return=args.source_is_total_return
    )
    adv = build_adv_20d_dollar(panel)
    borrow = build_borrow_proxy(sorted(panel["symbol"].unique().to_list()))
    audit = audit_delistings(panel=panel, exits=exits)

    artifacts: dict[str, ManifestArtifact] = {}
    for df, name in (
        (bundle.tradable, "sp500_tradable_prices"),
        (bundle.split_adj, "sp500_split_adjusted_prices"),
        (bundle.total_return, "sp500_total_return_prices"),
        (dividends, "sp500_dividends"),
        (adv, "sp500_adv"),
        (borrow, "sp500_borrow_proxy"),
        (audit.audit_table, "sp500_delisting_audit"),
    ):
        key, art = _write_parquet_with_artifact(df, root=root, name=name)
        artifacts[key] = art

    membership_source = MembershipSource(args.membership_source)
    if args.membership_path and membership_source != MembershipSource.ABSENT_PROTOTYPE_ONLY:
        mem_df = pl.read_parquet(args.membership_path)
        _, mart = _write_parquet_with_artifact(mem_df, root=root, name="sp500_pit_membership")
        artifacts["sp500_pit_membership"] = mart

    unknown_in_holdout = int(audit.counters.get("unknown_exit", 0))
    label = classify_pit_quality(
        PITQualityInputs(
            membership_source=membership_source,
            audit=audit,
            unknown_exit_in_holdout=unknown_in_holdout,
        )
    )

    corporate_action_quality = (
        "vendor_total_return"
        if args.source_is_total_return
        else "split_adj_plus_external_dividends"
    )

    manifest = EquityManifest(
        pipeline_version="0.1.0",
        git_sha=_git_sha(),
        artifacts=artifacts,
        data_quality_label=label,
        corporate_action_quality=corporate_action_quality,
        borrow_source_quality="static_proxy_v1",
        pit_membership_source=membership_source.value,
        delisting_audit_quality=(
            "captured_above_threshold"
            if label == DataQualityLabel.PIT_SAFE
            else ("partial_capture" if not audit.audit_table.is_empty() else "audit_absent")
        ),
        delisting_audit_counters=DelistingAuditCounters(**audit.counters),
        build_command_line=" ".join(sys.argv),
        python_version=platform.python_version(),
        package_versions={"polars": pl.__version__},
        warnings=(
            ["dividend feed: public_snapshot_not_vendor_pit"]
            if args.dividends and "yfinance" in str(args.dividends)
            else []
        ),
    )
    write_manifest(root / "_manifest.json", manifest)
    console.print(f"[bold green]Manifest written:[/bold green] {root / '_manifest.json'}")
    console.print(f"  data_quality_label = {label.value}")
    console.print(f"  build at {datetime.utcnow().isoformat()}Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
