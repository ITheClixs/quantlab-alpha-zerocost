"""One-shot exploration of Massive.com flat files.

Reads credentials from .env. Lists object keys for each documented prefix,
samples one recent file per prefix to inspect schema and row count.

This is an audit-grade exploration, not a strategy. Output written to
reports/signal_research/microstructure/massive_exploration.md and JSON
sidecar.

Usage:
    PYTHONPATH=src uv run python scripts/explore_massive_flatfiles.py
"""

from __future__ import annotations

import gzip
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import polars as pl
from botocore.config import Config
from dotenv import load_dotenv
from rich.console import Console

console = Console()

PREFIXES_TO_EXPLORE: tuple[str, ...] = (
    "global_crypto",
    "global_forex",
    "us_indices",
    "us_options_opra",
    "us_stocks_sip",
)

MAX_KEYS_PER_PREFIX: int = 60  # enough to characterize structure


@dataclass(frozen=True)
class PrefixSummary:
    prefix: str
    n_keys_listed: int
    sample_keys: list[str] = field(default_factory=list)
    earliest_key: str = ""
    latest_key: str = ""
    sample_file_key: str = ""
    sample_file_size_bytes: int = 0
    sample_file_rows: int = 0
    sample_file_columns: list[str] = field(default_factory=list)
    sample_file_first_rows: list[dict] = field(default_factory=list)
    error: str = ""


def _client():
    load_dotenv()
    access_key = os.environ.get("MASSIVE_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY")
    endpoint = os.environ.get(
        "MASSIVE_S3_ENDPOINT_URL", "https://files.massive.com"
    )
    if not access_key or not secret_key:
        raise SystemExit(
            "Missing MASSIVE_S3_ACCESS_KEY_ID / MASSIVE_S3_SECRET_ACCESS_KEY. "
            "Copy .env.example to .env and populate."
        )
    session = boto3.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
    )
    # NOTE (2026-05-29): the Massive endpoint only accepts *path-style* addressing.
    # boto3 defaults to virtual-hosted style against a custom endpoint, which times
    # out / 403s — that, not the credentials, caused the original audit's blanket
    # 403. With path-style addressing ListObjects works, though GetObject still 403s
    # on the free tier (downloads are a paid entitlement).
    return session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
    )


def _explore_prefix(s3, *, bucket: str, prefix: str) -> PrefixSummary:
    try:
        keys: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                if len(keys) >= MAX_KEYS_PER_PREFIX:
                    break
            if len(keys) >= MAX_KEYS_PER_PREFIX:
                break
        if not keys:
            return PrefixSummary(
                prefix=prefix, n_keys_listed=0,
                error="empty listing — prefix may not exist or be inaccessible",
            )
        # Sample a real CSV-like file from the keys (pick the most recent
        # *.csv.gz if present, else first key)
        candidates = [k for k in keys if k.endswith(".csv.gz") or k.endswith(".csv")]
        sample_key = candidates[-1] if candidates else keys[-1]
        head_obj = s3.head_object(Bucket=bucket, Key=sample_key)
        size = int(head_obj.get("ContentLength", 0))
        # Stream the first ~5 MB to avoid downloading multi-GB files
        get_obj = s3.get_object(
            Bucket=bucket, Key=sample_key, Range="bytes=0-5242879",
        )
        body = get_obj["Body"].read()
        rows_list: list[dict] = []
        columns: list[str] = []
        n_rows_observed = 0
        try:
            if sample_key.endswith(".gz"):
                raw = gzip.decompress(body) if len(body) > 10 else b""
                # Could be truncated — only parse up to last newline
                cut = raw.rfind(b"\n")
                if cut > 0:
                    raw = raw[: cut + 1]
                df = pl.read_csv(io.BytesIO(raw)) if raw else pl.DataFrame()
            elif sample_key.endswith(".csv"):
                cut = body.rfind(b"\n")
                if cut > 0:
                    body = body[: cut + 1]
                df = pl.read_csv(io.BytesIO(body))
            else:
                df = pl.DataFrame()
            if not df.is_empty():
                columns = df.columns
                n_rows_observed = df.height
                rows_list = df.head(3).to_dicts()
        except Exception as exc:
            return PrefixSummary(
                prefix=prefix, n_keys_listed=len(keys),
                sample_keys=keys[:10],
                earliest_key=keys[0], latest_key=keys[-1],
                sample_file_key=sample_key, sample_file_size_bytes=size,
                error=f"parse error on sample file: {exc}",
            )
        return PrefixSummary(
            prefix=prefix, n_keys_listed=len(keys),
            sample_keys=keys[:10],
            earliest_key=keys[0], latest_key=keys[-1],
            sample_file_key=sample_key, sample_file_size_bytes=size,
            sample_file_rows=n_rows_observed,
            sample_file_columns=columns,
            sample_file_first_rows=rows_list,
        )
    except Exception as exc:
        return PrefixSummary(
            prefix=prefix, n_keys_listed=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    s3 = _client()
    bucket = "flatfiles"
    console.print(f"[cyan]Exploring[/cyan] Massive flat files bucket={bucket!r}")
    summaries: list[PrefixSummary] = []
    for prefix in PREFIXES_TO_EXPLORE:
        console.print(f"  [yellow]prefix[/yellow] {prefix}")
        s = _explore_prefix(s3, bucket=bucket, prefix=prefix)
        if s.error:
            console.print(f"    [red]error[/red]: {s.error}")
        else:
            console.print(
                f"    [green]ok[/green] keys={s.n_keys_listed} "
                f"sample={s.sample_file_key} size={s.sample_file_size_bytes / 1e6:.1f}MB "
                f"cols={len(s.sample_file_columns)} rows~{s.sample_file_rows}"
            )
        summaries.append(s)
    out_dir = Path("reports/signal_research/microstructure")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "massive_exploration.json").write_text(
        json.dumps([s.__dict__ for s in summaries], indent=2, default=str)
    )

    lines = [
        "# Massive.com Flat Files — Exploration Report",
        "",
        "Authored to inform the trade-flow v1 intake. **No strategy work.**",
        "Credentials loaded from `.env` (gitignored). Secret never touches "
        "the repository.",
        "",
        "## Prefixes inspected",
        "",
        "| Prefix | Keys listed | Earliest key | Latest key | Sample file | Size | Cols | Rows~ | Error |",
        "|---|---:|---|---|---|---:|---:|---:|---|",
    ]
    for s in summaries:
        size_mb = s.sample_file_size_bytes / 1e6
        lines.append(
            f"| `{s.prefix}` | {s.n_keys_listed} | "
            f"`{s.earliest_key[:60]}` | `{s.latest_key[:60]}` | "
            f"`{s.sample_file_key[:60]}` | {size_mb:.1f}MB | "
            f"{len(s.sample_file_columns)} | {s.sample_file_rows} | "
            f"{s.error[:50] if s.error else ''} |"
        )
    lines.append("")
    for s in summaries:
        if s.sample_file_columns:
            lines.append(f"### `{s.prefix}` sample schema")
            lines.append("")
            lines.append("Columns: `" + ", ".join(s.sample_file_columns) + "`")
            lines.append("")
            lines.append("First 3 rows:")
            lines.append("")
            for row in s.sample_file_first_rows[:3]:
                lines.append(f"- `{row}`")
            lines.append("")
    (out_dir / "massive_exploration.md").write_text("\n".join(lines))
    console.print(
        f"[green]Wrote[/green] {out_dir / 'massive_exploration.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
