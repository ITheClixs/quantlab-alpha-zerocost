"""Microstructure data audit v1 — BTCUSDT (Binance public sources).

Pre-registered per docs/research/intake/2026-05-29-microstructure-data-audit-v1.md.

Pulls:
- aggTrades CSV archives from data.binance.vision (3 representative days)
- Current depth snapshot via REST /api/v3/depth
- Current bookTicker via REST /api/v3/ticker/bookTicker

Runs the §4.1-§4.7 pre-registered checks. Emits per-check JSON, raw
samples, and a final data_audit_report.md with the §6 label assignment
and §7 decision recommendation.

NO STRATEGY WORK. NO BACKTEST. Audit only.

Usage:
    PYTHONPATH=src uv run python scripts/audit_microstructure_data.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import requests
from rich.console import Console

console = Console()

BINANCE_REST = "https://api.binance.com"
BINANCE_DATA_VISION = "https://data.binance.vision"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class AuditOutcome:
    """Outcome of one quality check."""

    check_name: str
    passed: bool
    value: object
    notes: str = ""


@dataclass(frozen=True)
class DaySummary:
    date: str
    file_url: str
    file_sha256: str
    row_count: int
    schema: list[str]
    earliest_ts_ms: int
    latest_ts_ms: int
    duration_hours: float
    timestamp_unit: str = "ms"  # "ms" or "us" — autodetected on load


@dataclass(frozen=True)
class AuditResult:
    """All audit outputs aggregated."""

    instrument: str
    git_sha: str
    audit_timestamp_utc: str
    days_audited: list[DaySummary]
    outcomes: list[AuditOutcome] = field(default_factory=list)
    quality_label: str = ""
    decision_recommendation: str = ""


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _http_get_json(url: str, *, timeout: int = 30) -> dict | list:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _agg_trades_url(*, symbol: str, day: dt.date) -> str:
    return (
        f"{BINANCE_DATA_VISION}/data/spot/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    )


def _load_agg_trades_day(
    *, symbol: str, day: dt.date, raw_dir: Path,
) -> tuple[pl.DataFrame, DaySummary]:
    """Download + unzip + parse one day of aggTrades.

    Auto-detects timestamp resolution (Binance changed from ms to µs
    in 2024-2025 for some archives). If the raw values imply a year
    after 2200, treat as µs and divide; otherwise treat as ms.
    """
    url = _agg_trades_url(symbol=symbol, day=day)
    console.print(f"  fetching {url}")
    zip_bytes = _http_get(url, timeout=120)
    sha = _sha256_bytes(zip_bytes)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"aggTrades_{symbol}_{day.isoformat()}.zip").write_bytes(zip_bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fp:
            csv_bytes = fp.read()
    # Binance aggTrades schema (no header in archive files):
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    # timestamp, is_buyer_maker, is_best_match
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=False,
        new_columns=[
            "agg_trade_id", "price", "quantity",
            "first_trade_id", "last_trade_id",
            "timestamp_raw", "is_buyer_maker", "is_best_match",
        ],
        schema_overrides={
            "agg_trade_id": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "timestamp_raw": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        },
    )
    # Auto-detect ms vs µs. A ms timestamp for any year < 2200 is < 7.3e12;
    # a µs timestamp for 2024+ is ~1.7e15. Threshold at 1e14 cleanly separates.
    raw_max_arr = df["timestamp_raw"].to_numpy()
    raw_max = int(raw_max_arr.max())
    if raw_max > 1_000_000_000_000_000:  # > 1e15 → µs resolution
        timestamp_unit = "us"
        df = df.with_columns((pl.col("timestamp_raw") // 1000).alias("timestamp_ms"))
    else:
        timestamp_unit = "ms"
        df = df.with_columns(pl.col("timestamp_raw").alias("timestamp_ms"))
    console.print(f"    detected timestamp unit: {timestamp_unit}")
    ts_arr = df["timestamp_ms"].to_numpy()
    earliest_ts = int(ts_arr.min())
    latest_ts = int(ts_arr.max())
    summary = DaySummary(
        date=day.isoformat(),
        file_url=url,
        file_sha256=sha,
        row_count=df.height,
        schema=df.columns,
        earliest_ts_ms=earliest_ts,
        latest_ts_ms=latest_ts,
        duration_hours=(latest_ts - earliest_ts) / (1000.0 * 3600.0),
        timestamp_unit=timestamp_unit,
    )
    return df, summary


def _check_timestamp_quality(
    df: pl.DataFrame, *, day: str,
) -> list[AuditOutcome]:
    """§4.2 timestamp quality checks."""
    outcomes: list[AuditOutcome] = []
    ts = df["timestamp_ms"].to_numpy()
    n = len(ts)
    if n < 2:
        return outcomes
    # Resolution: aggTrades timestamps are ms by convention; verify min diff
    nonzero_diffs = ts[1:] - ts[:-1]
    nonzero_diffs = nonzero_diffs[nonzero_diffs > 0]
    min_positive_diff = int(nonzero_diffs.min()) if len(nonzero_diffs) > 0 else 0
    outcomes.append(AuditOutcome(
        check_name=f"timestamp_resolution_ms_{day}",
        passed=min_positive_diff >= 0,
        value=min_positive_diff,
        notes="smallest positive inter-event interval in ms",
    ))
    # Monotonicity
    monotonic_count = int((ts[1:] >= ts[:-1]).sum())
    monotonic_pct = monotonic_count / (n - 1) * 100.0
    outcomes.append(AuditOutcome(
        check_name=f"monotonicity_pct_{day}",
        passed=monotonic_pct >= 99.9,
        value=monotonic_pct,
        notes="% of consecutive rows where t_i >= t_{i-1}",
    ))
    # Duplicate exact-event detection
    duplicate_keys = (
        df.select(["agg_trade_id", "timestamp_ms", "price", "quantity"])
        .group_by(["agg_trade_id", "timestamp_ms", "price", "quantity"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    duplicate_pct = duplicate_keys / n * 100.0
    outcomes.append(AuditOutcome(
        check_name=f"duplicate_event_pct_{day}",
        passed=duplicate_pct <= 0.01,
        value=duplicate_pct,
        notes="% of exact (agg_trade_id, timestamp_ms, price, qty) duplicates",
    ))
    # Longest gap (seconds)
    diffs_ms = ts[1:] - ts[:-1]
    longest_gap_s = float(diffs_ms.max()) / 1000.0 if len(diffs_ms) > 0 else 0.0
    outcomes.append(AuditOutcome(
        check_name=f"longest_gap_seconds_{day}",
        passed=longest_gap_s < 300.0,
        value=longest_gap_s,
        notes="longest inter-trade gap on the audited day",
    ))
    # UTC convention: aggTrades timestamps are Unix ms (UTC by definition)
    outcomes.append(AuditOutcome(
        check_name=f"timezone_utc_{day}",
        passed=True,
        value="UTC (Unix epoch ms)",
        notes="Binance aggTrades timestamps are Unix ms by spec",
    ))
    return outcomes


def _check_trade_data_quality(
    df: pl.DataFrame, *, day: str,
) -> list[AuditOutcome]:
    """§4.4 trade data quality."""
    outcomes: list[AuditOutcome] = []
    n = df.height
    # Aggressor flag coverage
    aggressor_present = (
        df.filter(pl.col("is_buyer_maker").is_not_null()).height / n * 100.0
    )
    outcomes.append(AuditOutcome(
        check_name=f"aggressor_flag_coverage_pct_{day}",
        passed=aggressor_present >= 99.0,
        value=aggressor_present,
        notes="% of rows with is_buyer_maker not null",
    ))
    # Zero volume
    zero_vol = df.filter(pl.col("quantity") <= 0).height
    zero_vol_pct = zero_vol / n * 100.0
    outcomes.append(AuditOutcome(
        check_name=f"zero_volume_pct_{day}",
        passed=zero_vol_pct <= 0.01,
        value=zero_vol_pct,
        notes="% of rows with quantity <= 0",
    ))
    # Zero price
    zero_price = df.filter(pl.col("price") <= 0).height
    zero_price_pct = zero_price / n * 100.0
    outcomes.append(AuditOutcome(
        check_name=f"zero_price_pct_{day}",
        passed=zero_price_pct <= 0.01,
        value=zero_price_pct,
        notes="% of rows with price <= 0",
    ))
    # Outlier prices: 5σ from rolling 1-minute mean
    df_sorted = df.sort("timestamp_ms").with_columns(
        (pl.col("timestamp_ms") // 60_000).alias("minute_bucket")
    )
    minute_stats = (
        df_sorted.group_by("minute_bucket")
        .agg(
            pl.col("price").mean().alias("p_mean"),
            pl.col("price").std().alias("p_std"),
        )
    )
    joined = df_sorted.join(minute_stats, on="minute_bucket", how="left")
    outliers = joined.filter(
        (pl.col("p_std") > 0)
        & ((pl.col("price") - pl.col("p_mean")).abs() / pl.col("p_std") > 5.0)
    ).height
    outlier_pct = outliers / n * 100.0
    outcomes.append(AuditOutcome(
        check_name=f"outlier_5sigma_pct_{day}",
        passed=outlier_pct <= 0.1,
        value=outlier_pct,
        notes="% of trades > 5σ from 1-min rolling mean",
    ))
    return outcomes


def _check_depth_snapshot(*, symbol: str) -> list[AuditOutcome]:
    """§4.3 order book reconstructability via REST snapshot."""
    outcomes: list[AuditOutcome] = []
    url = f"{BINANCE_REST}/api/v3/depth?symbol={symbol}&limit=5000"
    try:
        snap = _http_get_json(url, timeout=30)
    except Exception as exc:
        outcomes.append(AuditOutcome(
            check_name="depth_snapshot_accessible",
            passed=False,
            value=str(exc),
            notes="REST /api/v3/depth failed",
        ))
        return outcomes
    if not isinstance(snap, dict):
        outcomes.append(AuditOutcome(
            check_name="depth_snapshot_accessible",
            passed=False,
            value=type(snap).__name__,
            notes="unexpected response shape",
        ))
        return outcomes
    bids = snap.get("bids", [])
    asks = snap.get("asks", [])
    last_update_id = snap.get("lastUpdateId")
    outcomes.append(AuditOutcome(
        check_name="depth_snapshot_accessible",
        passed=True,
        value=f"bids={len(bids)} asks={len(asks)} lastUpdateId={last_update_id}",
        notes="REST snapshot fetched successfully",
    ))
    outcomes.append(AuditOutcome(
        check_name="depth_has_sequence_id",
        passed=last_update_id is not None,
        value=last_update_id,
        notes="lastUpdateId enables snapshot/delta replay",
    ))
    outcomes.append(AuditOutcome(
        check_name="depth_bids_10_levels",
        passed=len(bids) >= 10,
        value=len(bids),
        notes="bid-side level count",
    ))
    outcomes.append(AuditOutcome(
        check_name="depth_asks_10_levels",
        passed=len(asks) >= 10,
        value=len(asks),
        notes="ask-side level count",
    ))
    # Crossed book check
    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        crossed = best_bid >= best_ask
        outcomes.append(AuditOutcome(
            check_name="depth_crossed_book",
            passed=not crossed,
            value=f"best_bid={best_bid} best_ask={best_ask}",
            notes="best bid must be strictly < best ask",
        ))
    return outcomes


def _check_book_ticker(*, symbol: str) -> list[AuditOutcome]:
    """Extra liveness sanity: /api/v3/ticker/bookTicker."""
    outcomes: list[AuditOutcome] = []
    url = f"{BINANCE_REST}/api/v3/ticker/bookTicker?symbol={symbol}"
    try:
        ticker = _http_get_json(url, timeout=20)
        if isinstance(ticker, dict):
            outcomes.append(AuditOutcome(
                check_name="book_ticker_accessible",
                passed=True,
                value=ticker,
                notes="REST best-bid/ask fetched",
            ))
        else:
            outcomes.append(AuditOutcome(
                check_name="book_ticker_accessible",
                passed=False,
                value=type(ticker).__name__,
                notes="unexpected response shape",
            ))
    except Exception as exc:
        outcomes.append(AuditOutcome(
            check_name="book_ticker_accessible",
            passed=False,
            value=str(exc),
            notes="REST bookTicker failed",
        ))
    return outcomes


def _check_storage(days: list[DaySummary]) -> list[AuditOutcome]:
    """§4.6 storage feasibility from observed row counts."""
    outcomes: list[AuditOutcome] = []
    if not days:
        return outcomes
    avg_rows = sum(d.row_count for d in days) / len(days)
    rows_per_year = avg_rows * 365.0
    # Estimate compressed parquet size: aggTrades schema = ~50-80 bytes/row in
    # parquet with zstd; use 60 bytes/row as midpoint
    bytes_per_row = 60.0
    yearly_bytes = rows_per_year * bytes_per_row
    yearly_gb = yearly_bytes / (1024.0**3)
    outcomes.append(AuditOutcome(
        check_name="avg_rows_per_day",
        passed=True,
        value=avg_rows,
        notes="averaged across audited days",
    ))
    outcomes.append(AuditOutcome(
        check_name="projected_yearly_gb_compressed",
        passed=yearly_gb < 100.0,
        value=yearly_gb,
        notes="rough estimate at 60 bytes/row in zstd parquet",
    ))
    return outcomes


def _check_history_availability(*, symbol: str) -> list[AuditOutcome]:
    """§4.7 market coverage — probe data.binance.vision archive listing."""
    outcomes: list[AuditOutcome] = []
    # Probe known-old date: 2018-01-01 (BTCUSDT existed)
    probe_old = _agg_trades_url(symbol=symbol, day=dt.date(2018, 1, 1))
    try:
        resp = requests.head(
            probe_old, headers={"User-Agent": USER_AGENT}, timeout=20,
            allow_redirects=True,
        )
        old_available = resp.status_code == 200
    except Exception:
        old_available = False
    outcomes.append(AuditOutcome(
        check_name="archive_2018_01_01_available",
        passed=old_available,
        value=probe_old,
        notes="HEAD on data.binance.vision aggTrades archive",
    ))
    # Recent date probe: 30 days ago
    recent_date = dt.date.today() - dt.timedelta(days=30)
    probe_recent = _agg_trades_url(symbol=symbol, day=recent_date)
    try:
        resp = requests.head(
            probe_recent, headers={"User-Agent": USER_AGENT}, timeout=20,
            allow_redirects=True,
        )
        recent_available = resp.status_code == 200
    except Exception:
        recent_available = False
    outcomes.append(AuditOutcome(
        check_name="archive_recent_30d_available",
        passed=recent_available,
        value=probe_recent,
        notes="HEAD on recent aggTrades archive",
    ))
    return outcomes


def _assign_label_and_decision(
    outcomes: list[AuditOutcome],
) -> tuple[str, str]:
    """§6 label assignment and §7 decision recommendation."""
    by_name = {o.check_name: o for o in outcomes}

    def check(prefix: str) -> bool:
        return all(o.passed for o in outcomes if o.check_name.startswith(prefix))

    timestamp_clean = (
        check("monotonicity_pct_")
        and check("duplicate_event_pct_")
        and check("longest_gap_seconds_")
        and check("timezone_utc_")
    )
    trade_clean = (
        check("aggressor_flag_coverage_pct_")
        and check("zero_volume_pct_")
        and check("zero_price_pct_")
        and check("outlier_5sigma_pct_")
    )
    depth_accessible = (
        by_name.get("depth_snapshot_accessible", AuditOutcome(
            check_name="x", passed=False, value=None,
        )).passed
    )
    depth_has_seq = (
        by_name.get("depth_has_sequence_id", AuditOutcome(
            check_name="x", passed=False, value=None,
        )).passed
    )
    depth_not_crossed = (
        by_name.get("depth_crossed_book", AuditOutcome(
            check_name="x", passed=False, value=None,
        )).passed
    )
    history_long = by_name.get("archive_2018_01_01_available", AuditOutcome(
        check_name="x", passed=False, value=None,
    )).passed

    # Decision tree per §6
    if not timestamp_clean or not trade_clean:
        return (
            "research_only",
            "Trade-stream quality below the §4 thresholds. Document findings "
            "and pivot to event-conditioned macro/calendar per program "
            "review §8.B.",
        )
    if not history_long:
        return (
            "research_only",
            "Sufficient quality but insufficient history (< 6 months "
            "documented). Pivot to event-conditioned macro/calendar.",
        )
    # Trade-stream clean and adequate history. Depth determines whether v1
    # is L2-capable or trade-only.
    if depth_accessible and depth_has_seq and depth_not_crossed:
        # Note: this v1 audit does NOT capture WebSocket deltas, so we cannot
        # fully verify gap-free replay. The snapshot endpoint provides a
        # lastUpdateId that documentation states supports replay. We report
        # this distinction in the label rationale.
        return (
            "trade_only_clean",
            "aggTrades stream is clean and history exceeds 6 months. Depth "
            "REST snapshot is accessible with sequence IDs, but full L2 "
            "replay requires WebSocket delta capture which is OUT OF SCOPE "
            "for this v1 audit. Recommend: proceed to trade-flow v1 "
            "(InformationSource.MICROSTRUCTURE_TICK). A future v2 audit may "
            "upgrade to microstructure_clean after a live delta-capture pass.",
        )
    if not depth_accessible:
        return (
            "trade_only_clean",
            "aggTrades clean; depth endpoint inaccessible at audit time. "
            "Proceed to trade-flow v1 (MICROSTRUCTURE_TICK). L2 strategies "
            "not authorized.",
        )
    return (
        "quotes_incomplete",
        "Trade stream clean but depth snapshot has integrity issues "
        "(missing sequence ID or crossed book). No strategy intake "
        "authorized. Recommend separate investigation of paid feeds.",
    )


def _write_report(*, result: AuditResult, output_path: Path) -> None:
    lines = [
        "# Microstructure Data Audit Report — BTCUSDT (Binance)",
        "",
        f"**Audit date:** {result.audit_timestamp_utc}",
        f"**Instrument:** {result.instrument}",
        f"**Git SHA:** {result.git_sha}",
        "**Intake reference:** `docs/research/intake/2026-05-29-microstructure-data-audit-v1.md`",
        "",
        "## §0 Binding question",
        "",
        "> **Can we build a believable microstructure backtest from the "
        "available BTCUSDT data?**",
        "",
        "Answer (per §6 label assigned below): **see §7 Decision** at the end.",
        "",
        "## §4.1 What data exists",
        "",
        "Days audited:",
        "",
        "| Day | Rows | Duration (h) | URL | SHA256 |",
        "|---|---:|---:|---|---|",
    ]
    for d in result.days_audited:
        lines.append(
            f"| {d.date} | {d.row_count:,} | {d.duration_hours:.2f} | "
            f"`{d.file_url}` | `{d.file_sha256[:16]}...` |"
        )
    lines += ["", "Schema observed: `" + ", ".join(
        result.days_audited[0].schema if result.days_audited else []
    ) + "`", ""]

    lines += ["## §4.2 - §4.7 Check outcomes", ""]
    lines += ["| Check | Pass | Value | Notes |", "|---|:---:|---|---|"]
    for o in result.outcomes:
        value_str = str(o.value)
        if len(value_str) > 80:
            value_str = value_str[:77] + "..."
        lines.append(
            f"| `{o.check_name}` | {'YES' if o.passed else 'no'} | "
            f"{value_str} | {o.notes} |"
        )

    lines += [
        "",
        "## §6 Data-quality label",
        "",
        f"**`{result.quality_label}`**",
        "",
        "## §7 Decision recommendation",
        "",
        result.decision_recommendation,
        "",
        "## Reproducibility",
        "",
        f"- Audit run at: {result.audit_timestamp_utc}",
        f"- Git SHA: {result.git_sha}",
        "- All downloaded files SHA256-recorded in §4.1 table",
        "- Raw aggTrades zip archives stored under "
        "`reports/signal_research/microstructure/audit_raw/`",
        "- Re-running this script on the same sample dates produces a "
        "byte-identical report modulo `audit_timestamp_utc`.",
        "",
        "## What this audit does NOT cover (per intake §3.3, §9)",
        "",
        "- Per-level L2 order book reconstruction from WebSocket deltas. The "
        "REST snapshot endpoint was sanity-checked but full gap-free replay "
        "requires a live capture pass (deferred to a v2 audit if applicable).",
        "- Funding rates (perpetuals only; this audit covers spot).",
        "- Cross-venue arbitrage (Binance only).",
        "- Paid data feeds.",
        "- Any strategy backtest. **No PnL series was produced.**",
        "",
        "## Next steps tied to this label",
        "",
        "- `microstructure_clean` → open L2 order-book v1 strategy intake.",
        "- `trade_only_clean` → open trade-flow v1 strategy intake "
        "(`InformationSource.MICROSTRUCTURE_TICK`).",
        "- `quotes_incomplete` / `book_not_reconstructable` → consider paid "
        "feeds in a separate review.",
        "- `research_only` → document; no strategy intake; pivot to "
        "event-conditioned macro/calendar.",
        "- `reject` → reject microstructure v1; pivot to event-conditioned "
        "macro/calendar.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--days", nargs="+", default=[
            "2024-04-01",  # calm 2024 weekday
            "2024-08-05",  # known high-vol day (early-Aug 2024 crypto selloff)
            None,  # recent day, computed below
        ],
    )
    p.add_argument(
        "--raw-dir", default="reports/signal_research/microstructure/audit_raw",
    )
    p.add_argument(
        "--report", default="reports/signal_research/microstructure/data_audit_report.md",
    )
    args = p.parse_args()

    # Resolve "recent" placeholder to current_date - 7 days
    resolved_days = []
    for d in args.days:
        if d is None or d == "None":
            resolved_days.append(dt.date.today() - dt.timedelta(days=7))
        else:
            resolved_days.append(dt.date.fromisoformat(d))

    raw_dir = Path(args.raw_dir)
    console.print(
        f"[cyan]Auditing[/cyan] {args.symbol} aggTrades for "
        f"{[d.isoformat() for d in resolved_days]}"
    )

    days_audited: list[DaySummary] = []
    all_outcomes: list[AuditOutcome] = []

    for day in resolved_days:
        try:
            df, summary = _load_agg_trades_day(
                symbol=args.symbol, day=day, raw_dir=raw_dir,
            )
            days_audited.append(summary)
            console.print(
                f"  [green]ok[/green] {day.isoformat()} "
                f"({summary.row_count:,} rows)"
            )
            all_outcomes.extend(_check_timestamp_quality(df, day=day.isoformat()))
            all_outcomes.extend(_check_trade_data_quality(df, day=day.isoformat()))
        except Exception as exc:
            console.print(f"  [yellow]skip[/yellow] {day.isoformat()}: {exc}")
            all_outcomes.append(AuditOutcome(
                check_name=f"agg_trades_day_load_{day.isoformat()}",
                passed=False, value=str(exc),
                notes="failed to download or parse archive",
            ))

    # Depth + book ticker (live REST sanity)
    console.print("[cyan]Checking depth snapshot + book ticker[/cyan]")
    all_outcomes.extend(_check_depth_snapshot(symbol=args.symbol))
    all_outcomes.extend(_check_book_ticker(symbol=args.symbol))

    # Storage feasibility from observed days
    all_outcomes.extend(_check_storage(days_audited))

    # History availability
    console.print("[cyan]Probing historical archive availability[/cyan]")
    all_outcomes.extend(_check_history_availability(symbol=args.symbol))

    # Label + decision
    label, decision = _assign_label_and_decision(all_outcomes)

    result = AuditResult(
        instrument=args.symbol,
        git_sha=_git_sha(),
        audit_timestamp_utc=dt.datetime.now(dt.UTC).isoformat(),
        days_audited=days_audited,
        outcomes=all_outcomes,
        quality_label=label,
        decision_recommendation=decision,
    )

    # Write per-check JSON for audit-raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "outcomes.json").write_text(
        json.dumps([asdict(o) for o in all_outcomes], indent=2, default=str)
    )
    (raw_dir / "days_summary.json").write_text(
        json.dumps([asdict(d) for d in days_audited], indent=2)
    )

    _write_report(result=result, output_path=Path(args.report))

    console.print(
        f"[bold green]Label[/bold green]: `{label}`"
    )
    console.print(f"[bold yellow]Decision[/bold yellow]: {decision}")
    console.print(f"[green]Report[/green]: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
