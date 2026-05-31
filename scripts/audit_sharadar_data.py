"""Audit ingested Sharadar tables against the kill criterion (no strategy code).

Runs the 8-name delisted probe, CIK-mapping-loss vs the EDGAR 727-company universe,
window coverage (2010-2022, 2019-2023), corporate-actions summary, and a
return-panel no-drop check. If no tables are present, writes a TEMPLATE report
listing exactly what it will check after acquisition.

Usage:
    PYTHONPATH=src uv run python scripts/audit_sharadar_data.py --data-dir data/raw/sharadar
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.data.sharadar.audit import (
    EIGHT_NAMES,
    actions_summary,
    cik_mapping_loss,
    coverage_check,
    eight_name_probe,
)
from quant_research_stack.data.sharadar.loaders import load_all
from quant_research_stack.data.sharadar.return_panel import build_return_panel

console = Console()
_OUT = Path("reports/data/sharadar/sharadar_data_audit.md")
_WINDOWS = {"edgar_10k_2010_2022": ("2010-01-01", "2022-12-31"),
            "options_iv_2019_2023": ("2019-10-14", "2023-07-28")}


def _edgar_universe(edgar_dir: str) -> pl.DataFrame:
    files = sorted(glob.glob(f"{edgar_dir}/data/*.parquet"))
    if not files:
        return pl.DataFrame({"cik": [], "company": []})
    df = pl.concat([pl.read_parquet(f, columns=["cik", "company"]) for f in files], how="diagonal_relaxed")
    return df.unique(subset=["cik"])


def _template_report() -> str:
    names = ", ".join(EIGHT_NAMES)
    return "\n".join([
        "# Sharadar Data Audit — TEMPLATE (no data present)",
        "",
        f"**Built:** {datetime.now(UTC).isoformat()}",
        "No Sharadar tables found. After acquisition + `ingest_sharadar.py`, this audit will check:",
        "",
        "1. **Delisted-name probe** for: " + names + " — found?, permaticker, isdelisted, last price date,"
        " delisting/merger action, final-return computable.",
        "2. **CIK-mapping loss** vs the EDGAR 727-company universe (direct `cik` field if present, else a"
        " degraded name bridge) — kill criterion: ≥90% mapped.",
        "3. **Window coverage**: 2010-2022 (EDGAR 10-K) and 2019-10-14..2023-07-28 (options-IV).",
        "4. **Corporate actions**: splits, dividends, delistings present.",
        "5. **Return panel**: builds a date×instrument panel and asserts NO delisted name is dropped and NO"
        " future-survival filter is applied.",
        "6. **License**: `license_local_research_use` must be operator-confirmed true in the manifest.",
        "",
        "Kill criterion (all must pass to justify the purchase): delisted names present; survivorship-safe"
        " returns/fields; ticker changes + actions; ≥90% CIK mapping; window coverage; local-research license.",
        "",
        "_No purchase, no strategy code. Next external action: §6 feasibility check on a free sample._",
    ]) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit Sharadar tables vs kill criterion (no strategy code)")
    p.add_argument("--data-dir", default="data/raw/sharadar", type=Path)
    p.add_argument("--edgar-dir", default="data/raw/huggingface/jlohding__sp500-edgar-10k")
    p.add_argument("--manifest", default="manifests/sharadar/sharadar_data_manifest.json", type=Path)
    p.add_argument("--out", default=_OUT, type=Path)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_all(args.data_dir)
    if "SEP" not in loaded and "TICKERS" not in loaded:
        out.write_text(_template_report())
        console.print(f"[yellow]No Sharadar tables — wrote TEMPLATE audit[/yellow] -> {out}")
        return 0

    sep = loaded["SEP"].df if "SEP" in loaded else None
    tickers = loaded["TICKERS"].df if "TICKERS" in loaded else None
    actions = loaded["ACTIONS"].df if "ACTIONS" in loaded else None
    edgar = _edgar_universe(args.edgar_dir)

    probe = eight_name_probe(tickers, actions, sep)
    mapping = cik_mapping_loss(tickers, edgar)
    coverage = coverage_check(sep, _WINDOWS)
    actions_sum = actions_summary(actions)
    panel_ok, panel_note = True, "skipped (no SEP)"
    if sep is not None:
        try:
            panel = build_return_panel(sep, tickers=tickers, actions=actions)
            panel_ok = panel["ticker"].n_unique() == sep["ticker"].n_unique()
            panel_note = f"{panel.height:,} rows, {panel['ticker'].n_unique()} tickers (no-drop={panel_ok})"
        except Exception as exc:
            panel_ok, panel_note = False, f"ERROR {exc}"

    lic = None
    if Path(args.manifest).exists():
        lic = json.loads(Path(args.manifest).read_text()).get("license_local_research_use")

    found = sum(1 for r in probe if r["found"])
    kill = {
        "delisted_names_present": found >= 6,
        "survivorship_safe_returns": sep is not None and ("closeadj" in sep.columns or "dividends" in sep.columns),
        "ticker_changes_and_actions": actions_sum.get("has_delistings", False) and actions_sum.get("has_splits", False),
        "cik_mapping_ge_90pct": mapping["passes_90pct"],
        "window_coverage": all(v.get("covered") for v in coverage.values()),
        "license_local_research": lic is True,
        "return_panel_no_drop": panel_ok,
    }
    passes = all(kill.values())

    lines = [
        "# Sharadar Data Audit",
        f"\n**Built:** {datetime.now(UTC).isoformat()} | tables: {sorted(loaded)}",
        f"\n## Kill criterion: **{'PASS' if passes else 'FAIL'}**",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in kill.items()],
        "\n## 8-name delisted probe",
        "| ticker | found | permaticker | isdelisted | last price | delisting action | final ret computable |",
        "|---|:---:|---|:---:|---|---|:---:|",
        *[f"| {r['ticker']} | {r['found']} | {r['permaticker']} | {r['isdelisted']} | {r['last_price_date']} | "
          f"{(r['delisting_action'] or {}).get('action') if r['delisting_action'] else None} | "
          f"{r['final_return_computable']} |" for r in probe],
        f"\n## CIK mapping ({mapping['method']})",
        f"- EDGAR companies {mapping['total_edgar']}, mapped {mapping['mapped']} ({mapping['mapped_pct']}%), "
        f"≥90% pass: **{mapping['passes_90pct']}**. unmapped examples: {mapping['unmapped_examples'][:5]}",
        "\n## Window coverage",
        *[f"- {w}: covered={v.get('covered')} (data {v.get('data_min')}..{v.get('data_max')}, "
          f"symbols_in_window={v.get('symbols_in_window')})" for w, v in coverage.items()],
        f"\n## Corporate actions\n- by_action: {actions_sum.get('by_action')}",
        f"- splits={actions_sum.get('has_splits')} dividends={actions_sum.get('has_dividends')} "
        f"delistings={actions_sum.get('has_delistings')}",
        f"\n## Return panel (no-drop check)\n- {panel_note}",
        f"\n## License\n- license_local_research_use (operator-confirmed in manifest): **{lic}**",
        "\n## Decision",
        ("- **PASS** — kill criterion satisfied. Proceed to the first post-purchase experiment (10-Q assembly+audit"
         " else options-IV cross-section v1). Strategy code only after this audit passes."
         if passes else
         "- **FAIL** — one or more kill-criterion items unmet. Do NOT buy / do NOT proceed; address the ❌ items."),
        "\n_No strategy code, no backtest, no promotion claims. Paid data is infrastructure, not alpha._",
    ]
    out.write_text("\n".join(lines) + "\n")
    console.print(f"[bold]kill criterion {'PASS' if passes else 'FAIL'}[/bold] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
