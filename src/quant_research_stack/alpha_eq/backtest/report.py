"""Markdown report writer for the strict backtest (spec §5.16)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ReportInputs:
    run_id: str
    git_sha: str
    data_manifest_sha256: str
    data_quality_label: str
    cohort: str
    daily_returns: pl.DataFrame
    decomposition_bps: dict[str, float]
    sensitivity_rows: list[dict[str, str | float]] = field(default_factory=list)


def _banner(label: str) -> str:
    if label == "survivorship_prototype_only":
        return (
            "> ⚠️ **PROTOTYPE-ONLY** — `data_quality_label = survivorship_prototype_only`. "
            "The success gate is suspended and these results are research-only.\n\n"
        )
    if label == "partial_pit_universe":
        return (
            "> ℹ️ **Conditional research pass — `partial_pit_universe`** "
            "(NOT institutional-grade). See limitations.\n\n"
        )
    return ""


def write_report(path: Path, inputs: ReportInputs) -> None:
    out = path
    out.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append(f"# S1-EQ backtest report `{inputs.run_id}`\n\n")
    parts.append(_banner(inputs.data_quality_label))
    parts.append("## Configuration\n\n")
    parts.append(f"- `run_id`: `{inputs.run_id}`\n")
    parts.append(f"- `git_sha`: `{inputs.git_sha}`\n")
    parts.append(f"- `data_manifest_sha256`: `{inputs.data_manifest_sha256}`\n")
    parts.append(f"- `data_quality_label`: `{inputs.data_quality_label}`\n")
    parts.append(f"- `cohort`: `{inputs.cohort}`\n\n")
    parts.append("## PnL decomposition (bps/day)\n\n")
    for k, v in inputs.decomposition_bps.items():
        parts.append(f"- `{k}`: `{v:+.4f}`\n")
    parts.append("\n## Daily returns (head)\n\n```\n")
    parts.append(str(inputs.daily_returns.head(10)) + "\n```\n\n")
    if inputs.sensitivity_rows:
        parts.append("## Sensitivity sweeps\n\n")
        keys = list(inputs.sensitivity_rows[0].keys())
        parts.append("| " + " | ".join(keys) + " |\n")
        parts.append("| " + " | ".join("---" for _ in keys) + " |\n")
        for r in inputs.sensitivity_rows:
            parts.append("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
    parts.append("\n## Limitations\n\n")
    parts.append(
        "- VWAP proxy = HLC3 (labelled `vwap_proxy_hlc3`); no intraday VWAP.\n"
        "- Borrow proxy is `static_proxy_v1`; real PIT borrow feed deferred.\n"
        "- No market-impact, no MOC slippage, no factor neutrality (Phase 2).\n"
    )
    parts.append("\n---\n`not_investment_advice: true`\n")
    out.write_text("".join(parts))
