"""Generate a green/red promotion-gate report for stage promotion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_research_stack.execution.configs import load_promotion_config


def _count_audit_days(audit_root: Path) -> int:
    if not audit_root.exists():
        return 0
    return sum(1 for p in audit_root.glob("*.jsonl") if p.is_file())


def _count_kill_triggers(audit_root: Path, last_n_days: int) -> int:
    if not audit_root.exists():
        return 0
    files = sorted(audit_root.glob("*.jsonl"))[-last_n_days:]
    n = 0
    for path in files:
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") == "kill_trigger":
                n += 1
    return n


def build_report(
    from_stage: str,
    to_stage: str,
    promotion_config_path: Path,
    audit_root: Path,
    s1_metrics_path: Path | None,
    validation_parquet_dir: Path | None = None,
    validation_config_path: Path | None = None,
) -> dict[str, Any]:
    cfg = load_promotion_config(promotion_config_path)
    gate_row = cfg.paper_to_live_shadow if from_stage == "paper" else cfg.live_shadow_to_live
    audit_days = _count_audit_days(audit_root)
    kill_window_days = gate_row.no_kill_triggers_days or 14
    kill_triggers = _count_kill_triggers(audit_root, last_n_days=kill_window_days)
    gates: list[dict[str, Any]] = []

    if gate_row.min_days_in_paper:
        gates.append({
            "name": "min_days_in_paper",
            "required": gate_row.min_days_in_paper,
            "observed": audit_days,
            "passed": audit_days >= gate_row.min_days_in_paper,
        })
    if gate_row.min_days_in_live_shadow:
        gates.append({
            "name": "min_days_in_live_shadow",
            "required": gate_row.min_days_in_live_shadow,
            "observed": audit_days,
            "passed": audit_days >= gate_row.min_days_in_live_shadow,
        })
    if gate_row.no_kill_triggers_days is not None:
        gates.append({
            "name": "no_kill_triggers_days",
            "required": 0,
            "observed": kill_triggers,
            "passed": kill_triggers == 0,
        })
    if s1_metrics_path is not None and s1_metrics_path.exists():
        metrics = json.loads(s1_metrics_path.read_text())
        r2 = float(metrics.get("holdout_weighted_zero_mean_r2", 0.0))
        gates.append({
            "name": "s1_holdout_r2_above_target",
            "required": 0.012,
            "observed": r2,
            "passed": r2 >= 0.012,
        })

    # S4.1α: hit-rate gate from validation parquets (kept out of configs/promotion.yaml
    # to avoid triggering CLAUDE.md §1.13's two-person-review requirement).
    if validation_parquet_dir is not None and validation_config_path is not None:
        from quant_research_stack.validation import load_validation_config

        vcfg = load_validation_config(validation_config_path)
        files = sorted(validation_parquet_dir.glob("*.parquet"))
        last_n = files[-vcfg.window.min_trading_days:]
        if last_n:
            frames = [pl.read_parquet(p) for p in last_n]
            full = pl.concat(frames, how="diagonal")
            eligible = full.filter(
                (pl.col("predicted_dir") != 0) & (pl.col("weight") > 0)
            )
            if eligible.height > 0:
                eligible_with_hit = eligible.filter(pl.col("hit").is_not_null())
                weighted_num = float(
                    eligible_with_hit.filter(pl.col("hit")).select(pl.col("weight").sum()).item()
                )
                weighted_den = float(eligible_with_hit.select(pl.col("weight").sum()).item())
                observed = weighted_num / weighted_den if weighted_den > 0 else 0.0
                gates.append({
                    "name": "hit_rate_min",
                    "required": vcfg.thresholds.hit_rate_min,
                    "observed": observed,
                    "passed": observed >= vcfg.thresholds.hit_rate_min,
                })

    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "generated_utc": datetime.now(UTC).isoformat(),
        "all_passed": all(g["passed"] for g in gates) if gates else False,
        "gates": gates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Promotion report: {report['from_stage']} -> {report['to_stage']}",
        "",
        f"Generated: {report['generated_utc']}  ",
        f"All gates passed: **{report['all_passed']}**",
        "",
        "| Gate | Required | Observed | Passed |",
        "|---|---:|---:|---|",
    ]
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"| {gate['name']} | {gate['required']} | {gate['observed']} | {status} |")
    lines += [
        "",
        "## Operator Signature",
        "",
        "I have reviewed the gates above and authorize promotion. -- _signed name_, _date_.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate promotion-gate green/red report")
    parser.add_argument("--from-stage", choices=["paper", "live_shadow"], required=True)
    parser.add_argument("--to-stage", choices=["live_shadow", "live"], required=True)
    parser.add_argument("--promotion-config", default="configs/promotion.yaml")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--s1-metrics", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--validation-parquet-dir", default=None)
    parser.add_argument("--validation-config", default="configs/validation.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        promotion_config_path=Path(args.promotion_config),
        audit_root=Path(args.audit_root),
        s1_metrics_path=Path(args.s1_metrics) if args.s1_metrics else None,
        validation_parquet_dir=Path(args.validation_parquet_dir) if args.validation_parquet_dir else None,
        validation_config_path=Path(args.validation_config) if args.validation_config else None,
    )
    markdown = render_markdown(report)
    out = Path(args.out) if args.out else Path(f"docs/runbooks/{args.from_stage}_to_{args.to_stage}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
