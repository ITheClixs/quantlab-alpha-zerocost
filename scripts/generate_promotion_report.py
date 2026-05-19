"""Generate a green/red promotion-gate report for stage promotion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        promotion_config_path=Path(args.promotion_config),
        audit_root=Path(args.audit_root),
        s1_metrics_path=Path(args.s1_metrics) if args.s1_metrics else None,
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
