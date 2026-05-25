from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def _frame_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], *, columns: list[str] | None = None, limit: int = 20) -> str:
    if not rows:
        return "_No rows._\n"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(column)) for column in cols) + " |"
        for row in rows[:limit]
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_research_outputs(
    *,
    output_dir: Path,
    registry: pl.DataFrame,
    all_backtests: pl.DataFrame,
    pbo_payload: dict[str, Any],
    best_candidates: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    cost_sensitivity_rows: list[dict[str, Any]],
    failure_reasons: list[str],
    commands: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.write_parquet(output_dir / "strategy_registry.parquet")
    all_backtests.write_parquet(output_dir / "all_backtests.parquet")
    _write_json(output_dir / "pbo_report.json", pbo_payload)

    pbo_md = [
        "# PBO Report",
        "",
        f"- PBO: {_format_value(pbo_payload.get('pbo'))}",
        f"- Bucket: {_format_value(pbo_payload.get('pbo_bucket'))}",
        f"- Strategies tested: {_format_value(pbo_payload.get('strategy_count'))}",
        f"- Chronological blocks: {_format_value(pbo_payload.get('block_count'))}",
        f"- CSCV splits: {_format_value(pbo_payload.get('split_count'))}",
        "",
        "A strategy is rejected when the in-sample winner frequently ranks poorly out of sample.",
        "",
    ]
    (output_dir / "pbo_report.md").write_text("\n".join(pbo_md))

    candidate_columns = [
        "strategy_id",
        "family",
        "execution_profile",
        "period",
        "net_total_return",
        "net_daily_sharpe",
        "delay_net_total_return",
        "delay_net_daily_sharpe",
        "max_drawdown",
        "trade_count",
        "pass_gate",
    ]
    best_md = [
        "# Best Candidates",
        "",
        "Candidates are ranked on validation data before the permanent holdout is evaluated.",
        "",
        _markdown_table(best_candidates, columns=[c for c in candidate_columns if any(c in row for row in best_candidates)]),
        "",
        "## Reproduction Commands",
        "",
        *[f"- `{command}`" for command in commands],
        "",
    ]
    (output_dir / "best_candidates_report.md").write_text("\n".join(best_md))

    cost_md = [
        "# Cost Sensitivity Report",
        "",
        "Finalists are rerun under cost multipliers and delayed execution stress tests.",
        "",
        _markdown_table(
            cost_sensitivity_rows,
            columns=[
                column
                for column in [
                    "strategy_id",
                    "stress",
                    "execution_profile",
                    "cost_multiplier",
                    "execution_delay_bars",
                    "net_total_return",
                    "net_daily_sharpe",
                    "trade_count",
                ]
                if any(column in row for row in cost_sensitivity_rows)
            ],
            limit=100,
        ),
        "",
    ]
    (output_dir / "cost_sensitivity_report.md").write_text("\n".join(cost_md))

    holdout_md = [
        "# Holdout Report",
        "",
        "The holdout period is evaluated only for finalists selected from validation and PBO diagnostics.",
        "",
        _markdown_table(holdout_rows, columns=[c for c in candidate_columns if any(c in row for row in holdout_rows)]),
        "",
    ]
    (output_dir / "holdout_report.md").write_text("\n".join(holdout_md))

    if holdout_rows:
        _frame_from_rows(holdout_rows).write_parquet(output_dir / "holdout_backtests.parquet")
    if cost_sensitivity_rows:
        _frame_from_rows(cost_sensitivity_rows).write_parquet(output_dir / "cost_sensitivity.parquet")

    failure_md = [
        "# Failure Report",
        "",
        "No strategy is promoted unless it passes the predefined promotion gate after costs and overfitting controls.",
        "",
    ]
    if failure_reasons:
        failure_md.extend(f"- {reason}" for reason in failure_reasons)
    else:
        failure_md.append("- At least one candidate passed the configured gate.")
    failure_md.append("")
    (output_dir / "failure_report.md").write_text("\n".join(failure_md))
