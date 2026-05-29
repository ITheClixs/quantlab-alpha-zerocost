from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _candidate_lines(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["No candidate statuses supplied."]
    lines = ["| strategy | research | production | blockers |", "| --- | ---: | ---: | --- |"]
    for candidate in candidates:
        blockers = candidate.get("blockers", [])
        blocker_text = ", ".join(str(item) for item in blockers) if blockers else "none"
        strategy = str(candidate.get("strategy_id", candidate.get("name", "unknown")))
        lines.append(
            "| "
            f"{strategy} | "
            f"{bool(candidate.get('research_candidate', False))} | "
            f"{bool(candidate.get('production_candidate', False))} | "
            f"{blocker_text} |"
        )
    return lines


def render_perp_summary_markdown(payload: dict[str, Any]) -> str:
    pbo = payload.get("pbo", {})
    candidates = payload.get("candidate_statuses", [])
    if not isinstance(candidates, list):
        candidates = []
    lines = [
        "# Perpetual Futures Microstructure Validation",
        "",
        "## PBO",
        "",
        f"- status: `{pbo.get('status', 'unknown')}`",
        f"- pbo probability: `{pbo.get('pbo_probability')}`",
        f"- split count: `{pbo.get('split_count', 0)}`",
        "",
        "## Candidates",
        "",
        *_candidate_lines([item for item in candidates if isinstance(item, dict)]),
        "",
        "Production promotion is disabled for this free-data research slice.",
        "",
    ]
    return "\n".join(lines)


def write_perp_reports(output_dir: Path | str, payload: dict[str, Any]) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    summary_json = root / "perp_validation_summary.json"
    summary_markdown = root / "perp_validation_summary.md"
    best_candidates = root / "best_candidates_report.md"
    failure_report = root / "failure_report.md"

    summary_json.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n")
    markdown = render_perp_summary_markdown(safe_payload)
    summary_markdown.write_text(markdown)
    best_candidates.write_text(markdown)

    candidates = safe_payload.get("candidate_statuses", [])
    has_research_candidate = any(
        isinstance(candidate, dict) and bool(candidate.get("research_candidate", False)) for candidate in candidates
    )
    outputs = {
        "summary_json": summary_json,
        "summary_markdown": summary_markdown,
        "best_candidates_report": best_candidates,
    }
    if not has_research_candidate:
        failure_report.write_text(markdown)
        outputs["failure_report"] = failure_report
    return outputs
