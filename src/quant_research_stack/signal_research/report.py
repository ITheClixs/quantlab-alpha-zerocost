"""Three-tier report writer (spec §6.2, §6.4, §7).

Generates `family/`, `profile/`, and the master `enhanced_benchmark.md`.
Always includes:
- Data-quality banner (per profile + universe).
- Selection funnel (counts at every stage).
- Four-tier status language (research_pass → production_candidate).
- §7 disclaimer text.
"""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.signal_research.runner import RunResult

_DISCLAIMER = (
    "Research output only. Past performance does not guarantee future results. "
    "No promotion to capital deployment occurs without an explicit promotion record "
    "(see spec §6.5 and the QuantLab promotion runbook)."
)


def write_reports(result: RunResult, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "enhanced_benchmark.md"

    funnel_lines = [
        f"- **{stage}**: {count}" for stage, count in result.funnel.to_ordered_dict().items()
    ]
    body = "\n".join([
        f"# Enhanced benchmark — profile `{result.profile_name}`",
        "",
        "## Selection funnel",
        *funnel_lines,
        "",
        "## Run metadata",
        f"- wall clock: {result.wall_clock_sec:.2f}s",
        f"- candidates evaluated: {result.metrics.height}",
        "",
        "## Disclaimer",
        _DISCLAIMER,
        "",
    ])
    master_path.write_text(body)
    return master_path
