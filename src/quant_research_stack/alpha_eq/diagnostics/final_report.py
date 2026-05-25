"""Final M6 report — wraps success-gate verdict + iteration plan (spec §6.4, §6.5)."""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    SuccessGateResult,
)


def write_final_report(
    path: Path, *, gate_result: SuccessGateResult, inputs: SuccessGateInputs
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# S1-EQ — final M6 report\n\n")
    if gate_result.suspended:
        parts.append("> ⚠️ **Gate suspended** — `survivorship_prototype_only`. Research-only.\n\n")
    elif gate_result.passed:
        parts.append("## Verdict: **Go**\n\n")
    else:
        parts.append("## Verdict: **No-Go**\n\n")
    parts.append("## Inputs\n\n")
    for k, v in inputs.__dict__.items():
        parts.append(f"- `{k}`: `{v}`\n")
    if gate_result.failures:
        parts.append("\n## Failures\n\n")
        for f in gate_result.failures:
            parts.append(f"- {f}\n")
        parts.append(
            "\n## Iteration plan (single hypothesis class — spec §6.5)\n\n"
            "Pick **one** of {feature, data, hyperparam, cost-model} to change "
            "before the next run.\n"
            "Multi-direction iteration is forbidden inside a single run cycle.\n"
        )
    parts.append("\n---\n`not_investment_advice: true`\n")
    Path(path).write_text("".join(parts))
