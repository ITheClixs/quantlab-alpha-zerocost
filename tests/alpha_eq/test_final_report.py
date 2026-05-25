"""Final report writer ties success-gate result + iteration plan."""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.diagnostics.final_report import write_final_report
from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    evaluate_success_gate,
)


def test_final_report_writes_go_or_nogo(tmp_path: Path) -> None:
    inputs = SuccessGateInputs(
        data_quality_label=DataQualityLabel.PIT_SAFE,
        holdout_trading_days=800,
        delisting_capture_ratio=0.99,
        delisting_unknown_in_holdout=0,
        s1_eq_net_sharpe=1.2, family_b_net_sharpe=0.6, spy_sharpe=0.5, max_drawdown=-0.15,
        net_sharpe_borrow_2x=0.9, net_total_return_borrow_3x=0.10, js_overlay_net_sharpe=0.4,
        rolling_window_alpha_consistent=True,
        concentration_stock_violation=False,
        concentration_month_violation=False,
        concentration_sector_violation=False,
        ci_tests_green=True, artifacts_complete=True,
    )
    res = evaluate_success_gate(inputs)
    out = tmp_path / "final_report.md"
    write_final_report(out, gate_result=res, inputs=inputs)
    text = out.read_text()
    assert "Go" in text
    assert "not_investment_advice: true" in text
