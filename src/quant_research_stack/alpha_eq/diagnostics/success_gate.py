"""Success-gate evaluator implementing spec §6.4 (13 criteria) + §6.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel


@dataclass(frozen=True)
class SuccessGateInputs:
    data_quality_label: DataQualityLabel
    holdout_trading_days: int
    delisting_capture_ratio: float
    delisting_unknown_in_holdout: int
    s1_eq_net_sharpe: float
    family_b_net_sharpe: float
    spy_sharpe: float
    max_drawdown: float
    net_sharpe_borrow_2x: float
    net_total_return_borrow_3x: float
    js_overlay_net_sharpe: float
    rolling_window_alpha_consistent: bool
    concentration_stock_violation: bool
    concentration_month_violation: bool
    concentration_sector_violation: bool
    ci_tests_green: bool
    artifacts_complete: bool

    def model_copy(self, *, update: dict[str, Any]) -> SuccessGateInputs:
        d = self.__dict__.copy()
        d.update(update)
        return SuccessGateInputs(**d)


@dataclass(frozen=True)
class SuccessGateResult:
    passed: bool
    suspended: bool
    failures: list[str]


def evaluate_success_gate(inputs: SuccessGateInputs) -> SuccessGateResult:
    failures: list[str] = []
    if inputs.data_quality_label == DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY:
        return SuccessGateResult(
            passed=False, suspended=True, failures=["gate suspended (prototype-only)"]
        )

    if inputs.data_quality_label not in (
        DataQualityLabel.PIT_SAFE,
        DataQualityLabel.PARTIAL_PIT_UNIVERSE,
    ):
        failures.append("data_quality_label not eligible")

    if inputs.holdout_trading_days < 756:
        failures.append(f"holdout too short: {inputs.holdout_trading_days} < 756")

    if inputs.delisting_capture_ratio < 0.95 or inputs.delisting_unknown_in_holdout > 0:
        if inputs.data_quality_label == DataQualityLabel.PIT_SAFE:
            failures.append("delisting audit below pit_safe threshold")

    if inputs.s1_eq_net_sharpe < 0.7:
        failures.append(f"net Sharpe below standalone bar: {inputs.s1_eq_net_sharpe:.3f} < 0.7")

    if inputs.family_b_net_sharpe > 0:
        if inputs.s1_eq_net_sharpe < 1.5 * inputs.family_b_net_sharpe:
            failures.append("S1-EQ Sharpe < 1.5 × Family B Sharpe")
    else:
        if inputs.s1_eq_net_sharpe - inputs.family_b_net_sharpe < 0.5:
            failures.append("S1-EQ does not beat Family B by ≥ 0.5 Sharpe")

    if inputs.spy_sharpe > 0 and inputs.s1_eq_net_sharpe <= inputs.spy_sharpe:
        failures.append("S1-EQ Sharpe not strictly above SPY Sharpe")

    if inputs.max_drawdown < -0.25:
        failures.append(f"max drawdown worse than -25%: {inputs.max_drawdown:.3f}")

    if inputs.net_sharpe_borrow_2x <= 0:
        failures.append("net Sharpe non-positive at borrow ×2")
    if inputs.net_total_return_borrow_3x <= 0:
        failures.append("net total return non-positive at borrow ×3")

    if inputs.js_overlay_net_sharpe >= inputs.s1_eq_net_sharpe:
        failures.append("JS-overlay Sharpe ≥ S1-EQ Sharpe — retraining did not help")

    if not inputs.rolling_window_alpha_consistent:
        failures.append("rolling-window CV shows regime-concentrated alpha")

    if inputs.concentration_stock_violation:
        failures.append("single stock contributes > 25% of net PnL")
    if inputs.concentration_month_violation:
        failures.append("single month contributes > 35% of net PnL")
    if inputs.concentration_sector_violation:
        failures.append("single sector contributes > 50% of net PnL (and not justified)")

    if not inputs.ci_tests_green:
        failures.append("CI tests not green")
    if not inputs.artifacts_complete:
        failures.append("required artifacts missing")

    return SuccessGateResult(passed=(len(failures) == 0), suspended=False, failures=failures)
