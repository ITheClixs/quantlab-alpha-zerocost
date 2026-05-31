"""VRP — index-level variance-risk-premium experiment.

See docs/research/intake/2026-05-28-vrp-index-v1.md for the intake.
"""

from quant_research_stack.signal_research.vrp.data import (
    UNDERLYING_TICKERS,
    VIX_FAMILY_TICKERS,
    VRPFetchResult,
    fetch_vrp_data,
)
from quant_research_stack.signal_research.vrp.features import (
    compute_realized_variance_annual,
    compute_vrp_features,
    vrp_zscore_60d,
)
from quant_research_stack.signal_research.vrp.runner import (
    VRPCrossMetrics,
    VRPRunReport,
    VRPSpec,
    VRPStrategyResult,
    render_vrp_report,
    run_vrp_pipeline,
)
from quant_research_stack.signal_research.vrp.timing_backtest import (
    TimingBacktestResult,
    TimingCostConfig,
    run_timing_backtest,
)

__all__ = [
    "TimingBacktestResult",
    "TimingCostConfig",
    "UNDERLYING_TICKERS",
    "VIX_FAMILY_TICKERS",
    "VRPCrossMetrics",
    "VRPFetchResult",
    "VRPRunReport",
    "VRPSpec",
    "VRPStrategyResult",
    "compute_realized_variance_annual",
    "compute_vrp_features",
    "fetch_vrp_data",
    "render_vrp_report",
    "run_timing_backtest",
    "run_vrp_pipeline",
    "vrp_zscore_60d",
]
