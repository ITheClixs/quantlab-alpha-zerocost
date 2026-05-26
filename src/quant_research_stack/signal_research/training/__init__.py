"""Training utilities for signal_research."""

from quant_research_stack.signal_research.training.backtest_diagnostics import (
    StrictBacktestDiagnosticsConfig,
    StrictBacktestDiagnosticsResult,
    render_strict_backtest_report,
    run_strict_backtest_diagnostics,
    write_strict_backtest_artifacts,
)
from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    MetaLabelWalkForwardResult,
    train_meta_label_walk_forward,
    write_meta_label_walk_forward_artifacts,
)

__all__ = [
    "MetaLabelWalkForwardConfig",
    "MetaLabelWalkForwardResult",
    "StrictBacktestDiagnosticsConfig",
    "StrictBacktestDiagnosticsResult",
    "render_strict_backtest_report",
    "run_strict_backtest_diagnostics",
    "train_meta_label_walk_forward",
    "write_strict_backtest_artifacts",
    "write_meta_label_walk_forward_artifacts",
]
