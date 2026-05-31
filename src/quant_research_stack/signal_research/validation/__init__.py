"""Production validation pipeline.

Single entrypoint for vetting any new strategy proposal under hedge-fund-
grade discipline. See `pipeline.validate_strategy` and the strategy intake
protocol at docs/research/STRATEGY_INTAKE.md.
"""

from quant_research_stack.signal_research.validation.concentration import (
    ConcentrationReport,
    concentration_by_period,
)
from quant_research_stack.signal_research.validation.cost_decomposition import (
    CostDecomposition,
    cost_decomposition,
)
from quant_research_stack.signal_research.validation.delay_stress import (
    shift_signal_by_n_bars,
)
from quant_research_stack.signal_research.validation.pipeline import (
    PipelineReport,
    SignalFn,
    StrategyValidationResult,
    render_pipeline_report,
    validate_strategy,
)
from quant_research_stack.signal_research.validation.sanity import (
    inverted_signal,
    random_signal,
)
from quant_research_stack.signal_research.validation.spec import (
    InformationSource,
    ValidationSpec,
)

__all__ = [
    "ConcentrationReport",
    "CostDecomposition",
    "InformationSource",
    "PipelineReport",
    "SignalFn",
    "StrategyValidationResult",
    "ValidationSpec",
    "concentration_by_period",
    "cost_decomposition",
    "inverted_signal",
    "random_signal",
    "render_pipeline_report",
    "shift_signal_by_n_bars",
    "validate_strategy",
]
