"""Zero-cost deployable strategy data layer (P0).

Free, directly-tradable instruments + timestamp-safe market-priced macro only.
Intake/plan: docs/research/intake/2026-05-30-zero-cost-deployable-v1.md.
Constraint: docs/research/2026-05-30-ZERO-COST-CONSTRAINT.md.
Hard rule: revised macro aggregates (GDP/CPI/NFP/UNRATE) are FORBIDDEN as features
unless point-in-time vintages are used. Everything here is observable at close t and
used only at t+1.
"""

from quant_research_stack.signal_research.zero_cost.data import (
    FORBIDDEN_SERIES,
    INSTRUMENTS,
    MACRO_REGISTRY,
    load_instrument,
    load_macro,
)

__all__ = ["FORBIDDEN_SERIES", "INSTRUMENTS", "MACRO_REGISTRY", "load_instrument", "load_macro"]
