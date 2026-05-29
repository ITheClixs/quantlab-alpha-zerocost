"""Event-conditioned macro/calendar layer (event_conditioned_macro_v1).

Leak-free calendar conditioning: all features derive from the ex-ante event
schedule and the date itself — never from future prices. See intake
docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md.
"""

from quant_research_stack.signal_research.events.calendar import (
    EARNINGS_WINDOWS,
    attach_event_features,
    load_fomc_dates,
)

__all__ = ["EARNINGS_WINDOWS", "attach_event_features", "load_fomc_dates"]
