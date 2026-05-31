"""Crypto perpetual funding-rate carry data layer (free Binance public archives).

Funding-carry is the crypto-native carry premium (the analogue of the futures carry
we could not source for traditional markets). Funding settles every 8h; the realized
rate is timestamped at settlement, so conditioning on funding<=t and earning the
next settlement is leak-safe. All data is free from data.binance.vision.
Audit-first: see reports/signal_research/funding_carry_v1/funding_carry_data_audit.md.
"""

from quant_research_stack.crypto_research.funding.data import (
    FUNDING_BASE,
    funding_day_url,
    load_funding,
    normalize_funding,
)

__all__ = ["FUNDING_BASE", "funding_day_url", "load_funding", "normalize_funding"]
