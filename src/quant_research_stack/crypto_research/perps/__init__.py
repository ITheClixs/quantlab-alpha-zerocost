"""Perpetual futures microstructure research helpers."""

from quant_research_stack.crypto_research.perps.backtest import (
    PerpBacktestConfig,
    PerpBacktestResult,
    run_event_backtest,
)
from quant_research_stack.crypto_research.perps.events import (
    normalize_agg_trade,
    normalize_book_ticker,
    normalize_depth_update,
)
from quant_research_stack.crypto_research.perps.features import build_l1_features
from quant_research_stack.crypto_research.perps.manifest import build_dataset_manifest
from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    PerpWalkForwardResult,
    train_perp_walk_forward,
)

__all__ = [
    "PerpBacktestConfig",
    "PerpBacktestResult",
    "PerpWalkForwardConfig",
    "PerpWalkForwardResult",
    "build_dataset_manifest",
    "build_l1_features",
    "normalize_agg_trade",
    "normalize_book_ticker",
    "normalize_depth_update",
    "run_event_backtest",
    "train_perp_walk_forward",
]
