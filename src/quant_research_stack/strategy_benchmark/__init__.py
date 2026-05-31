"""Strategy-benchmark framework — runs ~1500 quant strategies through a
single-asset walk-forward backtest, then applies Bailey/Lopez de Prado PBO
and Deflated Sharpe Ratio (DSR) to penalise multiple-testing bias.

Built explicitly as a benchmark to anchor what's realistic on free daily
data, not as a live-trade engine.  See docs (TODO add report path).
"""
