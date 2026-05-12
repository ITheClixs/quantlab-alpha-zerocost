# Quant Research Machine Learning Workspace

This workspace is organized for building a local quantitative-research model stack:

- market-data datasets for supervised price, volatility, and order-flow prediction
- paper-derived research corpora for retrieval, hypothesis generation, and LLM mentoring
- local time-series and finance LLM models that fit a MacBook Air M4 with 24 GB unified memory
- reproducible manifests and scripts with a hard combined artifact budget

Raw datasets and model weights are intentionally ignored by git. Recreate them from:

```bash
uv run python scripts/download_hf_artifacts.py --dry-run --sort size
uv run python scripts/download_hf_artifacts.py --sort size
uv run python scripts/prepare_market_data.py
uv run python scripts/prepare_research_corpus.py
```

The default artifact budget is `100 GB` across Hugging Face datasets, Hugging Face models,
and downloaded paper PDFs. The downloader sorts candidates by estimated size and stops before
crossing the budget.

## Stack

The recommended stack keeps prediction and explanation separate:

1. Market features: OHLCV, returns, realized volatility, spreads, order-flow imbalance,
   depth imbalance, and forward returns.
2. Labels: next-horizon direction, forward return, volatility regime, and simple
   triple-barrier-style labels.
3. Forecasting models: Chronos, Kronos, Granite TTM, TimeMoE, plus conventional baselines.
4. LLM/research role: explain signals, retrieve papers, generate strategy hypotheses,
   critique validation, and mentor the terminal workflow.
5. Validation: walk-forward splits, transaction-cost-aware backtests, and leakage controls.

## Layout

```text
configs/                 global stack and budget settings
manifests/               dataset, model, and paper manifests
scripts/                 download, inventory, and preparation entrypoints
src/quant_research_stack shared Python utilities
data/raw/                ignored raw datasets and PDFs
data/processed/          ignored train-ready parquet/jsonl artifacts
models/huggingface/      ignored local model snapshots
reports/                 generated inventory/download reports
```

## Important

This workspace prepares research and training data. It does not claim that any model is
profitable. Trading performance must be measured with out-of-sample walk-forward tests,
costs, slippage, latency, and drawdown constraints before any live use.

