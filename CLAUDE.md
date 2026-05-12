# CLAUDE.md

This file defines how Claude Code or any similar coding agent must behave inside this repository.

Project name:

```text
QuantLab
```

Project goal:

```text
Build a rigorous local quantitative finance machine learning research system on Apple Silicon.
```

Target machine:

```text
MacBook Air M4
24 GB unified memory
macOS
Apple Silicon MPS available through PyTorch
MLX available for local LLM inference
```

Primary pipeline:

```text
raw data
  -> cleaned panel data
  -> features
  -> labels
  -> walk-forward validation
  -> model training
  -> predictions
  -> backtest
  -> paper trading simulator
```

---

## 1. Absolute project rules

 Some Rules

```text
1. Do not use random train-test split for financial time series.
2. Do not use future data in features.
3. Do not fit scalers, imputers, encoders, or normalizers on validation or test data.
4. Do not evaluate only accuracy.
5. Do not report a strategy without transaction costs and turnover.
9. Do not full-fine-tune a 12B to 14B parameter model locally.
10. Do not download massive datasets without checking disk and user intent.
```

---

## 2. Preferred implementation order

Implement in this order:

```text
1. Project scaffolding
2. Environment validation
3. Dataset download scripts
4. Raw to Parquet conversion
5. Canonical panel schema
6. Feature generation
7. Label generation
8. Walk-forward split
9. Leakage tests
10. Ridge baseline
11. LightGBM baseline
12. Cross-sectional backtest
13. Report generation
14. G-Research crypto pipeline
15. Optiver pipeline
16. Jane Street pipeline
17. FinBERT sentiment features
18. Small PyTorch MLP
19. Sequence models
20. Paper trading simulator
21. Optional MLX local LLM assistant
```

---

## 3. Repository structure

Expected structure:

```text
QuantLab/
  data/
    raw/
    processed/
    features/
    labels/
    splits/
    backtests/
    paper_trading/
  models/
    tree/
    torch/
    sentiment/
    embeddings/
    llm/
    mlx/
  notebooks/
  reports/
  experiments/
  logs/
  config/
  scripts/
  src/
    quantlab/
      data/
      features/
      labels/
      splits/
      models/
      backtest/
      paper/
      execution/
      research/
      utils/
  tests/
```

Do not put important project logic only in notebooks. Notebooks may be used for exploration, but reusable logic must live under `src/quantlab`.

---

## 4. Coding style

Use:

```text
Python 3.11
type hints
dataclasses or pydantic for configuration
Polars or DuckDB for large preprocessing
Pandas for model interface when necessary
Parquet for intermediate storage
Joblib for tree model artifacts
Torch save for neural model artifacts
JSON for metrics
YAML for configs
```

Prefer:

```text
small composable modules
deterministic functions
explicit config
clear file paths
reproducible outputs
```

Avoid:

```text
hidden global state
implicit notebook variables
hardcoded absolute paths except $HOME/QuantLab
large monolithic scripts
silent exception swallowing
```

---

## 5. Financial ML methodology

### 5.1 Features

Every feature at time `t` must be computable with information available at or before time `t`.

Allowed feature families:

```text
past returns
rolling volatility
moving average distance
volume surprise
cross-sectional ranks
sector-neutral ranks if sector metadata exists
market beta estimates using only past windows
sentiment aggregates from already-published news
```

Forbidden features:

```text
future returns
future rolling statistics
future volume
target-derived transformations
post-event data not available at prediction time
```

### 5.2 Labels

Use forward labels:

```text
fwd_ret_h = close[t + h] / close[t] - 1
label_up_h = 1 if fwd_ret_h > threshold else 0
rank_label_h = cross-sectional rank of fwd_ret_h within date
```

Always document the prediction horizon.

### 5.3 Splits

Use walk-forward validation:

```text
train on past
validate on later period
test on still later period
roll forward
```

Never shuffle financial observations across time.

---

## 6. Model hierarchy

Implement models in this order:

```text
1. naive momentum or mean reversion
2. Ridge regression
3. Logistic regression
4. LightGBM
5. XGBoost
6. CatBoost
7. small PyTorch MLP
8. LSTM or GRU
9. temporal CNN
10. compact Transformer encoder
```

Tree models are the default for tabular financial data. Neural networks must beat strong tabular baselines before they are treated as useful.

---

## 7. Backtesting requirements

Backtests must include:

```text
transaction costs
slippage
turnover
gross returns
net returns
equity curve
Sharpe ratio
Sortino ratio
maximum drawdown
hit rate
average position count
average turnover
```

For cross-sectional equity strategy:

```text
sort assets by prediction each date
long top quantile
short bottom quantile if allowed
equal-weight or score-weight
clip position size
compute next-period realized return
subtract costs
```

Do not claim profitability from an in-sample or validation-only result.

---

## 8. Paper trading requirements

Paper trading must come after backtesting.

Required modules:

```text
paper/data_feed.py
paper/feature_service.py
paper/model_service.py
paper/portfolio_service.py
paper/risk_service.py
paper/broker_simulator.py
paper/state_store.py
paper/report.py
```

Paper trading must use the same feature functions as backtesting.

---

## 9. Apple Silicon policy

Use PyTorch MPS when available:

```python
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

If MPS fails for a model, fallback to CPU and log the reason.

Use MLX for local LLM inference and optional tiny LoRA experiments.

12B to 14B parameter models are allowed only as quantized inference or very small LoRA experiments. They are not the trading model.

---

## 10. Local LLM policy

Allowed use of local LLMs:

```text
coding assistant
research summarization
feature brainstorming
documentation summarization
small financial text classification experiments
```

Disallowed use:

```text
direct trading decisions
unvalidated market prediction
training from scratch
full fine-tuning large models locally
```

---

## 11. Testing requirements

Add tests for:

```text
forward return label direction
no future leakage in features
walk-forward fold ordering
backtest cost calculation
portfolio weight constraints
model output shape
missing data handling
```

Run:

```zsh
PYTHONPATH=src pytest -q
ruff check src tests
```

---

## 12. Completion criteria for first milestone

The first milestone is complete only if this exists:

```text
experiments/jpx_baseline/
  metadata.json
  ridge/
    metrics.json
    predictions.parquet
  lightgbm/
    metrics.json
    predictions.parquet
    backtest.parquet
    backtest_summary.json
```

The result must be reproducible from a clean repository using documented commands.

new:

now it is time for you to code a large language model for quantitative trading and quantitative research
  with the inspiration of current models that you have downloaded in this repo. The model that you will code
  will be medium to big sized and will then be integrated into a trading bot (ca. 20B-params) which will
  detect signals and upside downs given the order flow or any ingestion of a market data. For this purpose
  please reedit the README.md and AGENTS.md and also CLAUDE.md and also enhance them in a way that they are
  way more efficient. The model needs to be capable of  JaneStreet kaggle competition market prediction and
  needs to rank really high in that purpose. you are free to use any of your skills and HF and paper search
  skills for that purpose and also your other skills such as /superpowers:writing-plans or your plan mode
