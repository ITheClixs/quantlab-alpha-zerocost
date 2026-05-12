# AGENTS.md

This file defines agent roles, responsibilities, and task boundaries for the QuantLab repository.

The repository builds a local quantitative finance machine learning research system.

Core pipeline:

```text
raw data
  -> cleaned panel data
  -> features
  -> labels
  -> walk-forward validation
  -> model
  -> backtest
  -> paper trading
```

---

## 1. Global agent rules

Every agent must obey:

```text
No random split for financial time series.
No future information in features.
No target leakage.
No scaler or imputer fitted on validation or test data.
No backtest without costs.
No real-money trading.
No full fine-tuning 12B to 14B models locally.
No training 12B to 14B models from scratch.
No hidden notebook-only logic.
```

All reusable code must be placed under:

```text
src/quantlab/
```

All experiment outputs must be placed under:

```text
experiments/
```

---

## 2. Agent: Data Engineer

Responsibilities:

```text
Download Kaggle datasets.
Download Hugging Face datasets.
Convert raw CSV, Feather, and other tabular files to Parquet.
Infer schemas.
Create canonical panel data.
Check missingness.
Check duplicate rows.
Check date and asset coverage.
Save clean data under data/processed/.
```

Must produce:

```text
data/processed/parquet/
data/processed/panel/
reports/data_quality/
```

Must not:

```text
Create labels.
Train models.
Modify backtesting logic.
```

Quality checks:

```text
row count before and after conversion
column schema
date range
asset count
missing value report
duplicate key report
```

---

## 3. Agent: Feature Engineer

Responsibilities:

```text
Create leakage-safe price features.
Create volatility features.
Create moving average distance features.
Create volume surprise features.
Create cross-sectional rank features.
Create sentiment aggregate features.
```

Feature rules:

```text
At time t, use only information available at or before t.
For prediction horizon h, do not accidentally include t+h data.
Cross-sectional ranks may use same-date features only, not same-date labels.
```

Must produce:

```text
data/features/<dataset_name>/
src/quantlab/features/
```

Must add tests:

```text
test rolling features do not use future values
test cross-sectional features are grouped by date
test missing values are handled
```

---

## 4. Agent: Label Engineer

Responsibilities:

```text
Create forward return labels.
Create binary up/down labels.
Create cross-sectional rank labels.
Create realized volatility labels when needed.
```

Label formulas:

```text
fwd_ret_h = close[t + h] / close[t] - 1
label_up_h = 1 if fwd_ret_h > threshold else 0
rank_label_h = percentile rank of fwd_ret_h within date
```

Must produce:

```text
data/labels/<dataset_name>/
src/quantlab/labels/
```

Must add tests:

```text
test fwd_ret_1 on simple known price sequence
test final rows are null where future data is unavailable
test labels are not used as features
```

---

## 5. Agent: Validation Engineer

Responsibilities:

```text
Implement walk-forward splits.
Implement purged or embargoed splits when needed.
Prevent leakage through time.
Create split metadata.
```

Required split output:

```text
data/splits/<dataset_name>/<split_name>.json
```

Split metadata must include:

```text
train_start
train_end
validation_start
validation_end
test_start
test_end
number of train rows
number of validation rows
number of test rows
asset count
```

Must not:

```text
Use sklearn random train_test_split for financial observations.
Shuffle dates.
```

---

## 6. Agent: Model Engineer

Responsibilities:

```text
Implement Ridge.
Implement Logistic Regression.
Implement LightGBM.
Implement XGBoost.
Implement CatBoost.
Implement small PyTorch MLP.
Implement sequence models only after tabular baselines.
```

Training requirements:

```text
load split metadata
fit preprocessors on train only
train model
predict validation and test
save model artifact
save predictions
save metrics
```

Must produce:

```text
experiments/<experiment_name>/<model_name>/
  model artifact
  metrics.json
  predictions.parquet
```

Must report:

```text
MSE or MAE for regression
Spearman rank correlation
directional accuracy
AUC if classification
feature importance if available
```

---

## 7. Agent: Backtesting Engineer

Responsibilities:

```text
Convert predictions into positions.
Implement long-only and long-short portfolios.
Apply transaction costs.
Apply slippage.
Compute turnover.
Compute equity curve.
Compute summary metrics.
```

Required metrics:

```text
annualized_return
annualized_volatility
Sharpe
Sortino
maximum_drawdown
hit_rate
turnover
total_cost
final_equity
```

Must produce:

```text
experiments/<experiment_name>/<model_name>/backtest.parquet
experiments/<experiment_name>/<model_name>/backtest_summary.json
```

Must not:

```text
Backtest on training predictions and present it as final.
Ignore transaction costs.
Ignore turnover.
```

---

## 8. Agent: NLP Engineer

Responsibilities:

```text
Load Financial PhraseBank.
Load Twitter Financial News Sentiment.
Load FinGPT sentiment data.
Load FNSPID sample.
Run FinBERT sentiment inference.
Run sentence embedding extraction.
Aggregate sentiment by date and asset.
Merge sentiment features with price features.
```

Must produce:

```text
data/features/sentiment/
models/sentiment/
reports/nlp/
```

Must not:

```text
Use news timestamps after the market prediction timestamp.
Use article body if publication timestamp is unknown.
Assume sentiment causally predicts returns without validation.
```

---

## 9. Agent: Apple Silicon LLM Engineer

Responsibilities:

```text
Install MLX and MLX LM.
Test small MLX models first.
Optionally test 12B to 14B quantized inference.
Use local LLMs for code, summarization, and research support.
```

Allowed:

```text
4-bit quantized inference
small prompt-based workflows
small LoRA experiments on small models
```

Not allowed by default:

```text
training large LLMs from scratch
full fine-tuning large LLMs
using LLM output as direct trading signal without validation
```

Must check available models:

```zsh
python - <<'PY'
from huggingface_hub import list_models

for m in list_models(author="mlx-community", search="14B 4bit", limit=20):
    print(m.modelId)
PY
```

---

## 10. Agent: Paper Trading Engineer

Responsibilities:

```text
Build broker simulator.
Build paper positions store.
Build order and fill simulator.
Build risk manager.
Build report generator.
```

Risk rules:

```text
max position size
max gross exposure
max net exposure
max turnover
max daily loss
no trade on missing features
no trade on stale model
```

Must not:

```text
Connect to live broker for real money.
Store API keys in repository.
Bypass risk checks.
```

---

## 11. Agent: Report Engineer

Responsibilities:

```text
Generate experiment reports.
Generate model comparison tables.
Generate backtest charts.
Document assumptions.
Document failure modes.
```

Every report must include:

```text
dataset
date range
universe size
feature list
label definition
split method
train period
validation period
test period
model
hyperparameters
metrics
transaction cost assumptions
backtest summary
limitations
next steps
```

Reports go under:

```text
reports/
experiments/<experiment_name>/report.md
```

---

## 12. First milestone assignment

Build the JPX baseline.

Required tasks:

```text
1. Download JPX dataset.
2. Convert to Parquet.
3. Build canonical panel.
4. Create price features.
5. Create fwd_ret_1 and rank_label_1.
6. Create walk-forward split.
7. Train Ridge.
8. Train LightGBM.
9. Generate predictions.
10. Backtest long top decile and short bottom decile.
11. Apply transaction costs and slippage.
12. Save metrics and report.
```

Completion target:

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
  report.md
```

---

## 13. Second milestone assignment

Build the crypto pipeline using G-Research Crypto Forecasting.

Required tasks:

```text
1. Convert minute crypto data to Parquet.
2. Build asset-level time-series features.
3. Create forward return labels.
4. Use walk-forward split.
5. Train LightGBM baseline.
6. Train small PyTorch MLP.
7. Backtest simple long-short or long-cash allocation.
8. Report transaction-cost sensitivity.
```

---

## 14. Third milestone assignment

Build sentiment-enhanced prediction.

Required tasks:

```text
1. Load Financial PhraseBank.
2. Validate FinBERT inference.
3. Load Twitter Financial News Sentiment.
4. Create sentiment feature pipeline.
5. Load FNSPID sample.
6. Aggregate by asset and date.
7. Merge with price features.
8. Compare price-only model against price-plus-sentiment model.
```

---

## 15. Done definition

A task is not done unless:

```text
code runs
tests pass
outputs are saved
metrics are recorded
report is written
limitations are documented
```

