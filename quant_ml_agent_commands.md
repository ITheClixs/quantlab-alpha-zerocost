# Quantitative Finance and Local ML Dataset Setup Commands for Coding Agent

Purpose: set up a reproducible local quantitative research and machine learning workspace on an Apple Silicon MacBook Air M4 with 24 GB RAM.

Target machine:
- macOS on Apple Silicon
- 24 GB unified memory
- 1 TB SSD
- Local experimentation only
- Avoid full-scale deep learning workloads
- Prefer tabular models, efficient Parquet storage, walk-forward validation, and small neural networks

Core principle:
Do not build a live trading bot first. Build a research system first.

```text
raw data -> cleaned data -> features -> labels -> time-based split -> model -> backtest -> paper trading
```

Important safety note:
These commands download public datasets and models for research. They do not constitute financial advice. Any later trading system must use strict paper trading, risk limits, transaction cost modeling, and legal API usage.

---

## 0. Source links verified for this setup

### Kaggle quantitative finance competitions and datasets

```text
Jane Street Real-Time Market Data Forecasting:
https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting

Jane Street Market Prediction:
https://www.kaggle.com/competitions/jane-street-market-prediction

Optiver Trading at the Close:
https://www.kaggle.com/competitions/optiver-trading-at-the-close

Optiver Realized Volatility Prediction:
https://www.kaggle.com/competitions/optiver-realized-volatility-prediction

G-Research Crypto Forecasting:
https://www.kaggle.com/competitions/g-research-crypto-forecasting

JPX Tokyo Stock Exchange Prediction:
https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction

Stock Market Signal: Predict Next-Day Returns:
https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns

Two Sigma: Using News to Predict Stock Movements:
https://www.kaggle.com/c/two-sigma-financial-news

Winton Stock Market Challenge:
https://www.kaggle.com/c/the-winton-stock-market-challenge

9000+ Tickers of Stock Market Data, Full History:
https://www.kaggle.com/datasets/jakewright/9000-tickers-of-stock-market-data-full-history

US Stock Market Historical OHLCV Dataset:
https://www.kaggle.com/datasets/asadullahcreative/us-stock-market-historical-ohlcv-dataset
```

### Hugging Face financial datasets

```text
Financial PhraseBank:
https://huggingface.co/datasets/takala/financial_phrasebank

Twitter Financial News Sentiment:
https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

Twitter Financial News Topic:
https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic

FNSPID:
https://huggingface.co/datasets/Zihan1004/FNSPID

FNSPID paper page:
https://huggingface.co/papers/2402.06698

FinGPT Sentiment Train:
https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train

KrossKinetic S&P 500 Financial News Articles Time Series:
https://huggingface.co/datasets/KrossKinetic/SP500-Financial-News-Articles-Time-Series

Multimodal Financial Time-Series Dataset:
https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
```

### Hugging Face general ML datasets

```text
TinyStories:
https://huggingface.co/datasets/roneneldan/TinyStories

WikiText:
https://huggingface.co/datasets/Salesforce/wikitext

CIFAR-10:
https://huggingface.co/datasets/uoft-cs/cifar10

IMDB:
https://huggingface.co/datasets/stanfordnlp/imdb
```

### Hugging Face models useful for local finance and ML work

```text
FinBERT:
https://huggingface.co/ProsusAI/finbert

Qwen2.5 0.5B Instruct:
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Qwen2.5 0.5B Base:
https://huggingface.co/Qwen/Qwen2.5-0.5B

Sentence Transformers all-MiniLM-L6-v2:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

TinyStories 33M model:
https://huggingface.co/roneneldan/TinyStories-33M
```

### Platform and tooling references

```text
Apple PyTorch MPS backend:
https://developer.apple.com/metal/pytorch/

Kaggle API:
https://github.com/Kaggle/kaggle-api

Hugging Face Hub Python Library:
https://huggingface.co/docs/huggingface_hub/index

Hugging Face Datasets Library:
https://huggingface.co/docs/datasets/index

Polars:
https://pola.rs/

DuckDB:
https://duckdb.org/

LightGBM:
https://lightgbm.readthedocs.io/

XGBoost:
https://xgboost.readthedocs.io/

CatBoost:
https://catboost.ai/
```

---

## 1. Create base workspace

Run in zsh.

```zsh
mkdir -p "$HOME/QuantLab"
mkdir -p "$HOME/QuantLab/data/raw/kaggle"
mkdir -p "$HOME/QuantLab/data/raw/huggingface"
mkdir -p "$HOME/QuantLab/data/raw/external"
mkdir -p "$HOME/QuantLab/data/processed/parquet"
mkdir -p "$HOME/QuantLab/data/features"
mkdir -p "$HOME/QuantLab/data/labels"
mkdir -p "$HOME/QuantLab/models/tree"
mkdir -p "$HOME/QuantLab/models/torch"
mkdir -p "$HOME/QuantLab/models/sentiment"
mkdir -p "$HOME/QuantLab/models/embeddings"
mkdir -p "$HOME/QuantLab/notebooks"
mkdir -p "$HOME/QuantLab/reports"
mkdir -p "$HOME/QuantLab/experiments"
mkdir -p "$HOME/QuantLab/src/data"
mkdir -p "$HOME/QuantLab/src/features"
mkdir -p "$HOME/QuantLab/src/models"
mkdir -p "$HOME/QuantLab/src/backtest"
mkdir -p "$HOME/QuantLab/src/execution"
mkdir -p "$HOME/QuantLab/src/research"
mkdir -p "$HOME/QuantLab/logs"
mkdir -p "$HOME/QuantLab/config"
mkdir -p "$HOME/QuantLab/scripts"

cd "$HOME/QuantLab"
```

Create a `.gitignore`.

```zsh
cat > .gitignore <<'EOF'
# Data and model artifacts
data/raw/
data/processed/
data/features/
data/labels/
models/
experiments/
logs/

# Python
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env
.kaggle/
.DS_Store

# Large local files
*.zip
*.tar
*.tar.gz
*.parquet
*.csv
*.h5
*.pt
*.pth
*.safetensors
*.onnx
*.gguf
EOF
```

Create a README skeleton.

```zsh
cat > README.md <<'EOF'
# QuantLab

Local quantitative finance and machine learning research workspace.

Pipeline:

```text
raw data -> processed Parquet -> features -> labels -> walk-forward validation -> model -> backtest
```

Rules:
1. Never use random train-test split for financial time series.
2. Never allow look-ahead leakage.
3. Always log data version, feature code version, label definition, model parameters, validation window, and transaction cost assumptions.
4. Paper trade before any real execution.
EOF
```

---

## 2. Install system dependencies

Install Homebrew first if it is not installed.

```zsh
if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew from https://brew.sh before continuing."
  exit 1
fi
```

Install core packages.

```zsh
brew update

brew install \
  git \
  git-lfs \
  wget \
  curl \
  tree \
  jq \
  unzip \
  cmake \
  ninja \
  pkg-config \
  llvm \
  libomp \
  python@3.11 \
  uv \
  duckdb \
  sqlite \
  htop \
  tmux
```

Enable Git LFS.

```zsh
git lfs install
```

Optional developer tools.

```zsh
brew install --cask visual-studio-code
```

---

## 3. Create Python virtual environment

Use `uv` for fast reproducible Python environment management.

```zsh
cd "$HOME/QuantLab"

uv venv .venv --python 3.11
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Install core ML and data packages.

```zsh
uv pip install \
  numpy \
  pandas \
  polars \
  pyarrow \
  duckdb \
  scipy \
  scikit-learn \
  statsmodels \
  matplotlib \
  plotly \
  jupyterlab \
  ipykernel \
  tqdm \
  rich \
  pydantic \
  pyyaml \
  python-dotenv
```

Install ML packages.

```zsh
uv pip install \
  lightgbm \
  xgboost \
  catboost \
  optuna \
  joblib \
  mlflow
```

Install PyTorch and Hugging Face tooling.

```zsh
uv pip install \
  torch \
  torchvision \
  torchaudio \
  transformers \
  datasets \
  accelerate \
  evaluate \
  safetensors \
  tokenizers \
  sentence-transformers \
  huggingface_hub
```

Install Kaggle and market data helper libraries.

```zsh
uv pip install \
  kaggle \
  yfinance \
  pandas-datareader \
  exchange-calendars
```

Register Jupyter kernel.

```zsh
python -m ipykernel install --user --name quantlab --display-name "Python (QuantLab)"
```

Freeze environment.

```zsh
uv pip freeze > requirements.txt
```

---

## 4. Verify Apple Silicon PyTorch MPS support

Create a script.

```zsh
cat > scripts/check_mps.py <<'PY'
import platform
import torch

print("Python platform:", platform.platform())
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    print("MPS matmul ok:", y.shape, y.device)
else:
    print("MPS unavailable. CPU fallback only.")
PY

python scripts/check_mps.py
```

Expected:
- `MPS built: True`
- `MPS available: True`

If MPS is unavailable, continue with CPU. For most tabular quant research, CPU is sufficient.

---

## 5. Configure Kaggle API

Manual prerequisite:
1. Go to `https://www.kaggle.com/settings/account`
2. Create an API token.
3. Download `kaggle.json`.
4. Place it at `~/.kaggle/kaggle.json`.

Run:

```zsh
mkdir -p "$HOME/.kaggle"
chmod 700 "$HOME/.kaggle"

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Missing ~/.kaggle/kaggle.json. Download it from Kaggle account settings."
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"
kaggle --version
```

Important:
Some competition data downloads require manually accepting competition rules on the Kaggle website before the API download works.

---

## 6. Configure Hugging Face CLI

Login is optional for public datasets, but useful for caching and gated models.

```zsh
huggingface-cli --help >/dev/null
```

Optional login:

```zsh
huggingface-cli login
```

Set cache directories inside QuantLab to keep storage organized.

```zsh
cat >> .env <<'EOF'
HF_HOME=$HOME/QuantLab/data/raw/huggingface/.hf_cache
HF_DATASETS_CACHE=$HOME/QuantLab/data/raw/huggingface/.datasets_cache
TRANSFORMERS_CACHE=$HOME/QuantLab/models/.transformers_cache
TOKENIZERS_PARALLELISM=false
EOF
```

For the current terminal session:

```zsh
export HF_HOME="$HOME/QuantLab/data/raw/huggingface/.hf_cache"
export HF_DATASETS_CACHE="$HOME/QuantLab/data/raw/huggingface/.datasets_cache"
export TRANSFORMERS_CACHE="$HOME/QuantLab/models/.transformers_cache"
export TOKENIZERS_PARALLELISM=false
```

---

## 7. Download Kaggle quantitative finance datasets

Create dataset directories.

```zsh
mkdir -p "$HOME/QuantLab/data/raw/kaggle/competitions"
mkdir -p "$HOME/QuantLab/data/raw/kaggle/datasets"
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.1 JPX Tokyo Stock Exchange Prediction

Best first serious cross-sectional equity ranking dataset.

Source:
`https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction`

```zsh
mkdir -p competitions/jpx-tokyo-stock-exchange-prediction
cd competitions/jpx-tokyo-stock-exchange-prediction

kaggle competitions download -c jpx-tokyo-stock-exchange-prediction

unzip -n jpx-tokyo-stock-exchange-prediction.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.2 Stock Market Signal: Predict Next-Day Returns

Clean starter supervised learning problem.

Source:
`https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns`

```zsh
mkdir -p competitions/stock-market-signal-predict-next-day-returns
cd competitions/stock-market-signal-predict-next-day-returns

kaggle competitions download -c stock-market-signal-predict-next-day-returns

unzip -n stock-market-signal-predict-next-day-returns.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.3 G-Research Crypto Forecasting

Good for crypto time-series research and paper-trading prototypes.

Source:
`https://www.kaggle.com/competitions/g-research-crypto-forecasting`

```zsh
mkdir -p competitions/g-research-crypto-forecasting
cd competitions/g-research-crypto-forecasting

kaggle competitions download -c g-research-crypto-forecasting

unzip -n g-research-crypto-forecasting.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.4 Optiver Trading at the Close

Microstructure-style intraday forecasting.

Source:
`https://www.kaggle.com/competitions/optiver-trading-at-the-close`

```zsh
mkdir -p competitions/optiver-trading-at-the-close
cd competitions/optiver-trading-at-the-close

kaggle competitions download -c optiver-trading-at-the-close

unzip -n optiver-trading-at-the-close.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.5 Optiver Realized Volatility Prediction

Useful for volatility modeling.

Source:
`https://www.kaggle.com/competitions/optiver-realized-volatility-prediction`

```zsh
mkdir -p competitions/optiver-realized-volatility-prediction
cd competitions/optiver-realized-volatility-prediction

kaggle competitions download -c optiver-realized-volatility-prediction

unzip -n optiver-realized-volatility-prediction.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.6 Jane Street Market Prediction

Good for utility-based decision modeling.

Source:
`https://www.kaggle.com/competitions/jane-street-market-prediction`

```zsh
mkdir -p competitions/jane-street-market-prediction
cd competitions/jane-street-market-prediction

kaggle competitions download -c jane-street-market-prediction

unzip -n jane-street-market-prediction.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.7 Jane Street Real-Time Market Data Forecasting

Most serious but heavier. Use after easier datasets.

Source:
`https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting`

```zsh
mkdir -p competitions/jane-street-real-time-market-data-forecasting
cd competitions/jane-street-real-time-market-data-forecasting

kaggle competitions download -c jane-street-real-time-market-data-forecasting

unzip -n jane-street-real-time-market-data-forecasting.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.8 Two Sigma: Using News to Predict Stock Movements

Older but useful for price + news modeling.

Source:
`https://www.kaggle.com/c/two-sigma-financial-news`

```zsh
mkdir -p competitions/two-sigma-financial-news
cd competitions/two-sigma-financial-news

kaggle competitions download -c two-sigma-financial-news

unzip -n two-sigma-financial-news.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.9 Winton Stock Market Challenge

Older return forecasting benchmark.

Source:
`https://www.kaggle.com/c/the-winton-stock-market-challenge`

```zsh
mkdir -p competitions/the-winton-stock-market-challenge
cd competitions/the-winton-stock-market-challenge

kaggle competitions download -c the-winton-stock-market-challenge

unzip -n the-winton-stock-market-challenge.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.10 9000+ Tickers of Stock Market Data, Full History

Large daily OHLCV equity universe.

Source:
`https://www.kaggle.com/datasets/jakewright/9000-tickers-of-stock-market-data-full-history`

```zsh
mkdir -p datasets/9000-tickers-of-stock-market-data-full-history
cd datasets/9000-tickers-of-stock-market-data-full-history

kaggle datasets download -d jakewright/9000-tickers-of-stock-market-data-full-history

unzip -n 9000-tickers-of-stock-market-data-full-history.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

### 7.11 US Stock Market Historical OHLCV Dataset

Smaller equity starter dataset.

Source:
`https://www.kaggle.com/datasets/asadullahcreative/us-stock-market-historical-ohlcv-dataset`

```zsh
mkdir -p datasets/us-stock-market-historical-ohlcv-dataset
cd datasets/us-stock-market-historical-ohlcv-dataset

kaggle datasets download -d asadullahcreative/us-stock-market-historical-ohlcv-dataset

unzip -n us-stock-market-historical-ohlcv-dataset.zip
cd "$HOME/QuantLab/data/raw/kaggle"
```

---

## 8. Download Hugging Face financial datasets

Create script.

```zsh
cd "$HOME/QuantLab"

cat > scripts/download_hf_finance_datasets.py <<'PY'
from pathlib import Path
from datasets import load_dataset

BASE = Path.home() / "QuantLab" / "data" / "raw" / "huggingface"
BASE.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # Small and safe to download fully
    {
        "repo": "takala/financial_phrasebank",
        "config": "sentences_allagree",
        "name": "financial_phrasebank_allagree",
        "streaming": False,
    },
    {
        "repo": "takala/financial_phrasebank",
        "config": "sentences_75agree",
        "name": "financial_phrasebank_75agree",
        "streaming": False,
    },
    {
        "repo": "zeroshot/twitter-financial-news-sentiment",
        "config": None,
        "name": "twitter_financial_news_sentiment",
        "streaming": False,
    },
    {
        "repo": "zeroshot/twitter-financial-news-topic",
        "config": None,
        "name": "twitter_financial_news_topic",
        "streaming": False,
    },
    {
        "repo": "FinGPT/fingpt-sentiment-train",
        "config": None,
        "name": "fingpt_sentiment_train",
        "streaming": False,
    },
]

def save_dataset(repo: str, config: str | None, name: str, streaming: bool) -> None:
    out_dir = BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {repo} config={config} streaming={streaming}")
    if config is None:
        ds = load_dataset(repo, streaming=streaming)
    else:
        ds = load_dataset(repo, config, streaming=streaming)

    if streaming:
        print(f"Skipping full save for streaming dataset {repo}")
        return

    for split, split_ds in ds.items():
        path = out_dir / f"{split}.parquet"
        print(f"Saving {repo}:{split} -> {path}")
        split_ds.to_parquet(str(path))

for item in DATASETS:
    save_dataset(**item)

print("Done.")
PY

python scripts/download_hf_finance_datasets.py
```

### 8.1 Streaming or subset-only financial datasets

The following datasets may be large. Do not automatically download the full dataset unless explicitly needed.

Create a subset script.

```zsh
cat > scripts/sample_large_hf_finance_datasets.py <<'PY'
from pathlib import Path
from itertools import islice
from datasets import load_dataset, Dataset

BASE = Path.home() / "QuantLab" / "data" / "raw" / "huggingface" / "large_samples"
BASE.mkdir(parents=True, exist_ok=True)

LARGE_DATASETS = [
    {
        "repo": "Zihan1004/FNSPID",
        "name": "fnspid_sample_10000",
        "n": 10000,
    },
    {
        "repo": "KrossKinetic/SP500-Financial-News-Articles-Time-Series",
        "name": "sp500_financial_news_articles_time_series_sample_10000",
        "n": 10000,
    },
    {
        "repo": "Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting",
        "name": "multimodal_financial_timeseries_sample_10000",
        "n": 10000,
    },
]

def sample_streaming_dataset(repo: str, name: str, n: int) -> None:
    out_dir = BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample.parquet"

    print(f"Streaming sample: {repo}, n={n}")
    ds = load_dataset(repo, split="train", streaming=True)
    rows = list(islice(ds, n))
    sampled = Dataset.from_list(rows)
    sampled.to_parquet(str(out_path))
    print(f"Saved {out_path}")

for item in LARGE_DATASETS:
    try:
        sample_streaming_dataset(**item)
    except Exception as e:
        print(f"FAILED: {item['repo']} -> {e}")

print("Done.")
PY

python scripts/sample_large_hf_finance_datasets.py
```

If any dataset fails:
- The dataset may have a different split name.
- The dataset may require trust_remote_code.
- The dataset may be too large or use nonstandard files.
- The coding agent should inspect the dataset card and adjust the loader.

---

## 9. Download Hugging Face general ML datasets

Create script.

```zsh
cd "$HOME/QuantLab"

cat > scripts/download_hf_general_ml_datasets.py <<'PY'
from pathlib import Path
from datasets import load_dataset

BASE = Path.home() / "QuantLab" / "data" / "raw" / "huggingface" / "general_ml"
BASE.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "repo": "roneneldan/TinyStories",
        "config": None,
        "name": "tinystories",
        "splits": None,
    },
    {
        "repo": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "name": "wikitext_103_raw",
        "splits": None,
    },
    {
        "repo": "uoft-cs/cifar10",
        "config": None,
        "name": "cifar10",
        "splits": None,
    },
    {
        "repo": "stanfordnlp/imdb",
        "config": None,
        "name": "imdb",
        "splits": None,
    },
]

def save_dataset(repo: str, config: str | None, name: str, splits) -> None:
    out_dir = BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {repo} config={config}")
    if config is None:
        ds = load_dataset(repo)
    else:
        ds = load_dataset(repo, config)

    for split, split_ds in ds.items():
        path = out_dir / f"{split}.parquet"
        print(f"Saving {repo}:{split} -> {path}")
        split_ds.to_parquet(str(path))

for item in DATASETS:
    save_dataset(**item)

print("Done.")
PY

python scripts/download_hf_general_ml_datasets.py
```

---

## 10. Download useful local models

Use Hugging Face Hub snapshot downloads.

```zsh
cd "$HOME/QuantLab"

cat > scripts/download_hf_models.py <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

BASE = Path.home() / "QuantLab" / "models" / "huggingface"
BASE.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "repo_id": "ProsusAI/finbert",
        "local_dir": BASE / "ProsusAI_finbert",
    },
    {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": BASE / "sentence-transformers_all-MiniLM-L6-v2",
    },
    {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": BASE / "Qwen_Qwen2.5-0.5B-Instruct",
    },
    {
        "repo_id": "Qwen/Qwen2.5-0.5B",
        "local_dir": BASE / "Qwen_Qwen2.5-0.5B",
    },
    {
        "repo_id": "roneneldan/TinyStories-33M",
        "local_dir": BASE / "roneneldan_TinyStories-33M",
    },
]

for item in MODELS:
    print(f"Downloading {item['repo_id']} -> {item['local_dir']}")
    snapshot_download(
        repo_id=item["repo_id"],
        local_dir=str(item["local_dir"]),
        local_dir_use_symlinks=False,
    )

print("Done.")
PY

python scripts/download_hf_models.py
```

Notes:
- `ProsusAI/finbert` is for financial sentiment.
- `sentence-transformers/all-MiniLM-L6-v2` is for compact text embeddings.
- `Qwen2.5-0.5B` and `Qwen2.5-0.5B-Instruct` are small enough for local experiments.
- `TinyStories-33M` is useful for tiny language-model experiments.

---

## 11. Validate model loading locally

Create script.

```zsh
cat > scripts/check_hf_models.py <<'PY'
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

BASE = Path.home() / "QuantLab" / "models" / "huggingface"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# FinBERT
finbert_dir = BASE / "ProsusAI_finbert"
tok = AutoTokenizer.from_pretrained(finbert_dir)
model = AutoModelForSequenceClassification.from_pretrained(finbert_dir).to(device)
inputs = tok("Apple reports stronger quarterly earnings than expected.", return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs)
print("FinBERT logits shape:", out.logits.shape)

# Sentence embeddings
emb_dir = BASE / "sentence-transformers_all-MiniLM-L6-v2"
embedder = SentenceTransformer(str(emb_dir), device=device)
vec = embedder.encode(["Market volatility increased after the rate decision."])
print("Embedding shape:", vec.shape)

# Tiny Causal LM
qwen_dir = BASE / "Qwen_Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(qwen_dir)
lm = AutoModelForCausalLM.from_pretrained(qwen_dir).to(device)
inputs = tok("In quantitative finance, a signal is", return_tensors="pt").to(device)
with torch.no_grad():
    generated = lm.generate(**inputs, max_new_tokens=20)
print(tok.decode(generated[0], skip_special_tokens=True))

print("All model checks completed.")
PY

python scripts/check_hf_models.py
```

If MPS errors occur:
- rerun with CPU by editing `device = "cpu"`;
- some transformer operations may still fall back or fail depending on PyTorch and model internals.

---

## 12. Convert CSV datasets to Parquet

Financial datasets can be large. Parquet is usually faster and smaller than CSV.

Create conversion script.

```zsh
cd "$HOME/QuantLab"

cat > scripts/convert_csv_to_parquet.py <<'PY'
from pathlib import Path
import polars as pl

RAW = Path.home() / "QuantLab" / "data" / "raw" / "kaggle"
OUT = Path.home() / "QuantLab" / "data" / "processed" / "parquet" / "kaggle_csv_converted"
OUT.mkdir(parents=True, exist_ok=True)

csv_files = list(RAW.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files.")

for csv_path in csv_files:
    rel = csv_path.relative_to(RAW)
    out_path = OUT / rel.with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Skip existing {out_path}")
        continue

    print(f"Converting {csv_path} -> {out_path}")
    try:
        lf = pl.scan_csv(
            csv_path,
            infer_schema_length=10000,
            ignore_errors=True,
        )
        lf.sink_parquet(out_path, compression="zstd")
    except Exception as e:
        print(f"FAILED: {csv_path} -> {e}")

print("Done.")
PY

python scripts/convert_csv_to_parquet.py
```

Inspect output size.

```zsh
du -sh "$HOME/QuantLab/data/raw/kaggle"
du -sh "$HOME/QuantLab/data/processed/parquet"
```

---

## 13. Create DuckDB inspection database

Create script.

```zsh
cat > scripts/inspect_parquet_with_duckdb.py <<'PY'
from pathlib import Path
import duckdb

PARQUET = Path.home() / "QuantLab" / "data" / "processed" / "parquet"
DB = Path.home() / "QuantLab" / "data" / "processed" / "quantlab.duckdb"

con = duckdb.connect(str(DB))

files = list(PARQUET.rglob("*.parquet"))
print(f"Found {len(files)} parquet files.")

for i, path in enumerate(files[:50], start=1):
    print(f"\n[{i}] {path}")
    try:
        q = f"SELECT * FROM read_parquet('{path}') LIMIT 5"
        print(con.execute(q).df())
    except Exception as e:
        print("FAILED:", e)

con.close()
print(f"DuckDB path: {DB}")
PY

python scripts/inspect_parquet_with_duckdb.py
```

---

## 14. Create first baseline quant research project

This creates a minimal baseline that the coding agent can later improve.

```zsh
mkdir -p "$HOME/QuantLab/src/baselines"

cat > "$HOME/QuantLab/src/baselines/README.md" <<'EOF'
# Baselines

Baseline order:

1. Naive momentum baseline.
2. Ridge regression.
3. LightGBM regression/classification.
4. Walk-forward validation.
5. Transaction-cost-adjusted backtest.

Forbidden:
- Random train-test split.
- Feature leakage from future prices.
- Evaluating only accuracy.
- Reporting returns without turnover and drawdown.
EOF
```

Create a generic feature specification.

```zsh
cat > "$HOME/QuantLab/config/feature_spec.yaml" <<'YAML'
price_features:
  returns:
    windows: [1, 2, 5, 10, 20, 60]
  rolling_volatility:
    windows: [5, 10, 20, 60]
  moving_average_distance:
    windows: [5, 10, 20, 60]
  volume_features:
    windows: [5, 20, 60]

labels:
  forward_return:
    horizons: [1, 5]
  classification:
    up_down_threshold: 0.0

validation:
  type: walk_forward
  min_train_days: 252
  validation_days: 63
  test_days: 63

backtest:
  initial_capital: 100000
  transaction_cost_bps: 5
  slippage_bps: 5
  max_position_weight: 0.02
  long_quantile: 0.9
  short_quantile: 0.1
YAML
```

---

## 15. Install optional local services for ML engineering

Optional but useful for a serious MLE-style setup.

```zsh
brew install docker
```

If using Docker Desktop instead:

```zsh
brew install --cask docker
```

Create local service compose file.

```zsh
cat > docker-compose.yml <<'YAML'
services:
  postgres:
    image: postgres:16
    container_name: quantlab_postgres
    environment:
      POSTGRES_USER: quantlab
      POSTGRES_PASSWORD: quantlab
      POSTGRES_DB: quantlab
    ports:
      - "5432:5432"
    volumes:
      - ./data/processed/postgres:/var/lib/postgresql/data

  redis:
    image: redis:7
    container_name: quantlab_redis
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:latest
    container_name: quantlab_qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/processed/qdrant:/qdrant/storage

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: quantlab_mlflow
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ./experiments/mlflow:/mlflow
YAML
```

Start services.

```zsh
docker compose up -d
```

Stop services.

```zsh
docker compose down
```

---

## 16. Storage hygiene commands

Check total QuantLab size.

```zsh
du -sh "$HOME/QuantLab"
du -sh "$HOME/QuantLab/data/raw/kaggle"/*
du -sh "$HOME/QuantLab/data/raw/huggingface"/*
du -sh "$HOME/QuantLab/models"/*
```

Find huge files.

```zsh
find "$HOME/QuantLab" -type f -size +2G -print
```

Clean Python caches.

```zsh
find "$HOME/QuantLab" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$HOME/QuantLab" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
```

Docker cleanup.

```zsh
docker system df
docker image prune -f
docker container prune -f
docker volume prune -f
```

Do not run aggressive cleanup unless you understand the consequences:

```zsh
# docker system prune -a --volumes
```

---

## 17. Recommended execution order for the coding agent

The agent should execute commands in this order:

```text
1. Create folder layout.
2. Install Homebrew packages.
3. Create Python environment.
4. Verify PyTorch MPS.
5. Configure Kaggle API.
6. Configure Hugging Face cache.
7. Download only these first:
   a. JPX Tokyo Stock Exchange Prediction
   b. Stock Market Signal: Predict Next-Day Returns
   c. Financial PhraseBank
   d. Twitter Financial News Sentiment
   e. CIFAR-10
   f. WikiText
8. Convert CSV to Parquet.
9. Build baseline feature pipeline.
10. Train Ridge and LightGBM baselines.
11. Add G-Research Crypto Forecasting.
12. Add Optiver Trading at the Close.
13. Add Jane Street datasets later.
14. Add FNSPID only as streaming/subset first.
15. Add local FinBERT and sentence embeddings.
```

Reason:
The first datasets are small enough or structured enough to debug the complete research pipeline. Jane Street and FNSPID are valuable but should not be the first implementation target.

---

## 18. Commands for minimal first run

Use this if the agent should avoid downloading everything.

```zsh
cd "$HOME/QuantLab"
source .venv/bin/activate

# Kaggle starter downloads
mkdir -p data/raw/kaggle/competitions/jpx-tokyo-stock-exchange-prediction
cd data/raw/kaggle/competitions/jpx-tokyo-stock-exchange-prediction
kaggle competitions download -c jpx-tokyo-stock-exchange-prediction
unzip -n jpx-tokyo-stock-exchange-prediction.zip

cd "$HOME/QuantLab"
mkdir -p data/raw/kaggle/competitions/stock-market-signal-predict-next-day-returns
cd data/raw/kaggle/competitions/stock-market-signal-predict-next-day-returns
kaggle competitions download -c stock-market-signal-predict-next-day-returns
unzip -n stock-market-signal-predict-next-day-returns.zip

# Hugging Face finance and general ML starter downloads
cd "$HOME/QuantLab"
python scripts/download_hf_finance_datasets.py
python scripts/download_hf_general_ml_datasets.py

# Models
python scripts/download_hf_models.py

# Convert CSV to Parquet
python scripts/convert_csv_to_parquet.py

# Verify models
python scripts/check_hf_models.py
```

---

## 19. Research guardrails for the agent

The agent must obey these constraints:

```text
1. No random split for financial time series.
2. No fitting scalers on validation/test periods.
3. No using future prices, future volume, future labels, or target-derived features.
4. No evaluating a model only by accuracy.
5. Always report:
   - train period
   - validation period
   - test period
   - feature list
   - label horizon
   - transaction costs
   - turnover
   - Sharpe ratio
   - max drawdown
   - hit rate
   - rank correlation if cross-sectional
6. Prefer Parquet over CSV for repeated experiments.
7. Prefer Polars or DuckDB for large tabular preprocessing.
8. Prefer LightGBM / XGBoost / CatBoost before neural networks for tabular finance.
9. Use small neural networks only after strong baselines exist.
10. Never connect to a real brokerage API before paper trading is implemented and tested.
```

---

## 20. Local model training guidance for 24 GB RAM

Safe local workloads:

```text
- LightGBM / XGBoost / CatBoost on medium tabular datasets
- Ridge / Lasso / ElasticNet baselines
- Random forest on moderate subsets
- Small MLPs
- Small LSTM / GRU models
- Small temporal CNNs
- Tiny Transformer encoders
- FinBERT inference and small fine-tuning
- Sentence embedding extraction
- TinyStories-scale small language modeling
```

Avoid locally:

```text
- Training large LLMs from scratch
- Full-scale FNSPID text modeling without subsetting
- Full order book reconstruction over massive tick data
- Giant Transformer models on long financial histories
- High-frequency live trading infrastructure
```

---

## 21. What the agent should build after setup

Create these modules:

```text
src/data/
  kaggle_loader.py
  hf_loader.py
  parquet_registry.py

src/features/
  price_features.py
  cross_sectional_features.py
  sentiment_features.py
  leakage_checks.py

src/models/
  ridge_baseline.py
  lightgbm_baseline.py
  xgboost_baseline.py
  torch_mlp.py

src/backtest/
  walk_forward.py
  portfolio.py
  metrics.py
  transaction_costs.py

src/research/
  experiment_config.py
  run_experiment.py
  report.py
```

First deliverable:
A working JPX cross-sectional return prediction baseline with:
- time split
- feature creation
- LightGBM model
- daily ranking
- long-top-decile / short-bottom-decile simulation
- transaction cost adjustment
- report saved to `reports/`

Second deliverable:
Add FinBERT-derived sentiment features from Financial PhraseBank / news data.

Third deliverable:
Add G-Research crypto forecasting pipeline.

---

## 22. Final warning

A model that predicts well in a notebook is not a trading bot. A trading bot requires:

```text
data integrity
latency assumptions
broker/exchange API handling
risk management
position sizing
slippage modeling
paper trading
monitoring
kill switch
legal and tax awareness
```

The current setup is for research and engineering practice. Real-money execution is out of scope until the research system is validated.
