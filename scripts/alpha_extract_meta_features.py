from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rich.console import Console
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from quant_research_stack.alpha.meta_features import MetaFeatureCache, finbert_logits_cached

console = Console()


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _finbert_runner(model_dir: Path) -> callable:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device())
    model.eval()

    def run(texts: list[str]) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(_device())
                logits = model(**enc).logits.cpu().numpy()
                outputs.append(logits)
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 3))

    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache FinBERT-style sentiment features for the corpus.")
    p.add_argument("--input-jsonl", default="data/processed/research/research_corpus.jsonl")
    p.add_argument("--model-dir", default="models/huggingface/hasnain43__bert-stock-sentiment-v1")
    p.add_argument("--output-parquet", default="data/processed/research/finbert_features.parquet")
    p.add_argument("--cache-root", default="data/processed/research/meta_cache")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    with open(args.input_jsonl) as h:
        for i, line in enumerate(h):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({"id": rec["id"], "text": rec["text"]})
    if not rows:
        console.print("[red]No rows in input.[/red]")
        return 2
    texts = [r["text"] for r in rows]
    cache = MetaFeatureCache(root=Path(args.cache_root))
    runner = _finbert_runner(Path(args.model_dir))
    logits = finbert_logits_cached(texts, cache=cache, cache_key=f"finbert::{len(texts)}", runner=runner)
    df = pl.DataFrame({
        "id": [r["id"] for r in rows],
        "finbert_neg": logits[:, 0].tolist(),
        "finbert_neu": logits[:, 1].tolist(),
        "finbert_pos": logits[:, 2].tolist(),
    })
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output_parquet, compression="zstd")
    console.print(f"Wrote {df.height} rows to {args.output_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
