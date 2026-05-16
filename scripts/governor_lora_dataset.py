from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build governor-LoRA training JSONL from instructions corpus.")
    p.add_argument("--instructions-jsonl", default="data/processed/research/instructions.jsonl")
    p.add_argument("--corpus-parquet-dir", default="data/processed/research/parquet")
    p.add_argument("--out-jsonl", default="data/processed/research/lora_governor.jsonl")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    instructions = Path(args.instructions_jsonl)
    if not instructions.exists():
        console.print(f"[red]missing {instructions}; run paper_corpus_to_instructions.py first[/red]")
        return 2
    corpus = pl.read_parquet(list(Path(args.corpus_parquet_dir).glob("shard_*.parquet")))
    chunk_text_by_id = dict(zip(corpus["id"].to_list(), corpus["text"].to_list(), strict=True))

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with instructions.open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_text = chunk_text_by_id.get(rec.get("source_chunk_id", ""), "")
            if not chunk_text:
                continue
            messages = [
                {"role": "system", "content": "You are QuantLab's fast veto governor. Emit JSON only."},
                {"role": "user", "content": f"Chunk:\n{chunk_text}\n\nQuestion: {rec['prompt']}"},
                {"role": "assistant", "content": rec["response"]},
            ]
            dst.write(json.dumps({"messages": messages}) + "\n")
            written += 1
    console.print(f"Wrote {written} LoRA records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
