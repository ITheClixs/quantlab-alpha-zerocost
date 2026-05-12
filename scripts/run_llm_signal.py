from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console

from quant_research_stack.artifacts import read_yaml, write_json
from quant_research_stack.llm_quant import (
    build_signal_prompt,
    call_openai_compatible_local_model,
    choose_local_model,
    load_research_chunks,
    parse_quant_signal,
    retrieve_chunks,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate a local LLM quant signal.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--market-context-json", required=True)
    parser.add_argument("--query", default="Jane Street responder_6 order flow imbalance market prediction")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt and model choice without calling a server.")
    parser.add_argument("--report", default="reports/llm_signal.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    corpus = args.corpus or f"{config['paths']['processed_research_root']}/research_corpus.jsonl"
    market_context = json.loads(args.market_context_json)
    chunks = retrieve_chunks(args.query, load_research_chunks(corpus), top_k=4)
    prompt = build_signal_prompt(market_context, chunks)
    model_choice = choose_local_model(config)
    payload = {
        "model_choice": model_choice.as_dict(),
        "chunk_ids": [chunk.id for chunk in chunks],
        "prompt": prompt,
    }
    if not args.dry_run:
        raw = call_openai_compatible_local_model(prompt, base_url=args.base_url, model=model_choice.model_id)
        signal = parse_quant_signal(raw, {chunk.id for chunk in chunks})
        payload["raw_response"] = raw
        payload["signal"] = signal.as_dict()
    write_json(args.report, payload)
    console.print(payload)
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
