from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from quant_research_stack.artifacts import read_yaml, safe_repo_id

console = Console()


PROMPT_TEMPLATES = [
    "Summarize the key claim of this passage in 3 sentences.",
    "What is the main quantitative-finance concept introduced here, and how is it operationalized?",
    "Generate one well-formed exam-style question and answer pair grounded in this passage.",
]


@dataclass(frozen=True)
class ModelChoice:
    repo_id: str
    local_dir: Path
    mode: str  # "primary" | "fallback"


def find_model(primary_repo: str, fallback_repo: str, model_root: Path) -> ModelChoice | None:
    primary_dir = model_root / safe_repo_id(primary_repo)
    if primary_dir.exists() and any(primary_dir.iterdir()):
        return ModelChoice(repo_id=primary_repo, local_dir=primary_dir, mode="primary")
    fallback_dir = model_root / safe_repo_id(fallback_repo)
    if fallback_dir.exists() and any(fallback_dir.iterdir()):
        return ModelChoice(repo_id=fallback_repo, local_dir=fallback_dir, mode="fallback")
    return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def already_generated_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(record.get("source_chunk_id", "")))
    return done


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def generate_with_transformers(
    model_choice: ModelChoice,
    chunks: Iterable[dict[str, Any]],
    output_path: Path,
    max_per_chunk: int,
    max_new_tokens: int,
) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"Loading {model_choice.repo_id} ({model_choice.mode}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_choice.local_dir)
    model = AutoModelForCausalLM.from_pretrained(model_choice.local_dir).to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            chunk_id = str(chunk.get("id", ""))
            text = chunk.get("text", "")
            if not text:
                continue
            for index, template in enumerate(PROMPT_TEMPLATES[:max_per_chunk]):
                prompt = f"Passage:\n{text}\n\nInstruction: {template}\nResponse:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                record = {
                    "id": f"{chunk_id}#qa{index}",
                    "source_chunk_id": chunk_id,
                    "prompt": template,
                    "response": response,
                    "model_id": model_choice.repo_id,
                    "model_mode": model_choice.mode,
                    "generated_at": now_iso(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instruction-format Q&A from chunked research JSONL.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None, help="If set, process only the first N chunks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    corpus_cfg = config.get("research_corpus", {})
    input_path = Path(args.input or Path(paths["processed_research_root"]) / "research_corpus.jsonl")
    output_path = Path(args.output or Path(paths["processed_research_root"]) / "instructions.jsonl")
    model_root = Path(paths["model_root"])
    max_per_chunk = int(corpus_cfg.get("instruction_max_per_chunk", 3))
    max_new_tokens = int(corpus_cfg.get("instruction_max_new_tokens", 256))
    primary = corpus_cfg.get("instruction_primary_model", "Qwen/Qwen2.5-0.5B-Instruct")
    fallback = corpus_cfg.get("instruction_fallback_model", "roneneldan/TinyStories-33M")

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}.[/red]")
        return 2

    choice = find_model(primary, fallback, model_root)
    if choice is None:
        console.print(f"[red]Neither {primary} nor {fallback} is present under {model_root}.[/red]")
        return 3
    if choice.mode == "fallback":
        console.print(f"[yellow]Using fallback model {choice.repo_id} (primary {primary} unavailable).[/yellow]")

    if args.dry_run:
        console.print(f"Would use model: {choice.repo_id} ({choice.mode}) at {choice.local_dir}")
        console.print(f"Would read chunks from {input_path}; write Q&A to {output_path}")
        return 0

    done = already_generated_ids(output_path)

    def chunks_iter() -> Iterable[dict[str, Any]]:
        seen = 0
        for record in iter_jsonl(input_path):
            if str(record.get("id", "")) in done:
                continue
            yield record
            seen += 1
            if args.limit_chunks is not None and seen >= args.limit_chunks:
                return

    count = generate_with_transformers(choice, chunks_iter(), output_path, max_per_chunk, max_new_tokens)
    console.print(f"Wrote {count} Q&A records to {output_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "src")
    sys.exit(main())
