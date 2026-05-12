from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from rich.console import Console

from quant_research_stack.artifacts import read_yaml


console = Console()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(text: str, chunk_words: int, overlap_words: int, min_chunk_words: int) -> Iterable[str]:
    words = text.split()
    if len(words) < min_chunk_words:
        return
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if len(chunk) >= min_chunk_words:
            yield " ".join(chunk)
        if start + chunk_words >= len(words):
            break


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text(" ".join(pages))


def iter_text_sources(raw_paper_root: Path, raw_hf_root: Path) -> Iterable[tuple[str, str, str]]:
    for pdf in sorted(raw_paper_root.rglob("*.pdf")):
        try:
            yield "paper_pdf", str(pdf), read_pdf(pdf)
        except Exception as exc:
            console.print(f"Skipping {pdf}: {exc}")
    for path in sorted(raw_hf_root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".json", ".jsonl"}:
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            continue
        try:
            yield "hf_text", str(path), normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            console.print(f"Skipping {path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert papers and text datasets into chunked JSONL corpus.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    corpus_cfg = config["research_corpus"]
    output = Path(args.output or Path(paths["processed_research_root"]) / "research_corpus.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for source_type, source_path, text in iter_text_sources(Path(paths["raw_paper_root"]), Path(paths["raw_hf_root"])):
            for index, chunk in enumerate(
                chunk_words(
                    text,
                    int(corpus_cfg["chunk_words"]),
                    int(corpus_cfg["chunk_overlap_words"]),
                    int(corpus_cfg["min_chunk_words"]),
                )
            ):
                record = {
                    "id": f"{source_type}:{source_path}:{index}",
                    "source_type": source_type,
                    "source_path": source_path,
                    "chunk_index": index,
                    "text": chunk,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1
    console.print(f"Wrote {count} research chunks to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

