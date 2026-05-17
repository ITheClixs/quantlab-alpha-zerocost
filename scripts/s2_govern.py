from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from sentence_transformers import SentenceTransformer

from quant_research_stack.governor.audit import AuditWriter
from quant_research_stack.governor.bm25_index import load_bm25_index
from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import load_corpus
from quant_research_stack.governor.dense_index import load_dense_index
from quant_research_stack.governor.escalator import EscalationConfig, S1Signal, govern_signal
from quant_research_stack.governor.query_builder import build_query
from quant_research_stack.governor.reranker import CrossEncoderReranker
from quant_research_stack.governor.retrieval import HybridRetriever
from quant_research_stack.governor.runtime_tier1 import Tier1Runtime
from quant_research_stack.governor.runtime_tier2 import Tier2Runtime
from quant_research_stack.governor.runtime_tier3 import Tier3Runtime
from quant_research_stack.governor.transport import VerdictWriter

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 governor daemon — tail S1 predictions, write verdicts.")
    p.add_argument("--config", default="configs/governor.yaml")
    p.add_argument("--predictions", required=True, help="Path to S1 predictions Parquet (or directory of Parquets).")
    p.add_argument("--once", action="store_true", help="Process current rows then exit (CI smoke).")
    return p.parse_args()


def _today_path(root: Path) -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return root / f"{today}.jsonl"


def _signal_from_row(row: dict) -> S1Signal:
    return S1Signal(
        signal_id=str(row.get("signal_id") or row.get("id") or f"sig-{int(time.time()*1e6)}"),
        symbol=str(row.get("symbol", "UNKNOWN")),
        direction=int(row.get("direction", 0)),
        confidence=float(row.get("confidence", 0.0)),
        horizon_minutes=int(row.get("horizon_minutes", 15)),
        regime_hint=row.get("regime_hint"),
        recent_vol_label=str(row.get("recent_vol_label", "med")),
        trade_size_pct=float(row.get("trade_size_pct", 0.0)),
    )


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if Path("KILL_TRADING").exists():
        console.print("[red]KILL_TRADING present in repo root; refusing to start[/red]")
        return 4

    corpus = load_corpus(cfg["corpus"]["parquet_dir"])
    metadata_path = Path(cfg["retrieval"]["index_dir"]) / "index_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]missing {metadata_path}; run scripts/governor_build_indexes.py[/red]")
        return 3

    bm25 = load_bm25_index(Path(cfg["retrieval"]["index_dir"]) / "bm25_index.pkl")
    dense = load_dense_index(
        Path(cfg["retrieval"]["index_dir"]) / "dense_index.npy",
        Path(cfg["retrieval"]["index_dir"]) / "dense_index.faiss",
        chunk_ids=tuple(c.id for c in corpus),
    )
    reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model_dir"])
    retriever = HybridRetriever(corpus=corpus, bm25=bm25, dense=dense, reranker=reranker)
    embedder = SentenceTransformer(str(cfg["retrieval"]["embedding_model_dir"]))

    tier1 = Tier1Runtime(
        base_model_dir=Path(cfg["tiers"]["tier1"]["base_model_dir"]),
        adapter_dir=Path(cfg["tiers"]["tier1"]["adapter_dir"]) if Path(cfg["tiers"]["tier1"]["adapter_dir"]).exists() else None,
        max_new_tokens=int(cfg["tiers"]["tier1"]["max_new_tokens"]),
    )
    tier2 = None
    if bool(cfg["tiers"]["tier2"].get("enabled", True)):
        tier2 = Tier2Runtime(
            gguf_path=Path(cfg["tiers"]["tier2"]["gguf_path"]),
            n_ctx=int(cfg["tiers"]["tier2"]["n_ctx"]),
            n_gpu_layers=int(cfg["tiers"]["tier2"]["n_gpu_layers"]),
            max_new_tokens=int(cfg["tiers"]["tier2"]["max_new_tokens"]),
        )
    tier3 = None
    if bool(cfg["tiers"]["tier3"].get("enabled", True)):
        tier3 = Tier3Runtime(
            gguf_path=Path(cfg["tiers"]["tier3"]["gguf_path"]),
            output_path=_today_path(Path(cfg["transport"]["tier3_verdicts_dir"])),
            n_ctx=int(cfg["tiers"]["tier3"]["n_ctx"]),
            n_gpu_layers=int(cfg["tiers"]["tier3"]["n_gpu_layers"]),
            max_new_tokens=int(cfg["tiers"]["tier3"]["max_new_tokens"]),
        )
        tier3.start()

    class _Runtimes:
        pass

    runtimes = _Runtimes()
    runtimes.tier1 = tier1
    runtimes.tier2 = tier2
    runtimes.tier3 = tier3

    primary_writer = VerdictWriter(_today_path(Path(cfg["transport"]["primary_verdicts_dir"])))
    audit = AuditWriter(_today_path(Path(cfg["transport"]["audit_log_dir"])))
    esc_cfg = EscalationConfig(
        tier2_required_when_tier1_passes_above_confidence=float(cfg["tiers"]["tier2"]["triggered_when_tier1_passes_above_confidence"]),
        tier3_required_when_trade_size_pct_above=float(cfg["tiers"]["tier3"]["triggered_when_trade_size_pct_above"]),
        rerank_to_k=int(cfg["retrieval"]["rerank_to_k"]),
    )

    def retrieve_top_k(signal: S1Signal, k: int):
        query = build_query(signal)
        qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        return retriever.retrieve(query, bm25_n=int(cfg["retrieval"]["bm25_top_n"]), dense_n=int(cfg["retrieval"]["dense_top_n"]), k=k, query_vector=qv)

    stop = False

    def _handle(_signum, _frame):
        nonlocal stop
        stop = True
        console.print("[yellow]signal received; draining[/yellow]")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    def _process_rows(df: pl.DataFrame) -> int:
        n = 0
        for row in df.iter_rows(named=True):
            sig = _signal_from_row(row)
            audit.record(event="signal_received", payload={"signal_id": sig.signal_id, "symbol": sig.symbol})
            verdict = govern_signal(sig, esc_cfg, runtimes, corpus, retrieve_top_k)
            verdict, invalid = resolve_citations(verdict, corpus)
            audit.record(event="governor_verdict", payload={"signal_id": sig.signal_id, "decision": verdict.decision.value, "invalid_cited": invalid})
            primary_writer.write(verdict)
            n += 1
            if Path("KILL_TRADING").exists():
                return n
        return n

    seen_paths: set[str] = set()
    while not stop:
        candidate = Path(args.predictions)
        if candidate.is_file():
            df = pl.read_parquet(candidate)
            _process_rows(df)
        else:
            for shard in sorted(candidate.glob("*.parquet")):
                if str(shard) in seen_paths:
                    continue
                df = pl.read_parquet(shard)
                _process_rows(df)
                seen_paths.add(str(shard))
        if args.once:
            break
        time.sleep(2.0)

    if tier3 is not None:
        tier3.stop()
    primary_writer.close_and_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
