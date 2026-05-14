# ADR 0005: LLM governor outputs must cite local paper chunks

## Status
Accepted, 2026-05-14.

## Context
The 208 GB local corpus includes 48 open-access arXiv PDFs + chunked JSONL + paper Q&A
records. The LLM has been observed in prior projects to confidently produce false
explanations. We want every veto / pass decision to be traceable to evidence on disk.

## Decision
The S2 LLM governor's JSON schema requires `cited_paper_chunk_ids: [str, ...]` to be
non-empty. Outputs lacking citations are auto-rewritten to
`decision: insufficient_evidence` before reaching S4. The cited chunk IDs must resolve
to existing rows in `data/processed/research/parquet/`; otherwise the decision is
audited as `governor_citation_invalid` and treated as `insufficient_evidence`.

## Consequences
+ Every LLM verdict has an evidence trail.
+ Hallucinated explanations become detectable in audit logs.
- Governor latency increases (retrieval round-trip required before generation).
- Coverage gaps in the corpus produce more `insufficient_evidence` than `pass` early on.
