# ADR 0003: GBNF grammar constraints for LLM governor outputs

## Status
Accepted, 2026-05-14.

## Context
The LLM governor (S2) must emit decisions S4 can act on. Free-text outputs are
unreliable: the LLM can hallucinate fields, return malformed JSON, or write rationales
without citations. We need machine-parseable outputs by construction, not by post-hoc
validation.

## Decision
Every LLM governor inference uses llama.cpp's GBNF grammar to constrain token sampling
to a grammar that produces only valid JSON matching the signal schema. Non-JSON tokens
are physically impossible to sample. The schema requires `cited_paper_chunk_ids` to be
a non-empty array; outputs missing citations are auto-mapped to
`decision: insufficient_evidence`.

## Consequences
+ Zero hallucinated schema fields.
+ Zero non-JSON outputs reaching S4.
+ Citation invariant is enforceable.
- Slightly slower token sampling (grammar evaluation overhead, ~5-10%).
- Requires llama.cpp Python bindings, not raw transformers.
