"""S2 LLM Governor for QuantLab Alpha.

Three-tier cascade: Qwen 0.5B + LoRA (fast) → Mistral 22B Q4 (medium with RAG) →
Yi 34B Q4 (deep async). GBNF-constrained JSON outputs with mandatory paper citations.

Spec: docs/superpowers/specs/2026-05-16-quantlab-alpha-s2-governor-design.md
"""
