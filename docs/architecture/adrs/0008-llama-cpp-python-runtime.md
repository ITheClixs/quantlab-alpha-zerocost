# ADR 0008: Use llama-cpp-python with Metal backend for Tier 2 and Tier 3

## Status
Accepted, 2026-05-16.

## Context
Tier 2 (Mistral 22B) and Tier 3 (Yi 34B) are stored as Q4_K_M GGUF files on disk.
They must run with GBNF grammar enforcement (ADR 0003) and use Apple Silicon's
Metal backend for acceptable latency.

## Decision
`llama-cpp-python` is the runtime for Tier 2 and Tier 3. It is the only mainstream
Python library that:
- accepts a GBNF grammar string and constrains token sampling natively,
- loads GGUF Q4_K_M models without conversion,
- supports Apple Silicon Metal via the `n_gpu_layers=-1` flag.

Tier 1 (Qwen 0.5B + LoRA) uses `transformers` + `peft` because:
- LoRA + GGUF in llama-cpp-python is awkward,
- the model is small enough that HF-native inference on MPS is fast (< 500 ms target),
- HF supports a logits-processor JSON-only fallback if grammar enforcement breaks.

## Consequences
+ Single inference dependency for the heavy tiers.
+ GBNF grammar enforced at sampling time, not post-hoc.
- Two runtime libraries to install (`llama-cpp-python` + `transformers`).
- Building llama-cpp-python with Metal requires `CMAKE_ARGS="-DGGML_METAL=on"`.
