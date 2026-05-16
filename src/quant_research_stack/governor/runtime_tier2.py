from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.grammar import generate_full_grammar
from quant_research_stack.governor.prompts import SYSTEM_PROMPT, build_user_message
from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class Tier2Runtime:
    gguf_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_new_tokens: int = 384

    def __post_init__(self) -> None:
        self._llm = None
        self._grammar_text = generate_full_grammar()

    def _load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama, LlamaGrammar

        self._llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(self._grammar_text)

    def govern(self, signal, retrieval: Iterable[Chunk] | None) -> GovernorVerdict:
        self._load()
        chunks = list(retrieval or [])
        prompt = (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n{build_user_message(signal, chunks)} [/INST]"
        )
        out = self._llm(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            grammar=self._grammar,
        )
        text = out["choices"][0]["text"].strip()
        payload = json.loads(text)
        return GovernorVerdict.model_validate(payload)
