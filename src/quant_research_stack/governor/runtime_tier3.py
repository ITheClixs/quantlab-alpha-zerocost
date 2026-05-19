from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.grammar import generate_full_grammar
from quant_research_stack.governor.prompts import SYSTEM_PROMPT, build_user_message
from quant_research_stack.governor.signal_schema import GovernorVerdict
from quant_research_stack.governor.transport import VerdictWriter


@dataclass
class Tier3Runtime:
    gguf_path: Path
    output_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_new_tokens: int = 512
    queue: Queue = field(default_factory=Queue)
    _llm: Any | None = None
    _grammar: Any | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._grammar_text = generate_full_grammar()
        self._writer = VerdictWriter(self.output_path)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def schedule_async(self, signal, chunks: list[Chunk]) -> None:
        if self._thread is None:
            self.start()
        self.queue.put((signal, chunks))

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

    def _loop(self) -> None:
        self._load()
        assert self._llm is not None
        assert self._grammar is not None
        while not self._stop.is_set():
            item = self.queue.get()
            if item is None:
                break
            signal, chunks = item
            prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{build_user_message(signal, chunks)} [/INST]"
            out = self._llm(prompt, max_tokens=self.max_new_tokens, temperature=0.0, grammar=self._grammar)
            text = out["choices"][0]["text"].strip()
            payload = json.loads(text)
            verdict = GovernorVerdict.model_validate(payload)
            self._writer.write(verdict)
