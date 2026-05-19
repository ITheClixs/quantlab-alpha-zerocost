from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class Tier1Runtime:
    base_model_dir: Path
    adapter_dir: Path | None
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
        model_factory = cast(Any, AutoModelForCausalLM)
        model: Any = model_factory.from_pretrained(self.base_model_dir).to(device)
        if self.adapter_dir is not None and Path(self.adapter_dir).exists():
            model = PeftModel.from_pretrained(model, self.adapter_dir).to(device)
        model.eval()
        self._model = model
        self._device = device

    def govern(self, signal, retrieval: Iterable[Chunk] | None) -> GovernorVerdict:
        import torch

        self._load()
        prompt = self._render_prompt(signal)
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None
        assert model is not None
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self._device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            payload = json.loads(text)
            payload["cited_paper_chunk_ids"] = []
            payload["contradictions_flagged"] = []
            return GovernorVerdict.model_validate(payload)
        except Exception:
            return GovernorVerdict.model_validate({
                "signal_id": signal.signal_id,
                "decision": "insufficient_evidence",
                "direction": signal.direction,
                "confidence": 0.0,
                "horizon_minutes": signal.horizon_minutes,
                "regime_tag": signal.regime_hint or "unknown",
                "rationale_short": "tier1 parse failure",
                "cited_paper_chunk_ids": [],
                "contradictions_flagged": [],
            })

    @staticmethod
    def _render_prompt(signal) -> str:
        return (
            "<|im_start|>system\n"
            "You are QuantLab's fast veto governor. Output strict JSON with fields "
            "signal_id, decision (pass|veto), direction, confidence, horizon_minutes, "
            "regime_tag, rationale_short (<=120 chars), cited_paper_chunk_ids: [], "
            "contradictions_flagged: [].\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nSignal: {signal.signal_id} {signal.symbol} dir={signal.direction} "
            f"conf={signal.confidence:.4f} horizon={signal.horizon_minutes}m "
            f"regime={signal.regime_hint or 'unknown'}\nRespond with JSON only.\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
