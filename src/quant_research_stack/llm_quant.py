from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

VALID_DIRECTIONS = {"up", "down", "flat"}


@dataclass(frozen=True)
class ResearchChunk:
    id: str
    text: str
    source_path: str | None = None


@dataclass(frozen=True)
class LocalModelChoice:
    model_id: str
    local_dir: Path
    gguf_file: Path | None
    role: str

    @property
    def available(self) -> bool:
        return self.gguf_file is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "local_dir": str(self.local_dir),
            "gguf_file": str(self.gguf_file) if self.gguf_file else None,
            "role": self.role,
            "available": self.available,
        }


@dataclass(frozen=True)
class QuantSignal:
    signal_direction: str
    confidence: float
    horizon: int
    features_used: tuple[str, ...]
    research_support: tuple[str, ...]
    risk_flags: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_direction": self.signal_direction,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "features_used": list(self.features_used),
            "research_support": list(self.research_support),
            "risk_flags": list(self.risk_flags),
            "rationale": self.rationale,
        }


def choose_local_model(config: dict[str, Any], repo_root: str | Path = ".") -> LocalModelChoice:
    runtime = config.get("llm_runtime", {}) or {}
    root = Path(repo_root)
    candidates = [
        ("primary", runtime.get("primary_model_id"), runtime.get("primary_local_dir")),
        ("fallback", runtime.get("fallback_model_id"), runtime.get("fallback_local_dir")),
    ]
    for role, model_id, raw_dir in candidates:
        if not model_id or not raw_dir:
            continue
        local_dir = root / Path(raw_dir)
        gguf_files = sorted(local_dir.glob("*.gguf"))
        preferred_quant = str(runtime.get("preferred_quant", "Q4_K_M")).lower()
        preferred = [path for path in gguf_files if preferred_quant in path.name.lower()]
        selected = (preferred or gguf_files or [None])[0]
        if selected is not None:
            return LocalModelChoice(str(model_id), local_dir, selected, role)
    raw_dir = runtime.get("primary_local_dir", "models/huggingface")
    return LocalModelChoice(str(runtime.get("primary_model_id", "unknown")), root / Path(raw_dir), None, "missing")


def load_research_chunks(path: str | Path, limit: int | None = None) -> list[ResearchChunk]:
    chunks: list[ResearchChunk] = []
    corpus_path = Path(path)
    if not corpus_path.exists():
        return chunks
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(chunks) >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            chunks.append(
                ResearchChunk(
                    id=str(payload.get("id")),
                    text=str(payload.get("text", "")),
                    source_path=payload.get("source_path"),
                )
            )
    return chunks


def retrieve_chunks(query: str, chunks: list[ResearchChunk], *, top_k: int = 4) -> list[ResearchChunk]:
    terms = {term for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(term) > 2}
    scored = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, len(chunk.text), chunk))
    scored.sort(key=lambda row: (-row[0], row[1], row[2].id))
    return [chunk for _, _, chunk in scored[:top_k]]


def build_signal_prompt(market_context: dict[str, Any], chunks: list[ResearchChunk]) -> str:
    research = "\n".join(f"[{chunk.id}] {chunk.text[:900]}" for chunk in chunks)
    context = json.dumps(market_context, sort_keys=True)
    return (
        "You are a local quantitative research model. Produce one strict JSON object only.\n"
        "Use the market context and cited research chunks. Do not invent unavailable features.\n"
        "Required JSON keys: signal_direction, confidence, horizon, features_used, "
        "research_support, risk_flags, rationale.\n"
        "signal_direction must be one of: up, down, flat. confidence must be between 0 and 1.\n"
        f"Market context: {context}\n"
        f"Research chunks:\n{research}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")
    return json.loads(text[start : end + 1])


def parse_quant_signal(text: str, allowed_research_ids: set[str]) -> QuantSignal:
    payload = _extract_json_object(text)
    required = {"signal_direction", "confidence", "horizon", "features_used", "research_support", "risk_flags", "rationale"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"LLM signal is missing required keys: {missing}")

    direction = str(payload["signal_direction"]).lower()
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid signal_direction: {direction}")
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    horizon = int(payload["horizon"])
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    features_used = tuple(str(item) for item in payload["features_used"])
    research_support = tuple(str(item) for item in payload["research_support"])
    risk_flags = tuple(str(item) for item in payload["risk_flags"])
    if not features_used:
        raise ValueError("features_used must not be empty")
    if not research_support:
        raise ValueError("research_support must not be empty")
    invented = sorted(set(research_support) - allowed_research_ids)
    if invented:
        raise ValueError(f"research_support contains unknown chunk ids: {invented}")

    return QuantSignal(
        signal_direction=direction,
        confidence=confidence,
        horizon=horizon,
        features_used=features_used,
        research_support=research_support,
        risk_flags=risk_flags,
        rationale=str(payload["rationale"]),
    )


def call_openai_compatible_local_model(prompt: str, *, base_url: str = "http://localhost:8080/v1", model: str = "local-gguf", timeout: int = 120) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"])
