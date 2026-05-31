from __future__ import annotations

import hashlib
import random


def _seed_from(chunk_id: str, seed: int) -> int:
    h = hashlib.sha256(f"{chunk_id}::{seed}".encode()).hexdigest()
    return int(h[:8], 16)


def label_chunk_with_seed(chunk_id: str, chunk_text: str, *, seed: int) -> dict:
    rng = random.Random(_seed_from(chunk_id, seed))
    direction = rng.choice([-1, 0, 1])
    confidence = round(rng.random(), 4)
    horizon = rng.choice([1, 5, 15, 30, 60])
    regime_choices = ("trending", "mean_reverting", "high_vol", "low_vol", "unknown")
    regime = rng.choice(regime_choices)
    text_low = chunk_text.lower()
    if "mean reversion" in text_low and direction == 1 and horizon <= 5:
        decision = "veto"
        rationale = "long-direction at short horizon contradicts mean-reversion in cited chunk"
    elif "trending" in text_low and direction == 0:
        decision = "veto"
        rationale = "flat direction contradicts trending evidence"
    else:
        decision = "pass"
        rationale = "synthetic-pass case"
    return {
        "signal_id": f"sig-{_seed_from(chunk_id, seed):08x}",
        "decision": decision,
        "direction": direction,
        "confidence": confidence,
        "horizon_minutes": horizon,
        "regime_tag": regime,
        "rationale_short": rationale[:200],
        "cited_paper_chunk_ids": [chunk_id],
        "contradictions_flagged": [],
    }
