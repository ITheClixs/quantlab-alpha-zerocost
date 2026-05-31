from __future__ import annotations

import time
from pathlib import Path

import pytest

from quant_research_stack.governor.escalator import S1Signal
from quant_research_stack.governor.runtime_tier3 import Tier3Runtime


@pytest.mark.governor_slow
def test_tier3_writes_verdict_within_60_seconds(tmp_path: Path) -> None:
    out = tmp_path / "tier3_verdicts.jsonl"
    rt = Tier3Runtime(
        gguf_path=Path("models/huggingface/bartowski__Yi-1.5-34B-Chat-GGUF/Yi-1.5-34B-Chat-Q4_K_M.gguf"),
        output_path=out,
    )
    rt.start()
    sig = S1Signal(
        signal_id="sig-async-01", symbol="BTCUSDT", direction=1, confidence=0.95,
        horizon_minutes=15, regime_hint="trending", recent_vol_label="high", trade_size_pct=2.5,
    )
    rt.schedule_async(sig, [])
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if out.exists() and out.read_text().strip():
            break
        time.sleep(1.0)
    rt.stop()
    assert out.exists() and out.read_text().strip(), "tier3 did not write within 60s"
