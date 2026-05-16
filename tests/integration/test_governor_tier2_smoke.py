from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.governor.escalator import S1Signal
from quant_research_stack.governor.runtime_tier2 import Tier2Runtime
from quant_research_stack.governor.signal_schema import GovernorVerdict


@pytest.mark.governor_slow
def test_tier2_emits_valid_json_on_5_fixtures() -> None:
    rt = Tier2Runtime(
        gguf_path=Path("models/huggingface/bartowski__Mistral-Small-Instruct-2409-GGUF/Mistral-Small-Instruct-2409-Q4_K_M.gguf"),
    )
    fixtures = [
        S1Signal(signal_id=f"sig-{i:08d}", symbol="BTCUSDT", direction=1, confidence=0.8,
                 horizon_minutes=15, regime_hint="trending", recent_vol_label="med",
                 trade_size_pct=0.5)
        for i in range(5)
    ]
    for sig in fixtures:
        v = rt.govern(sig, retrieval=[])
        assert isinstance(v, GovernorVerdict)
