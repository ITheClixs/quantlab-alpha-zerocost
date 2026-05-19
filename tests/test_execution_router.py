from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.execution.router import BrokerRouter, UnknownBrokerError


def _cfg() -> dict:
    return yaml.safe_load(Path("configs/brokers.yaml").read_text())


def test_paper_stage_resolves_alpaca_paper_for_equity() -> None:
    router = BrokerRouter(_cfg())
    name = router.resolved_name("paper", asset_class="equity")
    assert name == "alpaca_paper"


def test_live_shadow_writes_to_null_broker() -> None:
    router = BrokerRouter(_cfg())
    name = router.resolved_name("live_shadow", asset_class="equity")
    assert name == "null_broker"


def test_live_route_raises_import_error_when_module_missing() -> None:
    router = BrokerRouter(_cfg())
    with pytest.raises(ImportError, match="Live broker not installed"):
        router.resolve("live", asset_class="equity")


def test_unknown_stage_raises() -> None:
    router = BrokerRouter(_cfg())
    with pytest.raises(UnknownBrokerError):
        router.resolve("does_not_exist", asset_class="equity")


def test_live_shadow_returns_null_broker_instance() -> None:
    router = BrokerRouter(_cfg())
    broker = router.resolve("live_shadow", asset_class="equity")
    assert isinstance(broker, NullBroker)
