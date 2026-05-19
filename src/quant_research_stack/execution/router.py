from __future__ import annotations

from typing import Any

from quant_research_stack.brokers.base import BrokerAdapter


class UnknownBrokerError(KeyError):
    pass


_NULL_ADAPTERS = {"null_broker"}
_LIVE_ADAPTERS = {"alpaca_live", "binance_live"}


class BrokerRouter:
    """Resolves a stage + asset class to a concrete BrokerAdapter instance."""

    def __init__(self, brokers_cfg: dict[str, Any]) -> None:
        routes = brokers_cfg.get("stage_routes")
        if not routes:
            raise ValueError("configs/brokers.yaml is missing 'stage_routes'")
        self._routes = routes
        self._brokers = brokers_cfg.get("brokers", {})

    def resolved_name(self, stage: str, asset_class: str) -> str:
        if stage not in self._routes:
            raise UnknownBrokerError(f"stage {stage!r} not in stage_routes")
        route = self._routes[stage]
        name = route.get(asset_class)
        if not name:
            raise UnknownBrokerError(f"asset_class {asset_class!r} not in stage_routes[{stage!r}]")
        return str(name)

    def resolve(self, stage: str, asset_class: str) -> BrokerAdapter:
        name = self.resolved_name(stage, asset_class)
        return self._instantiate(name)

    def _instantiate(self, name: str) -> BrokerAdapter:
        if name in _NULL_ADAPTERS:
            from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
            from quant_research_stack.brokers.null_broker import NullBroker

            return NullBroker(fill_model=FillModel(FillModelConfig()))
        if name == "alpaca_paper":
            from quant_research_stack.brokers.alpaca_paper import AlpacaPaper

            cfg = self._brokers.get("alpaca_paper", {})
            return AlpacaPaper(
                credentials_path=cfg.get("credentials_path", "~/.alpaca/paper_keys.json"),
                base_url=cfg.get("base_url", "https://paper-api.alpaca.markets"),
            )
        if name == "binance_testnet":
            from quant_research_stack.brokers.binance_testnet import BinanceTestnet

            cfg = self._brokers.get("binance_testnet", {})
            return BinanceTestnet(
                credentials_path=cfg.get("credentials_path", "~/.binance/testnet_keys.json"),
                rest_base_url=cfg.get("rest_base_url", "https://testnet.binance.vision"),
            )
        if name in _LIVE_ADAPTERS:
            raise ImportError(
                f"Live broker not installed: {name!r}. "
                "Live brokers (S4.1) require two-person review per CLAUDE.md §1.13.",
            )
        raise UnknownBrokerError(f"unknown broker {name!r}")
