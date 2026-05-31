"""Profile + universe configuration loader (spec §2.3, §6.6)."""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.signal_research.data.manifest import DataQualityTier
from quant_research_stack.signal_research.data.profiles import (
    ProfileConfig,
    list_profiles,
    load_profile,
)


def test_list_profiles_returns_four_canonical_profiles() -> None:
    profiles = list_profiles(Path("configs/signal_research_profiles"))
    assert set(profiles) == {"sp500", "nasdaq", "crypto", "futures_proxy"}


def test_load_nasdaq_profile_has_four_universes() -> None:
    cfg: ProfileConfig = load_profile(
        Path("configs/signal_research_profiles/nasdaq.yaml")
    )
    assert cfg.profile == "nasdaq"
    universe_names = {u.name for u in cfg.universes}
    assert universe_names == {
        "nasdaq_index_proxy", "nasdaq_100_current", "nasdaq_mega_cap_focus", "user_focus_tech"
    }


def test_load_nasdaq_profile_nasdaq_100_is_survivorship_warned() -> None:
    cfg = load_profile(Path("configs/signal_research_profiles/nasdaq.yaml"))
    ndx100 = next(u for u in cfg.universes if u.name == "nasdaq_100_current")
    assert ndx100.data_quality_label == DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY
    assert ndx100.constituent_survivorship_applicable is True


def test_load_crypto_profile_carries_directly_traded_semantics() -> None:
    cfg = load_profile(Path("configs/signal_research_profiles/crypto.yaml"))
    univ = cfg.universes[0]
    assert univ.constituent_survivorship_applicable is False
