from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha.meta_features import MetaFeatureCache, finbert_logits_cached, hash_input_dataframe


def test_hash_input_is_stable() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    h1 = hash_input_dataframe(df)
    h2 = hash_input_dataframe(df)
    assert h1 == h2


def test_meta_feature_cache_roundtrip(tmp_path: Path) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    key = "test_key"
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    cache.put(key, arr)
    out = cache.get(key)
    assert out is not None
    assert np.array_equal(out, arr)


def test_meta_feature_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    assert cache.get("absent") is None


def test_finbert_logits_cached_uses_cache(tmp_path: Path, monkeypatch) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
    cache.put("finbert::abc", arr)

    def fake_run(texts: list[str]) -> np.ndarray:
        raise AssertionError("should not be called when cached")

    out = finbert_logits_cached(["x", "y"], cache=cache, cache_key="finbert::abc", runner=fake_run)
    assert np.array_equal(out, arr)
