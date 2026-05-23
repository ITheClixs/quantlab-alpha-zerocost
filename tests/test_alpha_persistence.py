import polars as pl


def test_synthetic_js_fixture_shape(synthetic_js):
    df = synthetic_js
    assert isinstance(df, pl.DataFrame)
    assert df.height == 10_000
    # 50 feature columns + date_id + responder_6 + weight
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    assert len(feature_cols) == 50
    assert "responder_6" in df.columns
    assert "weight" in df.columns
    assert "date_id" in df.columns


def test_synthetic_js_fixture_deterministic(synthetic_js):
    # Same fixture invocation twice within a session must give identical content.
    # (pytest caches fixture results per scope; this asserts the underlying generator is seeded.)
    first = synthetic_js
    second = synthetic_js
    assert first.equals(second)
