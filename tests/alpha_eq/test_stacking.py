"""Linear stacker — L2-regularized, signed-diagnostic, large-negative-weight flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.stacking import (
    LinearStackerEq,
    StackerArtifact,
    flag_large_negative_weights,
)


def test_stacker_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((300, 3))
    y = oof.sum(axis=1) + rng.standard_normal(300) * 0.1
    s = LinearStackerEq(
        alpha=1e-3,
        prefer_non_negative=True,
        feature_order=("ridge", "lgb", "xgb"),
    )
    s.fit(oof_predictions=oof, y=y)
    p = s.predict(oof[:5])
    assert p.shape == (5,)
    assert np.all(s.weights >= -1e-9)


def test_stacker_signed_variant() -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((300, 3))
    y = oof[:, 0] - oof[:, 1] + rng.standard_normal(300) * 0.01
    s = LinearStackerEq(
        alpha=1e-3,
        prefer_non_negative=False,
        feature_order=("good", "bad", "noise"),
    )
    s.fit(oof_predictions=oof, y=y)
    assert np.any(s.weights < 0)


def test_flag_large_negative_weights() -> None:
    flagged = flag_large_negative_weights(
        weights=np.array([0.5, -0.30, 0.10]),
        names=("a", "b", "c"),
        threshold=-0.25,
    )
    assert "b" in flagged


def test_stacker_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((100, 3))
    y = oof.sum(axis=1)
    s = LinearStackerEq(alpha=1e-3, prefer_non_negative=True,
                       feature_order=("ridge", "lgb", "xgb"))
    s.fit(oof_predictions=oof, y=y)
    out = tmp_path / "stacker.joblib"
    s.save(out)
    s2 = LinearStackerEq.load(out)
    np.testing.assert_allclose(s.predict(oof[:5]), s2.predict(oof[:5]), atol=1e-12)
    art = StackerArtifact.from_model(s2)
    assert art.feature_order == ("ridge", "lgb", "xgb")
