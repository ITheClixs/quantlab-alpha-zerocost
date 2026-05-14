from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.alpha.registry import RunMetadata, RunRegistry


def test_create_run_writes_metadata(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={"x": "deadbeef"}, hyperparams={"alpha": 1.0})
    run_id = reg.create_run(meta)
    assert (tmp_path / run_id / "metadata.json").exists()
    loaded = json.loads((tmp_path / run_id / "metadata.json").read_text())
    assert loaded["git_sha"] == "abcd1234"
    assert loaded["version"] == "0.1.0"


def test_save_artifact_computes_sha256(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={}, hyperparams={})
    run_id = reg.create_run(meta)
    payload = b"some bytes"
    sha = reg.save_artifact(run_id, "model.bin", payload)
    expected = "0d22cdcc10e6d049dbe1af5123d50873fdfc1a4f58306e58cb6241be9472014d"  # sha256 of b"some bytes"
    assert sha == expected
    assert (tmp_path / run_id / "model.bin").read_bytes() == payload


def test_run_id_is_timestamped(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={}, hyperparams={})
    run_id = reg.create_run(meta)
    # run id starts with year
    assert run_id.startswith("20")
    assert len(run_id) >= 14  # YYYYMMDD-HHMMSS at minimum
