from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_funding_carry_paper", Path(__file__).resolve().parents[3] / "scripts" / "run_funding_carry_paper.py")
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_cli_exposes_main_and_build_source() -> None:
    assert hasattr(cli, "main")
    assert hasattr(cli, "build_rest_source")
