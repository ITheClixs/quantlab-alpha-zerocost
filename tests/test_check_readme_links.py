from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_readme_links", Path(__file__).resolve().parents[1] / "scripts" / "check_readme_links.py")
assert _spec is not None and _spec.loader is not None
crl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crl)


def test_extract_skips_http_and_anchors() -> None:
    md = "[a](https://x.com) [b](docs/y.md) [c](#section) [d](mailto:x@y.z) [e](docs/z.md#frag)"
    assert crl.extract_relative_links(md) == ["docs/y.md", "docs/z.md"]


def test_missing_links_detects_only_absent(tmp_path) -> None:
    (tmp_path / "README.md").write_text("[x](nope.md) [y](real.md)")
    (tmp_path / "real.md").write_text("hi")
    assert crl.missing_links(tmp_path / "README.md") == ["nope.md"]
