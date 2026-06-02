from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_readme_math", Path(__file__).resolve().parents[1] / "scripts" / "check_readme_math.py")
assert _spec is not None and _spec.loader is not None
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)


def test_operatorname_is_on_the_denylist() -> None:
    assert "operatorname" in crm.GITHUB_DISALLOWED_MACROS


def test_extract_math_blocks_finds_display_and_inline() -> None:
    md = "text $a+b$ more\n$$\\frac{x}{y}$$\nend"
    assert crm.extract_math_blocks(md) == ["\\frac{x}{y}", "a+b"]


def test_disallowed_flags_operatorname_but_not_mathrm() -> None:
    assert crm.disallowed_in_text(r"\operatorname{logit}(x)") == ["operatorname"]
    assert crm.disallowed_in_text(r"\mathrm{logit}(x)") == []


def test_whole_token_match_no_false_positive_on_left_or_default() -> None:
    # \let must not match inside \left; \def must not match inside \default.
    assert crm.disallowed_in_text(r"\left[ x \right] \mathrm{default}") == []


def test_scan_file_detects_disallowed(tmp_path) -> None:
    p = tmp_path / "doc.md"
    p.write_text(r"intro $\operatorname{P}(A)=1$ outro")
    assert crm.scan_file(p) == [("operatorname", r"\operatorname{P}(A)=1")]


def test_readme_has_no_disallowed_macros() -> None:
    """Regression guard: the live README must stay GitHub-renderable."""
    assert crm.scan_file(Path(__file__).resolve().parents[1] / "README.md") == []
