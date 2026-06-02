"""Verify README/report math uses no macros that GitHub's MathJax rejects.

GitHub renders ``$...$`` / ``$$...$$`` with MathJax in *safe* mode and additionally
strips a set of macros. When a blocked macro is used, GitHub shows a red box reading
"The following macros are not allowed: <name>" and the equation fails to render. KaTeX
(used elsewhere for syntax validation) happily accepts several of these, so a pure
KaTeX check is NOT sufficient — this denylist closes that gap.

GitHub does not publish an authoritative list (see github/markup#1688, which only
confirms ``\\operatorname``). This denylist is therefore a conservative, practical set:
  1. the confirmed failure (``\\operatorname`` and its sibling ``\\DeclareMathOperator``);
  2. the HTML / link / style / resource macros blocked by MathJax's ``safe`` extension;
  3. macro-definition / cross-reference commands GitHub does not support in math.
Add to ``GITHUB_DISALLOWED_MACROS`` as new offenders are discovered.

Run: PYTHONPATH=src uv run python scripts/check_readme_math.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Macro names WITHOUT the leading backslash.
GITHUB_DISALLOWED_MACROS: frozenset[str] = frozenset(
    {
        # (1) Confirmed blocked on GitHub (github/markup#1688).
        "operatorname",
        "DeclareMathOperator",
        # (2) MathJax `safe`-mode HTML / link / style / resource macros.
        "href",
        "unicode",
        "htmlClass",
        "htmlId",
        "htmlData",
        "htmlStyle",
        "cssId",
        "class",
        "style",
        "bbox",
        "require",
        "includegraphics",
        # (3) Macro-definition / cross-reference commands unsupported in GitHub math.
        "def",
        "newcommand",
        "renewcommand",
        "newenvironment",
        "renewenvironment",
        "let",
        "label",
        "ref",
        "eqref",
        "input",
        "include",
        "verb",
    }
)

# Default files that are rendered as math on GitHub.
DEFAULT_TARGETS: tuple[str, ...] = (
    "README.md",
    "reports/2026-06-02-COMPETITIVE-LANDSCAPE-PUBLIC-QUANT.md",
)

_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\\)\$([^$\n]+?)\$")
# Whole-token macro match: backslash + name not followed by another letter, so
# `\let` never matches inside `\left` and `\def` never matches inside `\default`.
_MACRO_RE = re.compile(
    r"\\(" + "|".join(sorted(GITHUB_DISALLOWED_MACROS, key=len, reverse=True)) + r")(?![A-Za-z])"
)


def extract_math_blocks(md: str) -> list[str]:
    """Return every inline and display math expression in a markdown string."""
    blocks = [m.strip() for m in _DISPLAY_RE.findall(md)]
    without_display = _DISPLAY_RE.sub("", md)
    blocks += [m.strip() for m in _INLINE_RE.findall(without_display)]
    return blocks


def disallowed_in_text(tex: str) -> list[str]:
    """Return the disallowed macro names present in one math expression."""
    return sorted({m.group(1) for m in _MACRO_RE.finditer(tex)})


def scan_file(md_path: Path) -> list[tuple[str, str]]:
    """Return ``(macro, snippet)`` findings for one markdown file."""
    findings: list[tuple[str, str]] = []
    for tex in extract_math_blocks(md_path.read_text()):
        for macro in disallowed_in_text(tex):
            findings.append((macro, tex))
    return findings


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or [Path(t) for t in DEFAULT_TARGETS]
    total = 0
    for path in targets:
        if not path.exists():
            continue
        for macro, tex in scan_file(path):
            total += 1
            print(f"DISALLOWED MACRO \\{macro} in {path}:")
            print(f"    {tex}")
    if total:
        print(f"\n{total} GitHub-disallowed macro use(s) found; "
              f"replace e.g. \\operatorname{{x}} -> \\mathrm{{x}}.")
        sys.exit(1)
    print("no GitHub-disallowed math macros found")


if __name__ == "__main__":
    main()
