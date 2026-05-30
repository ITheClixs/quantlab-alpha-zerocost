"""Verify every relative markdown link in README.md resolves to a real file.

Guards the research paper's in-repo hyperlink map ("blue underlines"). Skips http(s)/
mailto and pure #anchors; strips #fragments from relative links before existence-check.

Run: PYTHONPATH=src uv run python scripts/check_readme_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_relative_links(md: str) -> list[str]:
    out: list[str] = []
    for raw in _LINK_RE.findall(md):
        target = raw.split("#")[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append(target)
    return out


def missing_links(md_path: Path) -> list[str]:
    md = md_path.read_text()
    miss = [link for link in extract_relative_links(md)
            if not (md_path.parent / link).resolve().exists()]
    return sorted(set(miss))


def main() -> None:
    miss = missing_links(Path("README.md"))
    if miss:
        print("MISSING README LINKS:")
        for x in miss:
            print(f"  {x}")
        sys.exit(1)
    print("all README relative links resolve")


if __name__ == "__main__":
    main()
