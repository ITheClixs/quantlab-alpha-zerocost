from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dedupe_and_verify import (  # noqa: E402
    build_inventory,
    classify_bucket,
    file_sha256,
    group_duplicates,
)


def test_file_sha256_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"hello")
    assert file_sha256(path) == hashlib.sha256(b"hello").hexdigest()


def test_classify_bucket_handles_known_roots(tmp_path: Path) -> None:
    roots = {
        "hf_datasets": tmp_path / "data" / "raw" / "huggingface",
        "kaggle": tmp_path / "data" / "raw" / "kaggle",
        "models": tmp_path / "models" / "huggingface",
        "papers_and_derived": tmp_path / "data" / "raw" / "papers",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    f1 = roots["hf_datasets"] / "x" / "y.parquet"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("a")
    f2 = roots["models"] / "x" / "config.json"
    f2.parent.mkdir(parents=True, exist_ok=True)
    f2.write_text("b")
    f3 = roots["papers_and_derived"] / "x.pdf"
    f3.write_text("c")
    assert classify_bucket(f1, roots) == "hf_datasets"
    assert classify_bucket(f2, roots) == "models"
    assert classify_bucket(f3, roots) == "papers_and_derived"
    other = tmp_path / "outside" / "x.bin"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("d")
    assert classify_bucket(other, roots) == "other"


def test_group_duplicates_collapses_same_hash() -> None:
    items = [
        {"path": "a", "sha256": "X", "size_bytes": 1, "bucket": "hf_datasets"},
        {"path": "b", "sha256": "X", "size_bytes": 1, "bucket": "kaggle"},
        {"path": "c", "sha256": "Y", "size_bytes": 1, "bucket": "models"},
    ]
    dups = group_duplicates(items)
    assert dups == [{"sha256": "X", "paths": ["a", "b"]}]


def test_build_inventory_walks_files(tmp_path: Path) -> None:
    hf = tmp_path / "data" / "raw" / "huggingface" / "ds1"
    hf.mkdir(parents=True)
    (hf / "a.parquet").write_bytes(b"abc")
    (hf / "b.parquet").write_bytes(b"abc")
    models = tmp_path / "models" / "huggingface" / "m1"
    models.mkdir(parents=True)
    (models / "config.json").write_bytes(b"unique")

    roots = {
        "hf_datasets": tmp_path / "data" / "raw" / "huggingface",
        "kaggle": tmp_path / "data" / "raw" / "kaggle",
        "models": tmp_path / "models" / "huggingface",
        "papers_and_derived": tmp_path / "data" / "raw" / "papers",
    }
    inventory = build_inventory(roots)
    assert inventory["total_size_bytes"] == 3 + 3 + 6
    assert inventory["by_bucket"]["hf_datasets"] == 6
    assert inventory["by_bucket"]["models"] == 6
    assert len(inventory["items"]) == 3
    assert len(inventory["duplicates"]) == 1
