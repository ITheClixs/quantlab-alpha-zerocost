"""Equity data manifest — single source of truth for data-quality labels,
artifact hashes, and reproducibility metadata (spec §2.1, §2.7, §2.9)."""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataQualityLabel(enum.StrEnum):
    PIT_SAFE = "pit_safe"
    PARTIAL_PIT_UNIVERSE = "partial_pit_universe"
    SURVIVORSHIP_PROTOTYPE_ONLY = "survivorship_prototype_only"


class ManifestMismatchError(RuntimeError):
    """Raised when the manifest disagrees with the on-disk artifacts."""


class ManifestArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    row_count: int
    symbol_count: int
    date_range_start: str
    date_range_end: str
    schema_fingerprint: str
    source_url: str | None = None
    source_dataset_id: str | None = None
    source_snapshot_date: str | None = None


class DelistingAuditCounters(BaseModel):
    model_config = ConfigDict(frozen=True)

    delisted_captured: int = 0
    delisted_missing: int = 0
    merger_captured: int = 0
    merger_missing: int = 0
    ticker_changed: int = 0
    unknown_exit: int = 0


class EquityManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipeline_version: str
    git_sha: str
    artifacts: dict[str, ManifestArtifact]
    data_quality_label: DataQualityLabel
    corporate_action_quality: str
    borrow_source_quality: str
    pit_membership_source: str
    delisting_audit_quality: str
    delisting_audit_counters: DelistingAuditCounters | dict[str, int] = Field(
        default_factory=DelistingAuditCounters
    )
    build_command_line: str
    python_version: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, manifest: EquityManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    Path(path).write_bytes(_canonical_json(payload))


def load_and_verify_manifest(
    path: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> EquityManifest:
    if not Path(path).exists():
        raise ManifestMismatchError(f"manifest missing: {path}")
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ManifestMismatchError(f"manifest is not valid JSON: {exc}") from exc
    try:
        manifest = EquityManifest.model_validate(payload)
    except Exception as exc:
        raise ManifestMismatchError(f"manifest schema error: {exc}") from exc
    for key, sha in expected_sha256.items():
        if key not in manifest.artifacts:
            raise ManifestMismatchError(f"manifest missing artifact key: {key}")
        if manifest.artifacts[key].sha256 != sha:
            raise ManifestMismatchError(
                f"sha256 mismatch for {key}: expected={sha} got={manifest.artifacts[key].sha256}"
            )
    return manifest
