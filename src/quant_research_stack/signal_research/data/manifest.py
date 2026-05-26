"""5-tier data-quality manifest (spec §2.2, §2.5).

The 5 tiers:
- pit_safe
- partial_pit_universe
- public_snapshot_not_pit
- survivorship_prototype_only
- unknown (sentinel — must be resolved before downstream use)

`directly_traded_etf` is NOT a tier value. Directly-traded instruments
(SPY, QQQ, BTCUSDT, etc.) keep their tier (typically public_snapshot_not_pit)
and carry `constituent_survivorship_applicable: false` separately.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataQualityTier(enum.StrEnum):
    PIT_SAFE = "pit_safe"
    PARTIAL_PIT_UNIVERSE = "partial_pit_universe"
    PUBLIC_SNAPSHOT_NOT_PIT = "public_snapshot_not_pit"
    SURVIVORSHIP_PROTOTYPE_ONLY = "survivorship_prototype_only"
    UNKNOWN = "unknown"


class ManifestMismatchError(RuntimeError):
    pass


class DataSourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_url: str
    fetch_timestamp_utc: str
    path: str
    sha256: str
    row_count: int
    symbol_count: int
    date_range_start: str
    date_range_end: str
    schema_fingerprint: str
    data_quality_tier: DataQualityTier
    constituent_survivorship_applicable: bool
    vendor_disclosure: str
    timestamp_convention: str
    warnings: list[str] = Field(default_factory=list)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def write_manifest(path: Path, manifest: DataSourceManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(_canonical_json(manifest.model_dump(mode="json")))


def load_and_verify_manifest(
    path: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> DataSourceManifest:
    if not Path(path).exists():
        raise ManifestMismatchError(f"manifest missing: {path}")
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ManifestMismatchError(f"manifest is not valid JSON: {exc}") from exc
    try:
        m = DataSourceManifest.model_validate(payload)
    except Exception as exc:
        raise ManifestMismatchError(f"manifest schema error: {exc}") from exc
    for key, sha in expected_sha256.items():
        if m.source_name != key:
            continue
        if m.sha256 != sha:
            raise ManifestMismatchError(
                f"sha256 mismatch for {key}: expected={sha} got={m.sha256}"
            )
    return m
