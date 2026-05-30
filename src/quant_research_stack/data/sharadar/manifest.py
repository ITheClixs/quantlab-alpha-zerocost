"""Sharadar data manifest: record provenance + per-table metadata for reproducibility."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_research_stack.data.sharadar.loaders import LoadedTable

_MANIFEST_PATH = Path("manifests/sharadar/sharadar_data_manifest.json")
_REQUIRED_TABLES = ("SEP", "TICKERS", "ACTIONS")  # SF1 optional


def _pkg_versions() -> dict[str, str]:
    import polars
    out = {"python": platform.python_version(), "polars": polars.__version__}
    try:
        import numpy
        out["numpy"] = numpy.__version__
    except Exception:
        pass
    return out


def build_manifest(loaded: dict[str, LoadedTable], *, vendor: str = "sharadar",
                   license_local_research: bool | None = None) -> dict[str, Any]:
    tables = {name: lt.metadata for name, lt in loaded.items()}
    warnings = [w for lt in loaded.values() for w in lt.metadata.get("warnings", [])]
    missing_required = [t for t in _REQUIRED_TABLES if t not in loaded]
    if missing_required:
        warnings.append(f"missing required tables: {missing_required} (template/partial manifest)")
    return {
        "name": "sharadar_data_manifest",
        "vendor": vendor,
        "status": "template" if not loaded else ("partial" if missing_required else "complete"),
        "load_timestamp_utc": datetime.now(UTC).isoformat(),
        "package_versions": _pkg_versions(),
        # operator-confirmed metadata (manual): license permits local research use?
        "license_local_research_use": license_local_research,
        "tables": tables,
        "tables_present": sorted(loaded.keys()),
        "tables_missing": missing_required,
        "warnings": warnings,
    }


def write_manifest(manifest: dict[str, Any], path: Path | str = _MANIFEST_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p


def template_manifest() -> dict[str, Any]:
    """Manifest emitted when no data is present — documents the expected shape."""
    m = build_manifest({})
    m["note"] = ("No Sharadar files found. Place SEP/TICKERS/ACTIONS(/SF1) under the data dir "
                 "and re-run scripts/ingest_sharadar.py. Expected required columns per table are "
                 "defined in data.sharadar.schema.SCHEMAS.")
    return m
