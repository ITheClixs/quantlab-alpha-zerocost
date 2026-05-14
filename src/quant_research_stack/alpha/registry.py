from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    version: str
    git_sha: str
    data_hashes: dict[str, str]
    hyperparams: dict[str, Any]
    fold_definition: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRegistry:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def create_run(self, meta: RunMetadata) -> str:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(self.root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))
        return run_id

    def save_artifact(self, run_id: str, name: str, payload: bytes) -> str:
        path = Path(self.root) / run_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        index_path = Path(self.root) / run_id / "_artifact_sha256.json"
        index: dict[str, str] = {}
        if index_path.exists():
            index = json.loads(index_path.read_text())
        index[name] = sha
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        return sha
