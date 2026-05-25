"""Hash-verified parquet loaders for the equity processed root (spec §2.8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    EquityManifest,
    ManifestMismatchError,
    load_and_verify_manifest,
    sha256_of_file,
)


class LoaderHashError(RuntimeError):
    pass


@dataclass(frozen=True)
class EquityRootLoader:
    root: Path

    def _manifest(self) -> EquityManifest:
        return load_and_verify_manifest(self.root / "_manifest.json", expected_sha256={})

    def _verified_path(self, artifact_key: str) -> Path:
        m = self._manifest()
        if artifact_key not in m.artifacts:
            raise ManifestMismatchError(f"artifact key not in manifest: {artifact_key}")
        art = m.artifacts[artifact_key]
        path = self.root / art.path
        if not path.exists():
            raise FileNotFoundError(f"artifact missing on disk: {path}")
        actual_sha = sha256_of_file(path)
        if actual_sha != art.sha256:
            raise LoaderHashError(
                f"hash mismatch on {artifact_key}: manifest={art.sha256} disk={actual_sha}"
            )
        return path

    def load_tradable_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_tradable_prices"))

    def load_split_adjusted_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_split_adjusted_prices"))

    def load_total_return_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_total_return_prices"))

    def load_dividends(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_dividends"))

    def load_adv(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_adv"))

    def load_borrow_proxy(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_borrow_proxy"))

    def load_delisting_audit(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_delisting_audit"))

    def load_pit_membership(self) -> pl.DataFrame | None:
        """PIT membership is optional — absence implies prototype-only."""
        m = self._manifest()
        if "sp500_pit_membership" not in m.artifacts:
            return None
        return pl.read_parquet(self._verified_path("sp500_pit_membership"))
