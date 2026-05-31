"""Paper-signal abstract base classes (spec §3.1).

Every paper-derived module is exactly one of:
- StandaloneStrategy
- FeatureGenerator
- Wrapper
- ModelFamily
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class StandaloneStrategy(ABC):
    @abstractmethod
    def positions(self, panel: pl.DataFrame) -> pl.DataFrame: ...


class FeatureGenerator(ABC):
    @abstractmethod
    def features(self, panel: pl.DataFrame) -> pl.DataFrame: ...


class Wrapper(ABC):
    @abstractmethod
    def apply(self, positions: pl.Series) -> pl.Series: ...


class ModelFamily(ABC):
    @abstractmethod
    def fit(self, x: pl.DataFrame, y: pl.Series) -> None: ...

    @abstractmethod
    def predict(self, x: pl.DataFrame) -> pl.Series: ...
