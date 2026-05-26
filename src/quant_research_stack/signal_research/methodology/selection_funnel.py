"""Selection funnel (spec §6.4) — ordered per-stage counts."""

from __future__ import annotations

from collections import OrderedDict


class SelectionFunnel:
    def __init__(self) -> None:
        self._stages: OrderedDict[str, int] = OrderedDict()

    def record(self, stage: str, count: int) -> None:
        self._stages[stage] = int(count)

    def to_ordered_dict(self) -> OrderedDict[str, int]:
        return OrderedDict(self._stages)
