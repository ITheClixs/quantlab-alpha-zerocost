"""Dev-only invariant: methodology modules NEVER touch the permanent holdout.

Spec §4.9, §0 non-negotiable #2. Only `inference.evaluate_holdout` is allowed
to read holdout rows.
"""

from __future__ import annotations

from typing import Final

_ALLOWED_CALLERS: Final[frozenset[str]] = frozenset({"inference.evaluate_holdout"})


class HoldoutAccessError(RuntimeError):
    pass


def enforce_dev_only(
    *,
    caller: str,
    holdout_indices: list[int],
    accessed_indices: list[int],
) -> None:
    if caller in _ALLOWED_CALLERS:
        return
    overlap = set(holdout_indices).intersection(accessed_indices)
    if overlap:
        raise HoldoutAccessError(
            f"caller={caller} accessed {len(overlap)} holdout rows; "
            "methodology modules must operate on dev+validation data only (§4.9)"
        )
