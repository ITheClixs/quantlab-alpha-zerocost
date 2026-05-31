"""Foundation-model meta-features — disabled by default in v1 (spec §3.3-7)."""

from __future__ import annotations

from typing import Any


class MetaFeaturesDisabledError(RuntimeError):
    pass


def build_meta_features(
    *,
    panel: Any,
    enable: bool = False,
    audit_pass_token: str | None = None,
) -> Any:
    if not enable:
        raise MetaFeaturesDisabledError(
            "meta-features disabled by default in v1 (spec §3.3-7); "
            "enable only after timestamp audit, ablation, baseline comparison, "
            "and dev-window improvement"
        )
    if audit_pass_token is None:
        raise MetaFeaturesDisabledError(
            "meta-features require an audit_pass_token recorded in metadata.json"
        )
    raise NotImplementedError(
        "meta-features extractor wiring deferred to a follow-up task; "
        "v1 ships with this gate disabled"
    )
