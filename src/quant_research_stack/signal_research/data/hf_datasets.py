"""HuggingFace datasets loader — gated by default (spec §2.6).

Sentiment + fundamentals datasets enter the promoted benchmark only via
the FinBERT-style audit ladder (spec §3.3 #8). v1 default is to BLOCK
loading unless an audit token validated by the audit gate is provided.
"""

from __future__ import annotations


class HFDatasetGatedError(RuntimeError):
    pass


_AUDIT_TOKENS_ACCEPTED: frozenset[str] = frozenset()  # populated by audit pipeline


def load_hf_dataset_gated(*, dataset_id: str, audit_token: str | None) -> object:
    if audit_token is None:
        raise HFDatasetGatedError(
            f"HF dataset {dataset_id} is research_only_default. Provide an audit_token "
            "after passing the 10-criterion FinBERT-style audit gate (spec §3.3 FinBERT)."
        )
    if audit_token not in _AUDIT_TOKENS_ACCEPTED:
        raise HFDatasetGatedError(
            f"audit_token '{audit_token}' not in the accepted-tokens set; "
            "audit must be re-run and the token registered before this dataset can load."
        )
    from datasets import load_dataset  # local import
    return load_dataset(dataset_id)
