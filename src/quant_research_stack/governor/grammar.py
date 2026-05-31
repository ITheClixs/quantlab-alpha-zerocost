from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict

_PACKAGE_DIR = Path(__file__).parent
PACKAGE_GRAMMAR_FULL_PATH = _PACKAGE_DIR / "grammar.gbnf"
PACKAGE_GRAMMAR_TIER1_PATH = _PACKAGE_DIR / "grammar_tier1.gbnf"


_FULL_GRAMMAR = """root ::= "{" ws "\\"signal_id\\":" ws string ws "," ws "\\"decision\\":" ws decision ws "," ws "\\"direction\\":" ws direction ws "," ws "\\"confidence\\":" ws confidence ws "," ws "\\"horizon_minutes\\":" ws posint ws "," ws "\\"regime_tag\\":" ws regime ws "," ws "\\"rationale_short\\":" ws string ws "," ws "\\"cited_paper_chunk_ids\\":" ws arrayitems ws "," ws "\\"contradictions_flagged\\":" ws arrayitems ws "}" ws

decision ::= "\\"pass\\"" | "\\"veto\\"" | "\\"insufficient_evidence\\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\\"trending\\"" | "\\"mean_reverting\\"" | "\\"high_vol\\"" | "\\"low_vol\\"" | "\\"unknown\\""
string ::= "\\"" char* "\\""
char ::= [^"\\\\\\n] | "\\\\" ["\\\\nrt]
arrayitems ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
ws ::= [ \\t\\n]*
"""


_TIER1_GRAMMAR = """root ::= "{" ws "\\"signal_id\\":" ws string ws "," ws "\\"decision\\":" ws decision ws "," ws "\\"direction\\":" ws direction ws "," ws "\\"confidence\\":" ws confidence ws "," ws "\\"horizon_minutes\\":" ws posint ws "," ws "\\"regime_tag\\":" ws regime ws "," ws "\\"rationale_short\\":" ws string ws "," ws "\\"cited_paper_chunk_ids\\":" ws "[]" ws "," ws "\\"contradictions_flagged\\":" ws "[]" ws "}" ws

decision ::= "\\"pass\\"" | "\\"veto\\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\\"trending\\"" | "\\"mean_reverting\\"" | "\\"high_vol\\"" | "\\"low_vol\\"" | "\\"unknown\\""
string ::= "\\"" char* "\\""
char ::= [^"\\\\\\n] | "\\\\" ["\\\\nrt]
ws ::= [ \\t\\n]*
"""


def generate_full_grammar() -> str:
    return _FULL_GRAMMAR


def generate_tier1_grammar() -> str:
    return _TIER1_GRAMMAR


def validate_against_grammar_shape(text: str) -> bool:
    """Cheap structural validation that mirrors what the GBNF accepts.

    Used in unit tests to check fixtures without invoking llama.cpp.
    """
    try:
        payload = json.loads(text)
        GovernorVerdict.model_validate(payload)
        return True
    except Exception:
        return False
