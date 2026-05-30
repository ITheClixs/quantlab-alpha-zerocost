"""Loughran-McDonald-style sentiment lexicon — COMPACT SUBSET (v1 approximation).

This is a hand-curated, representative subset of the Loughran & McDonald (2011)
financial sentiment word lists, NOT the full ~85k-entry master dictionary (which is
not available offline). It captures high-frequency terms per category for a v1
classical baseline. **Limitation (pre-registered):** an incomplete lexicon will
under-count tone; a negative v1 result on tone features is therefore "not found
with this lexicon", not "no tone signal exists". The full LM dictionary is a v2
upgrade. All words are lowercase; matching is whole-word.
"""

from __future__ import annotations

NEGATIVE: frozenset[str] = frozenset({
    "loss", "losses", "decline", "declines", "declined", "declining", "adverse", "adversely",
    "litigation", "lawsuit", "lawsuits", "default", "defaults", "deficiency", "deficiencies",
    "impairment", "impairments", "impaired", "deteriorate", "deteriorated", "deterioration",
    "weak", "weakness", "weaknesses", "weaker", "downturn", "recession", "bankruptcy", "insolvency",
    "restructuring", "restructure", "layoff", "layoffs", "shortfall", "writedown", "write-down",
    "fail", "failed", "failure", "failures", "breach", "breaches", "violation", "violations",
    "penalty", "penalties", "fraud", "misconduct", "discontinued", "termination", "terminated",
    "negative", "unfavorable", "diminished", "erosion", "shortage", "disruption", "disruptions",
    "delays", "delay", "unable", "difficult", "difficulties", "concern", "concerns", "doubt",
})

POSITIVE: frozenset[str] = frozenset({
    "gain", "gains", "gained", "growth", "growing", "grew", "profit", "profits", "profitable",
    "improve", "improved", "improvement", "improvements", "improving", "strong", "stronger",
    "strength", "favorable", "favorably", "success", "successful", "successfully", "exceeded",
    "outperform", "outperformed", "advantage", "advantages", "innovation", "innovative", "leading",
    "leadership", "efficient", "efficiencies", "robust", "expansion", "expanding", "record",
    "achieve", "achieved", "achievement", "opportunity", "opportunities", "benefit", "benefits",
    "enhanced", "enhancement", "superior", "progress", "positive", "rewarding",
})

UNCERTAINTY: frozenset[str] = frozenset({
    "may", "could", "might", "uncertain", "uncertainty", "uncertainties", "risk", "risks", "risky",
    "possible", "possibly", "probable", "approximately", "estimate", "estimates", "estimated",
    "believe", "believes", "assume", "assumption", "assumptions", "depend", "depends", "depending",
    "fluctuate", "fluctuation", "fluctuations", "volatile", "volatility", "unpredictable",
    "anticipate", "anticipated", "predict", "predictability", "contingent", "contingency",
    "exposure", "exposures", "speculative", "indefinite", "tentative", "vary", "varying",
})

LITIGIOUS: frozenset[str] = frozenset({
    "litigation", "litigated", "lawsuit", "lawsuits", "plaintiff", "plaintiffs", "defendant",
    "defendants", "court", "courts", "judicial", "judgment", "judgments", "settlement", "settlements",
    "claim", "claims", "alleged", "allegation", "allegations", "regulatory", "regulation",
    "regulations", "subpoena", "indemnification", "indemnify", "liability", "liabilities",
    "damages", "injunction", "arbitration", "compliance", "noncompliance", "statute", "statutory",
    "enforcement", "prosecution", "testimony", "appeal", "appeals",
})

MODAL_STRONG: frozenset[str] = frozenset({
    "will", "must", "shall", "always", "never", "highest", "lowest", "best", "definitely",
    "clearly", "undoubtedly", "certain", "certainly",
})

MODAL_WEAK: frozenset[str] = frozenset({
    "may", "could", "might", "possible", "possibly", "perhaps", "depending", "appears",
    "suggests", "seldom", "occasionally", "generally", "somewhat",
})

CONSTRAINING: frozenset[str] = frozenset({
    "constrain", "constrained", "constraint", "constraints", "covenant", "covenants", "restrict",
    "restricted", "restriction", "restrictions", "limit", "limited", "limitation", "limitations",
    "require", "required", "requirement", "requirements", "obligation", "obligations", "mandatory",
    "prohibit", "prohibited", "prevent", "prevented", "compelled", "imposed", "binding",
})

CATEGORIES: dict[str, frozenset[str]] = {
    "negative": NEGATIVE,
    "positive": POSITIVE,
    "uncertainty": UNCERTAINTY,
    "litigious": LITIGIOUS,
    "modal_strong": MODAL_STRONG,
    "modal_weak": MODAL_WEAK,
    "constraining": CONSTRAINING,
}
