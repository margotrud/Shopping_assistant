# src/Shopping_assistant/nlp/models/result.py
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from Shopping_assistant.nlp.schema import (
    Axis,
    Clause,
    Constraint,
    Direction,
    Mention,
    MentionKind,
    NLPResult,
    Polarity,
    Span,
    Strength,
)


# ---------------------------------------------------------------------
# Builders (thin helpers, no logic)
# ---------------------------------------------------------------------

def make_clause(
    clause_id: int,
    text: str,
    *,
    elliptical_neg: bool = False,
    reason: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Clause:
    """Does: build a Clause object from parsed text span and metadata.
    Used by: clause splitting and downstream NLP assembly.
    """
    return Clause(
        clause_id=clause_id,
        text=text,
        elliptical_neg=elliptical_neg,
        reason=reason,
        meta=meta or {},
    )


def make_mention(
    clause_id: int,
    canonical: str,
    *,
    raw: Optional[str] = None,
    polarity: Polarity = Polarity.UNKNOWN,
    kind: MentionKind = MentionKind.COLOR,
    span: Optional[Span] = None,
    confidence: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Mention:
    """Does: build a Mention object from token span, kind, and polarity.
    Used by: lexicon and LLM-based mention extraction.
    """
    return Mention(
        span=span or Span(0, 0),
        raw=raw or canonical,
        canonical=canonical,
        kind=kind,
        polarity=polarity,
        confidence=confidence,
        clause_id=clause_id,
        meta=meta or {},
    )


def make_constraint(
    clause_id: int,
    *,
    axis: Axis,
    direction: Direction,
    strength: Strength,
    evidence: str,
    confidence: float = 0.0,
    scope: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Constraint:
    """Does: build a Constraint object from axis, direction, and strength.
    Used by: constraint normalization and preference resolution.
    """
    return Constraint(
        axis=axis,
        direction=direction,
        strength=strength,
        evidence=evidence,
        clause_id=clause_id,
        confidence=confidence,
        scope=scope,
        meta=meta or {},
    )


def build_nlp_result(
    text: str,
    *,
    clauses: Sequence[Clause] = (),
    mentions: Sequence[Mention] = (),
    constraints: Sequence[Constraint] = (),
    diagnostics: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> NLPResult:
    """Does: assemble the final NLPResult from clauses, mentions, and constraints.
    Returns: immutable NLPResult consumed by preference resolution and scoring.
    """
    return NLPResult(
        text=text,
        clauses=tuple(clauses),
        mentions=tuple(mentions),
        constraints=tuple(constraints),
        diagnostics=diagnostics or {},
        trace=trace,
    )
