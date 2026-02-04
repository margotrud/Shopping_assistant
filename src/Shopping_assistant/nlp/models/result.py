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
    return NLPResult(
        text=text,
        clauses=tuple(clauses),
        mentions=tuple(mentions),
        constraints=tuple(constraints),
        diagnostics=diagnostics or {},
        trace=trace,
    )
