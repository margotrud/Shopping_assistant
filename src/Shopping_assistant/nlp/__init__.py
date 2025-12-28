# src/Shopping_assistant/nlp/__init__.py
from __future__ import annotations

# Stable re-exports (keep this list SHORT).
from Shopping_assistant.nlp.models.schema import (  # type: ignore
    Axis,
    Direction,
    Strength,
    Constraint,
    NLPResult,
    Clause,
    Mention,
    MentionKind,
    Polarity,
    Span,
)

__all__ = [
    "Axis",
    "Direction",
    "Strength",
    "Constraint",
    "NLPResult",
    "Clause",
    "Mention",
    "MentionKind",
    "Polarity",
    "Span",
]
