# src/Shopping_assistant/nlp/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------

class MentionKind(str, Enum):
    COLOR = "colors"
    PRODUCT = "product"
    BRAND = "brand"
    FINISH = "finish"
    OTHER = "other"


class Polarity(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Axis(str, Enum):
    BRIGHTNESS = "brightness"
    SATURATION = "saturation"
    VIBRANCY = "vibrancy"
    DEPTH = "depth"
    CLARITY = "clarity"


class Direction(str, Enum):
    RAISE = "raise"
    LOWER = "lower"


class Strength(str, Enum):
    WEAK = "weak"
    MED = "med"
    STRONG = "strong"


# ---------------------------------------------------------------------
# Core NLP objects
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Span:
    """
    Does:
        Store character offsets into the original input text for traceability.
    """
    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Clause:
    """
    Does:
        Store a clause chunk plus its resolved polarity for downstream logic.
    """
    clause_id: int
    text: str
    polarity: Polarity = Polarity.UNKNOWN
    elliptical_neg: bool = False
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Mention:
    """
    Does:
        Represent an extracted entity mention with resolved polarity.
    """
    span: Span
    raw: str
    canonical: str
    kind: MentionKind = MentionKind.OTHER
    polarity: Polarity = Polarity.UNKNOWN
    confidence: float = 0.0
    clause_id: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # canonical is a logical key (tests, filtering, dedupe). Normalize deterministically.
        object.__setattr__(self, "canonical", (self.canonical or "").strip().lower())


@dataclass(frozen=True, slots=True)
class Constraint:
    """
    Does:
        Represent a numeric preference constraint mapped to a scoring axis.
    """
    axis: Axis
    direction: Direction
    strength: Strength
    evidence: str
    clause_id: int = 0
    confidence: float = 0.0
    scope: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# NLP output contract
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NLPResult:
    """
    Does:
        Provide a stable NLP output contract for downstream preference/scoring.
    """
    text: str
    clauses: Tuple[Clause, ...] = ()
    mentions: Tuple[Mention, ...] = ()
    constraints: Tuple[Constraint, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Dict[str, Any]] = None

    def mentions_by_polarity(self, polarity: Polarity) -> Tuple[Mention, ...]:
        return tuple(m for m in self.mentions if m.polarity == polarity)

    def constraints_by_axis(self, axis: Axis) -> Tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.axis == axis)


# Backward-compatible alias
ParsedRequest = NLPResult
