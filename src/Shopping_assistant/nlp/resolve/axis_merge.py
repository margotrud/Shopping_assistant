from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Polarity
from Shopping_assistant.nlp.resolve.axis_projection import AxisIntent


@dataclass(frozen=True, slots=True)
class AxisDecision:
    """
    Does:
        Provide a single merged decision per axis, with conflict diagnostics.
    """
    axis: Axis
    direction: Optional[Direction]        # None => canceled / undecidable
    strength: Optional[Strength]          # None => canceled / undecidable
    score: float                          # signed aggregate score
    conflicts: bool                       # True if both signs observed
    sources: Tuple[str, ...]              # e.g. ("target", "global")
    intents: Tuple[AxisIntent, ...]       # keep raw intents for debug


# -----------------------------
# Public API
# -----------------------------

def merge_axis_intents(intents_by_axis: Dict[Axis, List[AxisIntent]]) -> Dict[Axis, AxisDecision]:
    """
    Does:
        Merge possibly conflicting intents per axis into a single decision.
    """
    out: Dict[Axis, AxisDecision] = {}
    for axis, intents in intents_by_axis.items():
        out[axis] = _merge_one_axis(axis, intents)
    return out


# -----------------------------
# Core logic
# -----------------------------

def _merge_one_axis(axis: Axis, intents: List[AxisIntent]) -> AxisDecision:
    """
    Does:
        Merge intents for one axis using signed aggregation with conflict detection.
    """
    if not intents:
        return AxisDecision(
            axis=axis,
            direction=None,
            strength=None,
            score=0.0,
            conflicts=False,
            sources=(),
            intents=(),
        )

    signed: List[float] = []
    sources = tuple(_dedupe_preserve_order([i.source for i in intents]))

    for i in intents:
        w = _strength_weight(i.strength)
        s_dir = _dir_sign(i.direction)         # LOWER => -1, HIGHER => +1
        s_pol = _polarity_sign(i.polarity)     # LIKE => +1, DISLIKE => -1
        signed.append(float(w) * float(s_dir) * float(s_pol))

    score = sum(signed)
    has_pos = any(v > 0 for v in signed)
    has_neg = any(v < 0 for v in signed)
    conflicts = has_pos and has_neg

    # Cancellation rule: exact tie => undecidable (force downstream to not hallucinate)
    if score == 0.0:
        return AxisDecision(
            axis=axis,
            direction=None,
            strength=None,
            score=0.0,
            conflicts=conflicts,
            sources=sources,
            intents=tuple(intents),
        )

    pos_dir = _positive_direction_from(intents)
    neg_dir = _negative_direction_from(intents)

    if score > 0:
        direction = pos_dir
    else:
        direction = neg_dir
    strength = _score_to_strength(abs(score))

    return AxisDecision(
        axis=axis,
        direction=direction,
        strength=strength,
        score=score,
        conflicts=conflicts,
        sources=sources,
        intents=tuple(intents),
    )


# -----------------------------
# Helpers
# -----------------------------

def _strength_weight(s: Strength) -> float:
    """
    Does:
        Convert enum Strength to an additive weight (robust to new members).
    """
    name = getattr(s, "name", str(s)).upper()
    if "STRONG" in name:
        return 3.0
    if "MED" in name or "MEDIUM" in name:
        return 2.0
    if "WEAK" in name or "LOW" in name:
        return 1.0
    # fallback: treat unknown as MED
    return 2.0


def _score_to_strength(mag: float) -> Strength:
    """
    Does:
        Map absolute aggregate magnitude to a Strength bucket.
    """
    # With weights (WEAK=1, MED=2, STRONG=3), typical sums are in [1..N*3]
    if mag >= 3.0:
        return Strength.STRONG
    if mag >= 2.0:
        return Strength.MED
    # if you have Strength.WEAK in your enum, this returns it; else fallback to MED
    if hasattr(Strength, "WEAK"):
        return Strength.WEAK  # type: ignore[attr-defined]
    return Strength.MED


def _dir_sign(d: Direction) -> int:
    """
    Does:
        Map direction to sign.
    """
    name = getattr(d, "name", str(d)).upper()
    if "LOW" in name:
        return -1
    return +1


def _polarity_sign(p: Polarity) -> int:
    """
    Does:
        Map polarity to sign.
    """
    return +1 if p == Polarity.LIKE else -1


def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    """
    Does:
        Dedupe while keeping stable order.
    """
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _positive_direction_from(intents: List[AxisIntent]) -> Direction:
    """
    Does:
        Return the direction that maps to positive sign among observed intents.
    """
    for i in intents:
        if _dir_sign(i.direction) > 0:
            return i.direction
    # Fallback: if none are positive (shouldn't happen), return first
    return intents[0].direction


def _negative_direction_from(intents: List[AxisIntent]) -> Direction:
    """
    Does:
        Return the direction that maps to negative sign among observed intents.
    """
    for i in intents:
        if _dir_sign(i.direction) < 0:
            return i.direction
    # Fallback: if none are negative (shouldn't happen), return first
    return intents[0].direction

