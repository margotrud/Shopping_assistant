# src/Shopping_assistant/nlp/resolve/axis_merge.py
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
    score: float                          # signed aggregate score (direction-only, polarity-free)
    conflicts: bool                       # True if both signs observed
    sources: Tuple[str, ...]              # e.g. ("target", "global")
    intents: Tuple[AxisIntent, ...]       # keep raw intents for debug
    has_dislike: bool = False             # True if any intent came from DISLIKE polarity


def merge_axis_intents(intents_by_axis: Dict[Axis, List[AxisIntent]]) -> Dict[Axis, AxisDecision]:
    """
    Does:
        Merge axis intents (possibly from different scopes) into one AxisDecision per axis.

    Important:
        Polarity MUST NOT affect direction aggregation; it is carried only as diagnostics
        (has_dislike) and used later for threshold weighting (axis_thresholds.py).
    """
    out: Dict[Axis, AxisDecision] = {}
    for axis, intents in intents_by_axis.items():
        out[axis] = _merge_one_axis(axis, intents)
    return out


def _merge_one_axis(axis: Axis, intents: List[AxisIntent]) -> AxisDecision:
    """
    Does:
        Merge all intents for one axis into a single decision.

    Rule:
        Direction score is computed ONLY from (strength, direction).
        Polarity does not upweight the vote (to avoid double-boost with axis_thresholds.py).

    Fix:
        Strength must NOT be derived from aggregate score magnitude (downgrades single intents).
        Strength is preserved as the MAX strength among intents supporting the winning direction.
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
            has_dislike=False,
        )

    signed: List[float] = []
    sources = tuple(_dedupe_preserve_order([i.source for i in intents]))
    has_dislike = any(i.polarity == Polarity.DISLIKE for i in intents)

    for i in intents:
        w = _strength_weight(i.strength)
        s_dir = _dir_sign(i.direction)  # LOWER => -1, RAISE => +1
        signed.append(float(w) * float(s_dir))

    score = float(sum(signed))
    has_pos = any(v > 0 for v in signed)
    has_neg = any(v < 0 for v in signed)
    conflicts = has_pos and has_neg

    # tie => undecidable (float-safe)
    if abs(score) < 1e-9:
        return AxisDecision(
            axis=axis,
            direction=None,
            strength=None,
            score=0.0,
            conflicts=conflicts,
            sources=sources,
            intents=tuple(intents),
            has_dislike=has_dislike,
        )

    direction = Direction.RAISE if score > 0 else Direction.LOWER

    # ✅ Strength preservation (no downgrade): take MAX strength among winning-direction intents
    win_intents = [i for i in intents if i.direction == direction]
    strength = _max_strength(win_intents)

    return AxisDecision(
        axis=axis,
        direction=direction,
        strength=strength,
        score=score,
        conflicts=conflicts,
        sources=sources,
        intents=tuple(intents),
        has_dislike=has_dislike,
    )


def _strength_weight(s: Strength) -> float:
    name = getattr(s, "name", str(s)).upper()
    if "STRONG" in name:
        return 3.0
    if "MED" in name or "MEDIUM" in name:
        return 2.0
    return 1.0


def _dir_sign(d: Direction) -> int:
    name = getattr(d, "name", str(d)).upper()
    if "RAISE" in name or "HIGH" in name or "UP" in name:
        return 1
    return -1


def _max_strength(intents: List[AxisIntent]) -> Optional[Strength]:
    """
    Does:
        Return the maximum Strength among intents, using _strength_weight as ordering.
    """
    if not intents:
        return None
    best = intents[0].strength
    best_w = _strength_weight(best)
    for it in intents[1:]:
        w = _strength_weight(it.strength)
        if w > best_w:
            best = it.strength
            best_w = w
    return best


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


__all__ = ["AxisDecision", "merge_axis_intents", "_dir_sign"]
