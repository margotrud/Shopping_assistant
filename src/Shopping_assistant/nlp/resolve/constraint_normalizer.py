# src/Shopping_assistant/nlp/resolve/constraint_normalizer.py
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Iterable, List, Optional

from Shopping_assistant.nlp.schema import Axis, Constraint, Direction, Strength


_AXIS_DEFAULT_DIRECTION = {
    Axis.BRIGHTNESS: Direction.LOWER,
    Axis.VIBRANCY: Direction.LOWER,
    Axis.SATURATION: Direction.LOWER,
    Axis.DEPTH: Direction.RAISE,
    Axis.CLARITY: Direction.LOWER,
}

_ALLOWED_DIRECTIONS = {Direction.RAISE, Direction.LOWER}
_ALLOWED_STRENGTHS = {Strength.WEAK, Strength.MED, Strength.STRONG}


def _clamp01(x: Optional[float]) -> float:
    try:
        v = float(x) if x is not None else 0.0
    except Exception:
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _warn_or_raise(msg: str, *, strict: bool) -> None:
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def normalize_constraints(
    constraints: Iterable[Constraint],
    *,
    strict: bool = False,
) -> List[Constraint]:
    """
    Does:
        Normalize constraints into a fully-formed, scoring-safe contract.

    Rules:
        - direction: keep if valid else use default per axis else LOWER
        - strength: keep if valid else MED
        - confidence: clamp to [0,1]
        - evidence: must be non-empty (strict: raise / non-strict: keep but warn)
    """
    out: List[Constraint] = []

    for c in constraints:
        if c.axis is None:
            _warn_or_raise("Constraint has axis=None", strict=strict)
            continue

        # direction
        direction = c.direction
        if direction not in _ALLOWED_DIRECTIONS:
            direction = _AXIS_DEFAULT_DIRECTION.get(c.axis, Direction.LOWER)
            _warn_or_raise(
                f"Constraint direction invalid for axis={c.axis.value}; defaulting to {direction.value}",
                strict=strict,
            )

        # strength
        strength = c.strength
        if strength not in _ALLOWED_STRENGTHS:
            strength = Strength.MED
            _warn_or_raise(
                f"Constraint strength invalid for axis={c.axis.value}; defaulting to {strength.value}",
                strict=strict,
            )

        # confidence
        conf = _clamp01(getattr(c, "confidence", 0.0))

        # evidence
        ev = (c.evidence or "").strip()
        if not ev:
            _warn_or_raise(
                f"Constraint evidence is empty for axis={c.axis.value} (clause_id={c.clause_id})",
                strict=strict,
            )

        out.append(
            replace(
                c,
                direction=direction,
                strength=strength,
                confidence=conf,
                evidence=ev,
            )
        )

    return out
