from __future__ import annotations
from typing import Iterable, List
from Shopping_assistant.nlp.schema import Constraint, Strength, Direction, Axis

_AXIS_DEFAULT_DIRECTION = {
    Axis.BRIGHTNESS: Direction.LOWER,
    Axis.VIBRANCY: Direction.LOWER,
    Axis.SATURATION: Direction.LOWER,
    Axis.DEPTH: Direction.RAISE,
    Axis.CLARITY: Direction.LOWER,
}

def normalize_constraints(constraints: Iterable[Constraint]) -> List[Constraint]:
    out: List[Constraint] = []
    for c in constraints:
        # direction: explicit > default per axis
        direction = c.direction or _AXIS_DEFAULT_DIRECTION.get(c.axis)

        # strength: clamp
        strength = c.strength
        if strength not in (Strength.WEAK, Strength.MED, Strength.STRONG):
            strength = Strength.MED

        out.append(
            Constraint(
                axis=c.axis,
                direction=direction,
                strength=strength,
                evidence=c.evidence,
                confidence=c.confidence,
                scope=c.scope,
                clause_id=c.clause_id,
                meta=c.meta,
            )
        )
    return out
