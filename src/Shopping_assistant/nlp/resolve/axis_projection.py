from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Polarity
from Shopping_assistant.nlp.resolve.preference_resolver import (
    ResolvedPreference,
    ResolvedTarget,
    ResolvedConstraint,
)


# -----------------------------
# Axis intent (post-resolution)
# -----------------------------

@dataclass(frozen=True, slots=True)
class AxisIntent:
    """
    Does:
        Represent a resolved intent on a single semantic axis,
        without numeric thresholds.
    """
    axis: Axis
    direction: Direction
    strength: Strength
    polarity: Polarity
    source: str  # "target" | "constraint"


# -----------------------------
# Projection
# -----------------------------

def project_axes(pref: ResolvedPreference) -> Dict[Axis, List[AxisIntent]]:
    """
    Does:
        Project resolved preferences into axis-level intents,
        without assigning numeric values.
    """
    intents: Dict[Axis, List[AxisIntent]] = {}

    # 1) Constraints attached to liked / disliked targets
    for bucket, polarity in (
        (pref.liked, Polarity.LIKE),
        (pref.disliked, Polarity.DISLIKE),
    ):
        for target in bucket:
            _collect_constraints(
                intents=intents,
                constraints=target.constraints,
                polarity=polarity,
                source="target",
            )

    # 2) Global constraints (scope-safe)
    _collect_constraints(
        intents=intents,
        constraints=[gc.constraint for gc in pref.global_constraints],
        polarity=Polarity.LIKE,  # global constraints are preferences, not dislikes
        source="global",
    )

    return intents


# -----------------------------
# Helpers
# -----------------------------

def _collect_constraints(
    *,
    intents: Dict[Axis, List[AxisIntent]],
    constraints: List,
    polarity: Polarity,
    source: str,
) -> None:
    """
    Does:
        Append axis intents derived from constraints.
    """
    for c in constraints:
        intent = AxisIntent(
            axis=c.axis,
            direction=c.direction,
            strength=c.strength,
            polarity=polarity,
            source=source,
        )
        intents.setdefault(c.axis, []).append(intent)
