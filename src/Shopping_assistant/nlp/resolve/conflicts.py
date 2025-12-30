# src/Shopping_assistant/nlp/resolve/conflicts.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from Shopping_assistant.nlp.schema import Axis, Constraint, Direction, Polarity, Strength


def _strength_rank(s: Strength) -> int:
    if s == Strength.STRONG:
        return 3
    if s == Strength.MED:
        return 2
    return 1


def _get_clause_polarity_from_meta(meta: Dict[str, Any] | None) -> Polarity:
    if not isinstance(meta, dict):
        return Polarity.UNKNOWN
    v = meta.get("clause_polarity")
    if not isinstance(v, str):
        return Polarity.UNKNOWN
    vv = v.strip().lower()
    if vv == "like":
        return Polarity.LIKE
    if vv == "dislike":
        return Polarity.DISLIKE
    if vv == "neutral":
        return Polarity.NEUTRAL
    return Polarity.UNKNOWN


def _is_cap_constraint(c: Constraint) -> bool:
    """
    Heuristic: detect 'cap' language ("not too X") which should remain LOWER under DISLIKE.
    We use evidence string because spaCy structure is not always preserved downstream.
    """
    ev = (c.evidence or "").strip().lower()
    return ("too " in ev) or ev.startswith("too ")


def _constraint_priority(c: Constraint) -> Tuple[int, float]:
    """
    Higher is better.
    """
    # prefer stronger, then higher confidence
    return (_strength_rank(c.strength), float(c.confidence or 0.0))


def resolve_symbolic_conflicts(
    constraints: Tuple[Constraint, ...],
) -> Tuple[Tuple[Constraint, ...], Dict[str, Any]]:
    """
    Does:
        Resolve contradictions among constraints before scoring.

    Rules:
      1) For same Axis:
         - if multiple directions exist, keep the best candidate after contextual normalization.
      2) Context normalization:
         - if clause_polarity=DISLIKE and direction=RAISE:
             - if "cap" ('too') -> keep as LOWER (cap), do not invert
             - else invert to LOWER and mark as normalized
    Output:
      - kept constraints (tuple)
      - diagnostics dict with suppressed/normalized details
    """
    by_axis: Dict[Axis, List[Constraint]] = {}
    for c in constraints:
        by_axis.setdefault(c.axis, []).append(c)

    kept: List[Constraint] = []
    suppressed: List[Dict[str, Any]] = []
    normalized: List[Dict[str, Any]] = []

    for axis, items in by_axis.items():
        normed: List[Constraint] = []

        # Step A: normalize direction using clause polarity context
        for c in items:
            cp = _get_clause_polarity_from_meta(c.meta)
            if cp == Polarity.DISLIKE and c.direction == Direction.RAISE:
                if _is_cap_constraint(c):
                    # "not too bright" style: should be LOWER already usually, but keep safe
                    c2 = replace(c, direction=Direction.LOWER)
                    normed.append(c2)
                    normalized.append(
                        {
                            "axis": axis.value,
                            "kind": "cap_guard",
                            "from": "raise",
                            "to": "lower",
                            "evidence": c.evidence,
                            "clause_id": c.clause_id,
                        }
                    )
                else:
                    # dislike + raise is contradictory: normalize by inverting to LOWER
                    c2 = replace(c, direction=Direction.LOWER)
                    normed.append(c2)
                    normalized.append(
                        {
                            "axis": axis.value,
                            "kind": "invert_dislike_raise",
                            "from": "raise",
                            "to": "lower",
                            "evidence": c.evidence,
                            "clause_id": c.clause_id,
                        }
                    )
            else:
                normed.append(c)

        # Step B: pick best if directions conflict
        dirs = {c.direction for c in normed}
        if len(dirs) == 1:
            # Keep best single (dedupe multiple candidates)
            best = max(normed, key=_constraint_priority)
            for c in normed:
                if c is not best:
                    suppressed.append(
                        {
                            "axis": axis.value,
                            "reason": "dedupe_same_direction",
                            "kept_evidence": best.evidence,
                            "dropped_evidence": c.evidence,
                            "kept_clause_id": best.clause_id,
                            "dropped_clause_id": c.clause_id,
                        }
                    )
            kept.append(best)
            continue

        # directions conflict: choose best candidate by priority
        best = max(normed, key=_constraint_priority)
        for c in normed:
            if c is best:
                continue
            suppressed.append(
                {
                    "axis": axis.value,
                    "reason": "conflict_opposite_directions",
                    "kept": {
                        "direction": best.direction.value,
                        "strength": best.strength.value,
                        "confidence": float(best.confidence or 0.0),
                        "evidence": best.evidence,
                        "clause_id": best.clause_id,
                    },
                    "dropped": {
                        "direction": c.direction.value,
                        "strength": c.strength.value,
                        "confidence": float(c.confidence or 0.0),
                        "evidence": c.evidence,
                        "clause_id": c.clause_id,
                    },
                }
            )
        kept.append(best)

    diag = {
        "suppressed_constraints": suppressed,
        "normalized_constraints": normalized,
    }
    return tuple(kept), diag


__all__ = ["resolve_symbolic_conflicts"]
