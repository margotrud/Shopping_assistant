# src/Shopping_assistant/reco/_family_constraints.py
from __future__ import annotations

from typing import Sequence

import numpy as np

from Shopping_assistant.color.constraints import ConstraintSpec as FamilyConstraintSpec

from ._constants import _ALLOWED_Q
from ._utils import _get, _get_enum_value


def _snap_q(q: float) -> float:
    qf = float(q)
    i = int(np.argmin(np.abs(_ALLOWED_Q - qf)))
    return float(_ALLOWED_Q[i])


def _family_specs_from_nlp(nlp_constraints: Sequence[object]) -> list[FamilyConstraintSpec]:
    """
    Map existing NLP constraint axes into within-family axes.
    Generic mapping:
      - brightness lower/raise -> L below/above
      - saturation/vibrancy lower/raise -> C below/above

    IMPORTANT: quantiles must exist in label_distributions; we snap to _ALLOWED_Q.
    """
    if not nlp_constraints:
        return []

    specs: list[FamilyConstraintSpec] = []

    def _strength_params(strength: str, direction: str):
        s = (strength or "med").lower()
        d = (direction or "").lower()

        s_w = {"weak": 0.6, "med": 0.85, "strong": 1.0}.get(s, 0.85)

        if d in {"raise", "higher", "up", "increase"}:
            if s == "weak":
                q_lo, q_hi = 0.45, 0.65
            elif s == "strong":
                q_lo, q_hi = 0.60, 0.85
            else:
                q_lo, q_hi = 0.50, 0.75
        else:
            if s == "weak":
                q_lo, q_hi = 0.35, 0.55
            elif s == "strong":
                q_lo, q_hi = 0.15, 0.40
            else:
                q_lo, q_hi = 0.25, 0.50

        q_lo_s = _snap_q(q_lo)
        q_hi_s = _snap_q(q_hi)
        if q_hi_s <= q_lo_s:
            idx = int(np.where(_ALLOWED_Q == q_lo_s)[0][0])
            q_hi_s = float(_ALLOWED_Q[min(idx + 1, len(_ALLOWED_Q) - 1)])
        return float(q_lo_s), float(q_hi_s), float(s_w)

    for c in nlp_constraints:
        axis = (_get_enum_value(_get(c, "axis", None)) or "").lower()
        direction = (_get_enum_value(_get(c, "direction", None)) or "").lower()
        strength = (_get_enum_value(_get(c, "strength", None)) or "med").lower()

        if axis not in {"brightness", "saturation", "vibrancy"}:
            continue
        if direction not in {"lower", "raise"}:
            continue

        q_lo, q_hi, s_w = _strength_params(strength, direction)

        if axis == "brightness":
            specs.append(
                FamilyConstraintSpec(
                    axis="L",
                    direction=("below" if direction == "lower" else "above"),
                    strength=float(s_w),
                    q_lo=float(q_lo),
                    q_hi=float(q_hi),
                )
            )
        else:
            specs.append(
                FamilyConstraintSpec(
                    axis="C",
                    direction=("below" if direction == "lower" else "above"),
                    strength=float(s_w),
                    q_lo=float(q_lo),
                    q_hi=float(q_hi),
                )
            )

    return specs
