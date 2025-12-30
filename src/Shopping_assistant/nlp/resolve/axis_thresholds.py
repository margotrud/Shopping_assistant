# src/Shopping_assistant/nlp/resolve/axis_thresholds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from Shopping_assistant.nlp.schema import Axis, Strength
from Shopping_assistant.nlp.resolve.axis_merge import AxisDecision, _dir_sign


@dataclass(frozen=True, slots=True)
class AxisThreshold:
    """
    Does:
        Represent a numeric constraint window on an axis in [0, 1], plus a weight.
    """
    axis: Axis
    low: Optional[float]            # None => no lower bound
    high: Optional[float]           # None => no upper bound
    weight: float                   # strength-derived weight (already includes dislike boost)
    source: str                     # "merged"


_CUTPOINTS_LOW = {
    "STRONG": 0.65,
    "MED": 0.50,
    "WEAK": 0.40,
}

# Generic fallback (only used if axis not found)
_CUTPOINTS_HIGH = {
    "STRONG": 0.41,
    "MED": 0.46,
    "WEAK": 0.52,
}

# Axis-specific caps for LOWER direction.
# Notes:
#   - Brightness LOWER must be a *soft cap* (MED should generally snap to "medium", not "low")
#     given your calibration thresholds (light_hsl low~0.4117, medium~0.4608, high~0.5176).
#   - Vibrancy LOWER is allowed (needed for "not neon/flashy"), but DO NOT trigger it from "bright".
#     That separation is handled upstream in axis_projection.
_CUTPOINTS_HIGH_BY_AXIS: Dict[str, Dict[str, float]] = {
    # brightness -> light_hsl (soft cap policy)
    "brightness": {"STRONG": 0.41, "MED": 0.46, "WEAK": 0.52},

    # vibrancy -> sat_eff (proxy scale in [0,1]; scoring_adapter snaps against calibration)
    # Policy: for "not neon/flashy", MED/STRONG should cap reasonably (MED is meaningful).
    # Keep WEAK permissive.
    "vibrancy": {"STRONG": 0.58, "MED": 0.50, "WEAK": 0.64},

    # saturation -> sat_hsl (optional)
    "saturation": {"STRONG": 0.50, "MED": 0.56, "WEAK": 0.62},

    # clarity -> colorfulness (optional but supported)
    "clarity": {"STRONG": 0.50, "MED": 0.56, "WEAK": 0.62},
}

_STRENGTH_WEIGHT = {
    "STRONG": 3.0,
    "MED": 2.0,
    "WEAK": 1.0,
}

_DISLIKE_THRESHOLD_MULT = 1.5


def thresholds_from_decisions(decisions: Dict[Axis, AxisDecision]) -> Dict[Axis, AxisThreshold]:
    """
    Does:
        Convert per-axis decisions into numeric thresholds in [0,1], skipping undecidable axes.
    """
    out: Dict[Axis, AxisThreshold] = {}
    for axis, d in decisions.items():
        th = _one_axis(axis, d)
        if th is not None:
            out[axis] = th
    return out


def _one_axis(axis: Axis, d: AxisDecision) -> Optional[AxisThreshold]:
    """
    Does:
        Build thresholds for one axis; returns None if canceled/undecidable.

    Policy:
        - For brightness LOWER: keep it soft (MED stays MED; should snap to "medium" in calibration).
        - For vibrancy LOWER: allow it (needed for "not neon/flashy"), but ensure it is only triggered
          when upstream semantics call it (axis_projection decides that).
    """
    if d.direction is None or d.strength is None:
        return None

    sname = _strength_name(d.strength)
    w = float(_STRENGTH_WEIGHT.get(sname, 2.0))

    if getattr(d, "has_dislike", False):
        w *= float(_DISLIKE_THRESHOLD_MULT)

    sign = _dir_sign(d.direction)
    axis_name = getattr(axis, "value", str(axis)).strip().lower()

    # RAISE => set floor
    if sign > 0:
        low = _clamp01(_CUTPOINTS_LOW.get(sname, 0.50))
        return AxisThreshold(axis=axis, low=low, high=None, weight=w, source="merged")

    # LOWER => set cap
    by_axis = _CUTPOINTS_HIGH_BY_AXIS.get(axis_name)

    # brightness soft-cap policy: do not escalate MED->STRONG
    if axis_name == "brightness" and isinstance(by_axis, dict):
        if sname == "STRONG":
            high_raw = by_axis["STRONG"]
        elif sname == "MED":
            high_raw = by_axis["MED"]
        else:
            high_raw = by_axis["WEAK"]
    else:
        high_raw = (by_axis.get(sname) if isinstance(by_axis, dict) else None)
        if high_raw is None:
            high_raw = _CUTPOINTS_HIGH.get(sname, 0.50)

    high = _clamp01(float(high_raw))
    return AxisThreshold(axis=axis, low=None, high=high, weight=w, source="merged")


def _strength_name(s: Strength) -> str:
    name = getattr(s, "name", str(s)).upper()
    if "STRONG" in name:
        return "STRONG"
    if "MED" in name or "MEDIUM" in name:
        return "MED"
    if "WEAK" in name or "LOW" in name:
        return "WEAK"
    return "MED"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
