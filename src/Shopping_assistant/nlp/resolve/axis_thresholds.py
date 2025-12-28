# src/Shopping_assistant/nlp/resolve/axis_thresholds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from Shopping_assistant.nlp.schema import Axis, Direction, Strength
from Shopping_assistant.nlp.resolve.axis_merge import AxisDecision, _dir_sign


@dataclass(frozen=True, slots=True)
class AxisThreshold:
    """
    Does:
        Represent a numeric constraint window on an axis in [0, 1], plus a weight.
    """
    axis: Axis
    low: Optional[float]          # None => no lower bound
    high: Optional[float]         # None => no upper bound
    weight: float                 # strength-derived weight
    source: str                   # "merged"


# Default cutpoints for generic [0,1] axes
# Interpretation:
# - positive direction => prefer higher => apply lower bound (low)
# - negative direction => prefer lower  => apply upper bound (high)
_CUTPOINTS_LOW = {
    "STRONG": 0.65,
    "MED": 0.50,
    "WEAK": 0.40,
}
_CUTPOINTS_HIGH = {
    "STRONG": 0.35,
    "MED": 0.50,
    "WEAK": 0.60,
}

# Weight mapping for downstream scorers (no scoring here; just exposure)
_STRENGTH_WEIGHT = {
    "STRONG": 3.0,
    "MED": 2.0,
    "WEAK": 1.0,
}


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
    """
    if d.direction is None or d.strength is None:
        return None

    sname = _strength_name(d.strength)
    w = _STRENGTH_WEIGHT.get(sname, 2.0)

    sign = _dir_sign(d.direction)  # +1 or -1 (implementation in axis_merge is enum-name-robust)

    if sign > 0:
        low = _clamp01(_CUTPOINTS_LOW.get(sname, 0.50))
        return AxisThreshold(axis=axis, low=low, high=None, weight=w, source="merged")

    high = _clamp01(_CUTPOINTS_HIGH.get(sname, 0.50))
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
