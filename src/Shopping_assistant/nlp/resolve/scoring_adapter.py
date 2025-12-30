# src/Shopping_assistant/nlp/resolve/scoring_adapter.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from Shopping_assistant.nlp.schema import Axis
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold


# Keep aligned with Shopping_assistant.color.scoring._nlp_axis_to_dim()
_AXIS_TO_DIM: Dict[str, str] = {
    "brightness": "light_hsl",
    "saturation": "sat_hsl",
    "vibrancy": "sat_eff",
    "depth": "depth",
    "clarity": "colorfulness",
}

_ALLOWED_LEVELS = ("low", "medium", "high", "very_high")


def build_constraints_blob_from_thresholds(
    thresholds: Mapping[Axis, AxisThreshold],
    *,
    calibration: Mapping[str, Any] | None = None,
) -> str:
    """
    Does:
        Convert AxisThresholds into score_inventory-compatible constraint blob tokens:
        dim<=|>=LEVEL:weight;...

    Important:
        LEVEL is derived from the NUMERIC cutpoint (AxisThreshold.low/high) by snapping to a
        calibrated LEVEL for that dim.

        If a dim's calibration thresholds are on [0,1], we snap by nearest numeric distance.
        If a dim's calibration thresholds are NOT on [0,1] (e.g. sat_eff), we interpret the NLP
        cutpoint as a [0,1] rank/quantile and choose the corresponding LEVEL by index.

        If calibration is missing/unusable for a dim, the constraint is skipped (do not emit
        numeric cutpoints: they are on the NLP [0,1] scale and will break dims like sat_eff).
    """
    tokens: List[str] = []
    for axis, th in thresholds.items():
        axis_name = getattr(axis, "value", str(axis))
        dim = _AXIS_TO_DIM.get(str(axis_name))
        if dim is None:
            continue

        op, level = _threshold_to_op_level(axis=axis, th=th, dim=dim, calibration=calibration)
        if op is None or level is None:
            continue

        weight = float(th.weight)
        tokens.append(f"{dim}{op}{level}:{weight:.6g}")

    return ";".join(tokens)


def _threshold_to_op_level(
    *,
    axis: Axis,
    th: AxisThreshold,
    dim: str,
    calibration: Mapping[str, Any] | None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Does:
        Map an AxisThreshold (low/high present) to a (op, level) compatible with calibration thresholds.

    Rule:
        - If th.low is set: op=">=" (floor) and level is snapped from numeric th.low via calibration.
        - If th.high is set: op="<=" (cap) and level is snapped from numeric th.high via calibration.
        - If both low and high are set: not supported here (skip).
    """
    if th.low is not None and th.high is not None:
        return None, None

    if th.low is not None:
        op = ">="
        lvl = _level_from_numeric_threshold(dim=dim, value=float(th.low), calibration=calibration)
        return op, lvl

    if th.high is not None:
        op = "<="
        lvl = _level_from_numeric_threshold(dim=dim, value=float(th.high), calibration=calibration)
        return op, lvl

    return None, None


def _level_from_numeric_threshold(
    *,
    dim: str,
    value: float,
    calibration: Mapping[str, Any] | None,
) -> Optional[str]:
    """
    Does:
        Convert an NLP numeric cutpoint into a calibrated LEVEL for a given dim.

    Correct behavior:
        - If calibration thresholds for dim are in [0,1], snap by nearest numeric distance.
        - Otherwise (dim thresholds are on another scale), interpret NLP value as a [0,1] rank/quantile
          and choose the LEVEL by position in the sorted thresholds list.
    """
    if calibration is None:
        return None

    levels_map = _get_dim_thresholds(calibration, dim=dim)
    if not levels_map:
        return None

    items = [(k, float(v)) for k, v in levels_map.items() if k in _ALLOWED_LEVELS]
    if not items:
        return None

    # Sort levels by increasing numeric threshold so "low < medium < high < very_high"
    items_sorted = sorted(items, key=lambda kv: kv[1])
    vals = [v for _, v in items_sorted]
    vmin, vmax = min(vals), max(vals)

    # Case A: dim is calibrated on ~[0,1] => nearest numeric snapping is valid
    if 0.0 <= vmin and vmax <= 1.0:
        best_level, _ = min(items_sorted, key=lambda kv: abs(kv[1] - float(value)))
        return best_level

    # Case B: dim calibration scale != [0,1] => treat NLP cutpoint as rank/quantile
    q = float(value)
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0

    idx = int(round(q * (len(items_sorted) - 1)))
    idx = max(0, min(idx, len(items_sorted) - 1))
    return items_sorted[idx][0]


def _get_dim_thresholds(calibration: Mapping[str, Any], *, dim: str) -> Mapping[str, float]:
    """
    Does:
        Extract {level: numeric_threshold} for a dim from either:
        - full calibration dict: calibration["thresholds"][dim]
        - or a direct thresholds dict: calibration[dim]
    """
    ths = calibration.get("thresholds") if isinstance(calibration, Mapping) else None
    if isinstance(ths, Mapping):
        dm = ths.get(dim)
        if isinstance(dm, Mapping):
            return dm  # type: ignore[return-value]

    dm2 = calibration.get(dim) if isinstance(calibration, Mapping) else None
    if isinstance(dm2, Mapping):
        return dm2  # type: ignore[return-value]

    return {}


def score_inventory_kwargs(
    *,
    inventory,  # pd.DataFrame
    cluster_id: int,
    thresholds: Mapping[Axis, AxisThreshold],
    lambda_constraints: float = 1.0,
    lambda_preference: float = 0.0,
    calibration_path=None,
    preference_weights_path=None,
    prototypes=None,
    assignments_path=None,
    calibration: Mapping[str, Any] | None = None,
) -> Dict:
    """
    Does:
        Build kwargs for Shopping_assistant.color.scoring.score_inventory().

    Important:
        Constraint blob uses LEVEL tokens (calibration-aware). If calibration is missing,
        constraints that require snapping are skipped rather than emitting broken numeric cutpoints.
    """
    constraints_blob = build_constraints_blob_from_thresholds(thresholds, calibration=calibration)
    return {
        "inventory": inventory,
        "prototypes": prototypes,
        "assignments_path": assignments_path,
        "cluster_id": int(cluster_id),
        "constraints": constraints_blob,
        "lambda_constraints": float(lambda_constraints),
        "lambda_preference": float(lambda_preference),
        "calibration_path": calibration_path,
        "preference_weights_path": preference_weights_path,
    }
