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

# Default fallback cutpoints when calibration is missing.
# Only safe for dims that are natively on ~[0,1] in scoring space.
# DO NOT add sat_eff here (it is not on [0,1] and must remain calibration-driven).
_DEFAULT_LEVEL_CUTS_01: Dict[str, Tuple[float, float, float]] = {
    # representative thresholds for levels (monotone)
    "light_hsl": (0.35, 0.55, 0.75),
    "sat_hsl": (0.30, 0.50, 0.70),
    "depth": (0.30, 0.50, 0.70),
    "colorfulness": (0.30, 0.50, 0.70),
}


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

        Key fix:
          - For op=">=" (floor), snap by CEIL (choose the smallest level threshold >= value).
          - For op="<=" (cap), snap by FLOOR (choose the largest level threshold <= value).

        Rationale:
          "nearest" snapping can produce constraints that are trivially satisfied by the entire pool,
          making them no-ops (observed for bright red: light_hsl>=low).
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
        - If th.low is set: op=">=" (floor) and level snapped from numeric th.low.
        - If th.high is set: op="<=" (cap) and level snapped from numeric th.high.
        - If both low and high are set: not supported here (skip).
    """
    if th.low is not None and th.high is not None:
        return None, None

    if th.low is not None:
        op = ">="
        lvl = _level_from_numeric_threshold(dim=dim, value=float(th.low), op=op, calibration=calibration)
        return op, lvl

    if th.high is not None:
        op = "<="
        lvl = _level_from_numeric_threshold(dim=dim, value=float(th.high), op=op, calibration=calibration)
        return op, lvl

    return None, None


def _level_from_numeric_threshold(
    *,
    dim: str,
    value: float,
    op: str,
    calibration: Mapping[str, Any] | None,
) -> Optional[str]:
    """
    Does:
        Convert an NLP numeric cutpoint into a calibrated LEVEL for a given dim.

    Correct behavior:
        - If calibration thresholds for dim are in [0,1], snap MONOTONICALLY:
            op=">=" -> CEIL snap
            op="<=" -> FLOOR snap
        - Otherwise (dim thresholds are on another scale), interpret NLP value as a [0,1] rank/quantile
          and choose the LEVEL by position with CEIL/FLOOR depending on op.

    Fallback:
        - If calibration is missing, only dims known to live on [0,1] may be snapped using
          default cutpoints (_DEFAULT_LEVEL_CUTS_01). All other dims are skipped.
    """
    # --- Fallback when calibration missing: only safe for dims on ~[0,1] ---
    if calibration is None:
        cuts = _DEFAULT_LEVEL_CUTS_01.get(dim)
        if cuts is None:
            return None  # e.g. sat_eff must not be snapped without calibration

        a, b, c = cuts
        # build monotone level thresholds
        items_sorted = [("low", a), ("medium", b), ("high", c), ("very_high", 1.0)]
        v = float(value)

        if op == ">=":
            # CEIL: first threshold >= v, else max
            for lvl, thr in items_sorted:
                if thr >= v:
                    return lvl
            return items_sorted[-1][0]

        # "<=" FLOOR: last threshold <= v, else min
        for lvl, thr in reversed(items_sorted):
            if thr <= v:
                return lvl
        return items_sorted[0][0]

    # --- Calibration present ---
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

    # Case A: dim is calibrated on ~[0,1] => monotone snapping in numeric space
    if 0.0 <= vmin and vmax <= 1.0:
        v = float(value)

        if op == ">=":
            # CEIL: first threshold >= v, else max
            for lvl, thr in items_sorted:
                if thr >= v:
                    return lvl
            return items_sorted[-1][0]

        # "<=" FLOOR: last threshold <= v, else min
        for lvl, thr in reversed(items_sorted):
            if thr <= v:
                return lvl
        return items_sorted[0][0]

    # Case B: dim calibration scale != [0,1] => treat NLP cutpoint as rank/quantile
    q = float(value)
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0

    n = len(items_sorted)
    pos = q * (n - 1)

    if op == ">=":
        # CEIL index
        idx = int(pos) if abs(pos - int(pos)) < 1e-12 else int(pos) + 1
    else:
        # "<=" FLOOR index
        idx = int(pos)

    idx = max(0, min(idx, n - 1))
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
        Exception: dims known to be on [0,1] use deterministic fallback cutpoints.
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
