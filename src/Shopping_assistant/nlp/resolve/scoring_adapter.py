from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from Shopping_assistant.nlp.schema import Axis, Direction, Strength
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold


# Mirror the calibrated dim mapping used in color/scoring.py (private there).
# Keep this mapping aligned with scoring._nlp_axis_to_dim(). :contentReference[oaicite:1]{index=1}
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
) -> str:
    """
    Does:
        Convert AxisThresholds into score_inventory-compatible constraint blob tokens:
        dim<=|>=level:weight;...
    """
    tokens: List[str] = []
    for axis, th in thresholds.items():
        axis_name = getattr(axis, "value", str(axis))
        dim = _AXIS_TO_DIM.get(str(axis_name))
        if dim is None:
            continue

        op, level = _threshold_to_op_level(axis=axis, th=th)
        if op is None or level is None:
            continue

        weight = float(th.weight)
        tokens.append(f"{dim}{op}{level}:{weight:.6g}")

    return ";".join(tokens)


def _threshold_to_op_level(*, axis: Axis, th: AxisThreshold) -> Tuple[Optional[str], Optional[str]]:
    """
    Does:
        Map an AxisThreshold (low/high present) to a (op, level) compatible with calibration thresholds.
    """
    # score_inventory only supports "<=" and ">=" ops. :contentReference[oaicite:2]{index=2}
    if th.low is not None and th.high is not None:
        # Not representable as a single token in current scoring language.
        # Keep it strict: refuse rather than invent.
        return None, None

    # Determine direction from bound type:
    # low bound => prefer higher => ">="
    # high bound => prefer lower  => "<="
    if th.low is not None:
        op = ">="
        lvl = _strengthish_level_from_weight(th.weight, op=op)
        return op, lvl

    if th.high is not None:
        op = "<="
        lvl = _strengthish_level_from_weight(th.weight, op=op)
        return op, lvl

    return None, None


def _strengthish_level_from_weight(weight: float, *, op: str) -> str:
    """
    Does:
        Convert weight (typically 1/2/3) into a stable level token for calibration thresholds.
    """
    # Your scorer uses levels as quantiles with fixed thresholds. :contentReference[oaicite:3]{index=3}
    # We interpret higher weight as "more restrictive".
    # For "<=" : more restrictive => lower cap earlier => "medium" then "high"/"very_high" later in the tail
    # For ">=" : more restrictive => require more => "high" then "very_high"
    if weight >= 2.7:
        return "very_high" if op == ">=" else "high"
    if weight >= 1.7:
        return "high" if op == ">=" else "medium"
    return "medium" if op == ">=" else "very_high"


def score_inventory_kwargs(
    *,
    inventory,               # pd.DataFrame (kept untyped here to avoid importing pandas in NLP layer)
    cluster_id: int,
    thresholds: Mapping[Axis, AxisThreshold],
    lambda_constraints: float = 1.0,
    lambda_preference: float = 0.0,
    calibration_path=None,
    preference_weights_path=None,
    prototypes=None,
    assignments_path=None,
) -> Dict:
    """
    Does:
        Build kwargs for Shopping_assistant.color.scoring.score_inventory().
    """
    constraints_blob = build_constraints_blob_from_thresholds(thresholds)
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
