from Shopping_assistant.nlp.schema import Axis
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold
from Shopping_assistant.nlp.resolve.scoring_adapter import build_constraints_blob_from_thresholds


def _idx_level(blob: str) -> int:
    order = ["low", "medium", "high", "very_high"]
    for i, tok in enumerate(order):
        if tok in blob:
            return i
    return -1


def test_weight_monotonicity_levels_le_cap_is_stricter_when_weight_increases():
    # "<=" : higher weight => stricter => LOWER level token
    th_w1 = AxisThreshold(axis=Axis.BRIGHTNESS, low=None, high=0.35, weight=1.0, source="merged")
    th_w2 = AxisThreshold(axis=Axis.BRIGHTNESS, low=None, high=0.35, weight=2.0, source="merged")
    th_w3 = AxisThreshold(axis=Axis.BRIGHTNESS, low=None, high=0.35, weight=3.0, source="merged")

    b1 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w1})
    b2 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w2})
    b3 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w3})

    assert _idx_level(b1) >= _idx_level(b2) >= _idx_level(b3)


def test_weight_monotonicity_levels_ge_floor_is_stricter_when_weight_increases():
    # ">=" : higher weight => stricter => HIGHER level token
    th_w1 = AxisThreshold(axis=Axis.BRIGHTNESS, low=0.35, high=None, weight=1.0, source="merged")
    th_w2 = AxisThreshold(axis=Axis.BRIGHTNESS, low=0.35, high=None, weight=2.0, source="merged")
    th_w3 = AxisThreshold(axis=Axis.BRIGHTNESS, low=0.35, high=None, weight=3.0, source="merged")

    b1 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w1})
    b2 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w2})
    b3 = build_constraints_blob_from_thresholds({Axis.BRIGHTNESS: th_w3})

    assert _idx_level(b1) <= _idx_level(b2) <= _idx_level(b3)
