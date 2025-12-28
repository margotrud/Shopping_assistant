from Shopping_assistant.nlp.schema import Axis
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold
from Shopping_assistant.nlp.resolve.scoring_adapter import build_constraints_blob_from_thresholds


def test_build_constraints_blob_from_thresholds_smoke():
    thresholds = {
        Axis.BRIGHTNESS: AxisThreshold(axis=Axis.BRIGHTNESS, low=None, high=0.4, weight=2.0, source="merged"),
    }
    blob = build_constraints_blob_from_thresholds(thresholds)
    assert "light_hsl" in blob
    assert "<=" in blob
    assert ":" in blob
