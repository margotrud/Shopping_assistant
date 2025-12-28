from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Polarity
from Shopping_assistant.nlp.resolve.axis_projection import project_axes, AxisIntent
from Shopping_assistant.nlp.resolve.preference_resolver import ResolvedPreference


def test_axis_projection_empty_preference():
    pref = ResolvedPreference(text="test")
    intents = project_axes(pref)
    assert intents == {}
