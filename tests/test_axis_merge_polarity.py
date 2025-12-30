from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Polarity
from Shopping_assistant.nlp.resolve.axis_projection import AxisIntent
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents


def test_dislike_does_not_flip_direction():
    # "not too bright" => direction LOWER, polarity DISLIKE
    intents = {
        Axis.BRIGHTNESS: [
            AxisIntent(
                axis=Axis.BRIGHTNESS,
                direction=Direction.LOWER,
                strength=Strength.STRONG,
                polarity=Polarity.DISLIKE,
                source="global",
            )
        ]
    }
    decisions = merge_axis_intents(intents)
    d = decisions[Axis.BRIGHTNESS]
    assert d.direction == Direction.LOWER
    assert d.has_dislike is True
