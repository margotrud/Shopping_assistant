from Shopping_assistant.nlp.schema import Axis, Strength, Polarity, Direction
from Shopping_assistant.nlp.resolve.axis_projection import AxisIntent
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents


def test_axis_merge_cancels_on_exact_tie():
    dirs = list(Direction)
    assert len(dirs) >= 2, "Direction enum must have at least 2 members"
    d1, d2 = dirs[0], dirs[1]
    assert d1 != d2

    intents = {
        Axis.BRIGHTNESS: [
            AxisIntent(axis=Axis.BRIGHTNESS, direction=d1, strength=Strength.MED, polarity=Polarity.LIKE, source="global"),
            AxisIntent(axis=Axis.BRIGHTNESS, direction=d2, strength=Strength.MED, polarity=Polarity.LIKE, source="global"),
        ]
    }

    merged = merge_axis_intents(intents)
    dec = merged[Axis.BRIGHTNESS]

    # If the two directions are opposite in sign, it cancels. If they aren't,
    # score won't cancel; this test requires sign-opposition.
    # Enforce sign-opposition using internal sign mapping:
    from Shopping_assistant.nlp.resolve.axis_merge import _dir_sign  # ok in test

    assert _dir_sign(d1) == -_dir_sign(d2)

    assert dec.direction is None
    assert dec.strength is None
    assert dec.score == 0.0
    assert dec.conflicts is True
