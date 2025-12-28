# tests/test_axis_thresholds_smoke.py
from Shopping_assistant.nlp.schema import Axis, Direction, Strength
from Shopping_assistant.nlp.resolve.axis_merge import AxisDecision, _dir_sign
from Shopping_assistant.nlp.resolve.axis_thresholds import thresholds_from_decisions


def _pick_dir(sign: int) -> Direction:
    for d in list(Direction):
        if _dir_sign(d) == sign:
            return d
    raise AssertionError(f"No Direction member with sign={sign}")


def test_thresholds_skip_undecidable():
    decisions = {
        Axis.BRIGHTNESS: AxisDecision(
            axis=Axis.BRIGHTNESS,
            direction=None,
            strength=None,
            score=0.0,
            conflicts=True,
            sources=(),
            intents=(),
        )
    }
    out = thresholds_from_decisions(decisions)
    assert out == {}


def test_thresholds_positive_sets_low_bound():
    dpos = _pick_dir(+1)

    decisions = {
        Axis.BRIGHTNESS: AxisDecision(
            axis=Axis.BRIGHTNESS,
            direction=dpos,
            strength=Strength.MED,
            score=2.0,
            conflicts=False,
            sources=("global",),
            intents=(),
        )
    }
    out = thresholds_from_decisions(decisions)
    th = out[Axis.BRIGHTNESS]
    assert th.low is not None
    assert th.high is None
    assert 0.0 <= th.low <= 1.0
    assert th.weight > 0.0


def test_thresholds_negative_sets_high_bound():
    dneg = _pick_dir(-1)

    decisions = {
        Axis.BRIGHTNESS: AxisDecision(
            axis=Axis.BRIGHTNESS,
            direction=dneg,
            strength=Strength.MED,
            score=-2.0,
            conflicts=False,
            sources=("global",),
            intents=(),
        )
    }
    out = thresholds_from_decisions(decisions)
    th = out[Axis.BRIGHTNESS]
    assert th.low is None
    assert th.high is not None
    assert 0.0 <= th.high <= 1.0
    assert th.weight > 0.0
