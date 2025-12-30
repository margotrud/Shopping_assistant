from Shopping_assistant.nlp.schema import Axis, Direction, Polarity, Strength
from Shopping_assistant.nlp.models.result import make_constraint
from Shopping_assistant.nlp.resolve.conflicts import resolve_symbolic_conflicts


def test_conflict_same_axis_keeps_best_by_strength_then_confidence():
    c1 = make_constraint(
        0, axis=Axis.BRIGHTNESS, direction=Direction.LOWER, strength=Strength.MED, evidence="less bright", confidence=0.2,
        meta={"clause_polarity": Polarity.DISLIKE.value},
    )
    c2 = make_constraint(
        1, axis=Axis.BRIGHTNESS, direction=Direction.RAISE, strength=Strength.STRONG, evidence="more bright", confidence=0.1,
        meta={"clause_polarity": Polarity.LIKE.value},
    )
    out, diag = resolve_symbolic_conflicts((c1, c2))
    assert len(out) == 1
    assert out[0].axis == Axis.BRIGHTNESS
    # best is STRONG despite lower confidence
    assert out[0].strength == Strength.STRONG
    assert out[0].direction in (Direction.RAISE, Direction.LOWER)
    assert diag["suppressed_constraints"]


def test_dislike_raise_is_inverted_to_lower():
    c = make_constraint(
        0, axis=Axis.BRIGHTNESS, direction=Direction.RAISE, strength=Strength.MED, evidence="bright",
        meta={"clause_polarity": Polarity.DISLIKE.value},
    )
    out, diag = resolve_symbolic_conflicts((c,))
    assert len(out) == 1
    assert out[0].direction == Direction.LOWER
    assert diag["normalized_constraints"]


def test_cap_guard_too_is_lower_under_dislike():
    c = make_constraint(
        0, axis=Axis.BRIGHTNESS, direction=Direction.RAISE, strength=Strength.STRONG, evidence="not too bright",
        meta={"clause_polarity": Polarity.DISLIKE.value},
    )
    out, diag = resolve_symbolic_conflicts((c,))
    assert len(out) == 1
    assert out[0].direction == Direction.LOWER
    assert diag["normalized_constraints"]
