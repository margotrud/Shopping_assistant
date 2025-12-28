from __future__ import annotations

from Shopping_assistant.color.scoring import constraints_from_nlp
from Shopping_assistant.nlp.models.schema import Axis, Direction, Strength, Constraint as NLPConstraint


def test_constraints_from_nlp_maps_to_calibrated_dims():
    nlp = [
        NLPConstraint(axis=Axis.BRIGHTNESS, direction=Direction.LOWER, strength=Strength.STRONG, evidence="not too bright"),
        NLPConstraint(axis=Axis.VIBRANCY, direction=Direction.LOWER, strength=Strength.MED, evidence="not too neon"),
    ]
    out = constraints_from_nlp(nlp)

    assert len(out) == 2
    assert out[0].dim == "light_hsl" and out[0].op == "<="
    assert out[1].dim == "sat_eff" and out[1].op == "<="
