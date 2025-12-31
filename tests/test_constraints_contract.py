# tests/test_constraints_contract.py
from __future__ import annotations

from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.schema import Direction, Strength


def test_constraints_are_fully_formed_and_non_empty():
    r = interpret_nlp("I want red but not too neon", debug=False)

    assert len(r.constraints) >= 1

    for c in r.constraints:
        assert c.axis is not None
        assert c.direction in (Direction.RAISE, Direction.LOWER)
        assert c.strength in (Strength.WEAK, Strength.MED, Strength.STRONG)
        assert 0.0 <= float(c.confidence) <= 1.0
        assert isinstance(c.evidence, str) and c.evidence.strip() != ""
