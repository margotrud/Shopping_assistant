from Shopping_assistant.nlp.schema import (
    NLPResult, Clause, Mention, MentionKind, Polarity, Span,
    Constraint, Axis, Direction, Strength
)
from Shopping_assistant.nlp.resolve import resolve_preference


def test_preference_resolver_attaches_constraints_when_unambiguous():
    nlp = NLPResult(
        text="I want red but not too bright",
        clauses=(
            Clause(
                clause_id=1,
                text="I want red but not too bright",
                meta={},
            ),
        ),
        mentions=(
            Mention(
                span=Span(start=7, end=10),
                raw="red",
                canonical="red",
                kind=MentionKind.COLOR,
                polarity=Polarity.LIKE,
                confidence=1.0,
                clause_id=1,
                meta={},
            ),
        ),
        constraints=(
            Constraint(
                axis=Axis.BRIGHTNESS,
                direction=Direction.LOWER,
                strength=Strength.MED,
                evidence="not too bright",
                clause_id=1,
                confidence=1.0,
                scope=None,
                meta={},
            ),
        ),
        diagnostics={},
    )

    resolved = resolve_preference(nlp)

    assert len(resolved.liked) == 1
    assert resolved.liked[0].mention.canonical == "red"
    assert len(resolved.liked[0].constraints) == 1
    assert resolved.liked[0].constraints[0].axis == Axis.BRIGHTNESS
    assert len(resolved.global_constraints) == 0
