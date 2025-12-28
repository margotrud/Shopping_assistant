from Shopping_assistant.nlp.schema import (
    NLPResult, Clause, Mention, MentionKind, Polarity, Span,
    Constraint, Axis, Direction, Strength
)
from Shopping_assistant.nlp.resolve import resolve_preference


def test_preference_resolver_keeps_constraint_global_when_multiple_targets():
    nlp = NLPResult(
        text="I want red and pink, not too bright",
        clauses=(
            Clause(clause_id=1, text="I want red and pink, not too bright", meta={}),
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
            Mention(
                span=Span(start=15, end=19),
                raw="pink",
                canonical="pink",
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

    assert len(resolved.liked) == 2
    assert {t.mention.canonical for t in resolved.liked} == {"red", "pink"}

    # Critical invariant: no unsafe attachment
    assert all(len(t.constraints) == 0 for t in resolved.liked)
    assert len(resolved.global_constraints) == 1
    assert resolved.global_constraints[0].constraint.axis == Axis.BRIGHTNESS
