# tests/test_preference_interpreter_contract.py
from __future__ import annotations


def _run(text: str):
    from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
    return interpret_nlp(text)


def test_like_and_negated_brightness():
    text = "I want a red lipstick but not too bright"
    r = _run(text)

    assert "red" in [m.canonical for m in r.mentions]

    assert len(r.constraints) == 1
    c = r.constraints[0]
    assert c.axis.value == "brightness"
    assert c.direction.value == "lower"
    assert c.strength.value == "strong"
    assert c.evidence == "not too bright"

    gs = c.meta["evidence_global_start"]
    ge = c.meta["evidence_global_end"]
    assert text[gs:ge] == "not too bright"


def test_neon_as_constraint():
    text = "I want red but not too neon"
    r = _run(text)

    assert len(r.constraints) == 1
    c = r.constraints[0]
    assert c.axis.value == "vibrancy"
    assert c.direction.value == "lower"
    assert c.evidence == "not too neon"


def test_noise_adverb_not_in_evidence():
    text = "I want red but really not too bright"
    r = _run(text)

    assert len(r.constraints) == 1
    c = r.constraints[0]
    assert c.evidence == "not too bright"


def test_global_mention_span():
    text = "I want a red lipstick"
    r = _run(text)

    assert len(r.mentions) >= 1
    m = [m for m in r.mentions if m.canonical == "red"][0]
    assert text[m.span.start : m.span.end] == "red"


def test_color_not_reused_as_constraint():
    text = "I want red and pink"
    r = _run(text)

    assert len(r.constraints) == 0
