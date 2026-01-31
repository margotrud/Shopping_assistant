from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon


def test_no_modifier_singletons_dark():
    lex = load_default_lexicon()
    assert lex.resolve("dark") == []
    assert lex.resolve("dark rose")
