from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon


def test_resolve_exact_family_tokens():
    lex = load_default_lexicon()
    for k in ["purple", "violet", "beige", "mauve", "pink", "aubergine"]:
        out = lex.resolve(k)
        assert out and out[0].alias == k and out[0].hex.startswith("#")
