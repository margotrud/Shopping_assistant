import Shopping_assistant.nlp.parsing.polarity as polarity


def test_hybrid_stops_after_lexical(monkeypatch):
    polarity.make_hybrid_polarity_fn.cache_clear()

    def fail_semantic(*args, **kwargs):
        raise AssertionError(
            "Semantic backend should not be called."
        )

    monkeypatch.setattr(
        polarity,
        "make_free_polarity_fn",
        fail_semantic,
    )

    import Shopping_assistant.nlp.llm.local_polarity as local_polarity

    def fail_generative(*args, **kwargs):
        raise AssertionError(
            "Generative backend should not be called."
        )

    monkeypatch.setattr(
        local_polarity,
        "make_local_generative_polarity_fn",
        fail_generative,
    )

    fn = polarity.make_hybrid_polarity_fn()

    result = fn(
        "I want pink",
        ["pink"],
    )

    assert result == {
        "pink": "LIKE",
    }


def test_hybrid_uses_semantic_for_unresolved_mentions(
    monkeypatch,
):
    polarity.make_hybrid_polarity_fn.cache_clear()

    def fake_make_semantic(*args, **kwargs):
        def _fn(clause_text, mentions):
            assert mentions == ["mauve"]

            return {
                "mauve": "LIKE",
            }

        return _fn

    monkeypatch.setattr(
        polarity,
        "make_free_polarity_fn",
        fake_make_semantic,
    )

    import Shopping_assistant.nlp.llm.local_polarity as local_polarity

    def fail_generative(*args, **kwargs):
        raise AssertionError(
            "Generative backend should not be called "
            "when semantic resolves the mention."
        )

    monkeypatch.setattr(
        local_polarity,
        "make_local_generative_polarity_fn",
        fail_generative,
    )

    fn = polarity.make_hybrid_polarity_fn()

    result = fn(
        "I'm drawn toward mauve",
        ["mauve"],
    )

    assert result == {
        "mauve": "LIKE",
    }


def test_hybrid_uses_generative_after_semantic_abstains(
    monkeypatch,
):
    polarity.make_hybrid_polarity_fn.cache_clear()

    def fake_make_semantic(*args, **kwargs):
        def _fn(clause_text, mentions):
            return {
                mention: None
                for mention in mentions
            }

        return _fn

    monkeypatch.setattr(
        polarity,
        "make_free_polarity_fn",
        fake_make_semantic,
    )

    import Shopping_assistant.nlp.llm.local_polarity as local_polarity

    def fake_make_generative(*args, **kwargs):
        def _fn(clause_text, mentions):
            assert mentions == ["mauve"]

            return {
                "mauve": "LIKE",
            }

        return _fn

    monkeypatch.setattr(
        local_polarity,
        "make_local_generative_polarity_fn",
        fake_make_generative,
    )

    fn = polarity.make_hybrid_polarity_fn()

    result = fn(
        "I'm drawn toward mauve",
        ["mauve"],
    )

    assert result == {
        "mauve": "LIKE",
    }