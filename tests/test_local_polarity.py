from Shopping_assistant.nlp.llm.local_polarity import (
    _normalize_llm_output,
)


def test_normalize_llm_output_accepts_valid_json():
    result = _normalize_llm_output(
        '{"mauve": "LIKE", "coral": "DISLIKE"}',
        ["mauve", "coral"],
    )

    assert result == {
        "mauve": "LIKE",
        "coral": "DISLIKE",
    }


def test_normalize_llm_output_accepts_json_inside_markdown():
    result = _normalize_llm_output(
        '```json\n{"mauve": "LIKE"}\n```',
        ["mauve"],
    )

    assert result == {
        "mauve": "LIKE",
    }


def test_normalize_llm_output_rejects_unknown_label():
    result = _normalize_llm_output(
        '{"mauve": "UNKNOWN"}',
        ["mauve"],
    )

    assert result == {
        "mauve": None,
    }


def test_normalize_llm_output_rejects_invalid_label():
    result = _normalize_llm_output(
        '{"mauve": "POSITIVE"}',
        ["mauve"],
    )

    assert result == {
        "mauve": None,
    }


def test_normalize_llm_output_ignores_invented_keys():
    result = _normalize_llm_output(
        '{"pink": "LIKE", "coral": "DISLIKE"}',
        ["mauve"],
    )

    assert result == {
        "mauve": None,
    }


def test_normalize_llm_output_rejects_invalid_json():
    result = _normalize_llm_output(
        "mauve is LIKE",
        ["mauve"],
    )

    assert result == {
        "mauve": None,
    }