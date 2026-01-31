from Shopping_assistant.reco.recommend import resolve_anchor_from_text

EXPECTED = {
    "red": "#ff0000",
    "pink": "#fe46a5",
    "fuchsia": "#ff00ff",
    "purple": "#7e1e9c",
    "mauve": "#e0b0ff",
    "orange": "#ff8000",
    "coral": "#ff7f50",
    "peach": "#ffe5b4",
    "brown": "#964b00",
    "nude": "#e6c2b3",
    "beige": "#a67b5b",
    "blue": "#0000ff",
    "green": "#15b01a",
    "yellow": "#f5bd1f",
    "black": "#3b3c36",
    "white": "#ffffff",
    "gray": "#86949f",
}

def test_color_anchors_from_lexicon():
    for c, hx in EXPECTED.items():
        out = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        assert (out["anchor_hex"] or "").lower() == hx
