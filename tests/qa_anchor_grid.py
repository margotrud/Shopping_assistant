# tests/qa_anchor_grid.py
from __future__ import annotations

import re

from Shopping_assistant.reco.recommend import resolve_anchor_from_text

COLORS = [
    "red","pink","fuchsia","purple","mauve",
    "orange","coral","peach",
    "brown","nude","beige",
    "blue","green","yellow","black","white","gray",
]

TEMPLATES = [
    "I want a {c} lipstick",
    "I want {c} lipstick",
    "Looking for a {c} shade",
    "Give me a {c} lip color",
]

def _is_hex(h: str | None) -> bool:
    return isinstance(h, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", h) is not None

def test_anchor_grid_print():
    # This is a QA test: it prints results and asserts only minimal invariants.
    # Run with: pytest -q -s tests/qa_anchor_grid.py
    rows = []
    for c in COLORS:
        for t in TEMPLATES:
            q = t.format(c=c)
            out = resolve_anchor_from_text(q, debug=False)
            hx = out.get("anchor_hex")
            rows.append((c, q, hx, out.get("anchor_source"), out.get("anchor_key")))

    print("\n=== ANCHOR GRID ===")
    for c, q, hx, src, key in rows:
        print(f"{c:8s} | {hx} | src={src} | key={key} | q={q}")

    # Minimal invariants:
    # - If a color is known, anchor_hex must be a hex.
    # - We do NOT assert specific hexes here (those are in qa_color_anchors_lexicon.py already).
    for c, q, hx, src, key in rows:
        if c != "nude":  # nude can be lexicon-missing depending on your data; keep this flexible
            assert _is_hex(hx), f"Missing/invalid hex for {c} via q={q!r}: {hx}"
