# tests/qa_pool_sizes_plain_colors.py
from __future__ import annotations

from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.color.hard_color_pool import hard_color_pool, params_from_env
from Shopping_assistant.reco.recommend import resolve_anchor_from_text

COLORS = [
    "red","pink","fuchsia","purple","mauve",
    "orange","coral","peach",
    "brown","nude","beige",
    "blue","green","yellow","black","white","gray",
]

def test_pool_sizes_plain_colors():
    assets = load_default_assets()
    inv = assets.inventory.copy()
    p = params_from_env()

    print("\n=== POOL SIZES (hard_color_pool) ===")
    print(f"{'color':8s}  {'hex':10s}  {'pool_n':>6s}")

    for c in COLORS:
        out = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        lab = out.get("anchor_lab")
        hx = out.get("anchor_hex")
        if lab is None:
            print(f"{c:8s}  {str(hx):10s}  {'-':>6s}")
            continue
        dfp = hard_color_pool(inv, anchor_lab=tuple(lab), params=p)
        print(f"{c:8s}  {str(hx):10s}  {len(dfp):6d}")

    assert True
