from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.reco.recommend import recommend_from_text

COLORS = ["red","pink","fuchsia","purple","mauve","orange","coral","peach","brown","nude","beige","blue","green","yellow","black","white","gray"]

def test_pool_sizes_plain_colors_failsafe():
    assets = load_default_assets()

    print("\n=== POOL SIZES (recommend_from_text with failsafe) ===")
    print("color   topk_returned")
    for c in COLORS:
        df = recommend_from_text(f"I want a {c} lipstick", assets=assets, topk=20, debug=False)
        print(f"{c:6s} {len(df):4d}")

    # Guardrail: if has_color resolved, we want non-empty results even for missing colors.
    # (If you want stricter: assert >= 10 for all; but keep it soft for now.)
    df_blue = recommend_from_text("I want a blue lipstick", assets=assets, topk=20, debug=False)
    assert len(df_blue) > 0
