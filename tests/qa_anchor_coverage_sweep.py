import numpy as np
from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.reco.recommend import resolve_anchor_from_text
from Shopping_assistant.reco.typical_color import hex_to_lab
from Shopping_assistant.color.deltae import delta_e_ciede2000

COLORS = ["red","pink","fuchsia","purple","mauve","orange","coral","peach","brown","nude","beige","blue","green","yellow","black","white","gray"]
THRS = [18, 20, 22, 26, 30]

def test_anchor_coverage_sweep():
    assets = load_default_assets()
    inv = assets.inventory.copy()

    L = inv["L_lab"].to_numpy(float)
    a = inv["a_lab"].to_numpy(float)
    b = inv["b_lab"].to_numpy(float)
    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)

    print("\n=== ANCHOR COVERAGE SWEEP (ΔE00) ===")
    print("color   hex        minDE  " + "  ".join([f"<= {t:2d}" for t in THRS]))

    for c in COLORS:
        out = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        hx = out["anchor_hex"]
        lab = hex_to_lab(hx)
        d = np.full(len(inv), np.inf, float)
        d[m] = delta_e_ciede2000(L[m], a[m], b[m], float(lab[0]), float(lab[1]), float(lab[2]))
        mn = float(np.min(d))
        counts = [int(np.sum(d <= t)) for t in THRS]
        print(f"{c:6s} {hx:9s} {mn:6.2f}  " + "  ".join([f"{x:4d}" for x in counts]))
