import numpy as np

from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.color.hard_color_pool import params_from_env
from Shopping_assistant.color.deltae import delta_e_ciede2000
from Shopping_assistant.reco.typical_color import hex_to_lab
from Shopping_assistant.reco.recommend import resolve_anchor_from_text

# put at top of the test file
import os
from pathlib import Path

def _ensure_assets_env():
    # 1) enriched CSV
    if not os.environ.get("SA_ENRICHED_CSV_PATH"):
        # preferred canonical path in your repo
        p = Path("data/enriched_data/Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv")
        if not p.exists():
            # fallback: first match
            hits = sorted(Path("data").rglob("*enriched*.csv"))
            if not hits:
                raise FileNotFoundError("Cannot find any *enriched*.csv under data/.")
            p = hits[0]
        os.environ["SA_ENRICHED_CSV_PATH"] = str(p)

    # 2) calibration json
    if not os.environ.get("SA_CALIBRATION_JSON_PATH"):
        p = Path("data/models/color_scoring_calibration.json")
        if not p.exists():
            hits = sorted(Path("data").rglob("*calibration*.json"))
            if not hits:
                raise FileNotFoundError("Cannot find any *calibration*.json under data/.")
            p = hits[0]
        os.environ["SA_CALIBRATION_JSON_PATH"] = str(p)

def test_pool_thresholds_report():
    _ensure_assets_env()
    assets = load_default_assets()
    inv = assets.inventory.copy()
    p = params_from_env()

    colors = ["nude", "black", "white", "beige", "mauve"]
    print("\n=== POOL THRESHOLDS REPORT ===")
    print("color   anchor      C0     thr_used  <=18    <=thr")

    for c in colors:
        a = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        hx = a["anchor_hex"]
        lab0 = hex_to_lab(hx)
        L0, a0, b0 = map(float, lab0)
        C0 = float(np.hypot(a0, b0))

        L = inv["L_lab"].to_numpy(float)
        aa = inv["a_lab"].to_numpy(float)
        bb = inv["b_lab"].to_numpy(float)
        m = np.isfinite(L) & np.isfinite(aa) & np.isfinite(bb)

        d = np.full(len(inv), np.inf, float)
        d[m] = delta_e_ciede2000(L[m], aa[m], bb[m], L0, a0, b0)

        thr_used = float(p.de00_max_neutral if C0 < float(p.neutral_anchor_c_max) else p.de00_max)
        n18 = int(np.sum(d <= 18.0))
        nthr = int(np.sum(d <= thr_used))

        print(f"{c:6s}  {hx:8s}  {C0:6.2f}  {thr_used:7.2f}  {n18:5d}  {nthr:5d}")
