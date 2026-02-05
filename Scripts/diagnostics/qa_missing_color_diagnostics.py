import numpy as np

from Shopping_assistant.io.assets import load_default_assets
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

def _top_nearest(inv, lab0, k=10):
    L0, a0, b0 = map(float, lab0)
    L = inv["L_lab"].to_numpy(float)
    a = inv["a_lab"].to_numpy(float)
    b = inv["b_lab"].to_numpy(float)
    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)

    d = np.full(len(inv), np.inf, float)
    d[m] = delta_e_ciede2000(L[m], a[m], b[m], L0, a0, b0)

    idx = np.argsort(d)[:k]
    out = inv.iloc[idx].copy()
    out["_de00"] = d[idx]
    keep = [c for c in ["product_name", "shade_name", "chip_hex", "L_lab", "a_lab", "b_lab", "_de00"] if c in out.columns]
    return out[keep]


def test_missing_color_diagnostics():
    _ensure_assets_env()
    assets = load_default_assets()
    inv = assets.inventory.copy()

    colors = ["blue", "green", "gray"]
    for c in colors:
        a = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        hx = a["anchor_hex"]
        lab = hex_to_lab(hx)

        print(f"\n=== {c.upper()} === anchor={hx} lab={tuple(round(x, 2) for x in lab)}")
        nn = _top_nearest(inv, lab, k=10)
        print(nn.to_string(index=False))
        print(f"min_de00={float(nn['_de00'].min()):.2f}")
