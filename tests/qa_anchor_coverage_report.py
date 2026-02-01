# tests/qa_anchor_coverage_report.py
from __future__ import annotations

import numpy as np

from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.color.deltae import delta_e_ciede2000
from Shopping_assistant.reco.typical_color import hex_to_lab
from Shopping_assistant.reco.recommend import resolve_anchor_from_text

COLORS = [
    "red","pink","fuchsia","purple","mauve",
    "orange","coral","peach",
    "brown","nude","beige",
    "blue","green","yellow","black","white","gray",
]

def _best_de00(inv, lab0):
    L0, a0, b0 = map(float, lab0)
    L = inv["L_lab"].to_numpy(float)
    a = inv["a_lab"].to_numpy(float)
    b = inv["b_lab"].to_numpy(float)
    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    d = np.full(len(inv), np.inf, float)
    d[m] = delta_e_ciede2000(L[m], a[m], b[m], L0, a0, b0)
    return float(np.min(d)), int(np.sum(d <= 18.0)), int(np.sum(d <= 12.0))

def test_anchor_coverage_report():
    assets = load_default_assets()
    inv = assets.inventory.copy()

    print("\n=== ANCHOR COVERAGE (vs inventory) ===")
    print(f"{'color':8s}  {'hex':10s}  {'minDE':>6s}  {'<=18':>5s}  {'<=12':>5s}")

    bad = []
    for c in COLORS:
        out = resolve_anchor_from_text(f"I want a {c} lipstick", debug=False)
        hx = out.get("anchor_hex")
        if not hx:
            bad.append((c, None, None))
            print(f"{c:8s}  {str(hx):10s}  {'-':>6s}  {'-':>5s}  {'-':>5s}")
            continue

        lab = hex_to_lab(hx)
        mn, n18, n12 = _best_de00(inv, lab)
        print(f"{c:8s}  {hx:10s}  {mn:6.2f}  {n18:5d}  {n12:5d}")

        # diagnostics: flag obvious “out-of-domain for lipstick inventory”
        if n18 == 0:
            bad.append((c, hx, "no matches within DE<=18"))

    # Do NOT fail the test hard; this is a report.
    # But it should at least run and print.
    assert True

    if bad:
        print("\n--- FLAGS ---")
        for x in bad:
            print(x)
