from __future__ import annotations

import os
import numpy as np

from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.reco.recommend import (
    recommend_from_text,
    resolve_anchor_from_text,
    resolve_effective_anchor_from_text,
)

# -----------------------------
# Env defaults (repo-friendly)
# -----------------------------
os.environ["SA_ENRICHED_CSV_PATH"] = os.environ.get(
    "SA_ENRICHED_CSV_PATH",
    "data/enriched_data/Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv",
)
os.environ["SA_CALIBRATION_JSON_PATH"] = os.environ.get(
    "SA_CALIBRATION_JSON_PATH",
    "data/models/color_scoring_calibration.json",
)

assets = load_default_assets()

def C(lab):
    return float(np.hypot(lab[1], lab[2])) if lab is not None else None

def chk(msg, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(2)

# -----------------------------
# 0) Imports sanity
# -----------------------------
print("\n================= IMPORTS =================")
chk("imports ok", True)

# -----------------------------
# 1) NON-PLAIN GUARD
# -----------------------------
print("\n================= NON-PLAIN GUARD =================")
tests = [
    ("plain vivid",       "I want a vivid pink lipstick", True, 0.9),
    ("extra constraint",  "I want a vivid pink lipstick, not too bright", False, None),
    ("darker preference", "I want a vivid pink lipstick that is darker", False, None),
]
for name, q, exp_hi, exp_cq in tests:
    r = resolve_effective_anchor_from_text(q, assets=assets, debug=False)
    chk(f"{name}: hi_chroma=={exp_hi}", r.get("hi_chroma") is exp_hi)
    chk(f"{name}: chroma_q=={exp_cq}", r.get("chroma_q") == exp_cq)

# -----------------------------
# 2) VIVID increases chroma vs base (pink only)
# -----------------------------
print("\n================= VIVID CHROMA DELTA (PINK) =================")
r0 = resolve_effective_anchor_from_text("I want a pink lipstick", assets=assets)
r1 = resolve_effective_anchor_from_text("I want a vivid pink lipstick", assets=assets)
chk("vivid increases chroma vs base (pink)", C(r1["anchor_lab_effective"]) > C(r0["anchor_lab_effective"]))
print("base C :", C(r0["anchor_lab_effective"]))
print("vivid C:", C(r1["anchor_lab_effective"]))

# -----------------------------
# 3) Families where vivid may NOT increase chroma (still hi_chroma True)
# -----------------------------
print("\n================= VIVID PER FAMILY (NO dC GUARANTEE) =================")
pairs = [
    ("purple", "I want a purple lipstick", "I want a vivid purple lipstick"),
    ("nude",   "I want a nude lipstick",   "I want a vivid nude lipstick"),
]
for lab, q0, q1 in pairs:
    a0 = resolve_effective_anchor_from_text(q0, assets=assets)
    a1 = resolve_effective_anchor_from_text(q1, assets=assets)
    chk(f"{lab}: vivid triggers hi_chroma", a1.get("hi_chroma") is True and a1.get("chroma_q")==0.9)
    print(f"{lab:6s} C0={C(a0['anchor_lab_effective']):.3f} C1={C(a1['anchor_lab_effective']):.3f}")

# -----------------------------
# 4) _de00_anchor column sanity (numeric ΔE00)
# -----------------------------
print("\n================= _de00_anchor SANITY =================")
q = "I want a vivid pink lipstick"
df = recommend_from_text(q, assets=assets, topk=20, debug=False)
chk("df not empty", df is not None and not df.empty)
chk("_de00_anchor present", "_de00_anchor" in df.columns)
s = df["_de00_anchor"]
chk("_de00_anchor is numeric", np.issubdtype(s.dtype, np.number))
chk("_de00_anchor all finite", bool(np.isfinite(s.values).all()))
min_de = float(np.min(s.values))
chk("min ΔE00 reasonable (<6)", min_de < 6.0)
print("min ΔE00:", min_de)

# -----------------------------
# 5) Multi-query smoke (non-empty)
# -----------------------------
print("\n================= RECO SMOKE =================")
qs = [
    "I want a nude lipstick",
    "I want a vivid pink lipstick",
    "I want a vivid pink lipstick, not too bright",
    "I want a red lipstick",
    "I want a bright red lipstick",
    "I want a mauve lipstick",
    "I want a terracotta lipstick",
    "I want a beige lipstick",
    "I want a purple lipstick",
    "I want a brown lipstick that is darker",
]
for q in qs:
    df = recommend_from_text(q, assets=assets, topk=5, debug=False)
    chk(f"non-empty: {q}", df is not None and not df.empty)

print("\nALL OK")

