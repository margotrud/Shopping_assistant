# tests/qa_plain_color_neutrality.py
import os
import numpy as np
import pandas as pd

from Shopping_assistant.reco.recommend import recommend_from_text

COLORS = [
    "red",
    "pink",
    "purple",
    "beige",
    "terracotta",
    "brown",
    "mauve",
    "berry",
]

TOPK = 20
EXTREME_Q = 0.10  # 10% tails

# --- load inventory ---
path = os.environ.get("SA_ENRICHED_CSV_PATH")
assert path, "SA_ENRICHED_CSV_PATH not set"
inv = pd.read_csv(path)

for col in ["L_lab", "a_lab", "b_lab"]:
    assert col in inv.columns, inv.columns

# global thresholds
C_lab = np.sqrt(inv["a_lab"]**2 + inv["b_lab"]**2)
inv = inv.assign(C_lab=C_lab)

C_hi = inv["C_lab"].quantile(1 - EXTREME_Q)
L_lo = inv["L_lab"].quantile(EXTREME_Q)
L_hi = inv["L_lab"].quantile(1 - EXTREME_Q)

print("=== PLAIN COLOR NEUTRALITY TEST ===")

for color in COLORS:
    df = recommend_from_text(
        f"I want a {color} lipstick",
        topk=TOPK,
        debug=False,
    )

    assert len(df) >= TOPK, f"{color}: insufficient results"

    df = df.copy()
    df["C_lab"] = np.sqrt(df["a_lab"]**2 + df["b_lab"]**2)

    c_extreme = (df["C_lab"] > C_hi).mean()
    l_extreme = ((df["L_lab"] < L_lo) | (df["L_lab"] > L_hi)).mean()

    status = "PASS" if (c_extreme < 0.25 and l_extreme < 0.25) else "FAIL"

    print(
        f"{color:12s} | "
        f"C_extreme={c_extreme:.2f} | "
        f"L_extreme={l_extreme:.2f} | "
        f"{status}"
    )
