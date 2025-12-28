from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

TOP_K = 10
BOTTOM_K = 10

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INVENTORY_CSV = (
    PROJECT_ROOT
    / "data"
    / "enriched_data"
    / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
)

OUT_CSV = PROJECT_ROOT / "data" / "training" / "pairwise_preferences.csv"

QUERIES = [
    {"query_id": "q01", "text": "I want a deep red lipstick"},
    {"query_id": "q02", "text": "I like soft pink but not bright"},
    {"query_id": "q03", "text": "Looking for a nude shade"},
]


# ------------------------------------------------------------------
# MINIMAL LOCAL PRIMITIVES
# ------------------------------------------------------------------

def angular_diff_deg(h1: float, h2: float) -> float:
    """Signed angular difference in degrees ∈ [-180, 180]."""
    d = h1 - h2
    return (d + 180) % 360 - 180


def naive_score(query: Dict[str, str], row: pd.Series) -> float:
    """
    Score volontairement naïf.
    Sert UNIQUEMENT à ordonner pour créer les paires.
    """
    score = 0.0
    text = query["text"].lower()

    if "deep" in text:
        score += (100.0 - row["L_lab"]) * 0.01

    if "bright" in text:
        score += row["L_lab"] * 0.01

    if "soft" in text:
        score -= row["C_lab"] * 0.01

    if "nude" in text:
        score -= abs(row["H_lab_deg"] - 30.0) * 0.001  # beige-ish bias

    return score


# ------------------------------------------------------------------
# DATASET BUILDER
# ------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_CSV)

    rows: List[Dict] = []

    for q in QUERIES:
        inv = inventory.copy()
        inv["score"] = inv.apply(lambda r: naive_score(q, r), axis=1)

        ranked = inv.sort_values("score", ascending=False)

        top = ranked.head(TOP_K)
        bottom = ranked.tail(BOTTOM_K)

        for _, a in top.iterrows():
            for _, b in bottom.iterrows():
                dL = a["L_lab"] - b["L_lab"]
                dC = a["C_lab"] - b["C_lab"]
                dH = angular_diff_deg(a["H_lab_deg"], b["H_lab_deg"])

                # A préféré à B
                rows.append({
                    "query_id": q["query_id"],
                    "item_pos_id": a["shade_id"],
                    "item_neg_id": b["shade_id"],
                    "delta_L": dL,
                    "delta_C": dC,
                    "delta_H": dH,
                    "label": 1,
                })

                # B moins bon que A  → classe 0
                rows.append({
                    "query_id": q["query_id"],
                    "item_pos_id": b["shade_id"],
                    "item_neg_id": a["shade_id"],
                    "delta_L": -dL,
                    "delta_C": -dC,
                    "delta_H": -dH,
                    "label": 0,
                })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    assert INVENTORY_CSV.exists(), INVENTORY_CSV

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(OUT_CSV, index=False)

    print("[OK] Dataset generated")
    print("rows:", len(df))
    print(df.head())
