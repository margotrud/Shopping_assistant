# Scripts/colors/normalize_color_naming_dataset.py

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "data" / "colors" / "color_naming_dataset.parquet"
OUT_PATH = ROOT / "data" / "colors" / "color_naming_dataset_norm.parquet"

# Pivots : petit set stable (tu peux l’étendre ensuite)
PIVOTS = [
    # chromatic
    "red", "pink", "coral", "orange", "yellow",
    "green", "blue", "purple",

    # neutrals / earth
    "beige", "brown", "taupe", "gray",

    # light / dark
    "white", "black",
]


def main():
    df = pd.read_parquet(IN_PATH).dropna()

    # garder uniquement lignes avec label pivot disponible pour créer prototypes
    piv = {}
    for lbl in PIVOTS:
        sub = df[df["label"] == lbl]
        if len(sub) < 5:
            continue
        piv[lbl] = sub[["L", "a", "b"]].median().values.astype(float)

    if len(piv) < 8:
        raise RuntimeError(f"Not enough pivot labels found in dataset: got {len(piv)}")

    pivot_labels = list(piv.keys())
    pivot_mat = np.vstack([piv[k] for k in pivot_labels])  # (K,3)

    X = df[["L", "a", "b"]].values.astype(float)

    # nearest pivot by squared euclid in Lab
    # dist2 = ||x||^2 + ||p||^2 - 2 x.p
    x2 = (X**2).sum(axis=1, keepdims=True)
    p2 = (pivot_mat**2).sum(axis=1)[None, :]
    xp = X @ pivot_mat.T
    d2 = x2 + p2 - 2 * xp
    idx = d2.argmin(axis=1)

    df["label"] = [pivot_labels[i] for i in idx]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"[OK] wrote {len(df)} rows → {OUT_PATH}")
    vc = df["label"].value_counts()
    print("[INFO] n_labels:", vc.shape[0])
    print(vc.head(20))

if __name__ == "__main__":
    main()
