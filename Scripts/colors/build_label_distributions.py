# Scripts/colors/build_label_distributions.py

from __future__ import annotations

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "data" / "colors" / "chips_with_naming_probs.parquet"
OUT_PATH = ROOT / "data" / "colors" / "label_distributions.json"

P_THRESHOLD = 0.55
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

def main():
    df = pd.read_parquet(IN_PATH)

    labels = [
        c[2:]
        for c in df.columns
        if c.startswith("p_") and c not in ("p_top1", "p_top2")
    ]
    out = {}

    for lbl in labels:
        mask = df[f"p_{lbl}"] >= P_THRESHOLD
        sub = df.loc[mask]

        if len(sub) < 30:
            continue

        stats = {
            "n": int(len(sub)),
            "p_threshold": P_THRESHOLD,
            "L": sub["L"].quantile(QUANTILES).to_dict(),
            "C": sub["C"].quantile(QUANTILES).to_dict(),
            "h": sub["h"].quantile(QUANTILES).to_dict(),
        }

        out[lbl] = stats

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] wrote distributions → {OUT_PATH}")
    for k, v in out.items():
        print(f"{k:8s} n={v['n']}  L50={v['L'][0.5]:.1f}  C50={v['C'][0.5]:.1f}")

if __name__ == "__main__":
    main()
