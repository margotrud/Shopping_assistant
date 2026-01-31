# Scripts/colors/project_namer_on_chips.py

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHIPS_PATH = ROOT / "data" / "colors" / "chips_lab.parquet"
MODEL_PATH = ROOT / "data" / "colors" / "color_namer.pkl"
OUT_PATH = ROOT / "data" / "colors" / "chips_with_naming_probs.parquet"

def main():
    chips = pd.read_parquet(CHIPS_PATH).copy()
    pipe = joblib.load(MODEL_PATH)

    X = chips[["L", "a", "b"]].values.astype(float)
    probs = pipe.predict_proba(X)
    labels = list(pipe.classes_)

    # add probability columns
    for j, lbl in enumerate(labels):
        chips[f"p_{lbl}"] = probs[:, j].astype(float)

    # top-1 / top-2
    idx1 = probs.argmax(axis=1)
    p1 = probs[np.arange(len(probs)), idx1]
    chips["label_top1"] = [labels[i] for i in idx1]
    chips["p_top1"] = p1.astype(float)

    probs2 = probs.copy()
    probs2[np.arange(len(probs2)), idx1] = -1.0
    idx2 = probs2.argmax(axis=1)
    p2 = probs[np.arange(len(probs)), idx2]
    chips["label_top2"] = [labels[i] for i in idx2]
    chips["p_top2"] = p2.astype(float)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chips.to_parquet(OUT_PATH, index=False)

    print(f"[OK] wrote {len(chips)} rows → {OUT_PATH}")
    print("[INFO] top1 label distribution:")
    print(chips["label_top1"].value_counts())

    # sanity: average confidence
    print("[INFO] p_top1 mean:", float(chips["p_top1"].mean()))
    print("[INFO] p_top1 p10/p50/p90:", chips["p_top1"].quantile([0.1, 0.5, 0.9]).to_dict())

if __name__ == "__main__":
    main()
