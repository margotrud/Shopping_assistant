# Scripts/colors/train_color_namer.py

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------------------
# Bootstrap project root
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
DATASET_PATH = ROOT / "data" / "colors" / "color_naming_dataset_norm.parquet"
OUT_MODEL = ROOT / "data" / "colors" / "color_namer.pkl"

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    df = pd.read_parquet(DATASET_PATH)

    # Features / target
    X = df[["L", "a", "b"]].values.astype(float)
    y = df["label"].astype(str).values

    # Pipeline: scale + multinomial logreg
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    n_jobs=-1,
                    class_weight="balanced",
                )

            ),
        ]
    )
    print("[CHECK] n_rows:", len(df), "n_labels:", df["label"].nunique())
    print(df["label"].value_counts().head(20))

    pipe.fit(X, y)

    # Save model
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_MODEL)

    print(f"[OK] trained color namer on {len(df)} samples")
    print(f"[OK] saved model → {OUT_MODEL}")

    # Sanity check: predict on training medians
    med = np.median(X, axis=0).reshape(1, -1)
    probs = pipe.predict_proba(med)[0]
    labels = pipe.classes_

    top = sorted(zip(labels, probs), key=lambda x: -x[1])[:10]
    print("[SANITY] top predictions at median Lab:")
    for lbl, p in top:
        print(f"  {lbl:12s} {p:.3f}")

if __name__ == "__main__":
    main()
