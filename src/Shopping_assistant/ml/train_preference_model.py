from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------

# Racine du projet (pythonProject/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_CSV = PROJECT_ROOT / "data" / "training" / "pairwise_preferences.csv"
OUT_WEIGHTS_JSON = PROJECT_ROOT / "data" / "models" / "color_preference_weights.json"


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

assert DATASET_CSV.exists(), DATASET_CSV

df = pd.read_csv(DATASET_CSV)

X = df[["delta_L", "delta_C", "delta_H"]].values
y = df["label"].values

assert not np.isnan(X).any()
assert set(y) == {0, 1}


# ------------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------------

model = LogisticRegression(
    fit_intercept=False,
    penalty="l2",
    solver="lbfgs",
)

model.fit(X, y)

weights = model.coef_[0]


# ------------------------------------------------------------------
# OUTPUT (STDOUT)
# ------------------------------------------------------------------

print("\n[TRAINED PREFERENCE MODEL]")
print(f"w_L = {weights[0]:.4f}")
print(f"w_C = {weights[1]:.4f}")
print(f"w_H = {weights[2]:.4f}")

norm = np.linalg.norm(weights)
print(f"||w|| = {norm:.4f}")


# ------------------------------------------------------------------
# EXPORT WEIGHTS (AUTOMATIC, REPRODUCTIBLE)
# ------------------------------------------------------------------

OUT_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "model": "logistic_regression_pairwise",
    "trained_at": datetime.utcnow().isoformat(),
    "features": ["delta_L", "delta_C", "delta_H"],
    "weights": {
        "w_L": float(weights[0]),
        "w_C": float(weights[1]),
        "w_H": float(weights[2]),
    },
}

with OUT_WEIGHTS_JSON.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"\n[OK] Weights exported to {OUT_WEIGHTS_JSON}")
