# src/Shopping_assistant/ml/train_ranker.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_ml_dir() -> Path:
    return _project_root() / "data" / "ml"


def _default_train() -> Path:
    return _default_ml_dir() / "train.csv"


def _default_valid() -> Path:
    return _default_ml_dir() / "valid.csv"


def _default_test() -> Path:
    return _default_ml_dir() / "test.csv"


def _models_dir() -> Path:
    return _default_ml_dir() / "models"


def _reports_dir() -> Path:
    return _default_ml_dir() / "reports"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    feature_cols: Tuple[str, ...] = (
        "L_lab",
        "a_lab",
        "b_lab",
        "C_lab",
        "H_lab_deg",
        "depth",
        "warmth",
        "sat_eff",
    )
    label_col: str = "label"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _load_xy(path: Path, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    _require_cols(df, cfg.feature_cols, name=str(path))
    _require_cols(df, (cfg.label_col,), name=str(path))
    X = df[list(cfg.feature_cols)].to_numpy(dtype=np.float64)
    y = df[cfg.label_col].astype(int).to_numpy()
    return X, y, df


def _eval_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }
    return out


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_logreg(cfg: TrainConfig) -> Pipeline:
    # Strong baseline, interpretable coefficients
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=7,
            )),
        ]
    )


def build_hgb(cfg: TrainConfig) -> HistGradientBoostingClassifier:
    # Robust non-linear baseline (handles interactions)
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=25,
        l2_regularization=0.0,
        max_iter=400,
        random_state=7,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train a weakly-supervised colors ranker (binary) from data/ml splits.")
    p.add_argument("--ml-dir", type=str, default=str(_default_ml_dir()), help="Directory containing train/valid/test CSV.")
    p.add_argument("--model", type=str, default="hgb", choices=["hgb", "logreg"], help="Model type.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for reporting accuracy/CM.")
    args = p.parse_args()

    ml_dir = Path(args.ml_dir)
    train_path = ml_dir / "train.csv"
    valid_path = ml_dir / "valid.csv"
    test_path = ml_dir / "test.csv"

    cfg = TrainConfig()

    X_tr, y_tr, df_tr = _load_xy(train_path, cfg)
    X_va, y_va, df_va = _load_xy(valid_path, cfg)
    X_te, y_te, df_te = _load_xy(test_path, cfg)

    if args.model == "logreg":
        model = build_logreg(cfg)
        model_name = "logreg"
    else:
        model = build_hgb(cfg)
        model_name = "hgb"

    model.fit(X_tr, y_tr)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
    else:
        # HistGradientBoosting has predict_proba
        raise RuntimeError("Model does not support predict_proba; choose a different model.")

    report = {
        "model": model_name,
        "features": list(cfg.feature_cols),
        "train": _eval_binary(y_tr, p_tr, threshold=float(args.threshold)),
        "valid": _eval_binary(y_va, p_va, threshold=float(args.threshold)),
        "test": _eval_binary(y_te, p_te, threshold=float(args.threshold)),
    }

    # Feature importance / coefficients
    if model_name == "logreg":
        coef = model.named_steps["clf"].coef_[0].tolist()
        report["feature_weights"] = dict(zip(cfg.feature_cols, coef))
    else:
        # HGB: permutation importance is heavier; keep it simple and export predicted probs for analysis.
        report["note"] = "HGB trained. For importances, add permutation importance later."

    # Save artifacts
    models_dir = ml_dir / "models"
    reports_dir = ml_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}_color_ranker.joblib"
    report_path = reports_dir / f"{model_name}_report.json"

    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also export test predictions for inspection
    pred_df = df_te[["query_id", "product_id", "shade_id", "label"]].copy()
    pred_df["p_hat"] = p_te
    pred_df = pred_df.sort_values("p_hat", ascending=False)
    preds_path = reports_dir / f"{model_name}_test_predictions.csv"
    pred_df.to_csv(preds_path, index=False)

    print(str(model_path))
    print(str(report_path))
    print(str(preds_path))


if __name__ == "__main__":
    main()
