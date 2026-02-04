# src/Shopping_assistant/reco/_naming_probs.py
"""
Naming probability utilities.

Attaches and queries per-family naming probabilities on inventory items
to support domain anchoring, pooling, and within-family constraints.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from Shopping_assistant.color.constraints import load_label_distributions


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_chips_naming_probs() -> pd.DataFrame:
    root = _project_root()
    p = root / "data" / "colors" / "chips_with_naming_probs.parquet"
    if not p.exists():
        return pd.DataFrame()

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        df = table.to_pandas()
    except Exception:
        df = pd.read_parquet(p)

    if "chip_hex" not in df.columns:
        return pd.DataFrame()

    pcols = [c for c in df.columns if c.startswith("p_")]
    keep = ["chip_hex"] + pcols
    out = df[keep].copy()
    out["chip_hex"] = out["chip_hex"].astype(str).str.lower()
    return out.drop_duplicates(subset=["chip_hex"])


@lru_cache(maxsize=1)
def _load_family_label_distributions() -> dict:
    root = _project_root()
    p = root / "data" / "colors" / "label_distributions.json"
    if not p.exists():
        return {}
    return load_label_distributions(p)


def _attach_naming_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge p_<label> onto df using chip_hex.
    No hardcoding: uses whatever p_* exists in the parquet.
    """
    if df is None or df.empty:
        return df
    if "chip_hex" not in df.columns:
        return df

    probs = _load_chips_naming_probs()
    if probs.empty:
        return df

    out = df.copy()
    out["chip_hex"] = out["chip_hex"].astype(str).str.lower()

    if any(c.startswith("p_") for c in out.columns):
        return out

    return out.merge(probs, on="chip_hex", how="left")


def _naming_prob_label_supported(label: str, *, inv_with_probs: pd.DataFrame | None = None) -> bool:
    """
    Generic support check for label->p_<label> existence.
    No aliasing/hardcoding.
    """
    if not isinstance(label, str) or not label:
        return False
    pcol = f"p_{label.strip().lower()}"

    if inv_with_probs is not None and isinstance(inv_with_probs, pd.DataFrame) and (pcol in inv_with_probs.columns):
        return True

    probs = _load_chips_naming_probs()
    if probs.empty:
        return False
    return pcol in probs.columns
