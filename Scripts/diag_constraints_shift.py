from __future__ import annotations

import pandas as pd

from Shopping_assistant.io.assets import load_default_assets
from Shopping_assistant.reco.recommend import recommend_from_text

FEATURES = ["light_hsl", "sat_hsl", "sat_eff", "depth", "colorfulness", "warmth"]

# Candidate join keys between recommend_from_text() results and assets.inventory
_ID_CANDIDATES = [
    ("shade_id", "shade_id"),
    ("item_id", "item_id"),
    ("id", "id"),
    ("sku", "sku"),
]


def _find_join_keys(res: pd.DataFrame, inv: pd.DataFrame) -> tuple[str, str] | None:
    res_cols = set(res.columns)
    inv_cols = set(inv.columns)
    for rk, ik in _ID_CANDIDATES:
        if rk in res_cols and ik in inv_cols:
            return rk, ik
    return None


def _coerce_key_series_to_str(s: pd.Series) -> pd.Series:
    # robust: handles ints/floats/None; keep a stable representation
    return s.astype("string").fillna(pd.NA)


def _get_col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """
    Handle pandas merge suffixes: name, name_inv, name_x, name_y.
    Prefer the unsuffixed name when present.
    """
    if name in df.columns:
        return df[name]
    for cand in (f"{name}_inv", f"{name}_y", f"{name}_x"):
        if cand in df.columns:
            return df[cand]
    return None


def _summ(res: pd.DataFrame, inv: pd.DataFrame, k: int = 20) -> dict:
    top = res.head(k).copy()

    # If features are already present in results, compute directly.
    if all(c in top.columns for c in FEATURES):
        topf = top[FEATURES].apply(pd.to_numeric, errors="coerce")
        return {c: float(topf[c].mean()) for c in FEATURES}

    jk = _find_join_keys(top, inv)
    if jk is None:
        raise RuntimeError(
            "Cannot join recommend_from_text() results to assets.inventory.\n"
            f"Result columns (sample): {list(top.columns)[:40]}\n"
            f"Inventory columns (sample): {list(inv.columns)[:40]}\n"
            "Fix: add an ID column to results (e.g., shade_id/item_id) or extend _ID_CANDIDATES."
        )
    rk, ik = jk

    # Prepare inventory subset
    inv_need = [ik] + [c for c in FEATURES if c in inv.columns]
    inv_sub = inv[inv_need].copy()

    # Coerce join keys to same dtype (string) to avoid object/int mismatches
    top[rk] = _coerce_key_series_to_str(top[rk])
    inv_sub[ik] = _coerce_key_series_to_str(inv_sub[ik])

    # Use stable suffix to avoid accidental collisions
    merged = top.merge(inv_sub, how="left", left_on=rk, right_on=ik, suffixes=("", "_inv"))

    out = {}
    for c in FEATURES:
        s = _get_col(merged, c)
        if s is None:
            out[c] = None
        else:
            out[c] = float(pd.to_numeric(s, errors="coerce").mean())
    return out


def _run_pair(base_q: str, constrained_q: str, k: int = 20) -> None:
    assets = load_default_assets()

    a = recommend_from_text(base_q, assets=assets, topk=k, lambda_constraints=2.0, debug=True)
    b = recommend_from_text(constrained_q, assets=assets, topk=k, lambda_constraints=2.0, debug=True)

    sa = _summ(a, assets.inventory, k=k)
    sb = _summ(b, assets.inventory, k=k)

    print("\n====================")
    print("BASE:", base_q)
    print("CONS:", constrained_q)
    print("MEAN_TOPK(base):", sa)
    print("MEAN_TOPK(cons):", sb)

    delta = {kk: (None if sa[kk] is None or sb[kk] is None else (sb[kk] - sa[kk])) for kk in sa.keys()}
    print("DELTA(cons-base):", delta)


if __name__ == "__main__":
    _run_pair("I want a red lipstick", "I want a red lipstick not too bright", k=30)
    _run_pair("I want a red lipstick", "I want a red lipstick more vibrant", k=30)
    _run_pair("I want a nude lipstick", "I want a nude lipstick more saturated", k=30)
    _run_pair("I want a berry lipstick", "I want a berry lipstick less saturated", k=30)
