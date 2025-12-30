# scripts/reco_ab.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# -----------------------------
# Force local src/ on sys.path (avoid stale site-packages)
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]  # pythonProject/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _print_import_origins() -> None:
    import Shopping_assistant
    from Shopping_assistant.nlp.resolve import axis_merge, axis_thresholds, scoring_adapter
    from Shopping_assistant.reco import recommend
    from Shopping_assistant.color import scoring as color_scoring

    print("\n[debug][imports] sys.path[0]:", sys.path[0])
    print("[debug][imports] Shopping_assistant:", Shopping_assistant.__file__)
    print("[debug][imports] axis_merge:", axis_merge.__file__)
    print("[debug][imports] axis_thresholds:", axis_thresholds.__file__)
    print("[debug][imports] scoring_adapter:", scoring_adapter.__file__)
    print("[debug][imports] recommend:", recommend.__file__)
    print("[debug][imports] color_scoring:", color_scoring.__file__)
    print()


from Shopping_assistant.color import scoring as color_scoring
from Shopping_assistant.io.assets import load_assets
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference
from Shopping_assistant.nlp.resolve.axis_projection import project_axes
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents
from Shopping_assistant.nlp.resolve.axis_thresholds import thresholds_from_decisions
from Shopping_assistant.nlp.resolve.scoring_adapter import build_constraints_blob_from_thresholds

from Shopping_assistant.reco.recommend import (
    recommend_from_text,
    _select_like_cluster_id_from_text,
    _neighbor_clusters_from_like_cluster,
)


# -----------------------------
# NLP -> thresholds/blob (debug only)
# -----------------------------

def blob_for_text(text: str, *, calibration: dict | None, debug: bool = False) -> str:
    nlp_res = interpret_nlp(text, debug=False)
    resolved = resolve_preference(nlp_res)
    intents = project_axes(resolved)
    decisions = merge_axis_intents(intents)
    ths = thresholds_from_decisions(decisions)
    blob = build_constraints_blob_from_thresholds(ths, calibration=calibration)

    if debug:
        print("\n[debug][NLP] text:", repr(text))
        print("[debug][NLP] decisions:", decisions)
        print("[debug][NLP] thresholds:", ths)
        print("[debug][NLP] blob:", blob)

    return blob


# -----------------------------
# Helpers
# -----------------------------

def _col(df: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def summarize(scored: pd.DataFrame, k: int = 20) -> dict:
    top = scored.head(k).copy()
    c_light = _col(top, "c1__light_hsl", "light_hsl")
    c_sat_eff = _col(top, "c2__sat_eff", "sat_eff")
    return {
        "k": k,
        "n_rows": int(len(scored)),
        "mean_light_like": float(top[c_light].mean()) if c_light else None,
        "mean_sat_eff": float(top[c_sat_eff].mean()) if c_sat_eff else None,
        "mean_constraint_penalty_norm": float(top["constraint_penalty_norm"].mean())
        if "constraint_penalty_norm" in top.columns
        else None,
        "max_penalty_norm": float(top["constraint_penalty_norm"].max())
        if "constraint_penalty_norm" in top.columns
        else None,
        "columns": list(scored.columns),
    }


def debug_constraint_effect(scored: pd.DataFrame, *, calibration: dict | None = None) -> None:
    pcols = [c for c in scored.columns if c.startswith("penalty_") and "__" in c]
    if not pcols:
        print("[debug] no penalty_*__* columns found")
        return

    pmat = scored[pcols].astype(float)
    psum = pmat.sum(axis=1)
    frac_nz = float((psum > 0).mean())
    maxp = float(psum.max())
    print(f"[debug] penalties: n_cols={len(pcols)} frac_nz={frac_nz:.3f} max_sum={maxp:.6f}")

    top_cols = pmat.mean().sort_values(ascending=False).head(5)
    print("[debug] top penalty cols (mean):", top_cols.to_dict())

    active = [c for c in pcols if pmat[c].mean() > 0]
    if active:
        print("[debug] active penalty cols:", active[:10], ("..." if len(active) > 10 else ""))


def compare_topk(scored_A: pd.DataFrame, scored_B: pd.DataFrame, k: int = 20) -> None:
    need = {"shade_id", "score"}
    if not need.issubset(set(scored_A.columns)) or not need.issubset(set(scored_B.columns)):
        print("[debug] compare_topk skipped: missing shade_id/score")
        return

    a = scored_A.head(k)[["shade_id", "score"]].set_index("shade_id")
    b = scored_B.head(k)[["shade_id", "score"]].set_index("shade_id")
    inter = a.join(b, how="inner", lsuffix="_A", rsuffix="_B")
    inter["delta"] = inter["score_B"] - inter["score_A"]

    print("[debug] topk intersection:", len(inter), "/", k)
    if len(inter):
        desc = inter["delta"].describe()
        print("[debug] delta(score) describe:", {kk: float(v) for kk, v in desc.to_dict().items()})
        print("[debug] worst deltas (B-A):")
        print(inter.sort_values("delta").head(5).to_string())


def _maybe_print_cluster_profile(scored: pd.DataFrame, *, label: str) -> None:
    if "cluster_id" not in scored.columns:
        return
    s = scored["cluster_id"].dropna()
    if not len(s):
        return
    mode = int(s.astype(int).mode().iloc[0])
    print(f"[debug] {label}: topk dominant cluster_id={mode}")


def _drop_non_lipstick(df: pd.DataFrame) -> pd.DataFrame:
    if "product_name" in df.columns:
        bad = df["product_name"].str.lower().fillna("")
        df = df[~bad.str.contains(r"\b(?:blush|joues|tint)\b", regex=True)].copy()
    return df


def _assert_assets_has_preference_weights(assets) -> None:
    w = getattr(assets, "preference_weights", None)
    if w is None:
        return
    if not isinstance(w, dict):
        raise TypeError("assets.preference_weights must be a dict if present.")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    _print_import_origins()

    assets = load_assets(
        enriched_csv=Path(color_scoring._default_enriched_for_scripts_only()),
        prototypes_csv=Path(color_scoring._default_prototypes_for_scripts_only()),
        assignments_csv=Path(color_scoring._default_assignments_for_scripts_only()),
        calibration_json=Path(color_scoring._default_calibration_for_scripts_only()),
        # Optional: if your AssetBundle supports it, also load preference weights so you can test lambda_preference
        # preference_weights_json=Path(color_scoring._default_preference_weights_for_scripts_only()),
    )
    _assert_assets_has_preference_weights(assets)

    print("[debug] calibration thresholds light_hsl:", assets.calibration.get("thresholds", {}).get("light_hsl"))
    print("[debug] calibration thresholds sat_eff:", assets.calibration.get("thresholds", {}).get("sat_eff"))

    text_base = "I want a red lipstick"
    text_constraints = "I want a red lipstick but not neon / not flashy"

    # 1) NLP-side debug (constraints only)
    blob_B = blob_for_text(text_constraints, calibration=assets.calibration, debug=True)
    print("\nBLOB (constraints):", blob_B)

    # 2) AB-critical: freeze like_cluster_id + candidate pool from BASE ONLY
    like_cluster_id = _select_like_cluster_id_from_text(text_base, assets=assets)
    candidate_cluster_ids = _neighbor_clusters_from_like_cluster(
        like_cluster_id,
        assets=assets,
        topn=4,
    )
    print("\n[debug] AB frozen like_cluster_id:", like_cluster_id)
    print("[debug] AB frozen candidate_cluster_ids:", candidate_cluster_ids)

    # 3) True A/B: SAME TEXT, SAME CLUSTER, SAME CANDIDATE POOL.
    print("\n--- AB RUN (A) : constraints OFF ---")
    scored_A = recommend_from_text(
        text_constraints,
        assets=assets,
        like_cluster_id=like_cluster_id,
        candidate_cluster_ids=candidate_cluster_ids,
        topk=50,
        constraints_blob_override="",
        lambda_preference=0.0,
    )
    scored_A = _drop_non_lipstick(scored_A)
    _maybe_print_cluster_profile(scored_A, label="A")

    print("\n--- AB RUN (B) : constraints ON ---")
    scored_B = recommend_from_text(
        text_constraints,
        assets=assets,
        like_cluster_id=like_cluster_id,
        candidate_cluster_ids=candidate_cluster_ids,
        topk=50,
        constraints_blob_override=blob_B,
        lambda_preference=0.0,
    )
    scored_B = _drop_non_lipstick(scored_B)
    _maybe_print_cluster_profile(scored_B, label="B")

    print("\nSUMMARY A:", summarize(scored_A))
    print("SUMMARY B:", summarize(scored_B))

    same_top10 = scored_A.head(10)["shade_id"].tolist() == scored_B.head(10)["shade_id"].tolist()
    print("\nTOP10 identical?", same_top10)

    print("\n--- CONSTRAINT EFFECT (B) ---")
    debug_constraint_effect(scored_B, calibration=assets.calibration)

    compare_topk(scored_A, scored_B, k=20)

    cols_pref = [
        "rank",
        "chip_hex",
        "r", "g", "b",
        "brand_name",
        "product_name",
        "shade_name",
        "score",
        "preference_score",
        "constraint_penalty_norm",
        "constraint_penalty",
        "c1__light_hsl",
        "c2__sat_eff",
        "light_hsl",
        "sat_hsl",
        "sat_eff",
        "depth",
        "url",
    ]
    cols_A = [c for c in cols_pref if c in scored_A.columns]
    cols_B = [c for c in cols_pref if c in scored_B.columns]

    print("\nTOP 10 A (constraints OFF):")
    print(scored_A.head(10)[cols_A].to_string(index=False))

    print("\nTOP 10 B (constraints ON):")
    print(scored_B.head(10)[cols_B].to_string(index=False))


if __name__ == "__main__":
    main()
