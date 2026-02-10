# src/Shopping_assistant/reco/explain.py
"""Explainable recommendation trace.

Does: Runs the same pipeline as recommend_from_text() but returns intermediate artifacts
(anchor resolution, candidate pool diagnostics, constraints, scoring table) for UI explainability.
Public API: explain_recommendation().
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from Shopping_assistant.color.hard_color_pool import hard_color_pool, params_from_env
from Shopping_assistant.color.scoring import QuerySpec, score_shades
from Shopping_assistant.io.assets import AssetBundle, load_default_assets
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp

from ._anchor import (
    _anchor_from_nlp,
    _anchor_source_mode,
    _first_color_token_from_nlp,
    _has_color_like_mention,
    _is_plain_color_query,
    _seed_hex_from_nlp,
)
from ._colorconv import _hex_to_lab
from ._constants import (
    _DEFAULT_POOL_TOPN,
    _DOMAIN_ANCHOR_P_MIN,
    _FAMILY_FLOOR,
    _FAMILY_P_HI,
    _FAMILY_P_MIN,
    _HUE_FAILSAFE_BAND_DEG,
    _HUE_FALLBACK_MIN_POOL_N,
    _MAX_POOL_TOPN,
    _NAMING_POOL_MIN_N,
    _NAMING_POOL_P_MIN,
    _NAMING_POOL_TOPN,
    _POOL_FALLBACK_DE00_CAP,
    _POOL_FALLBACK_MIN_N,
)
from ._domain_anchor import _domain_anchor_from_naming_probs
from ._family_constraints import _family_specs_from_nlp
from ._filters import _filter_invalid_products
from ._naming_probs import (
    _attach_naming_probs,
    _load_family_label_distributions,
    _naming_prob_label_supported,
)
from ._pooling import (
    _adaptive_de00_pool,
    _anchor_inventory_coverage,
    _domain_pool_and_anchor,
    _ensure_de00_anchor_col,
    _hue_band_subset,
    _pool_by_naming_prob,
    _restore_feature_cols,
)

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = True) -> bool:
    v = os.environ.get(key, "1" if default else "0")
    return str(v).strip().lower() in {"1", "true", "yes"}


def _adaptive_hue_band_deg(anchor_C: float) -> float:
    """Does: compute hue band width (deg) from anchor chroma (low-C tight, high-C wider).
    Inputs: anchor_C (chroma in Lab).
    Returns: band width in degrees, clipped to [18, 36].
    """
    return float(np.clip(18.0 + 0.20 * float(anchor_C), 18.0, 36.0))


def _wants_hi_chroma(nlp_res) -> bool:
    """Does: decide whether to use a high-chroma domain anchor based on NLP constraints.
    Inputs: nlp_res with constraints (+ optional axis_debug for near-ties).
    Returns: True iff intent is chroma-focused AND not an explicit "darker/less bright" request.
    """
    cons = tuple(getattr(nlp_res, "constraints", ()) or ())
    if not cons:
        return False

    HI_AXES = {"saturation", "vibrancy", "chroma"}
    HI_DIRS = {"raise", "high", "up", "more"}

    LOWER_BLOCK_AXES = {"depth", "brightness"}
    LOWER_DIRS = {"lower", "down", "less"}

    for c in cons:
        axis = getattr(c, "axis", None)
        direction = getattr(c, "direction", None)
        axis = axis.value if hasattr(axis, "value") else axis
        direction = direction.value if hasattr(direction, "value") else direction
        if axis in LOWER_BLOCK_AXES and direction in LOWER_DIRS:
            return False

    MARGIN_RANKED = 0.03
    MARGIN_VOTE_AVG = 0.04

    def _near_tie_ranked(axis_debug) -> bool:
        ranked = axis_debug.get("ranked")
        if not isinstance(ranked, (list, tuple)) or len(ranked) < 2:
            return False
        try:
            top_score = float(ranked[0][1])
            best_hi = None
            for ax, sc in ranked:
                ax = str(ax)
                if ax in HI_AXES:
                    s = float(sc)
                    best_hi = s if best_hi is None else max(best_hi, s)
            return (best_hi is not None) and ((top_score - best_hi) <= MARGIN_RANKED)
        except Exception:
            return False

    def _near_tie_vote(axis_debug) -> bool:
        ws = axis_debug.get("vote_axis_weighted_sum")
        wsum = axis_debug.get("vote_axis_weight_sum")
        if not isinstance(ws, dict) or not isinstance(wsum, dict):
            return False
        try:

            def avg(ax: str) -> float:
                num = float(ws.get(ax, 0.0))
                den = float(wsum.get(ax, 0.0))
                return (num / den) if den > 0 else float("nan")

            top = avg("brightness")
            if not np.isfinite(top):
                return False

            best_hi = None
            for ax in HI_AXES:
                v = avg(ax)
                if np.isfinite(v):
                    best_hi = v if best_hi is None else max(best_hi, v)

            return (best_hi is not None) and ((top - best_hi) <= MARGIN_VOTE_AVG)
        except Exception:
            return False

    for c in cons:
        axis = getattr(c, "axis", None)
        direction = getattr(c, "direction", None)
        meta = getattr(c, "meta", None) or {}

        axis = axis.value if hasattr(axis, "value") else axis
        direction = direction.value if hasattr(direction, "value") else direction

        if axis in HI_AXES and direction in HI_DIRS:
            return True

        if axis == "brightness" and direction in HI_DIRS:
            axis_debug = meta.get("axis_debug") or {}
            if _near_tie_ranked(axis_debug):
                return True
            if _near_tie_vote(axis_debug):
                return True

    return False


def _summarize_constraints(nlp_constraints: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in nlp_constraints or ():
        axis = getattr(c, "axis", None)
        direction = getattr(c, "direction", None)
        strength = getattr(c, "strength", None)
        evidence = getattr(c, "evidence", None)
        meta = getattr(c, "meta", None)

        axis = axis.value if hasattr(axis, "value") else axis
        direction = direction.value if hasattr(direction, "value") else direction
        strength = strength.value if hasattr(strength, "value") else strength

        out.append(
            {
                "axis": axis,
                "direction": direction,
                "strength": strength,
                "evidence": evidence,
                "meta": meta,
            }
        )
    return out


def explain_recommendation(
    text: str,
    *,
    assets: AssetBundle | None = None,
    top_k: int = 6,
    candidate_pool_topn: int = _DEFAULT_POOL_TOPN,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Does: run full pipeline and return intermediate artifacts for explain UI.
    Inputs: text + optional assets/pool/scoring params.
    Returns: dict with nlp/anchor/pool/scoring_table/topk and pool/scoring diagnostics.
    """
    if assets is None:
        assets = load_default_assets()

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)

    pool_cap = int(min(max(int(candidate_pool_topn), 1), len(inv))) if len(inv) else 0
    pool_cap = int(min(pool_cap, int(_MAX_POOL_TOPN)))

    include_xkcd = _env_bool("SA_INCLUDE_XKCD", True)
    polarity_backend = os.environ.get("SA_POLARITY_BACKEND", "lexical")
    nlp_res = interpret_nlp(
        text,
        include_xkcd=include_xkcd,
        polarity_backend=polarity_backend,
        debug=debug,
    )
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)
    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    anchor_eff = anchor_lab

    inv = _attach_naming_probs(inv)
    dists = _load_family_label_distributions()

    family_label = None
    family_in_dists = False
    if has_color:
        tok = _first_color_token_from_nlp(nlp_res)
        if tok:
            tok = tok.strip().lower()
            keys = set(str(k).lower() for k in (dists or {}).keys())
            if tok in keys:
                family_label, family_in_dists = tok, True
            elif _naming_prob_label_supported(tok, inv_with_probs=inv):
                family_label, family_in_dists = tok, False

    nlp_constraints = tuple(getattr(nlp_res, "constraints", ()) or ())
    want_hiC = bool(_wants_hi_chroma(nlp_res)) and (len(nlp_constraints) == 1)
    chroma_q = 0.90 if want_hiC else None

    used_domain_anchor = False
    if has_color and anchor_eff is not None and family_label is not None and (_is_plain_color_query(nlp_res) or want_hiC):
        dom = _domain_anchor_from_naming_probs(
            inv,
            label=family_label,
            p_min=float(_DOMAIN_ANCHOR_P_MIN),
            anchor_lab_lexicon=tuple(map(float, anchor_lab)) if anchor_lab is not None else None,
            anchor_hex=anchor_hex,
            chroma_q=chroma_q,
        )
        if dom is not None:
            anchor_eff = dom
            used_domain_anchor = True

    # -------------------------
    # PHASE A — candidate pool
    # -------------------------
    pool_debug: Dict[str, Any] = {}
    pool_methods: List[str] = []
    pool_gate: Dict[str, Any] = {}

    inv_eff = inv
    fb_dbg: Dict[str, Any] = {"fallback_used": False}
    thr_used = None
    min_de_all = None
    n_support_all = None
    min_de_eff = None
    n_support_eff = None

    if has_color and anchor_eff is not None:
        p = params_from_env()
        aC_raw = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
        thr_used = float(p.de00_max_neutral if aC_raw < float(p.neutral_anchor_c_max) else p.de00_max)

        min_de_all, n_support_all = _anchor_inventory_coverage(inv, tuple(map(float, anchor_eff)), thr_used)

        if n_support_all < int(_HUE_FALLBACK_MIN_POOL_N):
            inv_eff, anchor_eff2, fb_dbg = _domain_pool_and_anchor(
                inv,
                anchor_lab=tuple(map(float, anchor_eff)),
                anchor_hex=anchor_hex,
            )
            pool_methods.append("hue_fallback_domain_pool")
        else:
            inv_eff, anchor_eff2, fb_dbg = inv, tuple(map(float, anchor_eff)), {
                "fallback_used": False,
                "anchor_C": float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2]))),
                "H0": None,
                "cand_n": None,
                "cthr": None,
                "anchor_lab_eff": tuple(map(float, anchor_eff)),
                "hue_source": None,
            }

        anchor_eff = anchor_eff2

        min_de_eff, n_support_eff = _anchor_inventory_coverage(inv_eff, tuple(map(float, anchor_eff)), thr_used)

        df_pool = hard_color_pool(inv_eff, anchor_lab=anchor_eff, params=p)
        pool_methods.append("hard_color_pool")

        if pool_cap > 0 and len(df_pool) > pool_cap:
            df_pool = df_pool.head(pool_cap).copy()
            pool_gate["pool_cap"] = int(pool_cap)

        # Constraint-driven widening (hue-bounded adaptive ΔE)
        if nlp_constraints:
            min_pool_cons = max(int(_POOL_FALLBACK_MIN_N), 120)
            if len(df_pool) < int(min_pool_cons):
                aC = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
                band_deg = float(_adaptive_hue_band_deg(aC))

                hue_cand, hdbg_cons = _hue_band_subset(
                    inv_eff,
                    anchor_hex=anchor_hex,
                    anchor_lab=tuple(map(float, anchor_eff)),
                    band_deg=float(band_deg),
                )
                base_df = hue_cand if len(hue_cand) > 0 else inv_eff

                df_pool2, adbg_cons = _adaptive_de00_pool(
                    base_df,
                    anchor_lab=tuple(map(float, anchor_eff)),
                    thr_base=float(thr_used),
                    min_n=int(min_pool_cons),
                    cap=float(_POOL_FALLBACK_DE00_CAP),
                )
                if not df_pool2.empty:
                    pool_n_before = int(len(df_pool))
                    df_pool = df_pool2
                    pool_methods.append("constraints_adaptive_de00")
                    if pool_cap > 0 and len(df_pool) > pool_cap:
                        df_pool = df_pool.head(pool_cap).copy()

                    pool_debug["constraints_pool_widening"] = {
                        "hue_band_used": bool(len(hue_cand) > 0),
                        "hue_band_deg": float(band_deg),
                        "hue_dbg": dict(hdbg_cons) if isinstance(hdbg_cons, dict) else hdbg_cons,
                        "adaptive_dbg": dict(adbg_cons) if isinstance(adbg_cons, dict) else adbg_cons,
                        "pool_n_before": int(pool_n_before),
                        "pool_n_after": int(len(df_pool)),
                        "min_pool_cons": int(min_pool_cons),
                        "thr_base": float(thr_used),
                    }

        # General failsafe: adaptive ΔE (optionally hue-band first)
        if df_pool.empty or (len(df_pool) < int(_POOL_FALLBACK_MIN_N)):
            df_pool2, adbg = _adaptive_de00_pool(
                inv_eff,
                anchor_lab=tuple(map(float, anchor_eff)),
                thr_base=float(thr_used),
                min_n=int(_POOL_FALLBACK_MIN_N),
                cap=float(_POOL_FALLBACK_DE00_CAP),
            )
            pool_methods.append("adaptive_de00_fallback")

            if df_pool2.empty or len(df_pool2) < int(_POOL_FALLBACK_MIN_N):
                hue_cand, hdbg = _hue_band_subset(
                    inv_eff,
                    anchor_hex=anchor_hex,
                    anchor_lab=tuple(map(float, anchor_eff)),
                    band_deg=float(_HUE_FAILSAFE_BAND_DEG),
                )
                if len(hue_cand) > 0:
                    df_pool3, adbg2 = _adaptive_de00_pool(
                        hue_cand,
                        anchor_lab=tuple(map(float, anchor_eff)),
                        thr_base=float(thr_used),
                        min_n=int(_POOL_FALLBACK_MIN_N),
                        cap=float(_POOL_FALLBACK_DE00_CAP),
                    )
                    if len(df_pool3) > 0:
                        df_pool2 = df_pool3
                        pool_methods.append("hue_band_then_adaptive_de00")
                        pool_debug["hue_band_failsafe"] = {
                            "hue_dbg": dict(hdbg) if isinstance(hdbg, dict) else hdbg,
                            "adaptive_dbg": dict(adbg2) if isinstance(adbg2, dict) else adbg2,
                        }

            df_pool = df_pool2
            if pool_cap > 0 and len(df_pool) > pool_cap:
                df_pool = df_pool.head(pool_cap).copy()

            pool_debug["adaptive_de00_dbg"] = dict(adbg) if isinstance(adbg, dict) else adbg

            if df_pool.empty:
                return {
                    "nlp": {
                        "text": text,
                        "has_color": bool(has_color),
                        "constraints": _summarize_constraints(nlp_constraints),
                        "raw": getattr(nlp_res, "__dict__", None),
                    },
                    "anchor": {
                        "anchor_source": _anchor_source_mode(),
                        "anchor_hex": anchor_hex,
                        "anchor_lab_lexicon": anchor_lab,
                        "anchor_lab_effective": anchor_eff,
                        "used_domain_anchor": bool(used_domain_anchor),
                        "hi_chroma": bool(want_hiC),
                        "chroma_q": chroma_q,
                        "family_label": family_label,
                        "family_label_in_dists": bool(family_in_dists),
                    },
                    "candidate_pool": {
                        "method": "+".join(pool_methods) if pool_methods else "none",
                        "n_before": int(len(inv)),
                        "n_after": 0,
                        "pool_cap": int(pool_cap),
                        "thr_used": thr_used,
                        "coverage_all": {"min_de00": min_de_all, "n_support": n_support_all},
                        "coverage_eff": {"min_de00": min_de_eff, "n_support": n_support_eff},
                        "fallback": fb_dbg,
                        "debug": pool_debug,
                    },
                    "scoring_table": pd.DataFrame(),
                    "topk": pd.DataFrame(),
                }

        # Naming-prob pool gate (rare, but explainable)
        req_label = _first_color_token_from_nlp(nlp_res)
        use_naming_pool = False
        if req_label and _naming_prob_label_supported(req_label, inv_with_probs=inv_eff):
            if (n_support_eff is not None and n_support_eff < int(_NAMING_POOL_MIN_N)) or (
                min_de_eff is not None and thr_used is not None and float(min_de_eff) > float(thr_used)
            ):
                use_naming_pool = True
            if len(df_pool) < int(_NAMING_POOL_MIN_N):
                use_naming_pool = True

        if use_naming_pool:
            name_pool, ndbg = _pool_by_naming_prob(
                inv_eff,
                label=req_label,
                pool_topn=int(_NAMING_POOL_TOPN),
                p_min=float(_NAMING_POOL_P_MIN),
                min_n=int(_NAMING_POOL_MIN_N),
            )
            if not name_pool.empty:
                df_pool = name_pool
                pool_methods.append("naming_prob_pool")
                if pool_cap > 0 and len(df_pool) > pool_cap:
                    df_pool = df_pool.head(pool_cap).copy()
                pool_debug["naming_prob_dbg"] = dict(ndbg) if isinstance(ndbg, dict) else ndbg
                pool_debug["naming_prob_gate"] = {
                    "req_label": req_label,
                    "n_support_eff": int(n_support_eff) if n_support_eff is not None else None,
                    "min_de_eff": float(min_de_eff) if min_de_eff is not None else None,
                    "thr_used": float(thr_used) if thr_used is not None else None,
                }

    else:
        df_pool = inv.head(pool_cap).copy()
        pool_methods.append("no_color_head_pool")
        if pool_cap > 0:
            pool_gate["pool_cap"] = int(pool_cap)

    if has_color and anchor_eff is not None:
        df_pool = _ensure_de00_anchor_col(df_pool, tuple(map(float, anchor_eff)))
        if "deltaE_norm" in df_pool.columns:
            df_pool["deltaE_norm"] = df_pool["deltaE_norm"].clip(lower=0.0, upper=1.0)

    # -------------------------
    # PHASE B — scoring
    # -------------------------
    query = QuerySpec(anchor_lab=anchor_eff, constraints=nlp_constraints)

    df_pool = _attach_naming_probs(df_pool)

    family_label_for_constraints = family_label if (family_label is not None and family_in_dists and bool(dists)) else None
    family_specs = _family_specs_from_nlp(nlp_constraints) if (family_label_for_constraints is not None) else []

    scored = score_shades(
        df_pool,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=getattr(assets, "preference_weights", None),
        constraint_label=family_label_for_constraints,
        constraint_specs=family_specs,
        label_distributions=dists if (family_label_for_constraints is not None and dists) else None,
        constraint_p_min=float(_FAMILY_P_MIN),
        constraint_p_hi=float(_FAMILY_P_HI),
        constraint_floor=float(_FAMILY_FLOOR),
    )

    scored = _restore_feature_cols(df_pool, scored)
    if has_color and anchor_eff is not None:
        scored = _ensure_de00_anchor_col(scored, tuple(map(float, anchor_eff)))

    # Re-apply the same deltaE_norm blending logic as recommend_from_text()
    if has_color and anchor_eff is not None and "deltaE_norm" in scored.columns:
        has_cons = bool(nlp_constraints)

        aC = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
        if has_cons:
            w = 2.25 if aC < 25.0 else 1.50
        else:
            w = 0.75

        base = scored["score_total"] if "score_total" in scored.columns else (
            scored["score"] if "score" in scored.columns else None
        )
        if base is not None:
            scored = scored.copy()
            scored["score_total"] = base - float(w) * scored["deltaE_norm"]

    score_col = "score_total" if "score_total" in scored.columns else ("score" if "score" in scored.columns else None)

    if score_col is None:
        topk_df = scored.head(int(top_k)).copy()
    else:
        topk_df = scored.sort_values(score_col, ascending=False).head(int(top_k)).copy()

    # Candidate pool summary (post-hoc)
    method = "+".join(pool_methods) if pool_methods else "none"
    cand_summary = {
        "method": method,
        "n_before": int(len(inv)),
        "n_after": int(len(df_pool)),
        "pool_cap": int(pool_cap),
        "thr_used": float(thr_used) if thr_used is not None else None,
        "coverage_all": {"min_de00": float(min_de_all) if min_de_all is not None else None, "n_support": int(n_support_all) if n_support_all is not None else None},
        "coverage_eff": {"min_de00": float(min_de_eff) if min_de_eff is not None else None, "n_support": int(n_support_eff) if n_support_eff is not None else None},
        "fallback": fb_dbg,
        "gate": pool_gate,
        "debug": pool_debug if debug else {},
    }

    # NLP summary
    nlp_payload = {
        "text": text,
        "has_color": bool(has_color),
        "constraints": _summarize_constraints(nlp_constraints),
    }
    if debug:
        nlp_payload["raw"] = getattr(nlp_res, "__dict__", None)

    anchor_payload = {
        "anchor_source": _anchor_source_mode(),
        "anchor_hex": anchor_hex,
        "anchor_lab_lexicon": anchor_lab,
        "anchor_lab_effective": anchor_eff,
        "used_domain_anchor": bool(used_domain_anchor),
        "hi_chroma": bool(want_hiC),
        "chroma_q": chroma_q,
        "family_label": family_label,
        "family_label_in_dists": bool(family_in_dists),
    }

    # Scoring table for UI (keep useful cols first when present)
    scored_for_ui = scored.copy()
    preferred_cols = [
        "product_id",
        "shade_id",
        "brand",
        "product_name",
        "shade_name",
        "hex",
        "cluster_id",
        "deltaE_norm",
        "score_total",
        "score",
    ]
    keep = [c for c in preferred_cols if c in scored_for_ui.columns]
    rest = [c for c in scored_for_ui.columns if c not in keep]
    scored_for_ui = scored_for_ui[keep + rest] if keep else scored_for_ui

    return {
        "nlp": nlp_payload,
        "anchor": anchor_payload,
        "candidate_pool": cand_summary,
        "queryspec": {
            "anchor_lab": tuple(map(float, anchor_eff)) if anchor_eff is not None else None,
            "n_constraints": int(len(nlp_constraints)),
            "constraint_label": family_label_for_constraints,
            "family_specs_n": int(len(family_specs)),
        },
        "scoring_table": scored_for_ui,
        "topk": topk_df,
    }


__all__ = ["explain_recommendation"]
