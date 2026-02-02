# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

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
from ._family_constraints import _family_specs_from_nlp
from ._filters import _filter_invalid_products
from ._naming_probs import _attach_naming_probs, _load_family_label_distributions, _naming_prob_label_supported
from ._pooling import (
    _adaptive_de00_pool,
    _anchor_inventory_coverage,
    _domain_pool_and_anchor,
    _ensure_de00_anchor_col,
    _hue_band_subset,
    _pool_by_naming_prob,
    _restore_feature_cols,
)
from ._domain_anchor import _domain_anchor_from_naming_probs


# =============================================================================
# ANCHOR-ONLY API (no scoring, no inventory)
# =============================================================================


def resolve_anchor_from_text(text: str, *, debug: bool = False) -> dict:
    """
    Anchor-only resolution for "plain colors" debugging.
    Returns a dict with anchor_hex, anchor_lab, and minimal NLP context.

    NOTE:
      - Uses SA_ANCHOR_SOURCE ('lexicon' default) to prevent polluted anchors.
      - Does NOT score / does NOT touch inventory.
    """
    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)

    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    out = {
        "text": text,
        "has_color": bool(has_color),
        "anchor_source": _anchor_source_mode(),
        "anchor_hex": anchor_hex,
        "anchor_lab": anchor_lab,
    }

    if debug:
        print("\n[ANCHOR-ONLY]")
        print(f"  text={text!r}")
        print(f"  has_color={has_color}")
        print(f"  anchor_source={out['anchor_source']!r}")
        print(f"  anchor_hex={anchor_hex}  anchor_lab={anchor_lab}")

    return out


def resolve_effective_anchor_from_text(
    text: str,
    *,
    assets: AssetBundle | None = None,
    debug: bool = False,
) -> dict:
    """
    Resolve BOTH:
      - lexicon anchor (same as resolve_anchor_from_text)
      - effective anchor used by recommend_from_text for plain-color queries
        (domain anchor via naming probs p_<label> when available)

    Returns:
      {
        anchor_hex,
        anchor_lab_lexicon,
        anchor_lab_effective,
        has_color,
        family_label,
        family_label_in_dists (bool),
        used_domain_anchor (bool),
      }
    """
    if assets is None:
        assets = load_default_assets()

    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)
    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)
    inv = _attach_naming_probs(inv)

    dists = _load_family_label_distributions()

    family_label = None
    in_dists = False
    if has_color:
        tok = _first_color_token_from_nlp(nlp_res)
        if tok:
            tok = tok.strip().lower()
            keys = set(str(k).lower() for k in (dists or {}).keys())
            if tok in keys:
                family_label, in_dists = tok, True
            elif _naming_prob_label_supported(tok, inv_with_probs=inv):
                family_label, in_dists = tok, False

    anchor_eff = anchor_lab
    used_domain = False

    if has_color and anchor_eff is not None and family_label is not None and _is_plain_color_query(nlp_res):
        dom = _domain_anchor_from_naming_probs(
            inv,
            label=family_label,
            p_min=float(_DOMAIN_ANCHOR_P_MIN),
            anchor_lab_lexicon=tuple(map(float, anchor_lab)) if anchor_lab is not None else None,
            anchor_hex=anchor_hex,
        )
        if dom is not None:
            anchor_eff = dom
            used_domain = True

    out = {
        "text": text,
        "has_color": bool(has_color),
        "anchor_source": _anchor_source_mode(),
        "anchor_hex": anchor_hex,
        "anchor_lab_lexicon": anchor_lab,
        "anchor_lab_effective": anchor_eff,
        "family_label": family_label,
        "family_label_in_dists": bool(in_dists),
        "used_domain_anchor": bool(used_domain),
    }

    if debug:
        print("\n[ANCHOR-EFFECTIVE]")
        print(f"  text={text!r}")
        print(f"  has_color={has_color}")
        print(f"  family_label={family_label!r}  in_dists={in_dists}")
        print(f"  anchor_hex={anchor_hex}")
        print(f"  anchor_lab_lexicon={anchor_lab}")
        print(f"  anchor_lab_effective={anchor_eff}")
        print(f"  used_domain_anchor={used_domain}")

    return out


# =============================================================================
# MAIN
# =============================================================================


def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle | None = None,
    topk: int = 20,
    candidate_pool_topn: int = _DEFAULT_POOL_TOPN,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Recommendation mode (scoring).
    If you want only anchors (no pool, no scoring), call resolve_anchor_from_text().
    """
    if assets is None:
        assets = load_default_assets()

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)

    pool_cap = int(min(max(int(candidate_pool_topn), 1), len(inv))) if len(inv) else 0
    pool_cap = int(min(pool_cap, int(_MAX_POOL_TOPN)))

    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)
    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    anchor_eff = anchor_lab

    # Attach naming probs onto inventory once (needed for domain anchor + naming pool + family constraints)
    inv = _attach_naming_probs(inv)

    # Precompute label_distributions + family label once
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

    # Domain anchor for plain color queries
    if has_color and anchor_eff is not None and family_label is not None and _is_plain_color_query(nlp_res):
        dom = _domain_anchor_from_naming_probs(
            inv,
            label=family_label,
            p_min=float(_DOMAIN_ANCHOR_P_MIN),
            anchor_lab_lexicon=tuple(map(float, anchor_lab)) if anchor_lab is not None else None,
            anchor_hex=anchor_hex,
        )
        if dom is not None:
            anchor_eff = dom
            if debug:
                print("\n[DOMAIN ANCHOR]")
                print(f"  family_label={family_label!r} p_min={_DOMAIN_ANCHOR_P_MIN} in_dists={family_in_dists}")
                print(f"  anchor_lab_lexicon={anchor_lab}")
                print(f"  anchor_lab_domain ={anchor_eff}")

    # ==========================
    # PHASE A — strict pool (+ hue-family fallback) + failsafes
    # ==========================
    if has_color and anchor_eff is not None:
        p = params_from_env()

        aC_raw = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
        thr_used = float(p.de00_max_neutral if aC_raw < float(p.neutral_anchor_c_max) else p.de00_max)

        # coverage gate BEFORE any restriction
        min_de_all, n_support_all = _anchor_inventory_coverage(inv, tuple(map(float, anchor_eff)), thr_used)

        # only restrict by hue when dataset support is weak at the strict threshold
        if n_support_all < int(_HUE_FALLBACK_MIN_POOL_N):
            inv_eff, anchor_eff2, fb_dbg = _domain_pool_and_anchor(
                inv,
                anchor_lab=tuple(map(float, anchor_eff)),
                anchor_hex=anchor_hex,
            )
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

        # coverage gate AFTER restriction
        min_de_eff, n_support_eff = _anchor_inventory_coverage(inv_eff, tuple(map(float, anchor_eff)), thr_used)

        df_pool = hard_color_pool(inv_eff, anchor_lab=anchor_eff, params=p)

        if pool_cap > 0 and len(df_pool) > pool_cap:
            df_pool = df_pool.head(pool_cap).copy()

        if debug:
            aC = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}")
            print(
                f"  anchor_source={_anchor_source_mode()!r}  "
                f"anchor_lab_raw={anchor_lab}  anchor_hex={anchor_hex}  "
                f"anchor_lab_eff={anchor_eff}  anchor_C_eff={aC:.2f}"
            )
            print(
                f"  hue_fallback_used={bool(fb_dbg.get('fallback_used'))}"
                f"  hue_source={fb_dbg.get('hue_source')}"
            )
            if fb_dbg.get("fallback_used"):
                h0 = fb_dbg.get("H0")
                print(f"  hue_H0={h0}  cand_n={fb_dbg.get('cand_n')}  cthr={fb_dbg.get('cthr')}")
            print(
                f"  de00_thr={thr_used:.2f}  pool_cap={pool_cap}  pool_n={len(df_pool)}  "
                f"inv_n={len(inv)}  inv_eff_n={len(inv_eff)}"
            )
            print(f"  coverage_all: min_de00={min_de_all:.2f}  n_support<=thr={n_support_all}")
            print(f"  coverage_eff: min_de00={min_de_eff:.2f}  n_support<=thr={n_support_eff}")

        # FAILSAFE PATHS
        if df_pool.empty or (len(df_pool) < int(_POOL_FALLBACK_MIN_N)):
            # Fallback 1: adaptive ΔE pool on inv_eff
            df_pool2, adbg = _adaptive_de00_pool(
                inv_eff,
                anchor_lab=tuple(map(float, anchor_eff)),
                thr_base=float(thr_used),
                min_n=int(_POOL_FALLBACK_MIN_N),
                cap=float(_POOL_FALLBACK_DE00_CAP),
            )

            # Fallback 2: if still empty/tiny -> hue-band subset then adaptive ΔE
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
                        if debug:
                            print("  FAILSAFE: hue-band subset used", hdbg, adbg2)

            df_pool = df_pool2
            if pool_cap > 0 and len(df_pool) > pool_cap:
                df_pool = df_pool.head(pool_cap).copy()

            if debug:
                print("  FAILSAFE: adaptive ΔE used", adbg)

            if df_pool.empty:
                return df_pool

        # UPDATED FAILSAFE (coherent gate): naming-prob pool when effective support is weak
        req_label = _first_color_token_from_nlp(nlp_res)
        use_naming_pool = False
        if req_label and _naming_prob_label_supported(req_label, inv_with_probs=inv_eff):
            if (n_support_eff < int(_NAMING_POOL_MIN_N)) or (float(min_de_eff) > float(thr_used)):
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
                if pool_cap > 0 and len(df_pool) > pool_cap:
                    df_pool = df_pool.head(pool_cap).copy()
                if debug:
                    ndbg = dict(ndbg)
                    ndbg["gate"] = {
                        "n_support_eff": int(n_support_eff),
                        "min_de_eff": float(min_de_eff),
                        "thr_used": float(thr_used),
                        "pool_n_before": int(len(name_pool)),
                    }
                    print("  FAILSAFE: naming-prob pool used", ndbg)

    else:
        df_pool = inv.head(pool_cap).copy()
        if debug:
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}  anchor_lab={anchor_lab}")
            print(f"  pool_cap={pool_cap}  pool_n={len(df_pool)}  inv_n={len(inv)}")

    # Always add _de00_anchor for inspection
    if has_color and anchor_eff is not None:
        df_pool = _ensure_de00_anchor_col(df_pool, tuple(map(float, anchor_eff)))

    # ==========================
    # PHASE B — scoring
    # ==========================
    nlp_constraints = tuple(getattr(nlp_res, "constraints", ()) or ())
    query = QuerySpec(anchor_lab=anchor_eff, constraints=nlp_constraints)

    if debug and nlp_constraints:
        print("\n[PHASE B][CONSTRAINTS]")
        for c in nlp_constraints:
            axis = getattr(c, "axis", None)
            direction = getattr(c, "direction", None)
            strength = getattr(c, "strength", None)
            evidence = getattr(c, "evidence", None)
            meta = getattr(c, "meta", None)
            axis = axis.value if hasattr(axis, "value") else axis
            direction = direction.value if hasattr(direction, "value") else direction
            strength = strength.value if hasattr(strength, "value") else strength
            print(f"  axis={axis} dir={direction} strength={strength} evidence={evidence!r} meta={meta}")

    df_pool = _attach_naming_probs(df_pool)

    # within-family constraints require: label in distributions + distributions loaded
    family_label_for_constraints = family_label if (family_label is not None and family_in_dists and bool(dists)) else None
    family_specs = _family_specs_from_nlp(nlp_constraints) if (family_label_for_constraints is not None) else []

    if debug:
        print("\n[PHASE B][FAMILY-CONSTRAINTS]")
        print(f"  family_label={family_label!r}  in_dists={family_in_dists}")
        print(f"  specs_n={len(family_specs)}")
        if family_label is not None:
            pcol = f"p_{family_label}"
            print(
                f"  pcol_present={pcol in df_pool.columns}  "
                f"p_min={_FAMILY_P_MIN}  p_hi={_FAMILY_P_HI}  floor={_FAMILY_FLOOR}"
            )
        if family_label_for_constraints is None:
            print("  within_family_applied=False")
        else:
            print("  within_family_applied=True")
            if family_specs:
                for s in family_specs:
                    print(f"   - {s.axis}:{s.direction} q_lo={s.q_lo} q_hi={s.q_hi} strength={s.strength}")

    scored = score_shades(
        df_pool,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=getattr(assets, "preference_weights", None),
        # within-family (only applied if label+specs exist AND label_distributions supports it)
        constraint_label=family_label_for_constraints,
        constraint_specs=family_specs,
        label_distributions=dists if (family_label_for_constraints is not None and dists) else None,
        constraint_p_min=float(_FAMILY_P_MIN),
        constraint_p_hi=float(_FAMILY_P_HI),
        constraint_floor=float(_FAMILY_FLOOR),
    )

    # Restore dropped feature columns (incl. light_hsl expected by tests)
    scored = _restore_feature_cols(df_pool, scored)

    score_col = "score_total" if "score_total" in scored.columns else ("score" if "score" in scored.columns else None)
    if score_col is None:
        return scored.head(int(topk)).copy()

    return scored.sort_values(score_col, ascending=False).head(int(topk)).copy()


__all__ = ["recommend_from_text", "resolve_anchor_from_text", "resolve_effective_anchor_from_text"]
