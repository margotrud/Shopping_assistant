# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from Shopping_assistant.io.assets import AssetBundle
from Shopping_assistant.nlp import Axis
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference
from Shopping_assistant.nlp.resolve.axis_projection import project_axes
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold, thresholds_from_decisions
from Shopping_assistant.nlp.resolve.scoring_adapter import build_constraints_blob_from_thresholds
from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index

from Shopping_assistant.color.scoring import (
    Constraint,
    QuerySpec,
    score_shades,
    _ensure_cluster_id as _ensure_cluster_id,  # keep for scripts/debug
)

# Keep consistent with Shopping_assistant.color.scoring._parse_constraint()
_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*((?:low|medium|high|very_high)|(?:[0-9]*\.?[0-9]+))\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)

# Neutral “plain color” regularizer policy:
# - MUST NOT expand candidate pools (keeps hue family)
# - MUST be low-weight (tie-breaker only)
# NOTE: with the updated calibration JSON, warmth can be used as a constraint dim;
#       we include it here only for "plain color" regularization.
_NEUTRAL_DIMS = ("light_hsl", "sat_hsl", "warmth")
_NEUTRAL_LEVEL = "medium"
_NEUTRAL_W = 0.10  # low weight: tie-breaker only

# Hue-lock neighborhood: AB-dominant cluster distance (prevents "bright" drifting to peach/coral)
_NEIGHBOR_WL = 0.05  # 0.0 = strict hue lock, 0.05 = slight lightness tolerance


def _parse_constraint_token(token: str) -> Constraint:
    raw = token.strip()
    m = _CONSTRAINT_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid constraint token: {raw!r}")

    dim, op, lvl_or_num, w_str = m.groups()
    if lvl_or_num in {"low", "medium", "high", "very_high"}:
        return Constraint(dim=str(dim), op=str(op), level=str(lvl_or_num), cutpoint=None, weight=float(w_str))
    return Constraint(dim=str(dim), op=str(op), level=None, cutpoint=float(lvl_or_num), weight=float(w_str))


# ---------------------------------------------------------------------
# Dynamic like_cluster_id selection (NO static color tables)
# ---------------------------------------------------------------------


def _hex_to_rgb01(hx: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(hx, str):
        return None
    s = hx.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return None


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _hex_to_lab(hx: str) -> Optional[Tuple[float, float, float]]:
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None

    r, g, b = rgb01
    r, g, b = _srgb01_to_linear(r), _srgb01_to_linear(g), _srgb01_to_linear(b)

    # sRGB D65
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return (float(L), float(a), float(b2))


def _anchor_lab_from_nlp(nlp_res) -> Optional[Tuple[float, float, float]]:
    """
    Does:
        Derive an anchor Lab (L*,a*,b*) from NLP color mentions.
        Priority:
          1) Mention.meta has lab_L/lab_a/lab_b (if present)
          2) Lookup canonical in world alias index -> hex -> Lab
    """
    idx = build_world_alias_index()
    best_lab: Optional[Tuple[float, float, float]] = None
    best_score = -1.0

    for m in getattr(nlp_res, "mentions", ()) or ():
        kind = getattr(getattr(m, "kind", None), "value", None)
        if kind != "color":
            continue

        pol = getattr(getattr(m, "polarity", None), "value", None)
        if pol not in {"like", "neutral", "unknown"}:
            continue

        conf = float(getattr(m, "confidence", 0.0) or 0.0)
        meta = getattr(m, "meta", {}) or {}

        L0 = meta.get("lab_L", None)
        a0 = meta.get("lab_a", None)
        b0 = meta.get("lab_b", None)
        if isinstance(L0, (int, float)) and isinstance(a0, (int, float)) and isinstance(b0, (int, float)):
            lab = (float(L0), float(a0), float(b0))
        else:
            lab = None
            canon = str(getattr(m, "canonical", "") or "").strip().lower()
            if canon:
                info = idx.get(canon)
                hx = None if not info else info.get("hex")
                if isinstance(hx, str) and hx:
                    lab = _hex_to_lab(hx)

        if lab is None:
            continue

        if conf > best_score:
            best_score = conf
            best_lab = lab

    return best_lab


def _select_like_cluster_id_from_nlp(nlp_res, *, assets: AssetBundle) -> int:
    """
    Does:
        Choose like_cluster_id by projecting NLP anchor Lab onto prototype Lab centers.
        Dynamic: no keyword tables, no static RGB anchors.
    """
    prot = assets.prototypes
    need = {"cluster_id", "L_lab", "a_lab", "b_lab"}
    if not need.issubset(set(prot.columns)):
        vals = pd.Series(prot.get("cluster_id", [])).dropna()
        return int(vals.min()) if len(vals) else 0

    anchor = _anchor_lab_from_nlp(nlp_res)
    if anchor is None:
        vc = prot["cluster_id"].astype(int).value_counts()
        return int(vc.index[0]) if len(vc) else int(prot["cluster_id"].astype(int).min())

    L0, a0, b0 = anchor

    p = prot.copy()
    p["cluster_id"] = p["cluster_id"].astype(int)

    L = p["L_lab"].astype(float).to_numpy(float)
    a = p["a_lab"].astype(float).to_numpy(float)
    b = p["b_lab"].astype(float).to_numpy(float)
    cid = p["cluster_id"].to_numpy(int)

    # Distance in Lab; downweight L so hue/chroma dominate.
    wL = 0.25
    d2 = wL * (L - float(L0)) ** 2 + (a - float(a0)) ** 2 + (b - float(b0)) ** 2

    i = int(np.argmin(d2))
    return int(cid[i])


def _neighbor_clusters_from_like_cluster(
    like_cluster_id: int,
    *,
    assets: AssetBundle,
    topn: int = 10,
) -> list[int]:
    """
    Does:
        Select candidate pool: like cluster + nearest neighbor clusters from prototype Lab centers.
        Uses AB-dominant distance to lock hue family (prevents "bright" drifting to peach/coral).
    """
    prot = assets.prototypes
    need = {"cluster_id", "L_lab", "a_lab", "b_lab"}
    if not need.issubset(set(prot.columns)):
        return [int(like_cluster_id)]

    p = prot.copy()
    p["cluster_id"] = p["cluster_id"].astype(int)
    row = p[p["cluster_id"] == int(like_cluster_id)]
    if row.empty:
        return [int(like_cluster_id)]

    center = row[["L_lab", "a_lab", "b_lab"]].iloc[0].to_numpy(float)
    P = p[["L_lab", "a_lab", "b_lab"]].to_numpy(float)
    pid = p["cluster_id"].to_numpy(int)

    # AB-dominant neighborhood
    wL = float(_NEIGHBOR_WL)
    d2 = wL * (P[:, 0] - center[0]) ** 2 + (P[:, 1] - center[1]) ** 2 + (P[:, 2] - center[2]) ** 2
    order = np.argsort(d2)

    k = max(1, int(topn))
    out = [int(pid[i]) for i in order[:k]]

    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    if int(like_cluster_id) not in seen:
        uniq.insert(0, int(like_cluster_id))
    return uniq


def _has_effective_thresholds(ths: dict[Axis, AxisThreshold] | None) -> bool:
    """
    Does:
        Determine whether thresholds would have non-zero impact.
        Dynamic: relies on weights, not axis names or hardcoded lists.
    """
    if not ths:
        return False
    for t in ths.values():
        w = float(getattr(t, "weight", 0.0) or 0.0)
        if w > 0.0:
            return True
    return False


def _has_color_like_mention(nlp_res) -> bool:
    for m in getattr(nlp_res, "mentions", ()) or ():
        kind = getattr(getattr(m, "kind", None), "value", None)
        if kind != "color":
            continue
        pol = getattr(getattr(m, "polarity", None), "value", None)
        if pol in {"like", "neutral", "unknown"}:
            return True
    return False


def _is_neutral_regularizer_only(cons: tuple[Constraint, ...]) -> bool:
    """
    Does:
        Detect the internal "plain color" neutral regularizer constraints.
        Criteria: exactly the +/- medium band on the neutral dims, with low weights.
    """
    if not cons:
        return False

    needed = {(d, ">=", _NEUTRAL_LEVEL) for d in _NEUTRAL_DIMS} | {(d, "<=", _NEUTRAL_LEVEL) for d in _NEUTRAL_DIMS}
    got = set()

    for c in cons:
        if c.cutpoint is not None:
            return False
        if c.level is None:
            return False
        if str(c.dim) not in _NEUTRAL_DIMS:
            return False
        if str(c.level) != _NEUTRAL_LEVEL:
            return False
        if c.op not in {">=", "<="}:
            return False

        got.add((str(c.dim), str(c.op), str(c.level)))

        if float(getattr(c, "weight", 0.0) or 0.0) > 0.35:
            return False

    return got == needed


def _is_brightness_only_constraints(nlp_res) -> bool:
    """
    Does:
        Return True if NLP produced constraints and ALL are brightness/lightness axis constraints.
        Uses nlp_res.trace["constraints_final"] when available.
    """
    trace = getattr(nlp_res, "trace", None) or {}
    cons = trace.get("constraints_final", []) or []
    if not cons:
        return False

    for c in cons:
        axis = None
        try:
            axis = c.get("axis", None)
        except Exception:
            axis = None

        # Conservative fallback: if structure unexpected, treat as NOT brightness-only.
        if not isinstance(axis, str):
            return False

        ax = axis.strip().lower()
        if ax not in {"brightness", "lightness"}:
            return False

    return True


# ---------------------------------------------------------------------
# Constraint-aware pool expansion (DYNAMIC, but NOT "all clusters")
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _ClusterScore:
    cluster_id: int
    d2: float
    feasible_frac: float
    n_rows: int


def _fixed_threshold_from_calibration(cal: dict, dim: str, level: str) -> float:
    th = cal.get("thresholds", {}).get(dim, {})
    if not isinstance(th, dict) or level not in th:
        raise KeyError(f"Missing calibration threshold for dim={dim!r} level={level!r}")
    return float(th[level])


def _constraint_threshold_for_pool(cal: dict, c: Constraint) -> float:
    # Pool-feasibility expansion is conservative; we use ABS thresholds only.
    # Relative mode is applied inside score_shades/_apply_constraints.
    if c.cutpoint is not None:
        return float(c.cutpoint)
    if c.level is None:
        raise ValueError(f"Constraint has neither cutpoint nor level: {c}")
    return _fixed_threshold_from_calibration(cal, c.dim, str(c.level))


def _satisfy_constraint_series(vals: pd.Series, c: Constraint, thr: float) -> pd.Series:
    x = pd.to_numeric(vals, errors="coerce")
    if c.op == "<=":
        return x <= float(thr)
    return x >= float(thr)


def _pick_candidate_clusters_dynamic(
    *,
    assets: AssetBundle,
    like_cluster_id: int,
    base_clusters: list[int],
    constraints: tuple[Constraint, ...],
    min_candidates: int,
    max_clusters: int,
    min_feasible_frac: float,
) -> list[int]:
    """
    Does:
        Expand candidate clusters only as needed to satisfy constraints,
        prioritizing Lab proximity to like_cluster prototype center.
        Never expands to "all clusters" blindly.
    """
    if not constraints:
        return base_clusters

    inv = assets.inventory
    if "cluster_id" not in inv.columns:
        return base_clusters

    prot = assets.prototypes
    needp = {"cluster_id", "L_lab", "a_lab", "b_lab"}
    if not needp.issubset(set(prot.columns)):
        return base_clusters

    p = prot.copy()
    p["cluster_id"] = p["cluster_id"].astype(int)
    row0 = p[p["cluster_id"] == int(like_cluster_id)]
    if row0.empty:
        return base_clusters

    c0 = row0[["L_lab", "a_lab", "b_lab"]].iloc[0].to_numpy(float)
    P = p[["L_lab", "a_lab", "b_lab"]].to_numpy(float)
    pid = p["cluster_id"].to_numpy(int)

    # IMPORTANT: keep expansion itself also AB-dominant (same rationale as neighborhood)
    wL = float(_NEIGHBOR_WL)
    d2 = wL * (P[:, 0] - c0[0]) ** 2 + (P[:, 1] - c0[1]) ** 2 + (P[:, 2] - c0[2]) ** 2

    cal = assets.calibration
    inv_c = inv.copy()
    inv_c["cluster_id"] = inv_c["cluster_id"].astype(int)

    needed_dims = {c.dim for c in constraints}
    missing_dims = [d for d in needed_dims if d not in inv_c.columns]
    if missing_dims:
        return base_clusters

    cluster_rows: list[_ClusterScore] = []
    for cid, di in zip(pid.tolist(), d2.tolist()):
        g = inv_c[inv_c["cluster_id"] == int(cid)]
        n = int(len(g))
        if n == 0:
            feas = 0.0
        else:
            m = pd.Series(True, index=g.index)
            for c in constraints:
                thr = _constraint_threshold_for_pool(cal, c)
                m &= _satisfy_constraint_series(g[c.dim], c, thr).fillna(False)
            feas = float(m.mean()) if n else 0.0
        cluster_rows.append(_ClusterScore(cluster_id=int(cid), d2=float(di), feasible_frac=feas, n_rows=n))

    cluster_rows.sort(key=lambda r: (r.d2, -r.feasible_frac))

    chosen: list[int] = []
    seen: set[int] = set()
    for x in base_clusters:
        xi = int(x)
        if xi not in seen:
            chosen.append(xi)
            seen.add(xi)

    def feasible_count_in_pool(cluster_ids: list[int]) -> int:
        sub = inv_c[inv_c["cluster_id"].isin(cluster_ids)]
        if sub.empty:
            return 0
        m = pd.Series(True, index=sub.index)
        for c in constraints:
            thr = _constraint_threshold_for_pool(cal, c)
            m &= _satisfy_constraint_series(sub[c.dim], c, thr).fillna(False)
        return int(m.sum())

    target_n = max(int(min_candidates), 1)
    max_k = max(int(max_clusters), len(chosen))
    cur_feas = feasible_count_in_pool(chosen)

    for r in cluster_rows:
        if len(chosen) >= max_k:
            break
        if cur_feas >= target_n:
            break
        cid = int(r.cluster_id)
        if cid in seen:
            continue
        if r.n_rows > 0 and (r.feasible_frac >= float(min_feasible_frac) or cur_feas < max(5, target_n // 4)):
            chosen.append(cid)
            seen.add(cid)
            cur_feas = feasible_count_in_pool(chosen)

    for r in cluster_rows:
        if len(chosen) >= max_k:
            break
        if cur_feas >= target_n:
            break
        cid = int(r.cluster_id)
        if cid in seen:
            continue
        chosen.append(cid)
        seen.add(cid)
        cur_feas = feasible_count_in_pool(chosen)

    return chosen


def _print_dim_stats(df: pd.DataFrame, dim: str) -> None:
    if dim not in df.columns or df.empty:
        return
    vals = pd.to_numeric(df[dim], errors="coerce").dropna()
    if vals.empty:
        return
    print(f"  {dim:<12} min={vals.min():.3f}  mean={vals.mean():.3f}  max={vals.max():.3f}")


def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle,
    like_cluster_id: int | None = None,
    candidate_cluster_ids: Iterable[int] | None = None,
    candidate_clusters_topn: int = 10,
    topk: int = 20,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    # AB/control hooks
    constraints_blob_override: str | None = None,
    thresholds_override: dict[Axis, AxisThreshold] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Does:
        Recommend shades from free text with optional debug instrumentation.

    Policy (target behavior):
      - Color mention => lock hue family (AB-dominant neighborhood).
      - Plain color => return "most neutral/central" shades within that family (low-weight regularizer).
      - Color + constraints => apply constraints within family; expand only if family has too few feasible candidates.
      - Brightness-only constraints => NEVER expand beyond the like_cluster family (prevents coral/peach drift).
    """
    # 0) NLP
    nlp_res = interpret_nlp(text, debug=debug)
    lambda_preference = float(lambda_preference)

    has_color = _has_color_like_mention(nlp_res)
    hue_lock = bool(has_color)  # color mention => hue family lock
    brightness_only = bool(has_color) and _is_brightness_only_constraints(nlp_res)

    # 1) like_cluster_id (dynamic from NLP anchor Lab)
    if like_cluster_id is None:
        like_cluster_id = _select_like_cluster_id_from_nlp(nlp_res, assets=assets)
    like_cluster_id = int(like_cluster_id)

    # 2) Thresholds / blob
    ths: dict[Axis, AxisThreshold] | None
    constraints_blob: str

    if constraints_blob_override is not None:
        constraints_blob = str(constraints_blob_override).strip()
        ths = None
    else:
        if thresholds_override is not None:
            ths = dict(thresholds_override)
        else:
            resolved = resolve_preference(nlp_res)
            intents_by_axis = project_axes(resolved)
            decisions = merge_axis_intents(intents_by_axis)
            ths = thresholds_from_decisions(decisions)

        constraints_blob = build_constraints_blob_from_thresholds(ths, calibration=assets.calibration)

    # 3) Parse blob
    cons: tuple[Constraint, ...] = ()
    if constraints_blob:
        tokens = [t.strip() for t in str(constraints_blob).split(";") if t.strip()]
        cons = tuple(_parse_constraint_token(t) for t in tokens)

    # 3b) Plain-color neutral regularizer:
    injected_neutral = False
    if (not cons) and has_color and (not _has_effective_thresholds(ths)):
        neutral_tokens: list[str] = []
        for d in _NEUTRAL_DIMS:
            neutral_tokens.append(f"{d}>={_NEUTRAL_LEVEL}:{_NEUTRAL_W}")
            neutral_tokens.append(f"{d}<={_NEUTRAL_LEVEL}:{_NEUTRAL_W}")
        constraints_blob = ";".join(neutral_tokens)
        cons = tuple(_parse_constraint_token(t) for t in neutral_tokens)
        injected_neutral = True

    is_neutral_only = _is_neutral_regularizer_only(cons)

    # 4) Base pool
    if candidate_cluster_ids is None:
        if brightness_only:
            # Hard family lock for brightness-only requests (prevents peach/coral drift)
            base_clusters = [like_cluster_id]
        else:
            base_clusters = _neighbor_clusters_from_like_cluster(
                like_cluster_id,
                assets=assets,
                topn=int(candidate_clusters_topn),
            )
    else:
        base_clusters = [int(x) for x in candidate_cluster_ids]
        if like_cluster_id not in base_clusters:
            base_clusters = [like_cluster_id] + base_clusters

    # 5) Pool selection policy:
    # - neutral-only => NEVER expand
    # - brightness-only => NEVER expand (even if constraints exist)
    # - hue_lock => expand only if too few feasible candidates within base family
    # - no color mention => bounded feasibility-driven expansion
    candidate_cluster_ids_final = base_clusters

    if cons and (not is_neutral_only) and (not brightness_only):
        if hue_lock:
            min_candidates = max(50, 5 * int(topk))

            inv = assets.inventory
            inv_c = inv.copy()
            if "cluster_id" in inv_c.columns:
                inv_c["cluster_id"] = inv_c["cluster_id"].astype(int)
            else:
                inv_c = inv_c.assign(cluster_id=-1)

            sub = inv_c[inv_c["cluster_id"].isin(base_clusters)]
            if sub.empty:
                feasible_n = 0
            else:
                m = pd.Series(True, index=sub.index)
                for c in cons:
                    thr = _constraint_threshold_for_pool(assets.calibration, c)
                    m &= _satisfy_constraint_series(sub[c.dim], c, thr).fillna(False)
                feasible_n = int(m.sum())

            if feasible_n < min_candidates:
                max_clusters = max(len(base_clusters), min(18, len(assets.prototypes)))
                candidate_cluster_ids_final = _pick_candidate_clusters_dynamic(
                    assets=assets,
                    like_cluster_id=like_cluster_id,
                    base_clusters=base_clusters,
                    constraints=cons,
                    min_candidates=min_candidates,
                    max_clusters=max_clusters,
                    min_feasible_frac=0.05,
                )
            else:
                candidate_cluster_ids_final = base_clusters
        else:
            min_candidates = max(50, 5 * int(topk))
            max_clusters = max(len(base_clusters), min(28, len(assets.prototypes)))
            candidate_cluster_ids_final = _pick_candidate_clusters_dynamic(
                assets=assets,
                like_cluster_id=like_cluster_id,
                base_clusters=base_clusters,
                constraints=cons,
                min_candidates=min_candidates,
                max_clusters=max_clusters,
                min_feasible_frac=0.05,
            )

    # ---- DEBUG
    if debug:
        print("\n[NLP constraints_final]")
        for c in (getattr(nlp_res, "trace", None) or {}).get("constraints_final", []):
            try:
                print(
                    f"  {c['axis']:>10}  {c['direction']:<5}  "
                    f"{c['strength']:<6}  conf={c['confidence']:.2f}  "
                    f"'{c['evidence']}'"
                )
            except Exception:
                print(f"  {c}")

        if ths:
            print("\n[Axis thresholds]")
            for ax, t in ths.items():
                print(f"  {ax.value:<12} low={t.low} high={t.high} weight={t.weight:.2f}")

        print("\n[Constraints blob]")
        print(f"  {constraints_blob!r}")
        print(f"  injected_neutral={injected_neutral}  neutral_only={is_neutral_only}")
        print(f"  has_color={has_color}  hue_lock={hue_lock}  brightness_only={brightness_only}")
        print(f"  neighbor_wL={_NEIGHBOR_WL}")

        anchor = _anchor_lab_from_nlp(nlp_res)
        print("\n[Anchor]")
        print(f"  target_lab={anchor}")
        print(f"  like_cluster_id={like_cluster_id}")

        print("\n[Candidate pools]")
        print(f"  base_clusters(n={len(base_clusters)}): {base_clusters}")
        print(f"  final_clusters(n={len(candidate_cluster_ids_final)}): {candidate_cluster_ids_final}")

    # 6) Inventory slice
    df = assets.inventory.copy()
    if "cluster_id" not in df.columns:
        raise KeyError(
            "assets.inventory missing 'cluster_id'. "
            "Fix AssetBundle loader to inject cluster_id from assignments before recommendation."
        )

    df["cluster_id"] = df["cluster_id"].astype(int)
    df = df[df["cluster_id"].isin([int(x) for x in candidate_cluster_ids_final])].copy()
    if df.empty:
        df = assets.inventory.copy()
        df["cluster_id"] = df["cluster_id"].astype(int)
        df = df[df["cluster_id"].isin([int(x) for x in base_clusters])].copy()

    # 7) Scoring
    query = QuerySpec(like_cluster_id=like_cluster_id, constraints=cons)
    scored = score_shades(
        df,
        assets.prototypes,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=None,
    )

    sort_col = "score_total" if "score_total" in scored.columns else "score"
    top = scored.sort_values(sort_col, ascending=False).head(int(topk))

    if debug:
        print("\n[Top-K dim distribution]")
        for dim in ["light_hsl", "sat_hsl", "warmth", "depth", "colorfulness", "C_lab", "L_lab"]:
            _print_dim_stats(top, dim)

    return top
