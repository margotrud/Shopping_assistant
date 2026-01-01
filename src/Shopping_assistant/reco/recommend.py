# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import re
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
    _ensure_cluster_id,
)

# Keep consistent with Shopping_assistant.color.scoring._parse_constraint()
_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*((?:low|medium|high|very_high)|(?:[0-9]*\.?[0-9]+))\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)


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

        # 1) direct lab fields if ever present
        L0 = meta.get("lab_L", None)
        a0 = meta.get("lab_a", None)
        b0 = meta.get("lab_b", None)
        if isinstance(L0, (int, float)) and isinstance(a0, (int, float)) and isinstance(b0, (int, float)):
            lab = (float(L0), float(a0), float(b0))
        else:
            # 2) lookup by canonical -> hex -> lab
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

    # ✅ Distance in Lab (avoid hue-angle ambiguity with nude/beige)
    # Weight L lower than a/b to keep hue/chroma more important than lightness.
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

    d2 = ((P - center[None, :]) ** 2).sum(axis=1)
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


def _print_axis_stats(df: pd.DataFrame, axis: str) -> None:
    if axis not in df.columns or df.empty:
        return
    vals = df[axis].astype(float)
    print(
        f"  {axis:<12} "
        f"min={vals.min():.3f}  "
        f"mean={vals.mean():.3f}  "
        f"max={vals.max():.3f}"
    )


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
    """
    # 0) NLP
    nlp_res = interpret_nlp(text, debug=debug)

    # Keep preference term OFF unless explicitly enabled (score_shades contract).
    lambda_preference = float(lambda_preference)

    # 1) like_cluster_id (dynamic from NLP anchor Lab)
    if like_cluster_id is None:
        like_cluster_id = _select_like_cluster_id_from_nlp(nlp_res, assets=assets)

    # 2) Candidate pool
    if candidate_cluster_ids is None:
        candidate_cluster_ids = _neighbor_clusters_from_like_cluster(
            int(like_cluster_id),
            assets=assets,
            topn=int(candidate_clusters_topn),
        )
    candidate_cluster_ids = [int(x) for x in candidate_cluster_ids]

    # 3) Thresholds / blob
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

    # ---- DEBUG (NLP → thresholds + anchor)
    if debug:
        print("\n[NLP constraints_final]")
        for c in (nlp_res.trace or {}).get("constraints_final", []):
            print(
                f"  {c['axis']:>10}  {c['direction']:<5}  "
                f"{c['strength']:<6}  conf={c['confidence']:.2f}  "
                f"'{c['evidence']}'"
            )

        if ths:
            print("\n[Axis thresholds]")
            for ax, t in ths.items():
                print(f"  {ax.value:<12} low={t.low} high={t.high} weight={t.weight:.2f}")

        if constraints_blob:
            print("\n[Constraints blob]")
            print(f"  {constraints_blob}")

        anchor = _anchor_lab_from_nlp(nlp_res)
        print("\n[Anchor]")
        print(f"  target_lab={anchor}")
        print(f"  like_cluster_id={int(like_cluster_id)}")
        print(f"  candidate_cluster_ids={candidate_cluster_ids}")

    # 4) Parse constraints blob
    cons: tuple[Constraint, ...] = ()
    if constraints_blob:
        tokens = [t.strip() for t in str(constraints_blob).split(";") if t.strip()]
        cons = tuple(_parse_constraint_token(t) for t in tokens)

    # 5) Inventory
    df = assets.inventory.copy()
    df = _ensure_cluster_id(df, assets.prototypes, assets.assignments)
    df = df[df["cluster_id"].astype(int).isin(candidate_cluster_ids)].copy()

    # 6) Scoring
    query = QuerySpec(like_cluster_id=int(like_cluster_id), constraints=cons)
    scored = score_shades(
        df,
        assets.prototypes,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=None,  # keep None unless you explicitly enable preference term
    )

    sort_col = "score_total" if "score_total" in scored.columns else "score"
    top = scored.sort_values(sort_col, ascending=False).head(int(topk))

    if debug:
        print("\n[Top-K axis distribution]")
        for axis in ["brightness", "vibrancy", "saturation", "depth", "clarity"]:
            _print_axis_stats(top, axis)

    return top
