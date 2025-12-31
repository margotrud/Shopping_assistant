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

# Minimal color lexicon -> RGB anchors (sRGB 0..255).
_COLOR_ANCHORS_RGB: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("brick", (150, 45, 35)),
    ("burgundy", (110, 20, 45)),
    ("wine", (120, 20, 60)),
    ("berry", (150, 30, 80)),
    ("plum", (120, 40, 90)),
    ("magenta", (200, 0, 120)),
    ("fuchsia", (220, 0, 130)),
    ("pink", (230, 80, 140)),
    ("rose", (200, 70, 110)),
    ("coral", (240, 90, 80)),
    ("orange", (240, 110, 40)),
    ("red", (210, 20, 30)),
    ("nude", (200, 150, 120)),
    ("beige", (215, 190, 150)),
    ("tan", (185, 140, 100)),
    ("brown", (140, 90, 60)),
)

_COLOR_TOKEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k, _ in _COLOR_ANCHORS_RGB) + r")\b",
    flags=re.IGNORECASE,
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


# -----------------------------
# Color anchor -> Lab (sRGB D65)
# -----------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_xyz(rgb: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = rgb
    lin = _srgb_to_linear(np.array([r, g, b], dtype=float))
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    return M @ lin


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t: float) -> float:
        d = 6 / 29
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b], dtype=float)


def _rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb))


def _extract_color_keyword(text: str) -> Optional[str]:
    m = _COLOR_TOKEN_RE.search(text)
    return m.group(1).lower() if m else None


def _select_like_cluster_id_from_text(text: str, *, assets: AssetBundle) -> int:
    """
    Does:
        Choose like_cluster_id from text by mapping a detected color keyword to a Lab anchor,
        then picking the nearest prototype center in Lab space.
    """
    prot = assets.prototypes
    need = {"cluster_id", "L_lab", "a_lab", "b_lab"}
    if not need.issubset(set(prot.columns)):
        vals = pd.Series(prot.get("cluster_id", [])).dropna()
        return int(vals.min()) if len(vals) else 0

    kw = _extract_color_keyword(text)
    if not kw:
        vc = prot["cluster_id"].astype(int).value_counts()
        return int(vc.index[0]) if len(vc) else int(prot["cluster_id"].astype(int).min())

    rgb = dict(_COLOR_ANCHORS_RGB).get(kw)
    if rgb is None:
        vc = prot["cluster_id"].astype(int).value_counts()
        return int(vc.index[0]) if len(vc) else int(prot["cluster_id"].astype(int).min())

    target = _rgb_to_lab(rgb)
    P = prot[["L_lab", "a_lab", "b_lab"]].to_numpy(float)
    pid = prot["cluster_id"].to_numpy(int)

    d2 = ((P - target[None, :]) ** 2).sum(axis=1)
    return int(pid[int(np.argmin(d2))])


def _neighbor_clusters_from_like_cluster(
    like_cluster_id: int,
    *,
    assets: AssetBundle,
    topn: int = 4,
) -> list[int]:
    """
    Does:
        Select a stable candidate pool: the like cluster + its nearest neighbor clusters,
        computed ONLY from prototype Lab centers (independent of full user text).
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

    # deterministic unique order
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    if int(like_cluster_id) not in seen:
        uniq.insert(0, int(like_cluster_id))
    return uniq


def _get_assets_preference_weights(assets: AssetBundle) -> Optional[dict[str, float]]:
    """
    Does:
        Best-effort extraction of preference weights from AssetBundle.

    Important:
        score_shades requires preference_weights when lambda_preference != 0.
        We support multiple possible AssetBundle layouts without importing extra IO here.
    """
    w = getattr(assets, "preference_weights", None)
    if isinstance(w, dict):
        return w  # type: ignore[return-value]
    w2 = getattr(assets, "pref_weights", None)
    if isinstance(w2, dict):
        return w2  # type: ignore[return-value]
    return None


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
    candidate_clusters_topn: int = 4,
    topk: int = 20,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    # AB/control hooks (lets you make A vs B comparable)
    constraints_blob_override: str | None = None,
    thresholds_override: dict[Axis, AxisThreshold] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Does:
        Recommend shades from free text, using:
        text -> NLP -> axis intents -> merged axis decisions -> thresholds -> constraint blob -> scoring.

    Important:
        - constraints_blob_override: if provided, bypasses NLP-derived thresholds and uses this blob verbatim.
        - thresholds_override: if provided (and constraints_blob_override is None), bypasses NLP-derived thresholds
          and builds blob from these thresholds (calibration-aware).
        - If lambda_preference != 0, preference weights must be available on assets (assets.preference_weights).
        - If debug=True, prints NLP constraints + thresholds + top-K axis distributions.
    """
    # 0) NLP (only for trace/debug + to avoid re-parsing text in downstream pipeline)
    nlp_res = interpret_nlp(text, debug=debug)

    # 1) Stable like_cluster_id
    if like_cluster_id is None:
        like_cluster_id = _select_like_cluster_id_from_text(text, assets=assets)

    # 1bis) Stable pool
    if candidate_cluster_ids is None:
        candidate_cluster_ids = _neighbor_clusters_from_like_cluster(
            int(like_cluster_id),
            assets=assets,
            topn=int(candidate_clusters_topn),
        )
    candidate_cluster_ids = [int(x) for x in candidate_cluster_ids]

    # 2) Build constraints blob (AB-safe)
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

    # ---- DEBUG (NLP → thresholds)
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

    # 3) Parse constraints blob
    cons: tuple[Constraint, ...] = ()
    if constraints_blob:
        tokens = [t.strip() for t in str(constraints_blob).split(";") if t.strip()]
        cons = tuple(_parse_constraint_token(t) for t in tokens)

    # 4) Inventory + deterministic cluster_id
    df = assets.inventory.copy()
    df = _ensure_cluster_id(df, assets.prototypes, assets.assignments)
    df = df[df["cluster_id"].astype(int).isin(candidate_cluster_ids)].copy()

    # 5) Preference weights (only if enabled)
    pref_w = None
    if float(lambda_preference) != 0.0:
        pref_w = _get_assets_preference_weights(assets)
        if pref_w is None:
            raise ValueError(
                "lambda_preference != 0 but no preference weights found on AssetBundle. "
                "Expected assets.preference_weights (dict with keys: w_L, w_C, w_H)."
            )

    # 6) Scoring
    query = QuerySpec(like_cluster_id=int(like_cluster_id), constraints=cons)
    scored = score_shades(
        df,
        assets.prototypes,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=pref_w,
    )

    sort_col = "score_total" if "score_total" in scored.columns else "score"
    top = scored.sort_values(sort_col, ascending=False).head(int(topk))

    # ---- DEBUG (results distribution)
    if debug:
        print("\n[Top-K axis distribution]")
        for axis in ["brightness", "vibrancy", "saturation", "depth", "clarity"]:
            _print_axis_stats(top, axis)

    return top
