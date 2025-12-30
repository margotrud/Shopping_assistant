# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import pandas as pd

from Shopping_assistant.io.assets import AssetBundle
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference
from Shopping_assistant.nlp.resolve.axis_projection import project_axes
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents
from Shopping_assistant.nlp.resolve.axis_thresholds import thresholds_from_decisions

from Shopping_assistant.color.scoring import (
    score_shades,
    QuerySpec,
    Constraint as ScoringConstraint,
    _ensure_cluster_id,
)

# Copie locale du mapping (doit rester aligné avec color/scoring.py)
_AXIS_TO_DIM = {
    "brightness": "light_hsl",
    "saturation": "sat_hsl",
    "vibrancy": "sat_eff",
    "depth": "depth",
    "clarity": "colorfulness",
}

# Copie locale des quantiles (doit rester aligné avec color/scoring.py)
_LEVEL_TO_Q = {
    "low": 0.35,
    "medium": 0.50,
    "high": 0.65,
    "very_high": 0.80,
}

def _level_from_cap_q(cap_q: float, *, op: str) -> str:
    """
    cap_q est une valeur "dans l'espace dim" (0..1) qu'on veut approximer
    par un level quantile supporté par le scorer.
    Pour "<=" : on veut un cap au moins aussi strict => choisir le plus petit level dont q <= cap_q ?
    En pratique: on choisit le level dont q est le plus proche de cap_q.
    """
    cap = float(cap_q)
    best = min(_LEVEL_TO_Q.items(), key=lambda kv: abs(kv[1] - cap))
    return str(best[0])

def _thresholds_to_scoring_constraints(thresholds) -> tuple[ScoringConstraint, ...]:
    out: list[ScoringConstraint] = []
    for axis, th in thresholds.items():
        axis_name = getattr(axis, "value", str(axis))
        dim = _AXIS_TO_DIM.get(str(axis_name))
        if dim is None:
            continue

        # score_shades/Constraint ne supporte que level (pas valeur numérique)
        # donc on approxime th.low/th.high en "low/medium/high/very_high".
        if th.low is not None and th.high is not None:
            continue  # non représentable en un seul token, on reste strict

        if th.low is not None:
            op = ">="
            lvl = _level_from_cap_q(th.low, op=op)
            out.append(ScoringConstraint(dim=dim, op=op, level=lvl, weight=float(th.weight)))
        elif th.high is not None:
            op = "<="
            lvl = _level_from_cap_q(th.high, op=op)
            out.append(ScoringConstraint(dim=dim, op=op, level=lvl, weight=float(th.weight)))

    return tuple(out)

def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle,
    like_cluster_id: int,
    topk: int = 20,
) -> pd.DataFrame:
    nlp_res = interpret_nlp(text)

    resolved = resolve_preference(nlp_res)
    intents_by_axis = project_axes(resolved)
    decisions = merge_axis_intents(intents_by_axis)

    # ✅ use thresholds (not strength->level heuristic)
    ths = thresholds_from_decisions(decisions)
    cons = _thresholds_to_scoring_constraints(ths)

    df = assets.inventory.copy()
    prototypes = assets.prototypes
    df = _ensure_cluster_id(df, prototypes, assets.assignments)

    query = QuerySpec(like_cluster_id=int(like_cluster_id), constraints=cons)

    scored = score_shades(
        df,
        prototypes,
        query,
        lambda_constraints=1.0,
        lambda_preference=0.0,
        calibration=assets.calibration,
    )
    sort_col = "score_total" if "score_total" in scored.columns else "score"
    return scored.sort_values(sort_col, ascending=False).head(topk)
