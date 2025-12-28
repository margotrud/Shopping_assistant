# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import pandas as pd

from Shopping_assistant.io.assets import AssetBundle
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp, build_preference_from_nlp
from Shopping_assistant.color.scoring import (
    score_shades,
    QuerySpec,
    constraints_from_nlp,
    _ensure_cluster_id,
)


def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle,
    like_cluster_id: int,
    topk: int = 20,
) -> pd.DataFrame:
    nlp_res = interpret_nlp(text)
    pref = build_preference_from_nlp(nlp_res)

    df = assets.inventory.copy()
    prototypes = assets.prototypes

    # cluster_id doit être assuré de manière déterministe via assignments explicites
    df = _ensure_cluster_id(df, prototypes, assets.assignments)

    cons = constraints_from_nlp(pref["hard_constraints"] + pref["soft_constraints"])
    query = QuerySpec(like_cluster_id=int(like_cluster_id), constraints=cons)

    scored = score_shades(
        df,
        prototypes,
        query,
        lambda_constraints=1.0,
        lambda_preference=0.0,
        calibration=assets.calibration,
    )
    return scored.sort_values("score_total", ascending=False).head(topk)
