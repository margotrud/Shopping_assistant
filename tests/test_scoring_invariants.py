# tests/test_scoring_invariants.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from Shopping_assistant.io.assets import load_assets
from Shopping_assistant.color.scoring import QuerySpec, score_shades
from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Constraint


def _project_root() -> Path:
    # pythonProject/tests -> pythonProject
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "data"


def _resolve_first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {[str(p) for p in candidates]}")


def _load_assets_defaults():
    data = _data_dir()

    enriched = _resolve_first_existing(
        [
            data / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv",
            data / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched_v2.csv",
        ]
    )

    calibration = _resolve_first_existing(
        [
            data / "models" / "color_scoring_calibration.json",
        ]
    )

    return load_assets(
        enriched_csv=enriched,
        calibration_json=calibration,
    )


def _score_top20(inv: pd.DataFrame, cal: dict, *, constraints=(), anchor_lab=None):
    query = QuerySpec(anchor_lab=anchor_lab, constraints=tuple(constraints))
    scored = score_shades(
        inv,
        query,
        lambda_constraints=1.0,
        lambda_preference=0.0,
        calibration=cal,
    )
    return scored.head(20)


def test_reproducible_top20_same_inputs():
    assets = _load_assets_defaults()

    out1 = _score_top20(assets.inventory, assets.calibration)
    out2 = _score_top20(assets.inventory, assets.calibration)

    key1 = list(zip(out1["product_id"].astype(str), out1["shade_id"].astype(str)))
    key2 = list(zip(out2["product_id"].astype(str), out2["shade_id"].astype(str)))
    assert key1 == key2


def test_invariance_to_row_order_shuffle():
    assets = _load_assets_defaults()

    inv = assets.inventory
    inv_shuf = inv.sample(frac=1.0, random_state=123).reset_index(drop=True)

    out1 = _score_top20(inv, assets.calibration)
    out2 = _score_top20(inv_shuf, assets.calibration)

    key1 = list(zip(out1["product_id"].astype(str), out1["shade_id"].astype(str)))
    key2 = list(zip(out2["product_id"].astype(str), out2["shade_id"].astype(str)))
    assert key1 == key2


def test_constraints_reduce_violations_in_top20():
    assets = _load_assets_defaults()

    # Preconditions for this invariant
    assert "sat_eff" in assets.inventory.columns, "inventory missing required column 'sat_eff'"

    base = _score_top20(assets.inventory, assets.calibration)

    # Enforce lower saturation in the score (sat_eff <= low)
    c = Constraint(
        axis=Axis.SATURATION,
        direction=Direction.LOWER,
        strength=Strength.STRONG,
        evidence="test_sat_low",
        meta={"axis_query": "sat_eff", "tok": "sat", "quality": "low"},
    )

    constrained = _score_top20(
        assets.inventory,
        assets.calibration,
        constraints=(c,),
    )

    inv_key = assets.inventory.copy()
    inv_key["product_id"] = inv_key["product_id"].astype(str)
    inv_key["shade_id"] = inv_key["shade_id"].astype(str)

    base_m = base.merge(
        inv_key[["product_id", "shade_id", "sat_eff"]],
        on=["product_id", "shade_id"],
        how="left",
    )
    cons_m = constrained.merge(
        inv_key[["product_id", "shade_id", "sat_eff"]],
        on=["product_id", "shade_id"],
        how="left",
    )

    assert base_m["sat_eff"].notna().all(), "sat_eff missing for some base top-20 rows"
    assert cons_m["sat_eff"].notna().all(), "sat_eff missing for some constrained top-20 rows"

    assert cons_m["sat_eff"].mean() <= base_m["sat_eff"].mean()
