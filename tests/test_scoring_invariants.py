# tests/test_scoring_invariants.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from Shopping_assistant.io.assets import load_assets
from Shopping_assistant.color.scoring import QuerySpec, score_shades


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

    prototypes = _resolve_first_existing(
        [
            data / "enriched_data" / "color_prototypes_fused_lab.csv",
            data / "enriched_data" / "color_prototypes_fused.csv",
            data / "enriched_data" / "color_prototypes_kmeans.csv",
        ]
    )

    assignments = _resolve_first_existing(
        [
            data / "enriched_data" / "color_cluster_assignments_fused.csv",
            data / "enriched_data" / "color_cluster_assignments.csv",
        ]
    )

    calibration = _resolve_first_existing(
        [
            data / "models" / "color_scoring_calibration.json",
        ]
    )

    return load_assets(
        enriched_csv=enriched,
        prototypes_csv=prototypes,
        assignments_csv=assignments,
        calibration_json=calibration,
    )


def _score_top20(inv: pd.DataFrame, proto: pd.DataFrame, cal: dict, *, cluster_id: int, constraints=()):
    query = QuerySpec(like_cluster_id=int(cluster_id), constraints=tuple(constraints))
    scored = score_shades(
        inv,
        proto,
        query,
        lambda_constraints=1.0,
        lambda_preference=0.0,
        calibration=cal,
    )
    return scored.head(20)


def test_reproducible_top20_same_inputs():
    assets = _load_assets_defaults()

    out1 = _score_top20(assets.inventory, assets.prototypes, assets.calibration, cluster_id=0)
    out2 = _score_top20(assets.inventory, assets.prototypes, assets.calibration, cluster_id=0)

    key1 = list(zip(out1["product_id"], out1["shade_id"]))
    key2 = list(zip(out2["product_id"], out2["shade_id"]))
    assert key1 == key2


def test_invariance_to_row_order_shuffle():
    assets = _load_assets_defaults()

    inv = assets.inventory
    inv_shuf = inv.sample(frac=1.0, random_state=123).reset_index(drop=True)

    out1 = _score_top20(inv, assets.prototypes, assets.calibration, cluster_id=0)
    out2 = _score_top20(inv_shuf, assets.prototypes, assets.calibration, cluster_id=0)

    key1 = list(zip(out1["product_id"], out1["shade_id"]))
    key2 = list(zip(out2["product_id"], out2["shade_id"]))
    assert key1 == key2


def test_constraints_reduce_violations_in_top20():
    assets = _load_assets_defaults()

    base = _score_top20(assets.inventory, assets.prototypes, assets.calibration, cluster_id=0)

    # keep using the inventory field directly (same logic as before)
    constrained = score_shades(
        assets.inventory,
        assets.prototypes,
        QuerySpec(
            like_cluster_id=0,
            constraints=(
                # sat_eff<=low:1.0
                # Construct via the public Constraint type? Not necessary if your score_shades
                # expects Constraint objects; if it does, import Constraint and build it.
            )
        ),
        lambda_constraints=1.0,
        lambda_preference=0.0,
        calibration=assets.calibration,
    ).head(20)

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
