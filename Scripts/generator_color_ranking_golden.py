from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from Shopping_assistant.color.scoring import score_inventory


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _find_first_existing(candidates: List[Path], *, label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {label}. Tried: {[str(c) for c in candidates]}")


def _find_scenarios_path() -> Path:
    # common locations
    candidates = [
        PROJECT_ROOT / "data" / "enriched_data" / "color_scenarios.csv",
        PROJECT_ROOT / "data" / "scenarios" / "color_scenarios.csv",
        PROJECT_ROOT / "data" / "color_scenarios.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: search under data/
    hits = sorted((PROJECT_ROOT / "data").glob("**/color_scenarios.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"color_scenarios.csv not found under {PROJECT_ROOT / 'data'}")

def _file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _stable_scenario_id(
    row: Dict[str, Any],
    *,
    calibration_path: Path,
    lambda_preference: float,
) -> str:
    payload = {
        "cluster_id": int(row["cluster_id"]),
        "constraints": str(row.get("constraints", "") or ""),
        "topk": int(row.get("topk", 20)),
        "lambda_constraints": float(row.get("lambda_constraints", 1.0)),
        "lambda_preference": float(lambda_preference),
        "calibration_sha1": _file_sha1(calibration_path),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _item_key(df: pd.DataFrame) -> pd.Series:
    return df["product_id"].astype(str) + "|" + df["shade_id"].astype(str)


@dataclass(frozen=True)
class GoldenScenario:
    scenario_id: str
    cluster_id: int
    constraints: str
    topk: int
    lambda_constraints: float
    lambda_preference: float
    calibration_sha1: str
    topk_keys: List[str]


def main() -> None:
    scenarios_path = _find_scenarios_path()
    inventory_path = PROJECT_ROOT / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    prototypes_path = PROJECT_ROOT / "data" / "enriched_data" / "color_prototypes_kmeans.csv"
    assignments_path = PROJECT_ROOT / "data" / "enriched_data" / "color_cluster_assignments.csv"
    calibration_path = PROJECT_ROOT / "data" / "models" / "color_scoring_calibration.json"

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Missing calibration file: {calibration_path}. "
            "Run: python scripts/build_color_scoring_calibration.py"
        )

    out_path = PROJECT_ROOT / "tests" / "goldens" / "color_ranking_goldens.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(scenarios_path)
    inventory = pd.read_csv(inventory_path)
    prototypes = pd.read_csv(prototypes_path)

    goldens: List[Dict[str, Any]] = []

    lambda_preference = 0.0  # explicit for goldens determinism
    cal_sha1 = _file_sha1(calibration_path)

    for _, r in scenarios.iterrows():
        row = r.to_dict()
        cluster_id = int(row["cluster_id"])
        constraints = str(row.get("constraints", "") or "")
        topk = int(row.get("topk", 20))
        lam = float(row.get("lambda_constraints", 1.0))

        scored = score_inventory(
            inventory=inventory,
            prototypes=prototypes,
            assignments_path=assignments_path,
            cluster_id=cluster_id,
            constraints=constraints,
            lambda_constraints=lam,
            lambda_preference=lambda_preference,
            calibration_path=calibration_path,
        )

        keys = _item_key(scored).head(topk).tolist()
        sid = _stable_scenario_id(row, calibration_path=calibration_path, lambda_preference=lambda_preference)

        goldens.append(asdict(GoldenScenario(
            scenario_id=sid,
            cluster_id=cluster_id,
            constraints=constraints,
            topk=topk,
            lambda_constraints=lam,
            lambda_preference=lambda_preference,
            calibration_sha1=cal_sha1,
            topk_keys=keys,
        )))

    out_path.write_text(json.dumps(goldens, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
