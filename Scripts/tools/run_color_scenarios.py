# scripts/run_color_scenarios.py
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return PROJECT_ROOT


def _stable_scenario_id(*, cluster_id: int, constraints: str, topk: int, lambda_constraints: float) -> str:
    payload = {
        "cluster_id": int(cluster_id),
        "constraints": str(constraints or ""),
        "topk": int(topk),
        "lambda_constraints": float(lambda_constraints),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _item_key(df: pd.DataFrame) -> pd.Series:
    if {"product_id", "shade_id"}.issubset(df.columns):
        return df["product_id"].astype(str) + "|" + df["shade_id"].astype(str)
    if "item_id" in df.columns:
        return df["item_id"].astype(str)
    raise ValueError("No stable item key found. Need (product_id, shade_id) or item_id.")


@dataclass(frozen=True)
class GoldenScenario:
    scenario_id: str
    cluster_id: int
    constraints: str
    topk: int
    lambda_constraints: float
    topk_keys: List[str]


def _load_goldens(path: Path) -> Dict[str, GoldenScenario]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, GoldenScenario] = {}
    for g in raw:
        gs = GoldenScenario(
            scenario_id=str(g["scenario_id"]),
            cluster_id=int(g["cluster_id"]),
            constraints=str(g.get("constraints", "") or ""),
            topk=int(g["topk"]),
            lambda_constraints=float(g.get("lambda_constraints", 1.0)),
            topk_keys=list(g["topk_keys"]),
        )
        out[gs.scenario_id] = gs
    return out


def _write_goldens(path: Path, goldens: List[GoldenScenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(g) for g in goldens]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenarios",
        type=str,
        default=str(_project_root() / "data" / "scenarios" / "color_scenarios.csv"),
    )
    p.add_argument(
        "--infile",
        type=str,
        default=str(_project_root() / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"),
    )
    p.add_argument(
        "--prototypes",
        type=str,
        default=str(_project_root() / "data" / "enriched_data" / "color_prototypes_kmeans.csv"),
    )
    p.add_argument(
        "--assignments",
        type=str,
        default=str(_project_root() / "data" / "enriched_data" / "color_cluster_assignments.csv"),
    )
    p.add_argument(
        "--goldens",
        type=str,
        default=str(_project_root() / "tests" / "goldens" / "color_ranking_goldens.json"),
    )
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--lambda-constraints", type=float, default=1.0)
    p.add_argument("--write-goldens", action="store_true", help="Generate/overwrite goldens from current code.")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()

    # Import here so running the script doesn't require import hacks
    from Shopping_assistant.color.scoring import score_inventory  # MUST exist (wrapper in scoring.py)

    scenarios_path = Path(args.scenarios)
    infile = Path(args.infile)
    prototypes_path = Path(args.prototypes)
    assignments_path = Path(args.assignments)
    goldens_path = Path(args.goldens)

    df = pd.read_csv(scenarios_path)
    required = {"cluster_id", "constraints"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"scenarios missing columns: {sorted(missing)}")

    # scenario_id is optional: we compute a stable one if absent
    have_scenario_id = "scenario_id" in df.columns

    inventory = pd.read_csv(infile)
    prototypes = pd.read_csv(prototypes_path)

    existing_goldens = _load_goldens(goldens_path) if (goldens_path.exists() and not args.write_goldens) else {}

    ok = 0
    fail = 0
    new_goldens: List[GoldenScenario] = []

    for row in df.itertuples(index=False):
        cluster_id = int(getattr(row, "cluster_id"))
        constraints = "" if pd.isna(getattr(row, "constraints")) else str(getattr(row, "constraints"))

        topk = int(getattr(row, "topk")) if hasattr(row, "topk") and not pd.isna(getattr(row, "topk")) else int(args.topk)
        lam = float(getattr(row, "lambda_constraints")) if hasattr(row, "lambda_constraints") and not pd.isna(getattr(row, "lambda_constraints")) else float(args.lambda_constraints)

        scenario_id = str(getattr(row, "scenario_id")) if have_scenario_id else _stable_scenario_id(
            cluster_id=cluster_id,
            constraints=constraints,
            topk=topk,
            lambda_constraints=lam,
        )

        print("\n" + "=" * 100)
        print(f"[scenario] {scenario_id} | cluster_id={cluster_id} | constraints={constraints} | topk={topk} | lambda={lam}")
        print("=" * 100)

        scored = score_inventory(
            inventory=inventory,
            prototypes=prototypes,
            assignments_path=assignments_path,
            cluster_id=cluster_id,
            constraints=constraints,
            lambda_constraints=lam,
        )

        result_keys = _item_key(scored).head(topk).tolist()

        if args.write_goldens:
            new_goldens.append(GoldenScenario(
                scenario_id=scenario_id,
                cluster_id=cluster_id,
                constraints=constraints,
                topk=topk,
                lambda_constraints=lam,
                topk_keys=result_keys,
            ))
            ok += 1
            continue

        golden = existing_goldens.get(scenario_id)
        if golden is None:
            fail += 1
            print(f"[FAIL] missing golden for scenario_id={scenario_id}", file=sys.stderr)
            if args.fail_fast:
                raise SystemExit(2)
            continue

        if golden.topk_keys != result_keys:
            fail += 1
            print(f"[FAIL] scenario_id={scenario_id} ranking mismatch", file=sys.stderr)
            print(f"expected_top1={golden.topk_keys[:1]}  got_top1={result_keys[:1]}", file=sys.stderr)
            if args.fail_fast:
                raise SystemExit(2)
            continue

        ok += 1
        print("[OK] ranking matches golden")

    if args.write_goldens:
        _write_goldens(goldens_path, new_goldens)
        print(f"\nWrote goldens: {goldens_path}")

    print("\n" + "-" * 60)
    print(f"Done. ok={ok} fail={fail}")
    print("-" * 60)

    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
