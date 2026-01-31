# Scripts/colors/qa_constraints.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from Shopping_assistant.color.constraints import (
    ConstraintSpec,
    constraint_factor,
    load_label_distributions,
)

ROOT = Path(__file__).resolve().parents[2]
CHIPS_PATH = ROOT / "data" / "colors" / "chips_with_naming_probs.parquet"
DISTS_PATH = ROOT / "data" / "colors" / "label_distributions.json"

P_MIN = 0.55

CASES: list[tuple[str, list[ConstraintSpec]]] = [
    # "bright red" -> higher chroma within the red family
    ("red", [ConstraintSpec(axis="C", direction="above", strength=1.0, q_lo=0.50, q_hi=0.75)]),
    # "soft pink" -> lower chroma within the pink family
    ("pink", [ConstraintSpec(axis="C", direction="below", strength=1.0, q_lo=0.25, q_hi=0.50)]),
    # "dark taupe" -> lower lightness within the taupe family
    ("taupe", [ConstraintSpec(axis="L", direction="below", strength=1.0, q_lo=0.25, q_hi=0.50)]),
    # "light coral" -> higher lightness within the coral family
    ("coral", [ConstraintSpec(axis="L", direction="above", strength=1.0, q_lo=0.50, q_hi=0.75)]),
]


def _specs_str(specs: list[ConstraintSpec]) -> str:
    return ",".join(f"{s.axis}:{s.direction}@{s.q_lo}-{s.q_hi}*{s.strength:.2f}" for s in specs)


def main() -> None:
    chips = pd.read_parquet(CHIPS_PATH)
    dists = load_label_distributions(DISTS_PATH)

    for label, specs in CASES:
        f = constraint_factor(chips, label=label, specs=specs, dists=dists, p_min=P_MIN)
        sub = chips.assign(f=f)

        pcol = f"p_{label}"
        if pcol not in sub.columns:
            raise KeyError(f"missing column: {pcol}")

        # Restrict to likely members; otherwise membership gate makes f~=1 everywhere.
        sub = sub[sub[pcol] >= P_MIN].copy()

        print("\n", label, _specs_str(specs), f"(n={len(sub)})")

        if len(sub) == 0:
            print("  [WARN] no members above threshold. Lower P_MIN or inspect p distribution.")
            continue

        # Weighted view (closer to production rerank): keep family membership in the ranking.
        sub["f_weighted"] = sub[pcol].astype(float) * sub["f"].astype(float)

        top = sub.sort_values("f_weighted", ascending=False).head(10)
        print(top[["chip_hex", "L", "C", pcol, "f", "f_weighted"]].to_string(index=False))

        bot = sub.sort_values("f_weighted", ascending=True).head(5)
        print("\n  worst:")
        print(bot[["chip_hex", "L", "C", pcol, "f", "f_weighted"]].to_string(index=False))


if __name__ == "__main__":
    main()
