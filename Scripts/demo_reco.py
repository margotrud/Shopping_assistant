# Scripts/demo_reco.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Shopping_assistant.color import scoring as color_scoring
from Shopping_assistant.io.assets import load_assets
from Shopping_assistant.reco.recommend import recommend_from_text


def _load_assets_defaults():
    return load_assets(
        enriched_csv=Path(color_scoring._default_enriched_for_scripts_only()),
        prototypes_csv=Path(color_scoring._default_prototypes_for_scripts_only()),
        assignments_csv=Path(color_scoring._default_assignments_for_scripts_only()),
        calibration_json=Path(color_scoring._default_calibration_for_scripts_only()),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default="I want a deep red lipstick but not too bright")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", type=str, default="data/reports/demo_topk.csv")
    args = p.parse_args()

    assets = _load_assets_defaults()
    df = recommend_from_text(args.text, assets=assets, topk=args.topk, debug=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print a compact view (best-effort columns)
    preferred = [c for c in ["score", "brand", "product_name", "shade_name", "url", "cluster_id"] if c in df.columns]
    view = df[preferred].head(10) if preferred else df.head(10)
    print(view.to_string(index=False))
    print(f"\n[demo] wrote {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
