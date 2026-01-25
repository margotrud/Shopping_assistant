from __future__ import annotations

from pathlib import Path

from Shopping_assistant.color import scoring as color_scoring
from Shopping_assistant.io.assets import load_assets
from Shopping_assistant.reco.recommend import recommend_from_text


def test_recommend_from_text_contract_smoke():
    assets = load_assets(
        enriched_csv=Path(color_scoring._default_enriched_for_scripts_only()),
        calibration_json=Path(color_scoring._default_calibration_for_scripts_only()),
    )

    df = recommend_from_text("I want a red lipstick", assets=assets, topk=10)

    assert len(df) > 0
    assert ("score_total" in df.columns) or ("score" in df.columns)
    assert "rank" in df.columns
    assert "shade_id" in df.columns
