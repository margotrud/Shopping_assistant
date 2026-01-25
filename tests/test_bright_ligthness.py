import os

os.environ["SA_ENRICHED_CSV_PATH"]="data/enriched_data/Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
os.environ["SA_CALIBRATION_JSON_PATH"]="data/models/color_scoring_calibration.json"

from Shopping_assistant.reco.recommend import recommend_from_text


def test_bright_increases_lightness_vs_plain_red():
    topk = 20
    plain = recommend_from_text("I want a red lipstick", topk=topk, debug=False)
    bright = recommend_from_text("I want a bright red lipstick", topk=topk, debug=False)

    m_plain = float(plain["light_hsl"].astype(float).median())
    m_bright = float(bright["light_hsl"].astype(float).median())

    assert m_bright > m_plain
