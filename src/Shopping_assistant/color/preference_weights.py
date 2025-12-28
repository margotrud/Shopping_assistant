# Shopping_assistant/color/preference_weights.py
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_JSON = PROJECT_ROOT / "data" / "models" / "color_preference_weights.json"

with WEIGHTS_JSON.open("r", encoding="utf-8") as f:
    _payload = json.load(f)

W_L = _payload["weights"]["w_L"]
W_C = _payload["weights"]["w_C"]
W_H = _payload["weights"]["w_H"]
