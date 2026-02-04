# tests/qa_color_coverage.py
import math
import os
import pandas as pd
import numpy as np
from Shopping_assistant.reco._colorconv import _hex_to_lab

# --- config ---
ANGLE_DEG = 25.0
TOPK = 20
N_MIN_FACTOR = 5  # n_close >= 5 * topk => OK

COLORS = {
    "red": "#d0312d",
    "pink": "#f69acd",
    "purple": "#a32cc4",
    "beige": "#ecdd9a",
    "terracotta": "#e2725b",
    "brown": "#231709",
    "mauve": "#7a4a88",
    "berry": "#241570",
}

# --- colors utils (NO external deps) ---
def hex_to_rgb01(h):
    h = h.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )

def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def rgb01_to_xyz(rgb):
    r, g, b = map(_srgb_to_linear, rgb)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return x, y, z

def _f(t):
    return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)

def xyz_to_lab(xyz):
    x, y, z = xyz
    x /= 0.95047
    z /= 1.08883
    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b

# --- load inventory ---
path = os.environ.get("SA_ENRICHED_CSV_PATH")
assert path, "SA_ENRICHED_CSV_PATH not set"
df = pd.read_csv(path)

assert {"a_lab", "b_lab"}.issubset(df.columns), df.columns

inv_angles = np.degrees(np.arctan2(df["b_lab"], df["a_lab"]))

# --- test ---
print("=== COLOR COVERAGE TEST ===")
for color, hx in COLORS.items():
    _, a, b = _hex_to_lab(hx)
    anchor_angle = math.degrees(math.atan2(b, a))

    delta = np.abs((inv_angles - anchor_angle + 180) % 360 - 180)
    n_close = int((delta <= ANGLE_DEG).sum())

    status = "PASS" if n_close >= N_MIN_FACTOR * TOPK else "FAIL"

    print(
        f"{color:12s} | n_close={n_close:4d} | "
        f"required>={N_MIN_FACTOR*TOPK:3d} | {status}"
    )
