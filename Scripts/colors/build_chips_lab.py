# Scripts/build_chips_lab.py

from pathlib import Path
import pandas as pd
import numpy as np
from Shopping_assistant.reco._colorconv import _hex_to_lab

# === CONFIG ===
INPUT_CSV = Path("data/enriched_data/Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv")
OUT_PARQUET = Path("data/colors/chips_lab.parquet")

# === COLOR UTILS (UNE implémentation) ===
def hex_to_rgb01(hex_str: str):
    hex_str = str(hex_str).lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return r, g, b

def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def rgb_to_xyz(r, g, b):
    r, g, b = map(srgb_to_linear, (r, g, b))
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return x, y, z

def xyz_to_lab(x, y, z):
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


# === MAIN ===
def main():
    df = pd.read_csv(INPUT_CSV)

    # Adapter le nom exact si besoin
    df = df[["chip_hex"]].dropna().drop_duplicates()

    labs = df["chip_hex"].apply(_hex_to_lab)
    df[["L", "a", "b"]] = pd.DataFrame(labs.tolist(), index=df.index)

    df["C"] = np.sqrt(df["a"] ** 2 + df["b"] ** 2)
    df["h"] = (np.degrees(np.arctan2(df["b"], df["a"])) + 360) % 360

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"[OK] wrote {len(df)} rows to {OUT_PARQUET}")

if __name__ == "__main__":
    main()
