# tests/qa_plain_color_bulk.py
import os
import math
import numpy as np
import pandas as pd

from Shopping_assistant.reco.recommend import recommend_from_text
from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

TOPK = 10
MAX_COLORS = 40
EXTREME_Q = 0.10
ANGLE_MED_FAIL = 35.0  # degrees


# ---------- colors helpers (no external deps) ----------

def _hex_to_rgb01(hx: str):
    s = hx.strip().lstrip("#")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _hex_to_lab(hx: str):
    r, g, b = _hex_to_rgb01(hx)
    r, g, b = _srgb01_to_linear(r), _srgb01_to_linear(g), _srgb01_to_linear(b)

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return float(L), float(a), float(b2)


def _anchor_angle_from_lex(lex, color: str):
    try:
        res = lex.resolve(color, topk=1, fuzzy_cutoff=75.0, use_semantic=False)
    except Exception:
        res = []
    if not res:
        return None

    hx = getattr(res[0], "hex", None)
    if not isinstance(hx, str) or not hx:
        return None

    _, a0, b0 = _hex_to_lab(hx)
    return math.degrees(math.atan2(b0, a0))


# ---------- main QA ----------

def main():
    path = os.environ.get("SA_ENRICHED_CSV_PATH")
    assert path, "SA_ENRICHED_CSV_PATH not set"

    inv = pd.read_csv(path)
    for c in ["L_lab", "a_lab", "b_lab"]:
        assert c in inv.columns, inv.columns

    inv = inv.copy()
    inv["C_lab"] = np.sqrt(inv["a_lab"] ** 2 + inv["b_lab"] ** 2)

    C_hi = inv["C_lab"].quantile(1 - EXTREME_Q)
    L_lo = inv["L_lab"].quantile(EXTREME_Q)
    L_hi = inv["L_lab"].quantile(1 - EXTREME_Q)

    lex = load_default_lexicon()

    names = []
    for e in getattr(lex, "entries", []) or []:
        name = getattr(e, "canonical", None) or getattr(e, "name", None)
        if isinstance(name, str) and name and name.isascii():
            names.append(name.strip().lower())

    if not names:
        names = [
            "red", "pink", "purple", "beige", "brown", "terracotta",
            "mauve", "berry", "coral", "nude", "plum", "wine", "rose", "brick", "burgundy",
        ]

    names = sorted(set(names))[:MAX_COLORS]

    rows = []
    for color in names:
        df = recommend_from_text(f"I want a {color} lipstick", topk=TOPK, debug=False)
        if df is None or df.empty:
            rows.append((color, "EMPTY", 1.0, 1.0, 999.0, True))
            continue

        df = df.copy()
        df["C_lab"] = np.sqrt(df["a_lab"] ** 2 + df["b_lab"] ** 2)

        c_ext = float((df["C_lab"] > C_hi).mean())
        l_ext = float(((df["L_lab"] < L_lo) | (df["L_lab"] > L_hi)).mean())

        # Prefer anchor emitted by recommend() if present; fallback to lex hex->Lab anchor.
        drift = float("nan")
        if {"_anchor_a_lab", "_anchor_b_lab"}.issubset(df.columns):
            try:
                a0 = float(df["_anchor_a_lab"].iloc[0])
                b0 = float(df["_anchor_b_lab"].iloc[0])
                anchor_ang = math.degrees(math.atan2(b0, a0))
                top_angles = np.degrees(np.arctan2(df["b_lab"].values, df["a_lab"].values))
                deltas = np.abs((top_angles - anchor_ang + 180.0) % 360.0 - 180.0)
                drift = float(np.median(deltas))
            except Exception:
                drift = float("nan")
        else:
            anchor_ang = _anchor_angle_from_lex(lex, color)
            if anchor_ang is not None:
                top_angles = np.degrees(np.arctan2(df["b_lab"].values, df["a_lab"].values))
                deltas = np.abs((top_angles - anchor_ang + 180.0) % 360.0 - 180.0)
                drift = float(np.median(deltas))

        low_cov = bool(df.get("_low_color_coverage", pd.Series([False])).iloc[0])

        rows.append((color, "OK", c_ext, l_ext, drift, low_cov))

    out = pd.DataFrame(
        rows,
        columns=[
            "colors",
            "status",
            "C_extreme",
            "L_extreme",
            "med_abs_hue_drift_deg",
            "low_coverage",
        ],
    )

    print("=== BULK PLAIN COLOR QA ===")
    print(out.head(50).to_string(index=False))

    fails = out[
        (out["status"] != "OK")
        | (out["C_extreme"] >= 0.25)
        | (out["L_extreme"] >= 0.25)
        | (
            (out["low_coverage"] == False)
            & (out["med_abs_hue_drift_deg"].notna())
            & (out["med_abs_hue_drift_deg"] >= ANGLE_MED_FAIL)
        )
    ]

    if not fails.empty:
        print("\nFAILS:")
        print(fails.to_string(index=False))
        raise SystemExit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
