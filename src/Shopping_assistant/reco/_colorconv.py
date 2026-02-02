# src/Shopping_assistant/reco/_colorconv.py
from __future__ import annotations

import colorsys
import numpy as np


def _hex_to_rgb01(hx: str):
    if not isinstance(hx, str):
        return None
    s = hx.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return None


def _hex_to_hsv_h_deg(hx: str):
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None
    r, g, b = rgb01
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360.0)


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _hex_to_lab(hx: str):
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None

    r, g, b = rgb01
    r, g, b = _srgb01_to_linear(r), _srgb01_to_linear(g), _srgb01_to_linear(b)

    # sRGB D65
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return (float(L), float(a), float(b2))


def _circ_dist_deg(a_deg: np.ndarray, b0_deg: float) -> np.ndarray:
    d = (a_deg - float(b0_deg) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _circular_diff_deg(h: np.ndarray, h0: float) -> np.ndarray:
    return (h - float(h0) + 180.0) % 360.0 - 180.0


def _circular_abs_diff_deg(h: np.ndarray, h0: float) -> np.ndarray:
    return np.abs(_circular_diff_deg(h, h0))
