# src/Shopping_assistant/color/deltae.py
from __future__ import annotations

import numpy as np

# Vectorized-ish CIEDE2000 implementation (single anchor vs array of Lab)
# Assumes Lab in D65, L in [0..100].
def delta_e_ciede2000(
    L1: np.ndarray, a1: np.ndarray, b1: np.ndarray,
    L2: float, a2: float, b2: float,
) -> np.ndarray:
    L1 = np.asarray(L1, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    b1 = np.asarray(b1, dtype=float)

    L2 = float(L2)
    a2 = float(a2)
    b2 = float(b2)

    # Helper
    def _hp(ap, b):
        h = np.degrees(np.arctan2(b, ap)) % 360.0
        return h

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)

    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar**7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0**7 + 1e-12)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = (L1 - L2)
    dCp = (C1p - C2p)

    dhp = h1p - h2p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)

    # mean hue
    hsum = h1p + h2p
    habp = np.where(np.abs(h1p - h2p) > 180.0, hsum + 360.0, hsum)
    habp = 0.5 * habp
    habp = habp % 360.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(habp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * habp))
        + 0.32 * np.cos(np.radians(3.0 * habp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * habp - 63.0))
    )

    dtheta = 30.0 * np.exp(-((habp - 275.0) / 25.0) ** 2)
    Rc = 2.0 * np.sqrt((Cbp**7) / (Cbp**7 + 25.0**7 + 1e-12))

    Sl = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T

    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    # kL=kC=kH=1
    dE = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE
