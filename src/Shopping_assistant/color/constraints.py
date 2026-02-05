# src/Shopping_assistant/color/constraints.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

Axis = Literal["L", "C", "a", "b", "h"]
Direction = Literal["above", "below", "near", "toward"]


@dataclass(frozen=True)
class LabelDistributions:
    """Does: hold empirical label distributions used for family-based constraints.
    Used by: naming-probability scoring and family-aware constraint application.
    """
    n: int
    p_threshold: float
    L: Mapping[str, float]
    C: Mapping[str, float]
    h: Mapping[str, float]
    a: Optional[Mapping[str, float]] = None
    b: Optional[Mapping[str, float]] = None


@dataclass(frozen=True)
class ConstraintSpec:
    """Does: represent a normalized constraint applied during scoring.
    Encodes axis, direction, quantile bounds, and strength.
    """
    axis: Axis
    direction: Direction
    strength: float = 1.0

    # for above/below
    q_lo: float = 0.50
    q_hi: float = 0.75

    # for near/toward
    q_target: float = 0.50
    q_span: float = 0.25  # +/- span in quantile space


def load_label_distributions(path: Path) -> Dict[str, LabelDistributions]:
    """
    Load per-label empirical distributions for within-family constraints.
    Args: path to label distribution JSON.
    Returns: dict of label -> distribution statistics.
    Raises: FileNotFoundError or ValueError on invalid data.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, LabelDistributions] = {}

    def _to_float_dict(x) -> Mapping[str, float]:
        if not isinstance(x, dict):
            return {}
        return {str(k): float(v) for k, v in x.items()}

    for lbl, v in raw.items():
        if not isinstance(v, dict):
            continue
        out[lbl] = LabelDistributions(
            n=int(v.get("n", 0) or 0),
            p_threshold=float(v.get("p_threshold", 0.0) or 0.0),
            L=_to_float_dict(v.get("L")),
            C=_to_float_dict(v.get("C")),
            h=_to_float_dict(v.get("h")),
            a=_to_float_dict(v.get("a")) if "a" in v else None,
            b=_to_float_dict(v.get("b")) if "b" in v else None,
        )
    return out

def _nearest_quantile_key(d: dict, q: float) -> str:
    # d keys are strings like "0.05", "0.5", "0.75"
    keys = []
    for k in d.keys():
        try:
            keys.append(float(k))
        except Exception:
            continue
    if not keys:
        raise KeyError("no numeric quantile keys available")
    qq = float(q)
    best = min(keys, key=lambda x: abs(x - qq))
    return f"{best:g}"

def _q(d: dict, q: float) -> float:
    k = f"{float(q):g}"
    if k in d:
        return float(d[k])

    # fallback: snap to nearest available quantile
    kn = _nearest_quantile_key(d, float(q))
    return float(d[kn])


def _axis_quantiles(dist: LabelDistributions, axis: Axis) -> Mapping[str, float]:
    if axis == "L":
        return dist.L
    if axis == "C":
        return dist.C
    if axis == "h":
        return dist.h
    if axis == "a":
        if dist.a is None:
            raise KeyError("label_distributions missing 'a' quantiles (rebuild json with a)")
        return dist.a
    if axis == "b":
        if dist.b is None:
            raise KeyError("label_distributions missing 'b' quantiles (rebuild json with b)")
        return dist.b
    raise ValueError(axis)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-z))


def _scale_from_iqr(q25: float, q75: float, *, min_scale: float = 1e-3) -> float:
    # Robust scale from IQR; makes scoring continuous (no saturation at hi/lo).
    return float(max(min_scale, 0.5 * abs(q75 - q25)))


def _score_above_sigmoid(x: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return _sigmoid((x - center) / max(scale, 1e-6))


def _score_below_sigmoid(x: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return _sigmoid((center - x) / max(scale, 1e-6))


def _band_score_near(x: np.ndarray, center: float, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.ones_like(x, dtype=float)
    d = np.abs(x - center) / radius
    t = 1.0 - np.clip(d, 0.0, 1.0)
    return _smoothstep(t)


def _circ_dist_deg(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _band_score_toward_h(h: np.ndarray, target: float, radius_deg: float) -> np.ndarray:
    if radius_deg <= 0:
        return np.ones_like(h, dtype=float)
    d = _circ_dist_deg(h, target) / radius_deg
    t = 1.0 - np.clip(d, 0.0, 1.0)
    return _smoothstep(t)


def _gate_from_membership(p: np.ndarray, p_min: float, p_hi: float) -> np.ndarray:
    if p_hi <= p_min:
        p_hi = min(0.90, p_min + 0.10)
    t = (p - p_min) / (p_hi - p_min)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t)


def constraint_factor(
    chips: pd.DataFrame,
    *,
    label: str,
    specs: Sequence[ConstraintSpec],
    dists: Mapping[str, LabelDistributions],
    p_col_prefix: str = "p_",
    p_min: float = 0.55,
    p_hi: float = 0.75,
    floor: float = 0.55,
) -> pd.Series:
    """
    Compute multiplicative score factor from within-family constraints.
    Uses label probabilities and geometric constraints in Lab space.
    Args: df with Lab + p_label, constraint specs, distributions.
    Returns: float array of per-row multiplicative factors.
    """

    if label not in dists:
        raise KeyError(f"label '{label}' missing from distributions")

    dist = dists[label]
    pcol = f"{p_col_prefix}{label}"
    if pcol not in chips.columns:
        raise KeyError(f"missing probability column: {pcol}")

    p = chips[pcol].to_numpy(dtype=float)
    gate = _gate_from_membership(p, p_min=p_min, p_hi=p_hi)

    col_cache: Dict[str, np.ndarray] = {}
    for ax in ("L", "C", "h", "a", "b"):
        if ax in chips.columns:
            col_cache[ax] = chips[ax].to_numpy(dtype=float)

    factor = np.ones(len(chips), dtype=float)

    for spec in specs:
        strength = float(np.clip(spec.strength, 0.0, 1.0))
        if strength <= 0:
            continue

        if spec.axis not in col_cache:
            raise KeyError(f"missing required column '{spec.axis}' for constraint axis={spec.axis}")

        x = col_cache[spec.axis]
        qdict = _axis_quantiles(dist, spec.axis)

        if spec.direction in ("above", "below"):
            # Continuous scoring using robust scale (IQR) to avoid saturation (binary f).
            q25 = _q(qdict, 0.25)
            q50 = _q(qdict, 0.50)
            q75 = _q(qdict, 0.75)

            scale = _scale_from_iqr(q25, q75)

            # Use q_lo/q_hi to adjust sharpness (smaller band => sharper)
            band = max(1e-6, abs(_q(qdict, float(spec.q_hi)) - _q(qdict, float(spec.q_lo))))
            k = max(0.6, min(2.5, float(band / max(scale, 1e-6))))  # clamp

            if spec.direction == "above":
                s = _sigmoid(k * (x - q50) / max(scale, 1e-6))
            else:
                s = _sigmoid(k * (q50 - x) / max(scale, 1e-6))


        elif spec.direction == "near":
            qt = float(spec.q_target)
            span = float(np.clip(spec.q_span, 0.01, 0.49))
            qlo = max(0.01, qt - span)
            qhi = min(0.99, qt + span)
            center = _q(qdict, qt)
            radius = max(1e-6, 0.5 * abs(_q(qdict, qhi) - _q(qdict, qlo)))
            s = _band_score_near(x, center=center, radius=radius)

        elif spec.direction == "toward":
            qt = float(spec.q_target)
            span = float(np.clip(spec.q_span, 0.01, 0.49))
            qlo = max(0.01, qt - span)
            qhi = min(0.99, qt + span)

            target = _q(qdict, qt)
            if spec.axis == "h":
                radius = max(10.0, 0.5 * abs(_q(qdict, qhi) - _q(qdict, qlo)))
                s = _band_score_toward_h(x, target=target, radius_deg=radius)
            else:
                radius = max(1e-6, 0.5 * abs(_q(qdict, qhi) - _q(qdict, qlo)))
                s = _band_score_near(x, center=target, radius=radius)

        else:
            raise ValueError(f"unknown direction: {spec.direction}")

        # soften: never hard-zero; strength controls bite.
        s_soft = float(floor) + (1.0 - float(floor)) * s
        s_mix = (1.0 - strength) + strength * s_soft
        factor *= s_mix

    # gate blend: if low membership, factor should revert toward 1
    factor = 1.0 - gate + gate * factor
    return pd.Series(np.clip(factor, 0.0, 1.0), index=chips.index, name="constraint_factor")
