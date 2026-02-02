# src/Shopping_assistant/reco/_constants.py
from __future__ import annotations

import re
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_POOL_TOPN = 1200
_MAX_POOL_TOPN = 2000

# Hue-family fallback (generic; restrict pool for low-chroma anchors; MUST NOT rewrite anchor)
_HUE_FALLBACK_BAND_DEG = 28.0
_HUE_FALLBACK_MIN_POOL_N = 20  # used as "support is weak" threshold (coverage gate)
_HUE_FALLBACK_CHROMA_Q = 0.50  # kept for debug/telemetry; not used to rewrite anchor
_HUE_FALLBACK_ANCHOR_C_MIN = 55.0

# Lexicon anchor selection (generic)
_LEX_TOPK = 10
_LEX_FUZZY_CUTOFF = 60.0
_LEX_SCORE_EPS = 0.03  # keep near-best score candidates when score is available

# within-family constraint gating
_FAMILY_P_MIN = 0.55
_FAMILY_P_HI = 0.75
_FAMILY_FLOOR = 0.35

# quantiles supported by label_distributions (prevents KeyError like 0.65 missing)
_ALLOWED_Q = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)

# Adaptive pool fallback when strict ΔE yields empty pool
_POOL_FALLBACK_MIN_N = 10
_POOL_FALLBACK_DE00_STEPS = (0.0, 2.0, 4.0, 8.0, 12.0)  # added to thr_used
_POOL_FALLBACK_DE00_CAP = 30.0

# High-chroma hue-band fallback when ΔE cannot find any support
_HUE_FAILSAFE_BAND_DEG = 32.0

# Naming-prob pool fallback (when anchor has weak dataset support at strict ΔE)
# No hardcoding: we use p_<label> if it exists in the parquet.
_NAMING_POOL_MIN_N = 20
_NAMING_POOL_TOPN = 180
_NAMING_POOL_P_MIN = 0.20

# Domain anchor (dataset-driven) using naming probs p_<label>
_DOMAIN_ANCHOR_P_MIN = 0.55
_DOMAIN_ANCHOR_MIN_N = 40
_DOMAIN_ANCHOR_TOPN = 250

# Domain-anchor FIX (ONLY for these labels; must not impact "perfect" colors)
_DOMAIN_ANCHOR_FIX_LABELS = {"purple", "violet", "brown"}
_DOMAIN_ANCHOR_HUE_BAND_DEG = 28.0  # generic hue window around reference hue (HSV)
_DOMAIN_ANCHOR_HUE_TRIM_Q = 0.10  # generic trimming (used for brown only)

# =============================================================================
# PRODUCT ELIGIBILITY FILTER
# =============================================================================

_BAD_PRODUCT_RE = re.compile(
    r"(?:\bclear\b|\buniversal\b|\brecharge\b|\brefill\b|\br?echargeable\b|\bécrin\b|\becrin\b|\bétui\b|\bcase\b|\bcap\b|\bmarbre\b|\bstrass\b)",
    re.I,
)
