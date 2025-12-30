# src/Shopping_assistant/nlp/resolve/axis_projection.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

from Shopping_assistant.nlp.schema import Axis, Direction, Strength, Polarity
from Shopping_assistant.nlp.resolve.preference_resolver import ResolvedPreference


# -----------------------------
# Axis intent (post-resolution)
# -----------------------------

@dataclass(frozen=True, slots=True)
class AxisIntent:
    """
    Does:
        Represent a resolved intent on a single semantic axis,
        without numeric thresholds.

    Important:
        polarity here must reflect the *constraint clause polarity* (LIKE/DISLIKE),
        not the target bucket polarity (liked/disliked target). The bucket only affects scoping.
    """
    axis: Axis
    direction: Direction
    strength: Strength
    polarity: Polarity
    source: str  # "target" | "global"


# -----------------------------
# Projection
# -----------------------------

def project_axes(pref: ResolvedPreference) -> Dict[Axis, List[AxisIntent]]:
    """
    Does:
        Project resolved preferences into axis-level intents,
        without assigning numeric values.
    """
    intents: Dict[Axis, List[AxisIntent]] = {}

    for bucket, bucket_polarity in (
        (pref.liked, Polarity.LIKE),
        (pref.disliked, Polarity.DISLIKE),
    ):
        for target in bucket:
            _collect_constraints(
                intents=intents,
                constraints=list(target.constraints),
                fallback_polarity=bucket_polarity,
                source="target",
            )

    _collect_constraints(
        intents=intents,
        constraints=[gc.constraint for gc in pref.global_constraints],
        fallback_polarity=Polarity.LIKE,
        source="global",
    )

    return intents


# -----------------------------
# Helpers
# -----------------------------

def _polarity_from_constraint_meta(c: Any) -> Polarity:
    meta = getattr(c, "meta", None)
    if not isinstance(meta, dict):
        return Polarity.UNKNOWN

    v = meta.get("clause_polarity")
    if not isinstance(v, str):
        return Polarity.UNKNOWN

    vv = v.strip().lower()
    if vv == "like":
        return Polarity.LIKE
    if vv == "dislike":
        return Polarity.DISLIKE
    if vv == "neutral":
        return Polarity.NEUTRAL
    return Polarity.UNKNOWN


def _canon(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


@lru_cache(maxsize=1)
def _axis_classifier():
    """
    Does:
        Build the embedding-based axis classifier if deps are installed.

    Important:
        If sentence-transformers isn't available, returns None (no-op bridges).
    """
    try:
        # src/Shopping_assistant/nlp/axes/classifier.py
        from Shopping_assistant.nlp.axes.classifier import make_axis_classifier_fn  # type: ignore
    except Exception:
        return None

    try:
        return make_axis_classifier_fn(debug=False)
    except Exception:
        return None


def _bridge_neon_flashy_from_evidence(c: Any, intent: AxisIntent) -> List[AxisIntent]:
    """
    Does:
        Add additional intents for neon/flashy ONLY when evidence is semantically classified
        as VIBRANCY/SATURATION/CLARITY via embeddings.

    Fix (required):
        If the classifier returns CLARITY for neon/flashy evidence, re-route to VIBRANCY.
        Rationale: in this project, "neon/flashy" must cap sat_eff (VIBRANCY). Capping CLARITY/colorfulness
        is not sufficient and often ineffective (e.g. snapping to very_high).

    Rules:
        - Only for LOWER direction (i.e., "not neon/flashy").
        - Keep bridge strength <= MED to avoid over-pruning.
        - Never creates new intents if classifier is unavailable.
    """
    if intent.direction != Direction.LOWER:
        return []

    evidence = getattr(c, "evidence", "") or ""
    if not isinstance(evidence, str) or not evidence.strip():
        return []

    clf = _axis_classifier()
    if clf is None:
        return []

    pred = clf(_canon(evidence), context="")
    ax = getattr(pred, "axis", None)
    if ax is None:
        return []

    # ---- critical re-route ----
    # neon/flashy should not land on CLARITY; if it does, treat it as VIBRANCY.
    if ax == Axis.CLARITY:
        ax = Axis.VIBRANCY
    # --------------------------

    if ax not in (Axis.VIBRANCY, Axis.SATURATION, Axis.CLARITY):
        return []

    s = intent.strength
    if s == Strength.STRONG:
        s = Strength.MED

    out: List[AxisIntent] = []

    # VIBRANCY => cap sat_eff
    if ax == Axis.VIBRANCY:
        out.append(
            AxisIntent(
                axis=Axis.VIBRANCY,
                direction=Direction.LOWER,
                strength=s,
                polarity=intent.polarity,
                source=intent.source,
            )
        )

    # SATURATION => cap sat_hsl (if used downstream)
    if ax == Axis.SATURATION:
        out.append(
            AxisIntent(
                axis=Axis.SATURATION,
                direction=Direction.LOWER,
                strength=s,
                polarity=intent.polarity,
                source=intent.source,
            )
        )

    # CLARITY => cap colorfulness (only when classifier truly means clarity, not neon)
    if ax == Axis.CLARITY:
        out.append(
            AxisIntent(
                axis=Axis.CLARITY,
                direction=Direction.LOWER,
                strength=s,
                polarity=intent.polarity,
                source=intent.source,
            )
        )

    return out


def _collect_constraints(
    *,
    intents: Dict[Axis, List[AxisIntent]],
    constraints: List,
    fallback_polarity: Polarity,
    source: str,
) -> None:
    for c in constraints:
        cp = _polarity_from_constraint_meta(c)
        pol = cp if cp != Polarity.UNKNOWN else fallback_polarity

        intent = AxisIntent(
            axis=c.axis,
            direction=c.direction,
            strength=c.strength,
            polarity=pol,
            source=source,
        )
        intents.setdefault(c.axis, []).append(intent)

        # dynamic bridge from evidence using embeddings (no static token lists)
        for bridged in _bridge_neon_flashy_from_evidence(c, intent):
            intents.setdefault(bridged.axis, []).append(bridged)
