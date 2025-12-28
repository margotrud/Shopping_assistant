# src/Shopping_assistant/nlp/axis_mapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from Shopping_assistant.nlp.schema import Axis


_AXIS_DESCRIPTIONS: Dict[Axis, str] = {
    Axis.BRIGHTNESS: "Brightness/lightness: how light or bright versus dark the color is.",
    Axis.SATURATION: "Saturation/chroma: how colorful, intense, or muted versus gray the color is.",
    Axis.VIBRANCY: "Vibrancy/neon: how vivid, electric, fluorescent, or neon-like the color feels.",
    Axis.DEPTH: "Depth: how deep, rich, dark, or heavy the color feels (opposite of light/airy).",
    Axis.CLARITY: "Clarity: how clean/crisp versus muddy/dull/dirty the color looks.",
}


@dataclass(frozen=True, slots=True)
class AxisMatch:
    axis: Axis
    score: float
    rationale: str


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AxisMapper:
    """
    Does:
        Map free-text adjectives (e.g. "vivid", "washed-out") to one of {brightness,saturation,vibrancy,depth,clarity}
        using transformer embeddings against axis descriptions.
    """

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # local import (fast fail if missing)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._axis_vecs: Dict[Axis, np.ndarray] = {}
        self._build_axis_vectors()

    def _build_axis_vectors(self) -> None:
        axes = list(_AXIS_DESCRIPTIONS.keys())
        texts = [f"Axis: {a.value}. {_AXIS_DESCRIPTIONS[a]}" for a in axes]
        vecs = self.model.encode(texts, normalize_embeddings=False)
        self._axis_vecs = {a: vecs[i] for i, a in enumerate(axes)}

    def map_adj_to_axis(
        self,
        adj: str,
        *,
        threshold: float = 0.35,
        return_topk: int = 1,
    ) -> Optional[AxisMatch] | List[AxisMatch]:
        """
        Does:
            Return best axis match for an adjective, or None if confidence below threshold.
        """
        q = f"Adjective describing a color preference: {adj}"
        qv = self.model.encode([q], normalize_embeddings=False)[0]

        scored: List[AxisMatch] = []
        for axis, av in self._axis_vecs.items():
            s = _cosine_sim(qv, av)
            scored.append(AxisMatch(axis=axis, score=s, rationale=_AXIS_DESCRIPTIONS[axis]))

        scored.sort(key=lambda x: x.score, reverse=True)

        if return_topk > 1:
            return scored[:return_topk]

        best = scored[0]
        if best.score < threshold:
            return None
        return best
