# src/Shopping_assistant/nlp/axes/predictor.py
from __future__ import annotations

from functools import lru_cache

from Shopping_assistant.nlp.axes.classifier import AxisPred, make_axis_classifier_fn


@lru_cache(maxsize=8)
def _get_axis_classifier(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    min_sim: float = 0.35,
    min_margin: float = 0.08,
    debug: bool = False,
):
    """
    Does:
        Provide a single shared axis classifier function across the NLP pipeline.
    """
    return make_axis_classifier_fn(
        model_name=model_name,
        min_sim=min_sim,
        min_margin=min_margin,
        debug=debug,
    )


def predict_axis(
    label: str,
    *,
    context: str = "",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_sim: float = 0.35,
    min_margin: float = 0.08,
    debug: bool = False,
) -> AxisPred:
    """
    Does:
        Predict an Axis for a free-text label with confidence + margin, or axis=None if unsure.
    """
    fn = _get_axis_classifier(model_name, min_sim=min_sim, min_margin=min_margin, debug=debug)
    return fn(label, context=context)


__all__ = ["predict_axis"]
