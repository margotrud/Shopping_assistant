# src/Shopping_assistant/nlp/axes/predictor.py
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable

from Shopping_assistant.nlp.axes.classifier import AxisPred, make_axis_classifier_fn
from Shopping_assistant.nlp.schema import Axis


@lru_cache(maxsize=16)
def _get_axis_classifier(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    min_sim: float = 0.35,
    min_margin: float = 0.08,
    debug: bool = False,
):
    """
    Does:
        Provide a shared axis classifier function across the NLP pipeline.
    """
    return make_axis_classifier_fn(
        model_name=model_name,
        min_sim=min_sim,
        min_margin=min_margin,
        debug=debug,
    )


def _clean_term(s: str) -> str | None:
    s = (s or "").replace("_", " ").strip().lower()
    if not s:
        return None
    for ch in s:
        if not (ch.isalpha() or ch == " "):
            return None
    if len(s) < 3 or len(s) > 40:
        return None
    return s


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


@lru_cache(maxsize=4096)
def _wordnet_expand_label(label: str) -> tuple[str, ...]:
    """
    Does:
        Expand a label using WordNet lemma relations (synonyms + derivational/pertainym forms).
    """
    from Shopping_assistant.utils.optional_deps import require

    nltk = require("nltk", extra="nltk", purpose="WordNet expansion for axis prediction")
    wn = getattr(getattr(nltk, "corpus"), "wordnet")

    seed = _clean_term(label) or (label or "").strip().lower()
    if not seed:
        return tuple()

    out: list[str] = [seed]

    for pos in (wn.ADJ, wn.NOUN, wn.ADV):
        for syn in wn.synsets(seed, pos=pos):
            for lemma in syn.lemmas():
                t = _clean_term(lemma.name())
                if t:
                    out.append(t)
                for drf in lemma.derivationally_related_forms():
                    tt = _clean_term(drf.name())
                    if tt:
                        out.append(tt)
                for p in lemma.pertainyms():
                    tt = _clean_term(p.name())
                    if tt:
                        out.append(tt)

    return tuple(_dedupe_keep_order(out))


@lru_cache(maxsize=8)
def _get_st_model(model_name: str):
    from Shopping_assistant.utils.optional_deps import require

    st = require("sentence_transformers", extra="sentence-transformers", purpose="variant similarity filter")
    SentenceTransformer = getattr(st, "SentenceTransformer")
    return SentenceTransformer(model_name)


def _filter_variants_by_similarity(
    label: str,
    variants: list[str],
    *,
    model_name: str,
    min_variant_sim: float,
) -> tuple[list[str], dict[str, float]]:
    """
    Does:
        Keep WordNet variants semantically close to label, and return cosine similarity weights.
        Returns (kept_variants, weights_by_variant).
    """
    import numpy as np

    if not variants:
        return variants, {}

    model = _get_st_model(model_name)
    vecs = model.encode([label] + variants, normalize_embeddings=True)
    sims = np.dot(vecs[1:], vecs[0])

    kept: list[str] = []
    weights: dict[str, float] = {}

    for v, s in zip(variants, sims):
        ss = float(s)
        if ss >= float(min_variant_sim):
            kept.append(v)
            weights[v] = ss

    # Always keep original label with weight 1.0
    weights[label] = 1.0
    kept = [label] + [v for v in kept if v != label]
    return kept, weights


def _with_meta(pred: AxisPred, extra: dict) -> AxisPred:
    meta = {**(pred.meta or {}), **(extra or {})}
    return AxisPred(
        axis=pred.axis,
        confidence=float(pred.confidence),
        label=pred.label,
        margin=float(pred.margin),
        meta=meta,
    )


def _weighted_ranked_vote(
    ranked_by_variant: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    *,
    topk: int,
) -> tuple[str | None, float, float, dict[str, float], dict[str, float]]:
    """
    Does:
        Aggregate full-ranked axis scores across variants using similarity weights (NOT top-1 only).
        For each variant, we add weight*score for axes in its top-k ranked list.
        Returns (best_axis_name|None, best_avg, margin, axis_weighted_sum, axis_weight_sum).
    """
    axis_wsum: dict[str, float] = defaultdict(float)
    axis_w: dict[str, float] = defaultdict(float)

    for v, ranked in ranked_by_variant.items():
        w = float(weights.get(v, 0.0))
        if w <= 0.0:
            continue
        for ax_name, score in (ranked or [])[: max(1, int(topk))]:
            axis_wsum[ax_name] += w * float(score)
            axis_w[ax_name] += w

    if not axis_wsum:
        return None, 0.0, 0.0, {}, {}

    axis_avg = {ax: axis_wsum[ax] / (axis_w[ax] + 1e-12) for ax in axis_wsum.keys()}
    ranked_avg = sorted(axis_avg.items(), key=lambda kv: kv[1], reverse=True)

    best_ax, best_avg = ranked_avg[0]
    second_avg = ranked_avg[1][1] if len(ranked_avg) > 1 else 0.0
    margin = float(best_avg - second_avg)
    return best_ax, float(best_avg), margin, dict(axis_wsum), dict(axis_w)


def predict_axis(
    label: str,
    *,
    context: str = "",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_sim: float = 0.35,
    min_margin: float = 0.08,
    debug: bool = False,
    min_variant_sim: float = 0.60,
    vote_topk: int = 3,
) -> AxisPred:
    """
    Does:
        Predict an Axis for a label.
        If classifier abstains, run WordNet label expansion and decide via similarity-weighted top-K ranked vote.
        If the vote abstains, fall back to raw label top-1 if it clears min_sim (marked ambiguous).
        Exposes raw ranked diagnostics when debug=True.
    """
    # Thresholded classifier (normal path)
    fn = _get_axis_classifier(model_name, min_sim=min_sim, min_margin=min_margin, debug=debug)

    label_clean = (label or "").strip()
    is_single_token = (" " not in label_clean) and ("-" not in label_clean) and bool(label_clean)

    pred0 = fn(label, context=context)
    if pred0.axis is not None:
        return pred0

    if not is_single_token:
        return pred0

    # WordNet variants + similarity weights
    try:
        variants0 = list(_wordnet_expand_label(label_clean))
        variants, weights = _filter_variants_by_similarity(
            label_clean,
            variants0,
            model_name=model_name,
            min_variant_sim=min_variant_sim,
        )
    except Exception as e:
        return _with_meta(pred0, {"wordnet_error": repr(e)}) if debug else pred0

    # Raw classifier: always provides ranked scores
    fn_raw = _get_axis_classifier(model_name, min_sim=0.0, min_margin=0.0, debug=True)

    raw_label_ranked = (fn_raw(label_clean, context=context).meta or {}).get("ranked") or []

    ranked_by_variant: dict[str, list[tuple[str, float]]] = {}
    debug_rows: list[dict] = []

    for v in variants:
        p = fn_raw(v, context=context)
        ranked = (p.meta or {}).get("ranked") or []
        ranked_by_variant[v] = [(a, float(s)) for a, s in ranked]

        if debug:
            debug_rows.append(
                {
                    "variant": v,
                    "weight": float(weights.get(v, 0.0)),
                    "ranked": ranked[: max(6, int(vote_topk))],
                }
            )

    best_name, best_avg, margin, axis_wsum, axis_w = _weighted_ranked_vote(
        ranked_by_variant,
        weights,
        topk=vote_topk,
    )

    vote_abstains = (
        best_name is None
        or float(best_avg) < float(min_sim)
        or float(margin) < float(min_margin)
    )

    if vote_abstains:
        # ---- HARD FALLBACK: never drop a reasonable single-token axis to None.
        # If raw label top1 clears min_sim, return it but mark ambiguous.
        if raw_label_ranked:
            top1_name, top1_score = raw_label_ranked[0]
            top2_score = raw_label_ranked[1][1] if len(raw_label_ranked) > 1 else 0.0
            top1_margin = float(top1_score) - float(top2_score)

            if float(top1_score) >= float(min_sim):
                ax = Axis(top1_name)
                meta = {
                    "source": "raw_label_top1_fallback_ambiguous",
                    "model": model_name,
                    "vote_topk": int(vote_topk),
                    "raw_label_ranked": raw_label_ranked,
                    "vote_best_avg": float(best_avg),
                    "vote_margin": float(margin),
                    "min_variant_sim": float(min_variant_sim),
                    "axis_ambiguous": True,
                }
                if debug:
                    meta.update(
                        {
                            "variants": variants,
                            "variant_weights": weights,
                            "debug_variants_scoring": debug_rows,
                            "vote_axis_weighted_sum": axis_wsum,
                            "vote_axis_weight_sum": axis_w,
                        }
                    )

                return AxisPred(
                    axis=ax,
                    confidence=float(top1_score),
                    label=label,
                    margin=float(top1_margin),
                    meta=meta,
                )

        if debug:
            return _with_meta(
                pred0,
                {
                    "source": "wordnet_weighted_ranked_vote_abstain",
                    "vote_topk": int(vote_topk),
                    "variants": variants,
                    "variant_weights": weights,
                    "raw_label_ranked": raw_label_ranked,
                    "debug_variants_scoring": debug_rows,
                    "vote_axis_weighted_sum": axis_wsum,
                    "vote_axis_weight_sum": axis_w,
                    "vote_best_avg": float(best_avg),
                    "vote_margin": float(margin),
                    "min_variant_sim": float(min_variant_sim),
                },
            )
        return pred0

    ax = Axis(best_name)

    meta = {
        "source": "wordnet_weighted_ranked_vote",
        "model": model_name,
        "vote_topk": int(vote_topk),
        "variants": variants,
        "variant_weights": weights,
        "vote_best_avg": float(best_avg),
        "vote_margin": float(margin),
        "min_variant_sim": float(min_variant_sim),
    }
    if debug:
        meta.update(
            {
                "raw_label_ranked": raw_label_ranked,
                "debug_variants_scoring": debug_rows,
                "vote_axis_weighted_sum": axis_wsum,
                "vote_axis_weight_sum": axis_w,
            }
        )

    return AxisPred(
        axis=ax,
        confidence=float(best_avg),
        label=label,
        margin=float(margin),
        meta=meta,
    )


__all__ = ["predict_axis"]
