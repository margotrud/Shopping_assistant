# streamlit_app/pages/2_Explain_Recommendation.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Page config MUST be first
# -----------------------------
st.set_page_config(page_title="Explain", layout="wide", initial_sidebar_state="collapsed")

try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

# -----------------------------
# Bootstrap: make src importable
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------------
# Shared UI chrome
# -----------------------------
from ui.components import render_hero  # noqa: E402
from ui.nav import top_nav  # noqa: E402
from ui.theme import inject_styles  # noqa: E402

inject_styles()
top_nav(active="Explain")

# -----------------------------
# App-level switches
# -----------------------------
os.environ.setdefault("SA_WARMUP_SEMANTIC", "0")
os.environ.setdefault("SA_INCLUDE_XKCD", "0")
os.environ.setdefault("SA_POLARITY_BACKEND", "lexical")

from ui.bootstrap import warmup_nlp_stack  # noqa: E402
from Shopping_assistant.reco.explain import explain_recommendation  # noqa: E402

# -----------------------------
# Copy helpers
# -----------------------------
_AXIS_HUMAN = {
    "brightness": "Brightness",
    "depth": "Depth",
    "saturation": "Saturation",
    "vibrancy": "Vividness",
    "clarity": "Clarity",
}
_DIR_HUMAN = {"lower": "less", "raise": "more", "higher": "more"}
_AXIS_HINT = {
    "brightness": "avoid too bright / too dark",
    "depth": "deeper vs softer",
    "saturation": "more vs less saturated",
    "vibrancy": "more vivid vs more muted",
    "clarity": "clean vs muted/greyish",
}

_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("nude", [r"\bnude\b", r"\bbeige\b", r"\bneutral\b"]),
    ("red", [r"\bred\b", r"\bscarlet\b", r"\bcrimson\b", r"\bcherry\b"]),
    ("pink", [r"\bpink\b", r"\brose\b", r"\brosy\b", r"\bfuchsia\b"]),
    ("berry", [r"\bberry\b", r"\braspberry\b", r"\bcranberry\b"]),
    ("mauve", [r"\bmauve\b"]),
    ("coral", [r"\bcoral\b"]),
    ("peach", [r"\bpeach\b"]),
    ("brown", [r"\bbrown\b", r"\bchocolate\b", r"\bcocoa\b"]),
    ("plum", [r"\bplum\b", r"\bprune\b"]),
    ("purple", [r"\bpurple\b", r"\bviolet\b", r"\blilac\b"]),
    ("orange", [r"\borange\b", r"\btangerine\b"]),
]

# UI-only leans tags (heuristic from names)
_LEAN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("brown-leaning", ["brown", "chocolate", "cocoa", "coffee", "mocha", "caramel", "toffee", "tan"]),
    ("nude-leaning", ["nude", "beige", "neutral", "sand", "latte", "cream"]),
    ("pink-leaning", ["pink", "rose", "rosy", "fuchsia", "blush"]),
    ("red-leaning", ["red", "scarlet", "crimson", "cherry"]),
    ("berry-leaning", ["berry", "raspberry", "cranberry"]),
    ("peach-leaning", ["peach", "apricot"]),
    ("coral-leaning", ["coral"]),
    ("plum-leaning", ["plum", "prune"]),
    ("purple-leaning", ["purple", "violet", "lilac"]),
    ("orange-leaning", ["orange", "tangerine"]),
]

_EXAMPLE_QUERIES: list[str] = [
    "I want a pink lipstick but not too bright",
    "Looking for a warm nude lipstick for everyday",
    "A deep berry lipstick, not too saturated",
    "A cool-toned mauve, slightly muted",
]


def _human_missing(v: object, *, fallback: str = "Model abstention (no reliable signal)") -> str:
    if v is None:
        return fallback
    s = str(v).strip()
    return s if s else fallback


def _detect_color_family_from_text(text: str) -> str | None:
    t = (text or "").lower()
    for fam, pats in _FAMILY_PATTERNS:
        for pat in pats:
            if re.search(pat, t, flags=re.IGNORECASE):
                return fam
    return None


def _constraints(trace: dict) -> list[dict]:
    nlp = trace.get("nlp", {}) or {}
    cons = nlp.get("constraints", []) if isinstance(nlp, dict) else []
    return [c for c in cons if isinstance(c, dict)]


def _fmt_constraint_human(c: dict) -> str:
    axis_key = c.get("axis")
    axis = _AXIS_HUMAN.get(axis_key, axis_key or "Constraint")
    hint = _AXIS_HINT.get(axis_key, "")
    direction = _DIR_HUMAN.get(c.get("direction"), c.get("direction") or "")
    evidence = (c.get("evidence") or "").strip()

    parts = [f"{axis} ({hint})" if hint else f"{axis}"]
    if direction:
        parts.append(f"→ **{direction}**")

    s = " ".join(parts).strip()
    if evidence:
        s += f' (from “{evidence}”)'
    return s


def _pill(text: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    st.markdown(
        f"""
        <span style="
          display:inline-block;
          padding:4px 10px;
          border-radius:999px;
          border:1px solid rgba(0,0,0,0.10);
          background:rgba(0,0,0,0.03);
          font-size:12px;
          line-height:1;
          margin-top:6px;
          margin-right:6px;
        ">{t}</span>
        """,
        unsafe_allow_html=True,
    )


def _lean_badge(row: dict) -> str:
    txt = " ".join([str(row.get("product_name") or ""), str(row.get("shade_name") or row.get("shade") or "")]).lower()
    for label, kws in _LEAN_KEYWORDS:
        if any(k in txt for k in kws):
            return label
    return "neutral-leaning"


def _render_intro() -> None:
    st.markdown('<div class="h2">What you’re looking at</div>', unsafe_allow_html=True)
    st.write(
        "A traceable lipstick recommender: it turns a natural-language request into ranked real shades by combining "
        "**NLP (intent + constraints)** with **perceptual color matching**."
    )
    st.caption(
        "Why this is hard: words like “nude”, “deep”, “not too bright” are subjective, and many shades are perceptually close."
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:6px;">NLP → intent & constraints</div>
              <div style="font-size:13px; opacity:.9;">
                Detects what you want + directional signals like <b>“less bright”</b>, <b>“more muted”</b>, <b>“deeper”</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:6px;">Color science → perceptual match</div>
              <div style="font-size:13px; opacity:.9;">
                Builds a <b>reference color</b> from your text and compares real shades by perceptual distance.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:6px;">Recommenders → filter & rank</div>
              <div style="font-size:13px; opacity:.9;">
                Reduces the catalog, then ranks what’s left. The decision stays <b>auditable</b> via intermediate signals.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_how_it_works() -> None:
    st.markdown('<div class="h2">How it works (in ~10 seconds)</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Understand** → interpret the sentence (intent + constraints)\n"
        "2. **Ground color** → pick a reference point for the requested family\n"
        "3. **Filter** → remove incompatible shades\n"
        "4. **Rank** → return closest matches (perceptual distance + constraint fit)\n"
        "5. **Explain** → surface what the system used and why"
    )
    st.caption("This is a full sentence-to-ranking pipeline (not a keyword filter).")


def _render_glossary() -> None:
    with st.expander("Glossary (quick definitions)", expanded=False):
        st.markdown(
            "- **Color family**: a broad group like *nude / red / pink / berry*.\n"
            "- **Constraints**: directional signals like *less bright*, *more muted*, *deeper*.\n"
            "- **Color reference point**: a representative color used for comparison (derived from your text).\n"
            "- **Cosmetics reference**: curated anchors for tricky families (used only when needed).\n"
            "- **Filtering strictness**: how aggressively we reduce the catalog before ranking."
        )


def _render_extracted_summary(trace: dict, query: str) -> str | None:
    st.markdown('<div class="h2">What we understood from your sentence</div>', unsafe_allow_html=True)

    nlp = trace.get("nlp", {}) or {}
    has_color = bool(nlp.get("has_color")) if isinstance(nlp, dict) else False

    anchor = trace.get("anchor", {}) or {}
    family_label = None
    if isinstance(anchor, dict):
        family_label = anchor.get("family_label") or anchor.get("family") or anchor.get("color_family")

    inferred_family = _detect_color_family_from_text(query) if not family_label else None
    if family_label:
        family_h = str(family_label).strip()
        family_src = "Detected by model"
        family_for_ui = family_h
    elif inferred_family:
        family_h = inferred_family
        family_src = "Detected from text"
        family_for_ui = inferred_family
    else:
        family_h = "Model abstention (below confidence threshold)"
        family_src = "Ambiguous / missing signal"
        family_for_ui = None

    anchor_src = _human_missing(anchor.get("anchor_source"), fallback="(internal)")
    used_domain_anchor = bool(anchor.get("used_domain_anchor")) if isinstance(anchor, dict) else False

    cons = _constraints(trace)
    cons_h = [_fmt_constraint_human(c) for c in cons[:3]]

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

    with c1:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:8px;">Request understanding</div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Color request detected: **{'Yes' if has_color else 'No'}**")
        st.write(f"Color family: **{family_h}**")
        st.caption(f"Source: {family_src}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("“Model abstention” means the system did not find a reliable signal in the sentence.")

    with c2:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:8px;">Constraints (signals extracted)</div>
            """,
            unsafe_allow_html=True,
        )
        if cons_h:
            st.markdown("\n".join([f"- {x}" for x in cons_h]))
            if len(cons) > 3:
                st.caption(f"+{len(cons) - 3} more (see technical details).")
        else:
            st.write("Model abstention (no constraint signal detected).")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Extracted from phrases like “not too bright”, “more muted”, “warm”, etc.")

    with c3:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:8px;">Color grounding</div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Anchor source: **{anchor_src if anchor_src else '(internal)'}**")
        st.write(f"Cosmetics reference used: **{'Yes' if used_domain_anchor else 'No'}**")
        st.caption("We do not display color chips here (UI choice).")
        st.markdown("</div>", unsafe_allow_html=True)

    return family_for_ui


def _render_pipeline_summary(trace: dict) -> None:
    st.markdown('<div class="h2">Pipeline summary</div>', unsafe_allow_html=True)

    cand = trace.get("candidate_pool", {}) or {}
    nb = cand.get("n_before") if isinstance(cand, dict) else None
    na = cand.get("n_after") if isinstance(cand, dict) else None

    if nb and na and nb > 0:
        kept_ratio = max(0.0, min(1.0, float(na) / float(nb)))
        kept_pct = int(round(100 * kept_ratio))
    else:
        kept_ratio, kept_pct = None, None

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown(
            """
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">
              <div style="font-weight:700; margin-bottom:8px;">Candidate reduction</div>
            """,
            unsafe_allow_html=True,
        )
        if nb is not None and na is not None:
            st.write(f"Start: **{nb}** shades (real products)")
            st.write(f"After intent + constraints: **{na}** kept")
            if kept_pct is not None:
                st.write(f"Kept **{kept_pct}%** of the catalog (less noise → more reliable ranking)")
        else:
            st.write("Pool sizes unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("**Filtering strictness**")
        if kept_ratio is not None and nb is not None and na is not None and nb > 0:
            strictness = max(0.0, min(1.0, 1.0 - kept_ratio))
            st.progress(strictness)
            st.caption(f"{na} / {nb} kept")
            if strictness >= 0.75:
                st.caption("High — very narrow set before ranking.")
            elif strictness >= 0.4:
                st.caption("Medium — some narrowing, still diverse options.")
            else:
                st.caption("Low — broad set (you may want to add more constraints).")
        else:
            st.caption("Unavailable.")


def _render_one_glance(query: str, trace: dict) -> None:
    nlp = trace.get("nlp", {}) or {}
    has_color = bool(nlp.get("has_color")) if isinstance(nlp, dict) else False

    cand = trace.get("candidate_pool", {}) or {}
    nb = cand.get("n_before") if isinstance(cand, dict) else None
    na = cand.get("n_after") if isinstance(cand, dict) else None

    cons = _constraints(trace)

    st.markdown('<div class="h2">At a glance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

    with c1:
        st.markdown("**1) Your request**")
        st.write(query)
        st.caption(f"Detected: {'color intent' if has_color else 'no explicit color intent'}")

    with c2:
        st.markdown("**2) Interpreted as**")
        if cons:
            ev = [(c.get("evidence") or "").strip() for c in cons if isinstance(c, dict)]
            ev = [e for e in ev if e]
            if ev:
                for x in ev[:3]:
                    st.markdown(f"- {x}")
            else:
                lines = [_fmt_constraint_human(c) for c in cons[:3]]
                st.markdown("\n".join([f"- {x}" for x in lines]))
        else:
            st.markdown("- Model abstention (no constraint signal)")
        st.caption("These are the signals the ranker tries to satisfy.")

    with c3:
        st.markdown("**3) How we narrowed it down**")
        if nb is not None and na is not None and nb > 0:
            pct = int(round(100 * float(na) / float(nb)))
            st.markdown(f"From **{nb}** shades → kept **{na}** compatible ones (**{pct}%**).")
            st.progress(max(0.0, min(1.0, float(na) / float(nb))))
            st.caption("Filtering reduces noise and makes ranking more reliable.")
        else:
            st.markdown("Candidate pool size unavailable")


def _render_why_this_result_line(trace: dict, *, family: str | None) -> None:
    cand = trace.get("candidate_pool", {}) or {}
    nb = cand.get("n_before") if isinstance(cand, dict) else None
    na = cand.get("n_after") if isinstance(cand, dict) else None

    anchor = trace.get("anchor", {}) or {}
    anchor_src = _human_missing(anchor.get("anchor_source"), fallback="(internal)")

    cons = _constraints(trace)
    ev = [(c.get("evidence") or "").strip() for c in cons if isinstance(c, dict)]
    ev = [e for e in ev if e]
    cons_short = ", ".join(ev[:3]) if ev else ("model_abstention" if not cons else "constraints extracted")

    bits: list[str] = []
    if family:
        bits.append(f"family={family}")
    bits.append(f"constraints={cons_short}")
    bits.append(f"anchor={anchor_src}")
    if nb is not None and na is not None and nb > 0:
        bits.append(f"kept={na}/{nb}")

    st.caption("**Why this looks like this:** " + " · ".join(bits))


def _render_best_matches(
    trace: dict,
    *,
    family: str | None,
    top_k: int,
) -> pd.DataFrame | None:
    topk = trace.get("topk")
    topk = topk if isinstance(topk, pd.DataFrame) else None
    if topk is None or topk.empty:
        st.caption("No results returned.")
        return None

    df = topk.copy()
    df["leans"] = df.apply(lambda r: _lean_badge(r.to_dict()), axis=1)

    st.markdown('<div class="h2">Best matches</div>', unsafe_allow_html=True)
    st.caption("Leaning tags are heuristic (based on shade naming).")
    if family:
        st.caption(f"Family context: **{family}** (results may include nearby neutrals if constraints dominate).")

    _render_why_this_result_line(trace, family=family)

    n_show = min(int(top_k), 6, len(df))
    if n_show <= 0:
        st.caption("No results to display.")
        return df

    cols = st.columns(3, gap="large")
    for i in range(n_show):
        row = df.iloc[i].to_dict()
        brand = (row.get("brand") or "").strip()
        product = (row.get("product_name") or "").strip()
        shade = (row.get("shade_name") or row.get("shade") or "").strip()
        leans = (row.get("leans") or "neutral-leaning").strip()

        with cols[i % 3]:
            st.markdown(
                '<div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px;">',
                unsafe_allow_html=True,
            )
            title = f"{brand} — {product}" if brand else product
            st.markdown(f"**{title}**")
            st.markdown(f"**Shade:** {shade}" if shade else "**Shade:** Model abstention (missing value)")
            _pill(leans)
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Refine your request (optional)", expanded=False):
        st.markdown(
            "- Add **warm vs cool** (e.g. “warm nude” / “cool nude”).\n"
            "- Add a direction: **more nude**, **more rosy**, **more muted**, **more vivid**.\n"
            "- If results look mixed: add **one more constraint** (often the biggest lever)."
        )

    return df


def _render_topk_table(topk: pd.DataFrame | None) -> None:
    st.markdown('<div class="h2">Top-K (table)</div>', unsafe_allow_html=True)
    if topk is None or topk.empty:
        st.caption("No table to display.")
        return
    cols = [c for c in ["leans", "brand", "product_name", "shade_name"] if c in topk.columns]
    view = topk[cols].copy() if cols else topk.copy()
    st.dataframe(view, width="stretch", hide_index=True)
    st.caption("Table view of the closest alternatives.")


def _render_technical(trace: dict, topk: pd.DataFrame | None) -> None:
    st.markdown('<div class="h2">Technical details (optional)</div>', unsafe_allow_html=True)

    with st.expander("Runtime warmup", expanded=False):
        if st.button("Warmup NLP stack"):
            ok = bool(warmup_nlp_stack())
            st.write("ok" if ok else "failed")

    with st.expander("Raw trace payloads", expanded=False):
        st.markdown("**nlp**")
        st.json(trace.get("nlp", {}) or {})
        st.markdown("**anchor**")
        st.json(trace.get("anchor", {}) or {})
        st.markdown("**candidate_pool**")
        st.json(trace.get("candidate_pool", {}) or {})

    with st.expander("Scoring table (raw)", expanded=False):
        scoring = trace.get("scoring_table")
        if isinstance(scoring, pd.DataFrame) and not scoring.empty:
            st.dataframe(scoring, width="stretch", hide_index=True)
        else:
            st.caption("No scoring_table returned.")

    with st.expander("Top-K with raw scores", expanded=False):
        if isinstance(topk, pd.DataFrame) and not topk.empty:
            st.dataframe(topk, width="stretch", hide_index=True)
        else:
            st.caption("No topk returned.")


# -----------------------------
# UI (state)
# -----------------------------
if "query" not in st.session_state:
    st.session_state["query"] = _EXAMPLE_QUERIES[0]
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 6
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False
if "last_trace" not in st.session_state:
    st.session_state["last_trace"] = None

render_hero(
    title="From language to shade ranking",
    subtitle="A traceable walkthrough of how the assistant turns a sentence into ranked, real-world lipstick shades.",
    kicker="NLP × color science × recommenders",
)

_render_intro()

with st.expander("Try a few example queries", expanded=False):
    st.caption("Pick one to see how the pipeline reacts to different intents and constraints.")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        if st.button(_EXAMPLE_QUERIES[0], use_container_width=True):
            st.session_state["query"] = _EXAMPLE_QUERIES[0]
            st.session_state["auto_run"] = True
            st.rerun()
        if st.button(_EXAMPLE_QUERIES[2], use_container_width=True):
            st.session_state["query"] = _EXAMPLE_QUERIES[2]
            st.session_state["auto_run"] = True
            st.rerun()
    with b2:
        if st.button(_EXAMPLE_QUERIES[1], use_container_width=True):
            st.session_state["query"] = _EXAMPLE_QUERIES[1]
            st.session_state["auto_run"] = True
            st.rerun()
        if st.button(_EXAMPLE_QUERIES[3], use_container_width=True):
            st.session_state["query"] = _EXAMPLE_QUERIES[3]
            st.session_state["auto_run"] = True
            st.rerun()

col1, col2 = st.columns([1.2, 0.6], gap="large")
with col1:
    st.text_input("Query", key="query", help="Natural language request.")
with col2:
    st.selectbox("Top-K", options=[3, 6, 10], index=[3, 6, 10].index(int(st.session_state["top_k"])), key="top_k")

run = st.button("Run", type="primary")
_run_now = bool(run) or bool(st.session_state.get("auto_run"))

_render_how_it_works()
_render_glossary()

# -----------------------------
# Run + render
# -----------------------------
if _run_now:
    st.session_state["auto_run"] = False
    with st.spinner("Running..."):
        trace = explain_recommendation(
            text=str(st.session_state["query"]),
            top_k=int(st.session_state["top_k"]),
            debug=False,
        )
    st.session_state["last_trace"] = trace

trace = st.session_state.get("last_trace")
if isinstance(trace, dict) and trace:
    family_for_ui = _render_extracted_summary(trace, str(st.session_state["query"]))
    _render_pipeline_summary(trace)

    st.divider()

    _render_one_glance(str(st.session_state["query"]), trace)

    best_df = _render_best_matches(
        trace,
        family=family_for_ui,
        top_k=int(st.session_state["top_k"]),
    )

    with st.expander("More (table + technical details)", expanded=False):
        _render_topk_table(best_df)
        _render_technical(trace, best_df)
