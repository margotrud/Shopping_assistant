# streamlit_app/pages/2_Explain_Text.py
"""
Explain (Text) — show, in plain language, how a user sentence is interpreted.

Requirements:
    - No code / no math / no formulas shown.
    - Visual, luxury layout consistent with other pages.
"""

from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st

from ui.nav import top_nav
from ui.theme import inject_styles


# -----------------------------
# Page config + global theme
# -----------------------------
st.set_page_config(page_title="Explain (Text)", layout="wide", initial_sidebar_state="collapsed")
inject_styles()
top_nav(active="Explain (Text)")


# -----------------------------
# Lightweight analysis (fallback)
# -----------------------------

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def _split_clauses(text: str) -> List[str]:
    """Does: split on simple discourse connectives; keeps order and trims."""
    t = _norm(text)
    if not t:
        return []
    parts = re.split(r"\b(?:but|however|though|although|yet|and|also)\b", t, flags=re.I)
    return [_norm(p) for p in parts if _norm(p)]


_NEG_PAT = re.compile(r"\b(?:not|no|never|avoid|without|less|don\s*'?t)\b", flags=re.I)


@dataclass(frozen=True, slots=True)
class Highlight:
    start: int
    end: int
    label: str  # "like" | "avoid" | "concept"


def _find_spans(text: str, pattern: str, label: str) -> List[Highlight]:
    spans: List[Highlight] = []
    for m in re.finditer(pattern, text, flags=re.I):
        spans.append(Highlight(m.start(), m.end(), label))
    return spans


def _dedupe_spans(spans: List[Highlight]) -> List[Highlight]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x.start, -(x.end - x.start)))
    out: List[Highlight] = []
    last_end = -1
    for s in spans:
        if s.start >= last_end:
            out.append(s)
            last_end = s.end
    return out


def _render_highlighted(text: str, spans: List[Highlight]) -> str:
    if not text:
        return ""
    if not spans:
        return html.escape(text)

    spans = sorted(spans, key=lambda x: x.start)
    chunks: List[str] = []
    cur = 0
    for s in spans:
        if s.start > cur:
            chunks.append(html.escape(text[cur:s.start]))

        frag = html.escape(text[s.start:s.end])
        cls = "hl-like" if s.label == "like" else "hl-avoid" if s.label == "avoid" else "hl"
        chunks.append(f'<span class="{cls}">{frag}</span>')
        cur = s.end

    if cur < len(text):
        chunks.append(html.escape(text[cur:]))

    return "".join(chunks)


def _heuristic_interpret(text: str) -> Tuple[List[str], List[Highlight], List[str]]:
    t = _norm(text)
    clauses = _split_clauses(t)

    color_terms = ["red", "pink", "nude", "coral", "berry", "plum", "mauve", "brown"]
    color_re = r"\b(?:" + "|".join(map(re.escape, color_terms)) + r")\b"

    spans: List[Highlight] = []
    spans += _find_spans(t, color_re, "like")

    # Brightness constraints
    spans += _find_spans(t, r"\bnot\s+too\s+bright\b", "avoid")
    spans += _find_spans(t, r"\btoo\s+bright\b", "avoid")
    spans += _find_spans(t, r"\bbright\b", "concept")
    spans += _find_spans(t, r"\btoo\s+light\b", "avoid")
    spans += _find_spans(t, r"\blight\b", "concept")

    spans = _dedupe_spans(spans)

    explanations: List[str] = []
    m = re.search(color_re, t, flags=re.I)
    if m:
        explanations.append(f"The app detected a preference for {m.group(0).lower()} shades.")

    if re.search(r"\bnot\s+too\s+bright\b", t, flags=re.I) or (
        re.search(r"\bbright\b", t, flags=re.I) and _NEG_PAT.search(t)
    ):
        explanations.append("The app detected a constraint against high brightness.")

    if len(clauses) >= 2:
        explanations.append("The sentence was split into simpler ideas before scoring shades.")
    elif clauses:
        explanations.append("The sentence was interpreted as a single idea.")

    if not explanations:
        explanations.append("No strong preference or constraint was detected in this sentence.")

    return clauses, spans, explanations


def _analyze(text: str) -> Tuple[List[str], List[Highlight], List[str]]:
    """Does: run analysis; uses project NLP if available, else heuristic."""
    t = _norm(text)
    if not t:
        return [], [], []

    try:
        from Shopping_assistant.nlp.interpretation.preference import interpret_nlp  # type: ignore
        from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference  # type: ignore

        nlp_res = interpret_nlp(t, debug=False)
        resolved = resolve_preference(nlp_res)

        clauses: List[str] = []
        if isinstance(nlp_res, dict) and "clauses" in nlp_res:
            raw = nlp_res.get("clauses") or []
            if isinstance(raw, list):
                clauses = [
                    _norm(getattr(x, "text", x.get("text") if isinstance(x, dict) else str(x)))
                    for x in raw
                ]
                clauses = [c for c in clauses if c]

        spans: List[Highlight] = []
        prefer_terms = []
        avoid_terms = []
        if isinstance(resolved, dict):
            prefer_terms = resolved.get("liked_tokens") or resolved.get("likes") or []
            avoid_terms = resolved.get("disliked_tokens") or resolved.get("avoids") or []

        def _as_terms(v) -> List[str]:
            if not v:
                return []
            if isinstance(v, str):
                return [v]
            if isinstance(v, list):
                out: List[str] = []
                for it in v:
                    if isinstance(it, str):
                        out.append(it)
                    elif isinstance(it, dict) and "text" in it:
                        out.append(str(it["text"]))
                    else:
                        out.append(str(it))
                return out
            return [str(v)]

        prefer_terms = _as_terms(prefer_terms)
        avoid_terms = _as_terms(avoid_terms)

        if prefer_terms or avoid_terms:
            for term in prefer_terms[:6]:
                spans += _find_spans(t, r"\b" + re.escape(term) + r"\b", "like")
            for term in avoid_terms[:6]:
                spans += _find_spans(t, r"\b" + re.escape(term) + r"\b", "avoid")
            spans = _dedupe_spans(spans)

        explanations: List[str] = []
        if prefer_terms:
            explanations.append(f"The app detected a preference for {prefer_terms[0]} shades.")
        if avoid_terms:
            explanations.append(f"The app detected a constraint to avoid {avoid_terms[0]}.")
        if clauses:
            explanations.append("The sentence was split into simpler ideas before scoring shades.")
        if explanations:
            return clauses or _split_clauses(t), spans, explanations

    except Exception:
        pass

    return _heuristic_interpret(t)


# -----------------------------
# Extra narrative helpers (no math, no code)
# -----------------------------

def _extract_signals(text: str, spans: List[Highlight]) -> Dict[str, List[str]]:
    """
    Does: build user-facing signals from highlighted spans.
    Output keys: preferences, constraints, concepts
    """
    t = text or ""
    prefs: List[str] = []
    cons: List[str] = []
    conc: List[str] = []

    for s in spans:
        frag = _norm(t[s.start:s.end]).lower()
        if not frag:
            continue
        if s.label == "like" and frag not in prefs:
            prefs.append(frag)
        elif s.label == "avoid" and frag not in cons:
            cons.append(frag)
        elif s.label == "concept" and frag not in conc:
            conc.append(frag)

    if any("bright" in x for x in cons) and "brightness" not in conc:
        conc.append("brightness")
    if prefs and "colors family" not in conc:
        conc.append("colors family")

    return {"preferences": prefs, "constraints": cons, "concepts": conc}


def _clause_explanations(clauses: List[str]) -> List[str]:
    """
    Does: produce plain-language explanations per clause.
    """
    out: List[str] = []
    for c in clauses:
        c_low = c.lower()
        if re.search(r"\bnot\s+too\s+bright\b", c_low) or (("bright" in c_low) and _NEG_PAT.search(c_low)):
            out.append("Adds a restriction: it limits brightness (avoids shades that look too bright).")
        elif re.search(r"\b(?:red|pink|nude|coral|berry|plum|mauve|brown)\b", c_low):
            out.append("Expresses a preference: it describes the colors family you want.")
        else:
            out.append("Adds context, but it was not strong enough to become a preference or a constraint.")
    return out


def _impact_lines(signals: Dict[str, List[str]]) -> List[str]:
    """
    Does: explain how this changes ranking in a concrete, non-mathy way.
    """
    prefs = signals.get("preferences") or []
    cons = signals.get("constraints") or []

    lines: List[str] = []
    if prefs:
        lines.append(f"Ranking will favor shades matching the {prefs[0]} family.")
    else:
        lines.append("Ranking will not strongly favor a specific colors family from this sentence.")

    if any("bright" in x for x in cons):
        lines.append("Very bright / very light shades will be pushed down in the ranking.")
        lines.append("Medium-to-deeper shades will be favored over light, vivid ones.")
    elif cons:
        lines.append(f'Shades matching “{cons[0]}” will be pushed down in the ranking.')
    else:
        lines.append("No strong constraint was detected, so ranking will mostly follow colors similarity.")

    lines.append("The final list is produced by combining all detected preferences and constraints.")
    return lines


# -----------------------------
# Luxe CSS
# -----------------------------

st.markdown(
    textwrap.dedent(
        """
        <style>
        :root{ --accent:#7A2E2E; --accent-soft: rgba(122, 46, 46, 0.12); }

        /* Tighter header (kicker closer + slightly stronger) */
        .page-hero{ margin: 0 0 18px 0; }
        .page-hero .kicker{ margin-bottom: 8px; opacity: 0.86; }
        .page-hero .h1{ margin: 0 0 10px 0; }
        .page-hero .p{ max-width: 72ch; }

        .surface{
          border: 1px solid rgba(19,42,99,0.10);
          background: linear-gradient(180deg, rgba(255,255,255,0.66), rgba(255,255,255,0.38));
          border-radius: 22px;
          padding: 18px 18px;
        }
        .surface + .surface{ margin-top: 14px; }

        .label{
          colors: rgba(19,42,99,0.62);
          font-size: 12px;
          letter-spacing:0.12em;
          text-transform: uppercase;
          font-weight: 600;
        }

        .hl, .hl-like, .hl-avoid{
          border-radius: 10px;
          padding: 2px 8px;
          display: inline-block;
          margin: 0 2px;
        }
        .hl{ background: rgba(19,42,99,0.08); }
        .hl-like{ background: rgba(122,46,46,0.12); border: 1px solid rgba(122,46,46,0.18); }
        .hl-avoid{ background: rgba(19,42,99,0.10); border: 1px solid rgba(19,42,99,0.18); text-decoration: line-through; text-decoration-thickness: 1px; }

        .clause-stack{ display: grid; gap: 10px; }
        .clause{
          padding: 12px 14px;
          border-radius: 16px;
          border: 1px solid rgba(19,42,99,0.10);
          background: rgba(255,255,255,0.32);
        }
        .clause .clause-k{ colors: rgba(19,42,99,0.62); font-size: 12px; font-weight: 600; letter-spacing:0.10em; text-transform: uppercase; margin-bottom: 6px; }
        .clause .clause-t{ colors: #132A63; font-size: 16px; line-height: 1.6; }

        .explain-list{ margin: 10px 0 0 0; padding: 0 0 0 18px; }
        .explain-list li{ margin: 8px 0; colors: rgba(19,42,99,0.78); line-height: 1.6; }

        /* Signals: 3 mini-cards */
        .signal-grid{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
        .signal{
          position: relative;
          border-radius: 18px;
          padding: 12px 12px 12px 14px;
          border: 1px solid rgba(19,42,99,0.10);
          background: rgba(255,255,255,0.34);
          overflow: hidden;
        }
        .signal:before{
          content: "";
          position: absolute;
          left: 0;
          top: 10px;
          bottom: 10px;
          width: 4px;
          border-radius: 999px;
          background: rgba(19,42,99,0.18);
        }

        .signal-k{ font-size: 12px; letter-spacing:0.10em; text-transform: uppercase; colors: rgba(19,42,99,0.62); font-weight: 700; }
        /* Slightly smaller + higher contrast (luxury hierarchy) */
        .signal-v{ margin-top: 6px; font-size: 13px; colors: rgba(19,42,99,0.86); line-height: 1.5; }

        .signal.pref{ border-colors: rgba(122,46,46,0.18); background: rgba(122,46,46,0.08); }
        .signal.pref:before{ background: rgba(122,46,46,0.30); }

        .signal.cons{ border-colors: rgba(19,42,99,0.18); background: rgba(19,42,99,0.06); }
        .signal.cons:before{ background: rgba(19,42,99,0.28); }

        .signal.conc{ border-colors: rgba(19,42,99,0.10); background: rgba(255,255,255,0.30); }
        .signal.conc:before{ background: rgba(19,42,99,0.16); }

        /* Input */
        .stTextArea textarea{ border-radius: 16px !important; }

        @media (max-width: 980px){
          .signal-grid{ grid-template-columns: 1fr; }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# -----------------------------
# Content
# -----------------------------

st.markdown(
    """
    <div class="page-hero">
      <div class="kicker">Explain</div>
      <div class="h1">Understanding the text</div>
      <p class="p">
        This page answers: <strong>“Does the app understand what I write?”</strong><br/>
        Type a sentence — the app will highlight what it kept as <em>preferences</em> and what it treated as <em>constraints</em>,
        then explain the result in plain language.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

default_text = "I want a red lipstick but not too bright"

text = st.text_area(
    "Your sentence",
    value=default_text,
    height=88,
    help="Example: I want a red lipstick but not too bright",
)

text_n = _norm(text)
clauses, spans, explanations = _analyze(text_n)
signals = _extract_signals(text_n, spans)

col_a, col_b = st.columns([1.25, 0.75], gap="large")

with col_a:
    rendered = _render_highlighted(text_n, spans)
    st.markdown(
        f"""
        <div class="surface">
          <div class="label">What the app highlighted</div>
          <div style="margin-top:10px; font-size:18px; line-height:1.75; colors:#132A63;">{rendered}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prefs = signals["preferences"]
    cons = signals["constraints"]
    conc = signals["concepts"]

    pref_txt = ", ".join(prefs) if prefs else "No explicit preference detected"
    cons_txt = ", ".join(cons) if cons else "No explicit constraint detected"
    conc_txt = ", ".join(conc) if conc else "General intent"

    st.markdown(
        f"""
        <div class="surface" style="margin-top:14px;">
          <div class="label">What the app understood</div>
          <div class="signal-grid">
            <div class="signal pref">
              <div class="signal-k">Preferences</div>
              <div class="signal-v">{html.escape(pref_txt)}</div>
            </div>
            <div class="signal cons">
              <div class="signal-k">Constraints</div>
              <div class="signal-v">{html.escape(cons_txt)}</div>
            </div>
            <div class="signal conc">
              <div class="signal-k">Key concepts</div>
              <div class="signal-v">{html.escape(conc_txt)}</div>
            </div>
          </div>

          <div style="margin-top:12px; colors: rgba(19,42,99,0.72); line-height:1.6;">
            In short, the app transforms your sentence into clear preferences and constraints that directly guide how shades are ranked.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if clauses:
        blocks: List[str] = []
        per_clause = _clause_explanations(clauses)

        for i, (c, exp) in enumerate(zip(clauses, per_clause), start=1):
            blocks.append(
                f'<div class="clause">'
                f'<div class="clause-k">Idea {i}</div>'
                f'<div class="clause-t">{html.escape(c)}</div>'
                f'<div style="margin-top:8px; colors: rgba(19,42,99,0.72); line-height:1.6;"><em>{html.escape(exp)}</em></div>'
                f"</div>"
            )

        st.markdown(
            f"""
            <div class="surface" style="margin-top:14px;">
              <div class="label">How the sentence was split</div>
              <div class="clause-stack" style="margin-top:10px;">
                {''.join(blocks)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_b:
    st.markdown(
        """
        <div class="surface">
          <div class="label">Plain-language explanation</div>
        """,
        unsafe_allow_html=True,
    )

    if explanations:
        bullets = "".join([f"<li>{html.escape(e)}</li>" for e in explanations])
        st.markdown(f"<ul class='explain-list'>{bullets}</ul>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='margin-top:10px; colors: rgba(19,42,99,0.72);'>Type a sentence to see the explanation.</p>",
            unsafe_allow_html=True,
        )

    impact = _impact_lines(signals)
    bullets2 = "".join([f"<li>{html.escape(x)}</li>" for x in impact])
    st.markdown(
        f"""
        <div style="height:14px"></div>
        <div class="label">What this changes in recommendations</div>
        <ul class="explain-list">{bullets2}</ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="height:10px"></div>
        <div class="label">What this page does not show</div>
        <ul class="explain-list">
          <li>No math.</li>
          <li>No formulas.</li>
          <li>No code.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
