# streamlit_app/Home.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import base64
import mimetypes

import streamlit as st

from ui.theme import inject_styles
from ui.nav import top_nav
from ui.bootstrap import get_nlp  # already exists in your project (spaCy cached)

# -----------------------------
# Warmup (spaCy + NLP stack)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _warmup_nlp_stack() -> bool:
    """
    Does:
        Warm up spaCy + polarity + world alias index + lexicon (+ optional semantic embeddings cache).
    """
    _ = get_nlp()

    from Shopping_assistant.nlp.parsing.polarity import make_free_polarity_fn
    make_free_polarity_fn()

    from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon
    lex = load_default_lexicon()
    _ = lex.raw_index

    from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index
    build_world_alias_index(include_xkcd=True)

    # optional: pre-build sentence-transformers key embeddings cache
    try:
        import os

        if os.environ.get("SA_WARMUP_SEMANTIC", "1").strip().lower() in {"1", "true", "yes"}:
            keys = list(lex.raw_index.keys())
            if keys:
                from Shopping_assistant.nlp.runtime.lexicon import (
                    _default_semantic_model,
                    _load_or_build_key_embeddings,
                )
                _load_or_build_key_embeddings(keys, _default_semantic_model())
    except Exception:
        pass

    return True


_warmup_nlp_stack()

st.set_page_config(
    page_title="Lipstick Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()
top_nav(active="Home")


@st.cache_data(show_spinner=False)
def _load_counts() -> Dict[str, Optional[int]]:
    """
    Does:
        Return products/shades/families counts if available, else None (never crashes).
    """
    return {"products": None, "shades": None, "families": None}


counts = _load_counts()
has_counts = all(v is not None for v in counts.values())

# -----------------------------
# Routes
# -----------------------------
PLAYGROUND = "pages/1_Playground.py"
EXPLAIN_TEXT = "pages/2_Explain_Text.py"
EXPLAIN_COLORS = "pages/3_Explain_Colors.py"
SHADE_LAB = "pages/4_Shade_Lab.py"
MODEL_CARD = "pages/5_Model_Card.py"

# -----------------------------
# Assets
# -----------------------------
assets_dir = Path(__file__).resolve().parent / "assets"
hero_img = assets_dir / "lipstick_smear.png"


def _img_data_uri(path: Path) -> Optional[str]:
    """
    Does:
        Return a data: URI for an image path, or None if missing/unreadable.
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _btn_page_link(page: str, label: str, primary: bool = False) -> str:
    """
    Does:
        Return a styled in-app link (HTML anchor) that matches .btn/.btn.primary.
    """
    cls = "btn primary" if primary else "btn"
    return f'<a class="{cls}" href="/{page}" target="_self">{label}</a>'


def _chip(title: str, body: str, href_page: str, cta: str) -> str:
    """
    Does:
        Return a compact “entry card” with a clear target page and CTA.
    """
    return (
        f'<a class="entry-card" href="/{href_page}" target="_self" style="text-decoration:none;">'
        f'<div class="entry-title">{title}</div>'
        f'<div class="entry-body">{body}</div>'
        f'<div class="entry-cta">{cta} →</div>'
        f"</a>"
    )


hero_data_uri = _img_data_uri(hero_img)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
:root{
  --accent: #7A2E2E;
  --accent-soft: rgba(122, 46, 46, 0.12);
}

/* Center the Streamlit page content */
section.main > div.block-container{
  max-width: 1120px;
  padding-left: 32px;
  padding-right: 32px;
}

/* Hero as a luxe surface */
section.hero{
  position: relative;
  border-radius: 22px;
  padding: 40px;
  overflow: hidden;

  border: 1px solid rgba(19,42,99,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.46));
  backdrop-filter: blur(6px);
}

.hero-split{
  display: grid;
  grid-template-columns: 1.22fr 0.78fr;
  gap: 40px;
  align-items: start;
}

.hero-left{
  max-width: 580px;
}

.hero-left .kicker{
  letter-spacing: 0.18em;
}

.hero-left .h1{
  margin-top: 12px;
  margin-bottom: 0;
}

.hero-left .home-rule{
  width: 44px;
  height: 3px;
  background: var(--accent);
  border-radius: 3px;
  margin: 18px 0 26px 0;
}

.hero-left .p{
  margin: 0 0 14px 0;
  line-height: 1.65;
  margin-bottom: 18px
}

/* CTA zone */
.hero-ctas{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;

  margin-top: 22px;
  margin-bottom: 22px;

  padding: 0;
  border: none;
  background: transparent;
}

.hero-ctas .btn:not(.primary){
  opacity: 0.85;
}
.hero-ctas .btn:not(.primary):hover{
  opacity: 1;
}

/* Media column */
.hero-media{
  display: flex;
  justify-content: center;
  align-items: flex-start;
  margin-top: 70px;
  overflow: hidden;
}

.hero-media img{
  width: 170%;
  max-width: 300px;
  height: auto;

  border-radius: 22px;
  border: 1px solid rgba(19,42,99,0.06);
  box-shadow: 0 18px 40px rgba(19,42,99,0.10);
  object-fit: cover;
  object-position: 55% 45%;
  transform: translateY(6px);
}

/* CTA */
a.btn.primary{
  background: var(--accent) !important;
  color: #fff !important;
  border-color: transparent !important;
}
a.btn.primary:hover{
  box-shadow: 0 10px 24px rgba(122, 46, 46, 0.32) !important;
  transform: translateY(-1px);
}

/* Steps */
.home-guide{
  margin-top: 6px;
}
.home-guide-line{
  line-height: 1.6;
}
.home-guide-step{
  border-color: rgba(122, 46, 46, 0.22) !important;
  background: rgba(122, 46, 46, 0.05) !important;
}

.entry-cta{ color: var(--accent) !important; }

.entry-grid a.entry-card:first-child{
  border-color: rgba(122, 46, 46, 0.55) !important;
  background: linear-gradient(180deg, rgba(122, 46, 46, 0.10), rgba(255,255,255,0.85)) !important;
}

/* Responsive */
@media (max-width: 900px){
  .hero-split{ grid-template-columns: 1fr; }
  .hero-left{ max-width: 100%; }
  .hero-media{ padding-top: 18px; }
  .hero-media img{ max-width: 100%; aspect-ratio: auto; }
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# HERO
# -----------------------------
counts_html = ""
if has_counts:
    counts_html = (
        f'<p class="small-note">'
        f'{counts["products"]} products · {counts["shades"]} shades · {counts["families"]} families'
        f"</p>"
    )

cta_html = (
    '<div class="hero-ctas">'
    + _btn_page_link(PLAYGROUND, "▶ Enter Playground", primary=True)
    + _btn_page_link(MODEL_CARD, "Model card")
    + "</div>"
)

if hero_data_uri:
    media_html = f'<div class="hero-media"><img src="{hero_data_uri}" alt="Lipstick texture smear"/></div>'
else:
    media_html = '<div class="hero-media"></div>'

hero_lines = [
    '<section class="hero">',
    '<div class="hero-split">',

    '<div class="hero-left">',
    '<div class="kicker">A NEW WAY TO CHOOSE LIPSTICK.</div>',
    '<div class="h1">Find the shade that feels right.</div>',
    '<div class="home-rule"></div>',
    '<p class="p">From words to shade — quietly, precisely.</p>',
    counts_html,
    cta_html,
    '<div class="home-guide">',
    '<div class="home-guide-line"><span class="home-guide-step">1</span> <strong>Describe</strong>a shade in natural language.</div>',
    '<div class="home-guide-line"><span class="home-guide-step">2</span> <strong>Understand</strong>the interpretation and scoring.</div>',
    '<div class="home-guide-line"><span class="home-guide-step">3</span> <strong>Explore</strong>the color space and palette.</div>',
    "</div>",
    "</div>",

    media_html,

    "</div>",
    "</section>",
]
st.markdown("\n".join([line for line in hero_lines if line]), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -----------------------------
# ENTRY MAP
# -----------------------------
st.markdown('<div class="h2">Explore the system</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="p">A walkthrough of the system, covering product flow, interpretation, and scoring.</p>',
    unsafe_allow_html=True,
)

entry_lines = [
    '<div class="entry-grid">',
    _chip("Playground", "Type a preference. Get ranked shades.", PLAYGROUND, "Start"),
    _chip("Explain (Text)", "Understand how words are interpreted into makeup intent.", EXPLAIN_TEXT, "Open"),
    _chip("Explain (Colors)", "Explore how intent becomes constraints in color space.", EXPLAIN_COLORS, "Open"),
    _chip("Shade Lab", "Explore shades across depth, brightness, and tone.", SHADE_LAB, "Explore"),
    _chip("Model Card", "Model assumptions, limitations, and design choices.", MODEL_CARD, "Read"),
    "</div>",
]
st.markdown("\n".join(entry_lines), unsafe_allow_html=True)

st.markdown(
    '<p class="small-note">This page remains intentionally non-technical. Deeper rationale is available in Explain and the Model Card.</p>',
    unsafe_allow_html=True,
)

if not hero_data_uri:
    st.caption(f"Image not found: {hero_img.as_posix()}")

# -----------------------------
# Local CSS for cards/steps
# -----------------------------
st.markdown(
    """
<style>
.home-guide { display: grid; gap: 10px; max-width: 860px; }
.home-guide-line { color: var(--muted); font-size: 15px; display: flex; align-items: center; gap: 10px; }
.home-guide-step {
  width: 22px; height: 22px; border-radius: 999px;
  border: 1px solid var(--border);
  display: inline-flex; align-items: center; justify-content: center;
  color: var(--ink); font-weight: 600; font-size: 12px;
  background: rgba(255,255,255,0.55);
}

.entry-grid {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 14px;
}
.entry-card {
  grid-column: span 4;
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 18px;
  padding: 16px 16px 14px 16px;
  display: block;
  transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}
.entry-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(19, 42, 99, 0.10);
  border-color: rgba(19, 42, 99, 0.22);
}
.entry-title { color: var(--ink); font-weight: 700; font-size: 16px; margin-bottom: 6px; }
.entry-body { color: var(--muted); font-size: 14px; line-height: 1.50; margin-bottom: 10px; }
.entry-cta { font-weight: 600; font-size: 13px; }

@media (max-width: 1100px) {
  .entry-card { grid-column: span 6; }
}
@media (max-width: 700px) {
  .entry-card { grid-column: span 12; }
}
</style>
""",
    unsafe_allow_html=True,
)
