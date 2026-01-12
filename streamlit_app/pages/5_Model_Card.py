# streamlit_app/pages/5_Model_Card.py
"""
Model Card — System Credibility (ML-Senior)

Purpose:
    Answer one question clearly:
    “Is this project technically serious?”

This page highlights modeling decisions explicitly,
without relying on ML jargon or opaque claims.
"""

from __future__ import annotations

import streamlit as st

from ui.nav import top_nav
from ui.theme import inject_styles


# -----------------------------
# Page config + theme
# -----------------------------
st.set_page_config(
    page_title="Model Card",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()
top_nav(active="Model Card")


# -----------------------------
# HERO — IMMEDIATE SIGNAL
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Model Card</div>
      <div class="h1">System credibility</div>

      <p class="p hero-sub">
        This page explains what is modeled, what is not,
        and how modeling choices shape the results.
      </p>

      <ul class="credibility-bullets">
        <li>Deterministic ML system — no black-box inference</li>
        <li>Continuous representations, not rule-based logic</li>
        <li>Explicitly scoped, calibrated, and reproducible</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# WHAT THIS SYSTEM IS
# -----------------------------
st.markdown(
    """
    <div class="section">
      <div class="h2">What this system is</div>

      <p class="p">
        This project is a deterministic, model-driven recommendation system
        for lipstick shades.
      </p>

      <p class="p">
        User preferences expressed in natural language are transformed into
        numeric constraints and applied to structured color representations
        to produce a ranked list of products.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# CORE ML — CENTRAL SECTION
# -----------------------------
st.markdown(
    """
    <div class="section highlight">
      <div class="h2">How modeling decisions shape the results</div>

      <p class="p">
        Although deterministic, this system is not rule-based.
        It relies on continuous representations and calibrated scoring
        to model preference alignment.
      </p>

      <ul class="p">
        <li>Each product is embedded in multiple complementary numeric color spaces</li>
        <li>User language is mapped to continuous constraints, not binary filters</li>
        <li>Soft preferences are combined through weighted aggregation</li>
        <li>All weights and thresholds are explicit, inspectable, and testable</li>
      </ul>

      <p class="p">
        This design prioritizes interpretability and control over opaque prediction.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# HARD SCOPE (MODELING DECISION)
# -----------------------------
st.markdown(
    """
    <div class="callout">
      <strong>
        This system is intentionally designed for color-driven cosmetics
        and works well for lipsticks, but not for skincare.
      </strong>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# DOES / DOES NOT
# -----------------------------
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown(
        """
        <div class="card positive">
          <div class="h3">What the system models well</div>
          <ul class="p">
            <li>Nuanced color preferences (e.g. “deep but not bright”)</li>
            <li>Trade-offs between multiple soft constraints</li>
            <li>Stable ranking behavior across similar inputs</li>
            <li>Consistent performance on color-driven products</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card negative">
          <div class="h3">What the system does not model</div>
          <ul class="p">
            <li>Implicit or unstated personal preferences</li>
            <li>Online learning or user feedback adaptation</li>
            <li>Skin chemistry or skincare effects</li>
            <li>Non color-based product attributes</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# DATA PROVENANCE
# -----------------------------
st.markdown(
    """
    <div class="section">
      <div class="h2">Where the data comes from</div>

      <p class="p">
        Product data is built from a curated lipstick inventory containing
        brand names, shade names, finishes, and official color chips.
      </p>

      <p class="p">
        Each shade is converted into numeric color representations
        using a fixed and fully reproducible preprocessing pipeline.
        No personal data is collected or stored.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# VERIFICATION
# -----------------------------
st.markdown(
    """
    <div class="section">
      <div class="h2">How correctness was verified</div>
      <ul class="p">
        <li>Unit tests covering language parsing and scoring behavior</li>
        <li>Explicit testing of edge cases and failure modes</li>
        <li>Cross-space consistency checks</li>
        <li>Qualitative validation on real, user-style queries</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# KEY FACTS — PROMOTED
# -----------------------------
st.markdown(
    """
    <div class="section">
      <div class="h2">Key facts</div>

      <div class="stats-grid">
        <div class="stat">
          <div class="value">1,500+</div>
          <div class="label">lipstick shades</div>
        </div>
        <div class="stat">
          <div class="value">6</div>
          <div class="label">modeled color dimensions</div>
        </div>
        <div class="stat">
          <div class="value">&lt;100 ms</div>
          <div class="label">deterministic local inference</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# FINAL NOTE
# -----------------------------
st.markdown(
    """
    <div class="section">
      <div class="h2">Final note</div>
      <p class="p">
        This project demonstrates applied machine learning through
        careful representation design, constraint modeling,
        and disciplined validation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
