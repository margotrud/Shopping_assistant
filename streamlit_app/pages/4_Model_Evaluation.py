# streamlit_app/pages/6_Model_Evaluation.py
from __future__ import annotations

import streamlit as st

from ui.theme import inject_styles
from ui.nav import top_nav
from ui.components import render_hero


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Model & Evaluation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()
top_nav(active="Model & Eval")

render_hero(
    title="Model & Evaluation",
    subtitle=(
        "Technical overview of the recommendation system: design decisions, "
        "evaluation strategy, and known limitations."
    ),
    kicker="Under the hood",
)

# ---------------------------------------------------------------------
# Helpers – match global typography / spacing
# ---------------------------------------------------------------------
def h2(text: str) -> None:
    st.markdown(f'<div class="h2">{text}</div>', unsafe_allow_html=True)


def p(text: str) -> None:
    st.markdown(f'<p class="p">{text}</p>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# 1. System Overview
# ---------------------------------------------------------------------
h2("1. System Overview")

p("This application is a <b>rule-guided, metric-driven recommendation system</b>.")
p(
    "The objective is not to predict popularity, but to translate subjective natural language "
    "into <b>explicit, measurable color constraints</b>, and rank candidate shades accordingly."
)
p(
    "The system is intentionally <b>not end-to-end</b> in order to remain interpretable, "
    "testable, and auditable."
)

st.markdown('<div class="kicker">High-level pipeline</div>', unsafe_allow_html=True)
st.code(
    """User query
→ NLP parsing (intent, polarity, constraints)
→ Axis inference (brightness, depth, saturation, clarity)
→ Candidate pool selection (anchors / clusters)
→ Color distance scoring (CIELAB)
→ Preference weighting & penalties
→ Final ranked shades""",
    language="text",
)

p(
    "Each stage is independently testable and exposes intermediate artifacts "
    "(constraints, scores, pools), enabling precise diagnosis of failures."
)

divider()

# ---------------------------------------------------------------------
# 2. NLP → Quantitative Constraints
# ---------------------------------------------------------------------
h2("2. NLP → Quantitative Constraints")

p("Natural language is mapped to <b>numeric constraints</b>, not free-form embeddings.")

st.markdown(
    """
- *“not too bright”* → upper bound on brightness axis  
- *“very nude”* → low chroma + narrow hue window  
- *“deep but soft”* → depth ↑, saturation ↓
""",
)


p(
    "This avoids semantic drift and ensures deterministic behavior under "
    "small linguistic perturbations."
)

divider()

# ---------------------------------------------------------------------
# 3. Color Representation & Scoring
# ---------------------------------------------------------------------
h2("3. Color Representation & Scoring")

st.markdown('<div class="kicker">Color space</div>', unsafe_allow_html=True)
p("All color distances are computed in <b>CIELAB</b> space.")
p(
    "RGB distances correlate poorly with human perception. "
    "Lab distances provide perceptual uniformity and stable ranking behavior."
)

st.markdown('<div class="kicker">Scoring logic (simplified)</div>', unsafe_allow_html=True)
st.code(
    """final_score =
    color_distance_score
  + axis_alignment_bonus
  + preference_weights
  - constraint_penalties""",
    language="text",
)

p(
    "There are <b>no hidden learned weights</b>. "
    "All coefficients are explicit, configurable, and inspectable."
)

divider()

# ---------------------------------------------------------------------
# 4. Evaluation Strategy
# ---------------------------------------------------------------------
h2("4. Evaluation Strategy")

p(
    "This project does not rely on labeled “ground truth” recommendations, "
    "which are unrealistic in subjective domains such as cosmetics."
)
p(
    "Evaluation focuses on <b>behavioral sanity</b>, consistency, and "
    "constraint fidelity."
)

st.markdown('<div class="kicker">Core metrics</div>', unsafe_allow_html=True)
st.markdown(
    """
- **Coverage**  
  Percentage of user queries yielding at least one valid recommendation.

- **Stability**  
  Ranking variance under small linguistic perturbations.

- **Constraint satisfaction**  
  Hard constraints are never violated (e.g. “not bright” never returns high-L shades).
"""
)

p("In this domain, <b>internal consistency</b> is more meaningful than synthetic accuracy scores.")

divider()

# ---------------------------------------------------------------------
# 5. Known Failure Modes
# ---------------------------------------------------------------------
h2("5. Known Failure Modes")

p("The system explicitly exposes its limits.")

st.markdown(
    """
- Ambiguous adjectives (“natural”, “clean”) producing weak constraints  
- Conflicting constraints requiring heuristic compromises  
- Sparse shade distributions overpowering language intent
"""
)

p("These are <b>design trade-offs</b>, not implementation bugs.")

divider()

# ---------------------------------------------------------------------
# 6. Design Decisions
# ---------------------------------------------------------------------
h2("6. Design Decisions")

st.markdown('<div class="kicker">Why not end-to-end ML?</div>', unsafe_allow_html=True)
st.markdown(
    """
- No reliable labeled dataset  
- Poor interpretability  
- Difficult failure analysis
"""
)

st.markdown('<div class="kicker">Why hybrid rules + scoring?</div>', unsafe_allow_html=True)
st.markdown(
    """
- Deterministic constraint enforcement  
- Human-interpretable diagnostics  
- Controlled calibration and extension
"""
)

st.markdown('<div class="kicker">Why not pure LLM reasoning?</div>', unsafe_allow_html=True)
st.markdown(
    """
- Non-deterministic outputs  
- Hard to evaluate  
- Not unit-testable in a meaningful way
"""
)

divider()

# ---------------------------------------------------------------------
# 7. Production Considerations (Out of Scope)
# ---------------------------------------------------------------------
h2("7. Production Considerations (Out of Scope)")

p("If productionized, the system would require:")

st.markdown(
    """
- Query drift monitoring  
- User feedback loops  
- A/B testing of ranking strategies  
- Continuous anchor and calibration updates
"""
)

p("These were deliberately excluded to keep the project focused and auditable.")

divider()

# ---------------------------------------------------------------------
# 8. What This Page Demonstrates
# ---------------------------------------------------------------------
h2("8. What This Page Demonstrates")

st.markdown(
    """
- Formalization of vague language into measurable constraints  
- Awareness of evaluation limits in subjective recommendation systems  
- Clear separation between prototype scope and production reality  
- Emphasis on interpretability and diagnostic power
"""
)


divider()

