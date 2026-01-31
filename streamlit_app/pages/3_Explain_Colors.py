# streamlit_app/pages/4_Explain_Choosing_Colors.py
"""
Explain — Choosing the colors

Goal:
    Answer: “Why this colors and not another?”
Constraints:
    - No heavy math shown (no formulas; keep numbers optional / minimal).
    - Explain like a human decision process (editorial, luxe).
    - Do NOT re-implement scoring logic. Only call existing project functions.
    - Everything else is CSS + layout + narration.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui.nav import top_nav
from ui.theme import inject_styles


# -----------------------------
# Page config + global theme
# -----------------------------
st.set_page_config(page_title="Explain — Choosing Colors", layout="wide", initial_sidebar_state="collapsed")
inject_styles()
top_nav(active="Explain — Choosing Colors")

# -----------------------------
# Bootstrap: make src importable
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------------
# HARD FIX: scope CSS reliably to this page
# -----------------------------
components.html(
    """
<script>
(() => {
  const doc = window.parent.document;
  doc.body.classList.add("page-explain-choosing-colors");
})();
</script>
""",
    height=0,
)

# -----------------------------
# Local luxe CSS (page-only)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --primaryColor: #6B1F2B;
  --ink: #132A63;
  --paper: rgba(255,255,255,0.82);
  --paper2: rgba(255,255,255,0.66);
  --border: rgba(19,42,99,0.10);
  --muted: rgba(19,42,99,0.62);
}

/* Layout */
body.page-explain-choosing-colors .block-container{ max-width: 1060px; }
body.page-explain-choosing-colors .page-wrap{ padding-top: 6px; }
body.page-explain-choosing-colors .section{ margin-top: 22px; }

body.page-explain-choosing-colors .stApp{ position: relative; }
body.page-explain-choosing-colors .stApp:before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(900px 420px at 18% 8%, rgba(107,31,43,0.055), transparent 60%),
    radial-gradient(760px 360px at 84% 18%, rgba(19,42,99,0.055), transparent 62%);
  z-index: 0;
}
body.page-explain-choosing-colors .stApp > div{ position: relative; z-index: 1; }

body.page-explain-choosing-colors .kicker{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.55);
  margin-bottom: 8px;
}
body.page-explain-choosing-colors .h1{
  font-family: "Libre Baskerville", serif;
  font-weight: 700;
  font-size: 52px;
  line-height: 1.05;
  colors: var(--ink);
  margin: 0 0 10px 0;
}
body.page-explain-choosing-colors .p{
  colors: rgba(19,42,99,0.70);
  font-size: 15px;
  line-height: 1.65;
  margin: 0;
}
body.page-explain-choosing-colors .h2{
  font-family: "Libre Baskerville", serif;
  font-weight: 700;
  font-size: 22px;
  colors: var(--ink);
  margin: 0 0 10px 0;
}
body.page-explain-choosing-colors .hairline{
  height: 1px;
  background: rgba(19,42,99,0.10);
  margin: 22px 0 16px;
}

/* CTA */
body.page-explain-choosing-colors div[data-testid="stFormSubmitButton"] button,
body.page-explain-choosing-colors .stButton > button{
  background: var(--primaryColor) !important;
  border: 1px solid rgba(107,31,43,0.55) !important;
  colors: #fff !important;
  box-shadow: 0 10px 30px rgba(19,42,99,0.10);
}
body.page-explain-choosing-colors div[data-testid="stFormSubmitButton"] button:hover,
body.page-explain-choosing-colors .stButton > button:hover{
  background: #5C1823 !important;
}

/* Input rail */
body.page-explain-choosing-colors .input-rail{ max-width: 760px; margin-top: 6px; }
body.page-explain-choosing-colors div[data-testid="stForm"],
body.page-explain-choosing-colors section[data-testid="stForm"],
body.page-explain-choosing-colors div[data-testid="stForm"] > div,
body.page-explain-choosing-colors section[data-testid="stForm"] > div,
body.page-explain-choosing-colors div[data-testid="stForm"] form{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
}
body.page-explain-choosing-colors div[data-testid="stTextInput"] small,
body.page-explain-choosing-colors div[data-testid="stTextInput"] [data-testid="stCaptionContainer"],
body.page-explain-choosing-colors div[data-testid="stTextInput"] [data-testid="stHelp"],
body.page-explain-choosing-colors div[data-testid="stTextInput"] [data-testid="InputInstructions"]{
  display: none !important;
}
body.page-explain-choosing-colors div[data-testid="stTextInput"] input{
  height: 46px !important;
  background: transparent !important;
  border: 0 !important;
  border-bottom: 1px solid rgba(19,42,99,0.18) !important;
  border-radius: 0 !important;
  outline: none !important;
  box-shadow: none !important;
  colors: var(--ink) !important;
  font-size: 16px !important;
  padding: 6px 2px !important;
  caret-colors: var(--primaryColor);
}
body.page-explain-choosing-colors div[data-testid="stTextInput"] input:focus{
  border-bottom-colors: rgba(107,31,43,0.72) !important;
  outline: none !important;
  box-shadow: none !important;
}
body.page-explain-choosing-colors div[data-testid="stFormSubmitButton"] button{
  height: 42px !important;
  border-radius: 999px !important;
  padding: 0 18px !important;
  font-size: 14px !important;
  letter-spacing: 0.02em !important;
}
body.page-explain-choosing-colors div[data-testid="stForm"] [data-testid="stHorizontalBlock"]{
  align-items: end !important;
  gap: 16px !important;
}
body.page-explain-choosing-colors .hint{
  max-width: 760px;
  margin-top: 10px;
  colors: rgba(19,42,99,0.55);
  font-size: 14px;
}

/* Preferences */
body.page-explain-choosing-colors .pref-note{ max-width: 860px; padding: 14px 0 0 0; }
body.page-explain-choosing-colors .pref-line{
  colors: rgba(19,42,99,0.72);
  font-size: 15px;
  line-height: 1.65;
  margin: 0 0 6px 0;
}
body.page-explain-choosing-colors .pref-line em{ font-style: normal; colors: rgba(19,42,99,0.90); }
body.page-explain-choosing-colors .pref-hair{ height: 1px; background: rgba(19,42,99,0.08); margin: 12px 0 0 0; }

/* Family step */
body.page-explain-choosing-colors .family-step{
  max-width: 980px;
  margin-top: 10px;
  padding-top: 12px;
  border-top: 1px solid rgba(19,42,99,0.08);
}
body.page-explain-choosing-colors .family-step-title{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.52);
  margin: 0 0 8px 0;
}
body.page-explain-choosing-colors .family-step-line{
  colors: rgba(19,42,99,0.70);
  font-size: 14px;
  line-height: 1.55;
  margin: 0 0 6px 0;
}

/* Verdict strip (global anchor) */
body.page-explain-choosing-colors .verdict-strip{
  max-width: 980px;
  margin: 12px 0 14px;
  padding: 12px 12px;
  border: 1px solid rgba(19,42,99,0.10);
  border-radius: 18px;
  background: rgba(255,255,255,0.74);
  backdrop-filter: blur(10px);
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}
body.page-explain-choosing-colors .v-item{
  border-radius: 14px;
  padding: 10px 10px;
  border: 1px solid rgba(19,42,99,0.08);
}
body.page-explain-choosing-colors .v-item.best{ border-colors: rgba(107,31,43,0.22); }
body.page-explain-choosing-colors .v-item.good{ border-colors: rgba(19,42,99,0.10); }
body.page-explain-choosing-colors .v-item.rej{ border-style: dashed; opacity: 0.88; }
body.page-explain-choosing-colors .v-k{
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.55);
  margin-bottom: 6px;
}
body.page-explain-choosing-colors .v-v{
  font-size: 13px;
  colors: rgba(19,42,99,0.84);
  line-height: 1.35;
}

/* Snapshot chart (thicker + more contrast) */
body.page-explain-choosing-colors .snap{
  max-width: 980px;
  margin: 10px 0 18px;
  padding: 14px 0 0;
  border-top: 1px solid rgba(19,42,99,0.08);
}
body.page-explain-choosing-colors .snap-head{
  display:flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
body.page-explain-choosing-colors .snap-title{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.56);
  margin: 0;
}
body.page-explain-choosing-colors .snap-note{
  font-size: 12px;
  colors: rgba(19,42,99,0.62);
  margin: 0;
  font-style: normal;
}
body.page-explain-choosing-colors .snap-row{
  display:grid;
  grid-template-columns: 160px 1fr;
  gap: 14px;
  align-items: center;
  margin: 14px 0;
}
body.page-explain-choosing-colors .snap-label{
  colors: rgba(19,42,99,0.72);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  white-space: nowrap;
}
body.page-explain-choosing-colors .bars{ display:grid; gap: 10px; }
body.page-explain-choosing-colors .bar{
  position: relative;
  height: 12px;                    /* thicker */
  border-radius: 999px;
  background: rgba(19,42,99,0.10); /* more contrast */
  overflow: hidden;
}
body.page-explain-choosing-colors .bar > span{
  position:absolute;
  inset:0;
  width: 0%;
  background: rgba(107,31,43,0.64); /* stronger */
}
body.page-explain-choosing-colors .bar.alt > span{ background: rgba(19,42,99,0.30); }
body.page-explain-choosing-colors .bar-meta{
  display:flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  colors: rgba(19,42,99,0.70);       /* stronger */
}
body.page-explain-choosing-colors .bar-meta b{
  colors: rgba(19,42,99,0.92);
  letter-spacing: 0.08em;
}

/* Cards head (stronger hierarchy) */
body.page-explain-choosing-colors .cards-head{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 22px;
  margin-top: 14px;
  margin-bottom: 10px;
}
body.page-explain-choosing-colors .cards-head .cap{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.62);
}
body.page-explain-choosing-colors .cards-head .cap b{
  colors: rgba(19,42,99,0.94);
  letter-spacing: 0.08em;
}

/* Card base (same presence across 3) */
body.page-explain-choosing-colors .choice-card{
  position: relative;
  border: 1px solid var(--border) !important;
  background: var(--paper2) !important;
  border-radius: 26px;
  padding: 20px 20px !important;   /* unify padding */
  box-shadow: 0 18px 54px rgba(19,42,99,0.09);
  backdrop-filter: blur(8px);
  min-height: 430px;               /* unify min height */
}

/* Status variations */
body.page-explain-choosing-colors .choice-card.is-chosen{
  border-colors: rgba(107,31,43,0.32) !important;
  background: var(--paper) !important;
  box-shadow:
    0 34px 96px rgba(19,42,99,0.18),
    0 0 0 4px rgba(107,31,43,0.06); /* premium halo */
}
body.page-explain-choosing-colors .choice-card.is-close{
  background: rgba(255,255,255,0.70) !important;
  box-shadow: 0 18px 54px rgba(19,42,99,0.09); /* same weight */
}
body.page-explain-choosing-colors .choice-card.is-rejected{
  opacity: 0.72;                   /* NOT too faded */
  filter: saturate(0.78);          /* keep legible */
  border-style: dashed !important;
  box-shadow: 0 18px 54px rgba(19,42,99,0.08); /* same presence */
  pointer-events: none;
}

/* Ribbon (top-left) */
body.page-explain-choosing-colors .ribbon{
  position:absolute;
  top: 14px;
  left: 14px;
  right: auto;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border: 1px solid rgba(19,42,99,0.14);
  colors: rgba(19,42,99,0.84);
  background: rgba(255,255,255,0.84);
}
body.page-explain-choosing-colors .choice-card.is-chosen .ribbon{
  border-colors: rgba(107,31,43,0.34);
  colors: rgba(107,31,43,0.94);
  background: rgba(255,255,255,0.92);
}
body.page-explain-choosing-colors .choice-card.is-rejected .ribbon{
  border-colors: rgba(19,42,99,0.12);
  colors: rgba(19,42,99,0.70);
  background: rgba(255,255,255,0.76);
}

/* Card header */
body.page-explain-choosing-colors .choice-head{
  display:flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 10px;
  padding-top: 30px; /* space for ribbon */
}
body.page-explain-choosing-colors .choice-title{
  font-family: "Libre Baskerville", serif;
  font-weight: 700;
  font-size: 16px;
  line-height: 1.35;
  colors: var(--ink);
  margin: 0;
}

/* Pills */
body.page-explain-choosing-colors .pills{
  display:flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 10px;
}
body.page-explain-choosing-colors .pill{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(19,42,99,0.12);
  colors: rgba(19,42,99,0.78);
  background: transparent;
}
body.page-explain-choosing-colors .pill .dot{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  border: 1px solid rgba(19,42,99,0.18);
}
body.page-explain-choosing-colors .pill.status.best{
  border-colors: rgba(107,31,43,0.28);
  colors: rgba(107,31,43,0.92);
}
body.page-explain-choosing-colors .pill.status.close{ border-colors: rgba(19,42,99,0.14); }
body.page-explain-choosing-colors .pill.status.reject{
  border-colors: rgba(19,42,99,0.12);
  colors: rgba(19,42,99,0.72);
}

/* Bullets */
body.page-explain-choosing-colors .bullets{
  margin: 10px 0 0 0;
  padding-left: 0;
  colors: rgba(19,42,99,0.74);
  font-size: 14px;
  line-height: 1.60;
  list-style: none;
}
body.page-explain-choosing-colors .bullets li{
  position: relative;
  padding-left: 16px;
  margin: 6px 0;
}
body.page-explain-choosing-colors .bullets li:before{
  content: "";
  width: 5px;
  height: 5px;
  border-radius: 999px;
  background: rgba(107,31,43,0.55);
  position: absolute;
  left: 0;
  top: 9px;
}
body.page-explain-choosing-colors .choice-card.is-rejected .bullets li:before{
  background: rgba(19,42,99,0.26);
}

/* Because / Proof / Mini chart */
body.page-explain-choosing-colors .because{
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(19,42,99,0.08);
}
body.page-explain-choosing-colors .because-title,
body.page-explain-choosing-colors .proof-title{
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  colors: rgba(19,42,99,0.52);
  margin: 0 0 8px 0;
}
body.page-explain-choosing-colors .because-line{
  colors: rgba(19,42,99,0.72);
  font-size: 14px;
  line-height: 1.55;
  margin: 0 0 6px 0;
}

/* Proof: no monospace */
body.page-explain-choosing-colors .proof{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(19,42,99,0.08);
}
body.page-explain-choosing-colors .proof-line{
  font-family: inherit;
  font-size: 12px;
  colors: rgba(19,42,99,0.66);
  margin: 0 0 6px 0;
}
body.page-explain-choosing-colors .proof-note{
  font-size: 12px;
  colors: rgba(19,42,99,0.56);
  margin: 6px 0 0 0;
  font-style: normal;
}

body.page-explain-choosing-colors .mini-chart{
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(19,42,99,0.08);
  display: grid;
  gap: 8px;
}
body.page-explain-choosing-colors .metric{
  display:grid;
  grid-template-columns: 1fr 96px;
  gap: 12px;
  align-items: center;
}
body.page-explain-choosing-colors .metric-label{
  colors: rgba(19,42,99,0.56);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  white-space: nowrap;
}
body.page-explain-choosing-colors .metric-grade{
  text-align: right;
  colors: rgba(19,42,99,0.76);
  font-size: 12px;
  white-space: nowrap;
}

@media (max-width: 1100px){
  body.page-explain-choosing-colors .verdict-strip{ grid-template-columns: 1fr; }
  body.page-explain-choosing-colors .cards-head{ grid-template-columns: 1fr; }
  body.page-explain-choosing-colors .choice-card{ min-height: auto; }
}
</style>
<div class="page-wrap"></div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Data hooks (NO re-implementation of scoring)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _assets():
    from Shopping_assistant.io.assets import load_assets  # type: ignore
    from Shopping_assistant.color import scoring as color_scoring  # type: ignore

    return load_assets(
        enriched_csv=color_scoring._default_enriched_for_scripts_only(),
        prototypes_csv=color_scoring._default_prototypes_for_scripts_only(),
        assignments_csv=color_scoring._default_assignments_for_scripts_only(),
        calibration_json=color_scoring._default_calibration_for_scripts_only(),
    )


def _safe_hex(x: object) -> str:
    s = str(x or "").strip()
    if not s:
        return "#FFFFFF"
    if s.startswith("#") and len(s) in (4, 7):
        return s
    return "#FFFFFF"


def _rule_lines_from_trace(nlp_res: object) -> List[str]:
    out: List[str] = []
    trace = getattr(nlp_res, "trace", None) or {}
    raw = trace.get("constraints_final", []) if isinstance(trace, dict) else []

    axis_to_text = {
        "brightness": ("Avoid very bright shades", "Prefer brighter shades"),
        "depth": ("Avoid very deep shades", "Prefer deeper shades"),
        "saturation": ("Avoid highly saturated shades", "Prefer more saturated shades"),
        "vibrancy": ("Avoid neon / overly vibrant shades", "Prefer more vibrant shades"),
        "clarity": ("Avoid overly clear / sharp shades", "Prefer cleaner shades"),
    }

    for c in raw[:6]:
        try:
            axis = str(c.get("axis", "")).strip()
            direction = str(c.get("direction", "")).strip()  # lower|raise
            pair = axis_to_text.get(axis)
            if not pair:
                continue
            out.append(pair[0] if direction == "lower" else pair[1])
        except Exception:
            continue

    seen = set()
    deduped: List[str] = []
    for r in out:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    return deduped


@st.cache_resource(show_spinner=False)
def _cluster_label_map() -> dict[int, str]:
    assets = _assets()
    prot = getattr(assets, "prototypes", None)
    if prot is None or not isinstance(prot, pd.DataFrame) or prot.empty:
        return {}

    need = {"cluster_id", "L_lab", "a_lab", "b_lab"}
    if not need.issubset(set(prot.columns)):
        return {}

    out: dict[int, str] = {}
    for _, r in prot.iterrows():
        try:
            cid = int(r["cluster_id"])
            L = float(r["L_lab"])
            a = float(r["a_lab"])
            b = float(r["b_lab"])

            h = (float(np.degrees(np.arctan2(b, a))) + 360.0) % 360.0
            C = float(np.hypot(a, b))

            light = "Deep" if L <= 38 else ("Medium" if L <= 55 else "Light")
            chroma = "Muted" if C <= 20 else ("Balanced" if C <= 40 else "Vivid")

            if h >= 335 or h < 18:
                fam = "Red"
            elif 18 <= h < 42:
                fam = "Coral"
            elif 42 <= h < 85:
                fam = "Warm nude"
            elif 85 <= h < 125:
                fam = "Beige"
            elif 125 <= h < 175:
                fam = "Unusual"
            elif 175 <= h < 235:
                fam = "Cool mauve"
            elif 235 <= h < 300:
                fam = "Berry"
            else:
                fam = "Fuchsia"

            out[cid] = f"Cluster {cid}" if fam == "Unusual" else f"{light} {chroma} {fam}"
        except Exception:
            continue
    return out


def _family_label_from_row(row: pd.Series) -> Optional[str]:
    for col in ("family", "color_family", "family_name"):
        if col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s

    if "cluster_id" in row.index and pd.notna(row["cluster_id"]):
        try:
            cid = int(row["cluster_id"])
            return _cluster_label_map().get(cid, f"Cluster {cid}")
        except Exception:
            return None
    return None


def _rules_to_editorial(rules: List[str]) -> Tuple[str, str, str]:
    base = [r.strip().rstrip(".") for r in rules if str(r).strip()][:3]
    if not base:
        return (
            "You’re steering toward shades that feel closest to your intent.",
            "We set aside options that clash with your constraints.",
            "Then we keep the one that reads most naturally on you.",
        )
    line1 = f"You’re leaning toward <em>{html.escape(base[0])}</em>."
    line2 = (
        f"You’re also asking to <em>{html.escape(base[1])}</em>."
        if len(base) > 1
        else "We keep the tone consistent with your intent."
    )
    line3 = (
        "Overall, we keep what feels closest — and set aside the rest."
        if len(base) < 3
        else f"Overall, we prioritize <em>{html.escape(base[2])}</em>."
    )
    return (line1, line2, line3)


def _relative_strength(chosen_v: float, alt_v: float) -> str:
    d = float(alt_v) - float(chosen_v)
    ad = abs(d)
    if ad >= 0.35:
        return "by a wide margin"
    if ad >= 0.18:
        return "noticeably"
    if ad >= 0.08:
        return "slightly"
    return "about the same"


def _because_lines(ch_delta: float, r_delta: float, ch_pen: float, r_pen: float) -> List[str]:
    closeness = _relative_strength(ch_delta, r_delta)
    if ch_delta < r_delta:
        line1 = f"Closer to your description — {closeness} closer than the alternative."
    elif ch_delta > r_delta:
        line1 = f"Color closeness is not the driver here — the alternative is {closeness} closer."
    else:
        line1 = "Color closeness is comparable — other constraints drive the final choice."

    fit = _relative_strength(ch_pen, r_pen)
    if ch_pen < r_pen:
        line2 = f"Breaks fewer of your constraints — {fit} fewer conflicts overall."
    elif ch_pen > r_pen:
        line2 = f"Constraint fit is tighter on the alternative — {fit} fewer conflicts there."
    else:
        line2 = "Constraint fit is comparable — we choose the shade that reads most coherent overall."

    return [line1, line2]


def _grade_lower_is_better(v: float) -> str:
    x = float(v)
    if x <= 0.15:
        return "Excellent"
    if x <= 0.35:
        return "Strong"
    if x <= 0.60:
        return "Okay"
    return "Weak"


def _proof_lines(ch_delta: float, r_delta: float, ch_pen: float, r_pen: float) -> List[str]:
    ch_delta = float(ch_delta)
    r_delta = float(r_delta)
    ch_pen = float(ch_pen)
    r_pen = float(r_pen)

    tight_de = abs(ch_delta - r_delta) <= 0.01
    tight_pen = abs(ch_pen - r_pen) <= 0.01
    tight = tight_de and tight_pen

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if tight else f"{x:.2f}"

    de_c = _fmt(ch_delta)
    de_r = _fmt(r_delta)
    de_d = _fmt(ch_delta - r_delta)
    pe_c = _fmt(ch_pen)
    pe_r = _fmt(r_pen)
    pe_d = _fmt(ch_pen - r_pen)

    if not str(de_d).startswith("-"):
        de_d = f"+{de_d}"
    if not str(pe_d).startswith("-"):
        pe_d = f"+{pe_d}"

    note = "Tight decision — small differences can change the ranking." if tight else "Clear separation on the key signals."
    return [
        f"Color distance: {de_c} vs {de_r}  (Δ {de_d})",
        f"Constraint conflicts: {pe_c} vs {pe_r}  (Δ {pe_d})",
        f"Lower is better for both. {note}",
    ]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _bar_score_higher_is_better_from_lower_metric(v: float, vmax: float) -> float:
    if vmax <= 1e-12:
        return 1.0
    return _clamp01(1.0 - (float(v) / float(vmax)))


def _render_snapshot(
    *,
    chosen_title: str,
    alt_title: str,
    chosen_delta: float,
    alt_delta: float,
    chosen_pen: float,
    alt_pen: float,
    max_delta: float,
    max_pen: float,
) -> str:
    c_close = _bar_score_higher_is_better_from_lower_metric(chosen_delta, max_delta)
    a_close = _bar_score_higher_is_better_from_lower_metric(alt_delta, max_delta)
    c_fit = _bar_score_higher_is_better_from_lower_metric(chosen_pen, max_pen)
    a_fit = _bar_score_higher_is_better_from_lower_metric(alt_pen, max_pen)

    tight_de = abs(float(chosen_delta) - float(alt_delta)) <= 0.01
    tight_pen = abs(float(chosen_pen) - float(alt_pen)) <= 0.01
    tight = tight_de and tight_pen

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if tight else f"{x:.2f}"

    lines = [
        '<div class="snap">',
        '<div class="snap-head">',
        '<p class="snap-title">Signal snapshot</p>',
        '<p class="snap-note">Two signals decide the winner: closeness + constraint fit.</p>',
        "</div>",
        '<div class="snap-row">',
        '<div class="snap-label">Closeness</div>',
        '<div class="bars">',
        f'<div class="bar"><span style="width:{c_close*100:.1f}%;"></span></div>',
        f'<div class="bar-meta"><span><b>BEST</b> — {html.escape(chosen_title)}</span><span>distance {html.escape(_fmt(chosen_delta))}</span></div>',
        f'<div class="bar alt"><span style="width:{a_close*100:.1f}%;"></span></div>',
        f'<div class="bar-meta"><span><b>CLOSE</b> — {html.escape(alt_title)}</span><span>distance {html.escape(_fmt(alt_delta))}</span></div>',
        "</div>",
        "</div>",
        '<div class="snap-row">',
        '<div class="snap-label">Fits rules</div>',
        '<div class="bars">',
        f'<div class="bar"><span style="width:{c_fit*100:.1f}%;"></span></div>',
        f'<div class="bar-meta"><span><b>BEST</b></span><span>conflicts {html.escape(_fmt(chosen_pen))}</span></div>',
        f'<div class="bar alt"><span style="width:{a_fit*100:.1f}%;"></span></div>',
        f'<div class="bar-meta"><span><b>CLOSE</b></span><span>conflicts {html.escape(_fmt(alt_pen))}</span></div>',
        "</div>",
        "</div>",
        "</div>",
    ]
    return "\n".join(lines)


def _dynamic_bullets(ch_delta: float, r_delta: float, ch_pen: float, r_pen: float) -> Tuple[List[str], List[str]]:
    ch_delta = float(ch_delta)
    r_delta = float(r_delta)
    ch_pen = float(ch_pen)
    r_pen = float(r_pen)

    d_close = ch_delta - r_delta
    d_pen = ch_pen - r_pen

    close_matters = abs(d_close) > 0.01
    pen_matters = abs(d_pen) > 0.01

    chosen_bul: List[str] = []
    alt_bul: List[str] = []

    if close_matters and ch_delta < r_delta:
        chosen_bul.append("Reads closer to your description overall.")
        alt_bul.append("Reads further from your description overall.")
    elif close_matters and ch_delta > r_delta:
        chosen_bul.append("Closeness is not the driver — other signals decide.")
        alt_bul.append("Closeness is not the driver — other signals decide.")
    else:
        chosen_bul.append("Closeness is essentially tied — we look at constraint fit.")
        alt_bul.append("Closeness is essentially tied — we look at constraint fit.")

    if pen_matters and ch_pen < r_pen:
        chosen_bul.append("Conflicts less with what you asked to avoid.")
        alt_bul.append("Introduces more conflicts with your constraints.")
    elif pen_matters and ch_pen > r_pen:
        chosen_bul.append("Fits your constraints about as well as the alternative.")
        alt_bul.append("Fits your constraints about as well as the alternative.")
    else:
        chosen_bul.append("Constraint fit is essentially tied.")
        alt_bul.append("Constraint fit is essentially tied.")

    chosen_bul.append("Keeps the tone coherent with your intent.")
    alt_bul.append("Less aligned with the tone you described.")
    return (chosen_bul[:3], alt_bul[:3])


def _render_choice_card(
    *,
    title: str,
    status_label: str,  # "BEST MATCH" | "GOOD ALTERNATIVE" | "REJECTED"
    status_key: str,  # "best"|"close"|"reject"
    card_kind: str,  # "chosen"|"close"|"rejected"
    family_label: str,
    swatch_hex: str,
    bullets: List[str],
    because_lines: List[str],
    fam_grade: str,
    fit_grade: str,
    close_grade: str,
    show_proof: bool = False,
    proof_lines: Optional[List[str]] = None,
) -> str:
    card_cls = f"choice-card is-{card_kind}"

    proof_html = ""
    if show_proof and proof_lines:
        proof_a = proof_lines[0] if len(proof_lines) > 0 else ""
        proof_b = proof_lines[1] if len(proof_lines) > 1 else ""
        proof_note = proof_lines[2] if len(proof_lines) > 2 else ""
        proof_html = (
            '<div class="proof">'
            '<div class="proof-title">Proof</div>'
            f'<p class="proof-line">{html.escape(proof_a)}</p>'
            f'<p class="proof-line">{html.escape(proof_b)}</p>'
            f'<p class="proof-note">{html.escape(proof_note)}</p>'
            "</div>"
        )

    bullets_html = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
    because_html = "".join(f"<p class='because-line'>{html.escape(x)}</p>" for x in because_lines[:2])

    fam_label = family_label.strip() or "Family"
    sw = html.escape(swatch_hex)

    return (
        f'<div class="{card_cls}">'
        f'<div class="ribbon">{html.escape(status_label)}</div>'
        '<div class="choice-head">'
        f'<div class="choice-title">{html.escape(title)}</div>'
        "</div>"
        '<div class="pills">'
        f'<div class="pill status {html.escape(status_key)}"><span class="dot" style="background:{sw};"></span>{html.escape(status_label)}</div>'
        f'<div class="pill"><span class="dot" style="background:{sw};"></span>{html.escape(fam_label)}</div>'
        "</div>"
        f'<ul class="bullets">{bullets_html}</ul>'
        '<div class="because"><div class="because-title">Because</div>'
        f"{because_html}"
        "</div>"
        f"{proof_html}"
        '<div class="mini-chart">'
        '<div class="metric"><div class="metric-label">Family</div>'
        f'<div class="metric-grade">{html.escape(fam_grade)}</div></div>'
        '<div class="metric"><div class="metric-label">Fits rules</div>'
        f'<div class="metric-grade">{html.escape(fit_grade)}</div></div>'
        '<div class="metric"><div class="metric-label">Closeness</div>'
        f'<div class="metric-grade">{html.escape(close_grade)}</div></div>'
        "</div>"
        "</div>"
    )


def _safe_title(row: pd.Series, fallback: str) -> str:
    brand = "" if pd.isna(row.get("brand_name")) else str(row.get("brand_name") or "").strip()
    shade = "" if pd.isna(row.get("shade_name")) else str(row.get("shade_name") or "").strip()
    s = f"{brand} — {shade}".strip(" —")
    return s or fallback


def _pick_close_runner_up(df_sorted: pd.DataFrame) -> pd.Series:
    return df_sorted.iloc[1] if len(df_sorted) > 1 else df_sorted.iloc[0]


def _pick_clear_rejected(df_sorted: pd.DataFrame, *, chosen_cluster: Optional[int]) -> pd.Series:
    if len(df_sorted) <= 2:
        return df_sorted.iloc[-1]

    tail = df_sorted.tail(min(20, len(df_sorted))).copy()

    if chosen_cluster is not None and "cluster_id" in tail.columns:
        out = tail[tail["cluster_id"].astype("Int64") != chosen_cluster]
        if not out.empty:
            if "constraint_penalty_norm" in out.columns:
                idx = pd.to_numeric(out["constraint_penalty_norm"], errors="coerce").fillna(0.0).idxmax()
                return df_sorted.loc[idx]
            return df_sorted.loc[out.index[0]]

    if "constraint_penalty_norm" in tail.columns:
        idx = pd.to_numeric(tail["constraint_penalty_norm"], errors="coerce").fillna(0.0).idxmax()
        return df_sorted.loc[idx]

    return df_sorted.iloc[-1]


# -----------------------------
# UI — Hero
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="kicker">Explain</div>
  <div class="h1">Choosing the colors</div>
  <p class="p">
    This page explains the selection like a human decision:
    we group shades into families, remove what doesn’t fit your intent, and keep what feels closest.
  </p>
  <p class="p" style="margin-top:6px; colors: rgba(19,42,99,0.60); font-size: 14px;">
    You’ll get: a short rationale, a simple signal snapshot, then a 3-way comparison (best, good alternative, rejected).
  </p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

# -----------------------------
# Input
# -----------------------------
st.markdown('<div class="section"><div class="input-rail">', unsafe_allow_html=True)
with st.form("explain_form", clear_on_submit=False):
    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        text = st.text_input(
            label="",
            placeholder="e.g. deep red, not too bright, no neon…",
            key="explain_choose_text",
        )
    with c2:
        run = st.form_submit_button("Explain", type="primary")
st.markdown("</div></div>", unsafe_allow_html=True)

if not run or not str(text or "").strip():
    st.markdown(
        """
<div class="section">
  <div class="hint">Type a short sentence above, then click <b>Explain</b>.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# -----------------------------
# Main logic: pipeline only
# -----------------------------
with st.spinner("Interpreting your request…"):
    assets = _assets()

    from Shopping_assistant.nlp.interpretation.preference import interpret_nlp  # type: ignore
    from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference  # type: ignore
    from Shopping_assistant.reco.recommend import recommend_from_text  # type: ignore

    nlp_res = interpret_nlp(text, debug=False)
    _ = resolve_preference(nlp_res)
    df = recommend_from_text(text, assets=assets, topk=60, debug=False)

if df is None or df.empty:
    st.markdown(
        """
<div class="section">
  <div class="hint">No shades were returned. Try a simpler sentence (one colors family + one constraint).</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# -----------------------------
# Preferences (editorial)
# -----------------------------
rules = _rule_lines_from_trace(nlp_res)
line1, line2, line3 = _rules_to_editorial(rules)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="h2">Your preferences</div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="pref-note">
  <p class="pref-line">{line1}</p>
  <p class="pref-line">{line2}</p>
  <p class="pref-line">{line3}</p>
  <div class="pref-hair"></div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Pick examples: best + close + rejected
# -----------------------------
df = df.reset_index(drop=True).copy()
score_col = "score_total" if "score_total" in df.columns else ("score" if "score" in df.columns else None)
if score_col is None:
    st.markdown(
        """
<div class="section">
  <div class="hint">Expected a score column from the pipeline (score / score_total), but none was found.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
chosen = df_sorted.iloc[0]
close_alt = _pick_close_runner_up(df_sorted)

chosen_cluster: Optional[int] = None
if "cluster_id" in chosen.index and pd.notna(chosen.get("cluster_id")):
    try:
        chosen_cluster = int(chosen.get("cluster_id"))
    except Exception:
        chosen_cluster = None

rejected = _pick_clear_rejected(df_sorted, chosen_cluster=chosen_cluster)

chosen_hex = _safe_hex(chosen.get("chip_hex", ""))
close_hex = _safe_hex(close_alt.get("chip_hex", ""))
rej_hex = _safe_hex(rejected.get("chip_hex", ""))

chosen_family = _family_label_from_row(chosen) or "Family"
close_family = _family_label_from_row(close_alt) or "Family"
rej_family = _family_label_from_row(rejected) or "Family"

chosen_delta = float(chosen.get("deltaE_norm", 0.0) or 0.0)
close_delta = float(close_alt.get("deltaE_norm", 0.0) or 0.0)
rej_delta = float(rejected.get("deltaE_norm", 0.0) or 0.0)

chosen_pen = float(chosen.get("constraint_penalty_norm", 0.0) or 0.0)
close_pen = float(close_alt.get("constraint_penalty_norm", 0.0) or 0.0)
rej_pen = float(rejected.get("constraint_penalty_norm", 0.0) or 0.0)

chosen_close_g = _grade_lower_is_better(chosen_delta)
close_close_g = _grade_lower_is_better(close_delta)
rej_close_g = _grade_lower_is_better(rej_delta)

chosen_fit_g = _grade_lower_is_better(chosen_pen)
close_fit_g = _grade_lower_is_better(close_pen)
rej_fit_g = _grade_lower_is_better(rej_pen)

chosen_fam_g = "Strong" if chosen_family else "Okay"
close_fam_g = "Strong" if close_family == chosen_family else "Okay"
rej_fam_g = "Weak" if rej_family != chosen_family else "Okay"

chosen_because = _because_lines(chosen_delta, close_delta, chosen_pen, close_pen)
close_because = _because_lines(close_delta, chosen_delta, close_pen, chosen_pen)
rej_because = _because_lines(rej_delta, chosen_delta, rej_pen, chosen_pen)

chosen_bullets, close_bullets = _dynamic_bullets(chosen_delta, close_delta, chosen_pen, close_pen)
_, rej_bullets = _dynamic_bullets(chosen_delta, rej_delta, chosen_pen, rej_pen)

# Make rejection reason explicit (use extracted rule text, not scoring logic)
if rules:
    rej_bullets = [f"Rejected because: {rules[0].rstrip('.')}."] + rej_bullets[:2]
else:
    rej_bullets = ["Rejected because it clashes with your constraints."] + rej_bullets[:2]

chosen_proof = _proof_lines(chosen_delta, close_delta, chosen_pen, close_pen)

max_delta = float(
    pd.to_numeric(df_sorted.get("deltaE_norm", pd.Series([1.0])), errors="coerce").fillna(0.0).max() or 1.0
)
max_pen = float(
    pd.to_numeric(df_sorted.get("constraint_penalty_norm", pd.Series([1.0])), errors="coerce").fillna(0.0).max()
    or 1.0
)
max_delta = max(max_delta, max(chosen_delta, close_delta, 1e-9))
max_pen = max(max_pen, max(chosen_pen, close_pen, 1e-9))

# -----------------------------
# Why this shade
# -----------------------------
total_candidates = int(len(df_sorted))
family_n = 0
if chosen_cluster is not None and "cluster_id" in df_sorted.columns:
    try:
        family_n = int((df_sorted["cluster_id"].astype("Int64") == chosen_cluster).sum())
    except Exception:
        family_n = 0

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="h2">Why this shade, and not another</div>', unsafe_allow_html=True)

family_line_a = f"Family detected: <b>{html.escape(chosen_family)}</b>."
family_line_b = (
    f"Shortlist within that family: <b>{family_n}</b> shades (out of <b>{total_candidates}</b> candidates)."
    if family_n > 0
    else f"Shortlist: <b>{total_candidates}</b> candidates."
)

st.markdown(
    f"""
<div class="family-step">
  <div class="family-step-title">Step 1 — family shortlist</div>
  <p class="family-step-line">{family_line_a}</p>
  <p class="family-step-line">{family_line_b}</p>
</div>
""",
    unsafe_allow_html=True,
)

chosen_title = _safe_title(chosen, "Best match")
close_title = _safe_title(close_alt, "Good alternative")
rej_title = _safe_title(rejected, "Rejected")

st.markdown(
    _render_snapshot(
        chosen_title=chosen_title,
        alt_title=close_title,
        chosen_delta=chosen_delta,
        alt_delta=close_delta,
        chosen_pen=chosen_pen,
        alt_pen=close_pen,
        max_delta=max_delta,
        max_pen=max_pen,
    ),
    unsafe_allow_html=True,
)

# -----------------------------
# Verdict global (repère immédiat)
# -----------------------------
st.markdown(
    f"""
<div class="verdict-strip">
  <div class="v-item best">
    <div class="v-k">Selected</div>
    <div class="v-v">{html.escape(chosen_title)}</div>
  </div>
  <div class="v-item good">
    <div class="v-k">Close alternative</div>
    <div class="v-v">{html.escape(close_title)}</div>
  </div>
  <div class="v-item rej">
    <div class="v-k">Rejected</div>
    <div class="v-v">{html.escape(rej_title)}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Cards: label each column (clear, immediate)
# -----------------------------
st.markdown(
    """
<div class="cards-head">
  <div class="cap"><b>BEST MATCH</b> — selected</div>
  <div class="cap"><b>GOOD</b> — close alternative</div>
  <div class="cap"><b>REJECTED</b> — clashes</div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 1, 1], gap="large")

with c1:
    st.markdown(
        _render_choice_card(
            title=chosen_title,
            status_label="BEST MATCH",
            status_key="best",
            card_kind="chosen",
            family_label=chosen_family,
            swatch_hex=chosen_hex,
            bullets=chosen_bullets,
            because_lines=chosen_because,
            fam_grade=chosen_fam_g,
            fit_grade=chosen_fit_g,
            close_grade=chosen_close_g,
            show_proof=True,
            proof_lines=chosen_proof,
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        _render_choice_card(
            title=close_title,
            status_label="GOOD ALTERNATIVE",
            status_key="close",
            card_kind="close",
            family_label=close_family,
            swatch_hex=close_hex,
            bullets=close_bullets,
            because_lines=close_because,
            fam_grade=close_fam_g,
            fit_grade=close_fit_g,
            close_grade=close_close_g,
            show_proof=False,
            proof_lines=None,
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        _render_choice_card(
            title=rej_title,
            status_label="REJECTED",
            status_key="reject",
            card_kind="rejected",
            family_label=rej_family,
            swatch_hex=rej_hex,
            bullets=rej_bullets,
            because_lines=rej_because,
            fam_grade=rej_fam_g,
            fit_grade=rej_fit_g,
            close_grade=rej_close_g,
            show_proof=False,
            proof_lines=None,
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="section">
  <p class="p">
    The app doesn’t look for the “best” colors. It looks for <em>your</em> colors.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
