from __future__ import annotations

import streamlit as st


# Streamlit multipage routes come from filenames in /pages.

_ROUTES = [
    ("Home", "/"),
    ("Playground", "/Playground"),
    ("Explain", "/Explain_Recommendation"),
    ("Model", "Model_Evaluation"),
]



def top_nav(active: str) -> None:
    """
    Does:
        Render a top navigation bar using pure HTML anchors (full CSS control).
    """
    items = []
    for label, href in _ROUTES:
        cls = "nav-link active" if label == active else "nav-link"
        items.append(f'<a class="{cls}" href="{href}" target="_self">{label}</a>')

    st.markdown(
        f"""
        <nav class="topnav">
          {''.join(items)}
        </nav>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )
