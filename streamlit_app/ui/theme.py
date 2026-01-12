from __future__ import annotations

from pathlib import Path
import streamlit as st


def inject_styles() -> None:
    """
    Does:
        Load ui/styles.css and inject it into the Streamlit app.
    """
    css_path = Path(__file__).with_name("styles.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
