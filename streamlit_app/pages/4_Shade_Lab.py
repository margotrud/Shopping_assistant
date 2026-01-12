from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from ui.theme import inject_styles
from ui.nav import top_nav


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Shade Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()
top_nav(active="Shade Lab")

# ============================================================
# Bootstrap src
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ============================================================
# Load EXISTING model artifacts (NO recompute)
# ============================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    colors_path = (
        ROOT
        / "data"
        / "enriched_data"
        / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    )
    clusters_path = (
        ROOT
        / "data"
        / "enriched_data"
        / "color_cluster_assignments_fused.csv"
    )

    df_colors = pd.read_csv(colors_path)
    df_clusters = pd.read_csv(clusters_path)

    df_colors = df_colors.rename(columns={"chip_hex": "hex"})
    df_clusters = df_clusters.rename(columns={"chip_hex": "hex"})

    if "hex" not in df_colors.columns:
        raise ValueError("chip_hex missing in colors CSV")
    if "hex" not in df_clusters.columns:
        raise ValueError("chip_hex missing in clusters CSV")

    df = df_colors.merge(df_clusters[["hex", "cluster_id"]], on="hex", how="inner")
    df = df.dropna(subset=["hex", "cluster_id"]).copy()
    df["hex"] = df["hex"].astype(str).str.strip().str.lower()

    # safety: keep only valid-looking hex
    df = df[df["hex"].str.match(r"^#[0-9a-f]{6}$", na=False)]

    return df[["hex", "cluster_id"]]


df = load_data()

# ============================================================
# Query param (selected shade)
# ============================================================
def get_query_param(name: str) -> str | None:
    try:
        v = st.query_params.get(name)
        if v is None:
            return None
        if isinstance(v, list):
            return v[0] if v else None
        return str(v)
    except Exception:
        qp = st.experimental_get_query_params()
        vals = qp.get(name)
        return vals[0] if vals else None


selected_hex = get_query_param("shade")
if selected_hex:
    selected_hex = selected_hex.strip().lower()

# ============================================================
# Header
# ============================================================
st.markdown(
    textwrap.dedent(
        """
        <div class="page-header">
          <h1 class="page-title">Shade Lab</h1>
          <div class="page-sub">Click a shade to see how the model groups colors.</div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)

# ============================================================
# HTML Grid (clickable color tiles)
# ============================================================
def render_color_grid(
    hexes: list[str],
    *,
    cols: int = 10,
    tile_h: int = 72,
    gap: int = 16,
) -> None:
    tiles_html = []
    for h in hexes:
        href = f"?shade={quote(h, safe='')}"  # encodes '#'
        tiles_html.append(
            f'<a class="shade-tile" href="{href}" target="_self" '
            f'title="{h}" aria-label="Pick {h}" '
            f'style="background:{h};"></a>'
        )

    html = textwrap.dedent(
        f"""
        <style>
          .shade-grid {{
            display: grid;
            grid-template-columns: repeat({cols}, minmax(0, 1fr));
            gap: {gap}px;
            margin-top: 8px;
          }}
          .shade-tile {{
            display: block;
            height: {tile_h}px;
            border-radius: 18px;
            border: 1px solid rgba(19,42,99,0.14);
            box-shadow: 0 1px 0 rgba(19,42,99,0.04);
            text-decoration: none;
            outline: none;
            transition: transform 120ms ease, border-color 120ms ease;
          }}
          .shade-tile:hover {{
            border-color: rgba(19,42,99,0.28);
            transform: translateY(-1px);
          }}
          .shade-tile:active {{
            transform: translateY(0px);
          }}
        </style>

        <div class="shade-grid">
          {''.join(tiles_html)}
        </div>
        """
    ).strip()

    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# Cluster view
# ============================================================
def show_cluster(df: pd.DataFrame, center_hex: str) -> None:
    st.markdown(
        """
        <div style="margin: 6px 0 18px 0;">
          <a href="?" target="_self"
             style="text-decoration:none; font-weight:600;">
             ← Back to all colors
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if center_hex not in set(df["hex"].values):
        st.warning("Selected shade not found in artifacts.")
        return

    cluster_id = df.loc[df["hex"] == center_hex, "cluster_id"].iloc[0]
    cluster_df = df[df["cluster_id"] == cluster_id]

    st.markdown("### Selected color")
    st.markdown(
        textwrap.dedent(
            f"""
            <div style="
              background:{center_hex};
              height:90px;
              border-radius:22px;
              margin-bottom:18px;
              border:2px solid rgba(19,42,99,0.25);
            "></div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <h3 style="margin-bottom:6px;">Colors grouped together by the model
              <span style="opacity:.6; font-weight:400;">(cluster {cluster_id})</span>
            </h3>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    render_color_grid(cluster_df["hex"].head(60).tolist(), cols=10, tile_h=56, gap=12)

# ============================================================
# Router
# ============================================================
if selected_hex:
    show_cluster(df, selected_hex)
else:
    st.markdown("### Pick a color")
    render_color_grid(df["hex"].head(120).tolist(), cols=10, tile_h=72, gap=16)
