from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import html

@dataclass(frozen=True, slots=True)
class InventoryStats:
    n_products: Optional[int]
    n_shades: Optional[int]
    n_families: Optional[int]


def _try_load_inventory_df() -> Optional[pd.DataFrame]:
    """
    Does:
        Best-effort load of an inventory CSV from common project locations.
    """
    candidates = [
        Path("assets/inventory_lipstick_with_numeric_axes.csv"),
        Path("data/inventory_lipstick_with_numeric_axes.csv"),
        Path("inventory_lipstick_with_numeric_axes.csv"),
        Path("src/Shopping_assistant/assets/inventory_lipstick_with_numeric_axes.csv"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    return None


def compute_inventory_stats(df: Optional[pd.DataFrame]) -> InventoryStats:
    """
    Does:
        Compute simple headline stats for the Home / Model Card pages.
    """
    if df is None or df.empty:
        return InventoryStats(None, None, None)

    # Heuristics: try common columns
    n_products = None
    for col in ["product_id", "sku", "product", "product_name"]:
        if col in df.columns:
            n_products = df[col].nunique(dropna=True)
            break

    # shades
    n_shades = None
    for col in ["shade_id", "shade", "shade_name", "variant_name"]:
        if col in df.columns:
            n_shades = df[col].nunique(dropna=True)
            break

    # families
    n_families = None
    for col in ["family", "color_family", "cluster_family", "family_name", "cluster_name"]:
        if col in df.columns:
            n_families = df[col].nunique(dropna=True)
            break

    return InventoryStats(n_products, n_shades, n_families)


def render_hero(
    title: str,
    subtitle: str,
    kicker: str = "Lipstick recommender",
) -> None:
    """
    Does:
        Render the Home hero block with luxe typography.
    """
    st.markdown(
        f"""
        <div class="hero">
          <div class="kicker">{kicker}</div>
          <div class="h1">{title}</div>
          <p class="p">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(stats: InventoryStats) -> None:
    """
    Does:
        Render 3 simple headline stats in a grid (with graceful fallback).
    """
    def fmt(x: Optional[int]) -> str:
        return "—" if x is None else f"{x:,}".replace(",", " ")

    st.markdown(
        f"""
        <div class="stats-grid">
          <div class="stat">
            <div class="value">{fmt(stats.n_products)}</div>
            <div class="label">Products</div>
          </div>
          <div class="stat">
            <div class="value">{fmt(stats.n_shades)}</div>
            <div class="label">Shades</div>
          </div>
          <div class="stat">
            <div class="value">{fmt(stats.n_families)}</div>
            <div class="label">Color families</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_card(
    brand: str,
    product: str,
    shade: str,
    hex_color: str,
    meta: str = "",
) -> None:
    """
    Does:
        Render a single recommendation card with a swatch + text.
    """
    hex_color = hex_color if (isinstance(hex_color, str) and hex_color.startswith("#")) else "#D9D2C8"
    st.markdown(
        f"""
        <div class="card">
          <div class="product">
            <div class="swatch" style="background:{hex_color};"></div>
            <div>
              <p class="title">{brand} — {product}</p>
              <p class="meta">{shade}{(" · " + meta) if meta else ""}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_row(items: list[str]) -> None:
    """Does:
        Render a row of small chips for extracted intents/constraints.
    """
    safe = [html.escape(str(x)) for x in items]
    chips = "".join([f'<div class="chip">{x}</div>' for x in safe])
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

def load_inventory_df_cached() -> Optional[pd.DataFrame]:
    """
    Does:
        Cached loader wrapper for inventory dataframe.
    """
    # streamlit cache wrapper
    @st.cache_data(show_spinner=False)
    def _load() -> Optional[pd.DataFrame]:
        return _try_load_inventory_df()

    return _load()
