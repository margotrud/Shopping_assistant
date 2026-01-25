# streamlit_app/pages/1_Playground.py
from __future__ import annotations

import base64
import colorsys
import html
import io
import os
import re
import sys
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# Page config MUST be first (Streamlit best practice)
# -----------------------------
st.set_page_config(page_title="Playground", layout="wide", initial_sidebar_state="collapsed")

# Fix noisy Streamlit+torch watcher crash (log spam / occasional UI weirdness)
try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

# -----------------------------
# Bootstrap (infra): make src importable
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # pythonProject/
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui.theme import inject_styles
from ui.nav import top_nav
from ui.bootstrap import warmup_nlp_stack  # ✅ single source of truth

# -----------------------------
# Accent color (sync with Home.py)
# -----------------------------
ACCENT = "#7A2E2E"
ACCENT_SOFT = "rgba(122, 46, 46, 0.12)"

inject_styles()
top_nav(active="Playground")


def _show_inline_loader(msg: str) -> Any:
    """
    Does:
        Render a small inline spinner + message; returns a placeholder to clear later.
    """
    ph = st.empty()
    ph.markdown(
        textwrap.dedent(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:12px 0 18px 0;">
              <div style="
                width:16px;height:16px;border-radius:50%;
                border:2px solid rgba(19,42,99,0.16);
                border-top-color: {ACCENT};
                animation: spin 0.8s linear infinite;
              "></div>
              <div style="color: rgba(19,42,99,0.70); font-size: 14px;">{html.escape(msg)}</div>
            </div>
            <style>
              @keyframes spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )
    return ph


_ID_FLOAT_RE = re.compile(r"^\s*(\d+)\.0\s*$")


def _norm_id(x: Any) -> str:
    """
    Does:
        Normalize an id from pandas (handles floats -> int-ish strings, trims, drops NaN).
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    m = _ID_FLOAT_RE.match(s)
    if m:
        return m.group(1)
    return s


def _render_html(markup: str) -> None:
    """
    Does:
        Render raw HTML without Streamlit markdown re-parsing (fixes HTML->code blocks).
    """
    markup = (markup or "").strip()
    if not markup:
        return

    if hasattr(st, "html"):
        # type: ignore[attr-defined]
        st.html(markup)  # pyright: ignore
        return

    import streamlit.components.v1 as components

    components.html(markup, height=560, scrolling=False)


# -----------------------------
# Luxe CSS (no gradients, stable heights, equal card sizes)
# -----------------------------
st.markdown(
    textwrap.dedent(
        f"""
        <style>
        :root {{
          --accent: {ACCENT};
          --accent-soft: {ACCENT_SOFT};
          --ink: rgba(19,42,99,0.96);
          --muted: rgba(19,42,99,0.62);
          --border: rgba(19,42,99,0.12);
          --card: rgba(255,255,255,0.86);

          --primary-color: var(--accent) !important;
          --primaryColor: var(--accent) !important;
          --st-color-primary: var(--accent) !important;
          --stPrimaryColor: var(--accent) !important;
        }}

        ::selection {{ background: rgba(122, 46, 46, 0.22) !important; }}
        ::-moz-selection {{ background: rgba(122, 46, 46, 0.22) !important; }}

        *:focus,
        *:focus-visible {{ outline-color: var(--accent) !important; }}

        button[data-testid="baseButton-primary"],
        div[data-testid="stFormSubmitButton"] button {{
          background-color: var(--accent) !important;
          border-color: var(--accent) !important;
          color: #fff !important;
          box-shadow: none !important;
          border-radius: 14px !important;
        }}
        button[data-testid="baseButton-primary"]:hover,
        div[data-testid="stFormSubmitButton"] button:hover {{
          box-shadow: 0 10px 22px rgba(122, 46, 46, 0.26) !important;
          transform: translateY(-1px);
        }}

        div[data-testid="stTextInput"] div[data-baseweb="input"] {{
          border-color: rgba(19,42,99,0.16) !important;
          box-shadow: none !important;
          border-radius: 14px !important;
        }}
        div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 3px rgba(122, 46, 46, 0.14) !important;
        }}

        .play-wrap {{
          max-width: 1120px;
          margin: 0 auto;
          padding: 6px 2px 0 2px;
        }}

        .hero {{
          margin: 18px 0 18px 0;
        }}
        .hero .kicker {{
          letter-spacing: 0.22em;
          text-transform: uppercase;
          font-size: 12px;
          color: rgba(19,42,99,0.60);
          margin-bottom: 8px;
        }}
        .hero .h1 {{
          font-family: "Libre Baskerville", serif;
          font-size: 44px;
          line-height: 1.05;
          color: var(--ink);
          margin-bottom: 8px;
        }}
        .hero .p {{
          color: rgba(19,42,99,0.62);
          font-size: 15px;
          margin-bottom: 0px;
        }}

        .hr {{
          height: 1px;
          background: rgba(19,42,99,0.10);
          margin: 16px 0 18px 0;
        }}

        .section-title {{
          letter-spacing: 0.18em;
          text-transform: uppercase;
          font-size: 12px;
          color: rgba(19,42,99,0.56);
          margin: 10px 0 10px 2px;
        }}

        /* force equal-height cards inside Streamlit columns */
        div[data-testid="column"] > div {{
          height: 100%;
        }}
        div[data-testid="column"] {{
          display: flex;
        }}
        div[data-testid="column"] > div {{
          display: flex;
          flex-direction: column;
          flex: 1 1 auto;
        }}

        .luxe-card {{
          height: 100%;
          min-height: 510px;
          display: flex;
          flex-direction: column;
          border: 1px solid var(--border);
          background: var(--card);
          border-radius: 22px;
          overflow: hidden;
          box-shadow: 0 10px 26px rgba(19,42,99,0.06);
          transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease;
        }}
        .luxe-card:hover {{
          transform: translateY(-2px);
          box-shadow: 0 14px 34px rgba(19,42,99,0.10);
          border-color: rgba(122, 46, 46, 0.24);
        }}

        .luxe-media {{
          position: relative;
          background: #FFFFFF;
          border-bottom: 1px solid rgba(19,42,99,0.08);
          flex: 0 0 auto;
        }}
        .luxe-media--card {{ aspect-ratio: 16 / 11; }}

        .luxe-media img {{
          width: 100%;
          height: 100%;
          object-fit: contain;
          object-position: center;
          display: block;
          padding: 18px;
        }}

        .luxe-body {{
          padding: 14px 16px 14px 16px;
          display: flex;
          flex-direction: column;
          flex: 1 1 auto;
        }}

        .luxe-brand {{
          letter-spacing: 0.22em;
          text-transform: uppercase;
          font-size: 11px;
          color: rgba(19,42,99,0.56);
          margin-bottom: 8px;
        }}

        .luxe-title {{
          font-size: 16px;
          font-weight: 700;
          color: rgba(19,42,99,0.96);
          line-height: 1.25;
          margin-bottom: 6px;

          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
          min-height: 40px;
        }}

        .luxe-shade {{
          font-size: 13px;
          color: rgba(19,42,99,0.72);
          margin-bottom: 10px;

          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
          min-height: 34px;
        }}

        .chips {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: auto;
        }}
        .chip {{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 7px 10px;
          border-radius: 999px;
          border: 1px solid rgba(19,42,99,0.10);
          background: rgba(255,255,255,0.82);
        }}
        .chip .dot {{
          width: 10px; height: 10px;
          border-radius: 999px;
          border: 1px solid rgba(19,42,99,0.18);
          background: #fff;
        }}
        .chip span {{
          font-size: 12px;
          color: rgba(19,42,99,0.68);
          white-space: nowrap;
        }}

        .pill {{
          display:inline-flex;
          align-items:center;
          gap:8px;
          padding: 7px 10px;
          border-radius: 999px;
          border: 1px solid rgba(122,46,46,0.22);
          background: rgba(122,46,46,0.08);
          margin-bottom: 10px;
        }}
        .pill .dot {{
          width: 8px; height: 8px;
          border-radius: 999px;
          background: var(--accent);
        }}
        .pill span {{
          font-size: 12px;
          color: rgba(19,42,99,0.78);
        }}
        </style>
        """
    ),
    unsafe_allow_html=True,
)


def _maybe_set_default_env_paths() -> None:
    """
    Does:
        Set SA_* asset env vars once per session if missing, using repo-relative defaults.
    """
    root = Path(os.environ.get("SA_ASSETS_ROOT", "data")).resolve()

    enriched_default = root / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    calib_default = root / "models" / "color_scoring_calibration.json"

    if not os.environ.get("SA_ENRICHED_CSV_PATH") and enriched_default.exists():
        os.environ["SA_ENRICHED_CSV_PATH"] = str(enriched_default)
    if not os.environ.get("SA_CALIBRATION_JSON_PATH") and calib_default.exists():
        os.environ["SA_CALIBRATION_JSON_PATH"] = str(calib_default)


@st.cache_resource(show_spinner=False)
def _assets() -> Any:
    """
    Does:
        Load assets from env vars or default repo paths (no cluster prototypes/assignments).
    """
    from Shopping_assistant.io.assets import load_assets  # type: ignore

    _maybe_set_default_env_paths()

    enriched = os.environ.get("SA_ENRICHED_CSV_PATH")
    calib = os.environ.get("SA_CALIBRATION_JSON_PATH")

    if enriched and calib:
        # IMPORTANT: load_assets expects Path-like objects (not str)
        return load_assets(enriched_csv=Path(enriched), calibration_json=Path(calib))

    raise FileNotFoundError(
        "Missing assets. Provide either:\n"
        "- SA_ENRICHED_CSV_PATH and SA_CALIBRATION_JSON_PATH env vars, or\n"
        "- files at data/enriched_data/Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv "
        "and data/models/color_scoring_calibration.json"
    )


def _pick_first(row: pd.Series, cols: list[str]) -> str:
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v:
                return v
    return ""


def _pick_hex(row: pd.Series) -> str:
    hx = _pick_first(row, ["chip_hex", "hex", "hex_color", "color_hex", "rgb_hex"])
    hx = (hx or "").strip()
    if hx and not hx.startswith("#") and len(hx) == 6:
        hx = f"#{hx}"
    if hx.startswith("#") and len(hx) == 7:
        return hx
    return ""


def _pick_id_any(row: pd.Series, cols: list[str]) -> str:
    return _pick_first(row, cols)


def _hex_to_rgb01(hx: str) -> Optional[tuple[float, float, float]]:
    if not isinstance(hx, str):
        return None
    s = hx.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return None


def _hex_to_hue_deg(hx: str) -> Optional[float]:
    rgb = _hex_to_rgb01(hx)
    if rgb is None:
        return None
    h, _s, _v = colorsys.rgb_to_hsv(*rgb)
    return float((h * 360.0) % 360.0)


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


def _canon_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("™", " ").replace("®", " ").replace("’", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _guess_mime_from_suffix(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    return "application/octet-stream"


@st.cache_data(show_spinner=False)
def _file_to_data_uri(path: Path, *, max_side_px: int = 1200, jpeg_quality: int = 86) -> str:
    mime = _guess_mime_from_suffix(path)
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side_px, max_side_px))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            b = buf.getvalue()

        enc = base64.b64encode(b).decode("ascii")
        return f"data:image/jpeg;base64,{enc}"
    except Exception:
        b = path.read_bytes()
        enc = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{enc}"


@st.cache_data(show_spinner=False)
def _discover_image_cache_dir() -> Path:
    candidates = [
        ROOT / "Scripts" / "tools" / "data" / "_product_image_cache",
        ROOT / "scripts" / "tools" / "data" / "_product_image_cache",
        ROOT / "Scripts" / "Tools" / "data" / "_product_image_cache",
        ROOT / "data" / "_product_image_cache",
        ROOT / "Data" / "_product_image_cache",
    ]

    def _stats(p: Path) -> tuple[int, bool, int]:
        if not p.exists() or not p.is_dir():
            return (0, False, 0)
        has_index = (p / "images_index.csv").exists()
        imgs = list(p.glob("*__*.*"))
        img_count = len([x for x in imgs if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        score = (100000 if has_index else 0) + img_count
        return (score, has_index, img_count)

    ranked: list[tuple[int, Path]] = []
    for p in candidates:
        score, _hi, _ic = _stats(p)
        if score > 0:
            ranked.append((score, p))

    if not ranked:
        try:
            for p in ROOT.rglob("_product_image_cache"):
                score, _hi, _ic = _stats(p)
                if score > 0:
                    ranked.append((score, p))
        except Exception:
            pass

    if ranked:
        ranked.sort(key=lambda t: (-t[0], len(str(t[1]))))
        return ranked[0][1]

    return ROOT / "Scripts" / "tools" / "data" / "_product_image_cache"


@st.cache_data(show_spinner=False)
def _load_local_image_maps() -> tuple[
    Path,
    dict[tuple[str, str], Path],
    dict[tuple[str, str, str], Path],
    dict[tuple[str, str], Path],
    dict[str, Path],
]:
    out_dir = _discover_image_cache_dir()
    idx_path = out_dir / "images_index.csv"

    id_map: dict[tuple[str, str], Path] = {}
    name_map: dict[tuple[str, str, str], Path] = {}
    name2_map: dict[tuple[str, str], Path] = {}
    shade_all: list[tuple[str, Path]] = []

    if idx_path.exists():
        df = pd.read_csv(idx_path, dtype=str)
        for _, r in df.iterrows():
            pid = _norm_id(r.get("product_id"))
            sid = _norm_id(r.get("shade_id"))
            if not pid or not sid:
                continue

            p_img: Optional[Path] = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = out_dir / f"{pid}__{sid}{ext}"
                if cand.exists():
                    p_img = cand
                    break
            if not p_img:
                continue

            id_map[(pid, sid)] = p_img

            b = _canon_text(r.get("brand_name") or "")
            pr = _canon_text(r.get("product_name") or "")
            sh = _canon_text(r.get("shade_name") or "")

            if b and pr and sh:
                name_map[(b, pr, sh)] = p_img
            if b and sh:
                name2_map[(b, sh)] = p_img
            if sh:
                shade_all.append((sh, p_img))

    if out_dir.exists():
        for p in out_dir.glob("*__*.*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            stem = p.stem
            if "__" not in stem:
                continue
            pid, sid = stem.split("__", 1)
            pid = _norm_id(pid)
            sid = _norm_id(sid)
            if pid and sid and (pid, sid) not in id_map:
                id_map[(pid, sid)] = p

    shade_u_map: dict[str, Path] = {}
    if shade_all:
        c = pd.Series([k for k, _ in shade_all]).value_counts()
        for sh, p in shade_all:
            if int(c.get(sh, 0)) == 1:
                shade_u_map[sh] = p

    return out_dir, id_map, name_map, name2_map, shade_u_map


def _pick_local_image_data_uri_any(
    row: pd.Series,
    *,
    id_map: dict[tuple[str, str], Path],
    name_map: dict[tuple[str, str, str], Path],
    name2_map: dict[tuple[str, str], Path],
    shade_u_map: dict[str, Path],
) -> Optional[str]:
    pid_raw = _pick_id_any(row, ["product_id", "productId", "pid"])
    sid_raw = _pick_id_any(row, ["shade_id", "shadeId", "sid", "variant_id", "sku_id", "sku"])
    pid = _norm_id(pid_raw)
    sid = _norm_id(sid_raw)

    if pid and sid:
        p = id_map.get((pid, sid))
        if p and p.exists():
            try:
                return _file_to_data_uri(p)
            except Exception:
                return None

        if not pid.startswith("P"):
            pid2 = f"P{pid}"
            p2 = id_map.get((pid2, sid))
            if p2 and p2.exists():
                try:
                    return _file_to_data_uri(p2)
                except Exception:
                    return None

    b = _canon_text(_pick_first(row, ["brand_name", "brand"]))
    pr = _canon_text(_pick_first(row, ["product_name", "product", "name"]))
    sh = _canon_text(_pick_first(row, ["shade_name", "shade", "variant_name"]))

    if b and pr and sh:
        p3 = name_map.get((b, pr, sh))
        if p3 and p3.exists():
            try:
                return _file_to_data_uri(p3)
            except Exception:
                return None

    if b and sh:
        p4 = name2_map.get((b, sh))
        if p4 and p4.exists():
            try:
                return _file_to_data_uri(p4)
            except Exception:
                return None

    if sh:
        p5 = shade_u_map.get(sh)
        if p5 and p5.exists():
            try:
                return _file_to_data_uri(p5)
            except Exception:
                return None

    return None


def _explain_from_resolved(text: str) -> dict[str, Any]:
    from Shopping_assistant.nlp.interpretation.preference import interpret_nlp  # type: ignore
    from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference  # type: ignore
    from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index  # type: ignore

    nlp_res = interpret_nlp(text, debug=False)
    resolved = resolve_preference(nlp_res)

    likes: list[str] = []
    dislikes: list[str] = []
    constraints: list[str] = []

    for t in resolved.liked:
        if t.mention.canonical:
            likes.append(t.mention.canonical)
        for c in t.constraints:
            if c.evidence:
                constraints.append(c.evidence)

    for t in resolved.disliked:
        if t.mention.canonical:
            dislikes.append(t.mention.canonical)
        for c in t.constraints:
            if c.evidence:
                constraints.append(c.evidence)

    for gc in resolved.global_constraints:
        if gc.constraint.evidence:
            constraints.append(gc.constraint.evidence)

    def _dedup(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x.strip())
        return out

    likes_u = _dedup(likes)
    dislikes_u = _dedup(dislikes)
    constraints_u = _dedup(constraints)

    target_hue: Optional[float] = None
    target_color: Optional[str] = None
    if likes_u:
        try:
            idx = build_world_alias_index(include_xkcd=True)
            cand = likes_u[0].strip().lower()
            info = idx.get(cand) if isinstance(idx, dict) else None
            hx = info.get("hex") if isinstance(info, dict) else None
            if isinstance(hx, str) and hx:
                target_hue = _hex_to_hue_deg(hx)
                target_color = cand
        except Exception:
            target_hue = None
            target_color = None

    return {
        "likes": likes_u,
        "dislikes": dislikes_u,
        "constraints": constraints_u,
        "target_color": target_color,
        "target_hue": target_hue,
    }


def _recommend(text: str, *, assets: Any, topk: int = 64) -> pd.DataFrame:
    from Shopping_assistant.reco.recommend import recommend_from_text  # type: ignore

    df = recommend_from_text(text, assets=assets, topk=topk)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("recommend_from_text() must return a pandas.DataFrame.")
    return df


def _chip(dot_color: Optional[str], label: str) -> str:
    dot = (
        dot_color
        if (isinstance(dot_color, str) and dot_color.startswith("#") and len(dot_color) == 7)
        else "#FFFFFF"
    )
    return (
        f'<div class="chip">'
        f'<span class="dot" style="background:{html.escape(dot)};"></span>'
        f"<span>{html.escape(label)}</span>"
        f"</div>"
    )


def _pill_top_match() -> str:
    return '<div class="pill"><span class="dot"></span><span>Top match</span></div>'


def _build_rationale_chips(row: pd.Series, *, top: pd.DataFrame, expl: dict[str, Any]) -> list[str]:
    chips: list[str] = []

    hx = _pick_hex(row)
    hue = _hex_to_hue_deg(hx) if hx else None
    target_hue = expl.get("target_hue")
    if hue is not None and isinstance(target_hue, float):
        ad = abs(_angle_diff_deg(hue, target_hue))
        chips.append(_chip(hx or None, f"Hue Δ{ad:.0f}°"))

    if "deltaE_norm" in top.columns and pd.notna(row.get("deltaE_norm", None)):
        try:
            de = float(row["deltaE_norm"])
            chips.append(_chip(None, f"ΔE {de:.2f}"))
        except Exception:
            pass

    if expl.get("constraints") and "constraint_penalty_norm" in top.columns:
        try:
            pen = float(row.get("constraint_penalty_norm", 0.0))
            if pen <= 0.05:
                chips.append(_chip(None, "Good constraint fit"))
        except Exception:
            pass

    return chips[:3]


def _render_card_html(
    *,
    is_top: bool,
    brand: str,
    product: str,
    shade: str,
    image_url: str,
    chips_html: str,
) -> None:
    b = html.escape(brand or "—")
    p = html.escape(product or "—")
    s = html.escape(shade or "—")
    top_badge = _pill_top_match() if is_top else ""

    markup = (
        f'<div class="luxe-card">'
        f'  <div class="luxe-media luxe-media--card">'
        f'    <img src="{html.escape(image_url)}" alt="product image"/>'
        f"  </div>"
        f'  <div class="luxe-body">'
        f"    {top_badge}"
        f'    <div class="luxe-brand">{b}</div>'
        f'    <div class="luxe-title">{p}</div>'
        f'    <div class="luxe-shade">{s}</div>'
        f'    <div class="chips">{chips_html}</div>'
        f"  </div>"
        f"</div>"
    )
    _render_html(markup)


# -----------------------------
# Page
# -----------------------------
st.markdown('<div class="play-wrap">', unsafe_allow_html=True)

st.markdown(
    textwrap.dedent(
        """
        <div class="hero">
          <div class="kicker">Playground</div>
          <div class="h1">Find the shade you mean.</div>
          <p class="p">Describe a lipstick in natural language. The ranking adapts to color, depth, brightness, and constraints</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# show "Results: 3/6" from the start (stateful)
if "show_n" not in st.session_state:
    st.session_state["show_n"] = 3

c_opt, _ = st.columns([1, 3], gap="large")
with c_opt:
    show_n = st.radio(
        "Results",
        options=[3, 6],
        horizontal=True,
        index=0 if st.session_state["show_n"] == 3 else 1,
        key="show_n",
    )

with st.form("playground_form", clear_on_submit=False):
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        text = st.text_input(
            label="Query",
            label_visibility="collapsed",
            placeholder="e.g. I want a deep red lipstick, not too bright.",
            key="playground_query",
        )
    with right:
        submitted = st.form_submit_button("Find shades", type="primary", use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

if submitted:
    if not text.strip():
        st.warning("Please enter a short description.")
        st.stop()

    # warmup ONLY when needed (cached inside warmup_nlp_stack)
    loader = _show_inline_loader("Loading engine…")
    try:
        warmup_nlp_stack()
    finally:
        loader.empty()

    st.caption("Curated selection. Always shown as a 3-column grid.")

    assets = _assets()
    out_dir, id_map, name_map, name2_map, shade_u_map = _load_local_image_maps()

    loader = _show_inline_loader("Understanding your request…")
    expl = _explain_from_resolved(text)
    loader.empty()

    loader = _show_inline_loader("Finding the best shades…")
    top = _recommend(text, assets=assets, topk=64)
    loader.empty()

    if top.empty:
        st.warning("No results found.")
        st.stop()

    items: list[dict[str, Any]] = []
    for i in range(len(top)):
        row = top.iloc[i]
        img_local = _pick_local_image_data_uri_any(
            row,
            id_map=id_map,
            name_map=name_map,
            name2_map=name2_map,
            shade_u_map=shade_u_map,
        )
        if not img_local:
            continue

        brand = _pick_first(row, ["brand_name", "brand"]) or "—"
        product = _pick_first(row, ["product_name", "product", "name"]) or "—"
        shade = _pick_first(row, ["shade_name", "shade", "variant_name"]) or "—"

        chips = _build_rationale_chips(row, top=top, expl=expl)
        chips_html = ("".join(chips) if chips else _chip(_pick_hex(row) or None, "Personalized match")).strip()

        items.append(
            {
                "img": img_local,
                "brand": brand,
                "product": product,
                "shade": shade,
                "chips_html": chips_html,
            }
        )
        if len(items) >= int(st.session_state["show_n"]):
            break

    if not items:
        st.warning(
            f"No results with local images found. "
            f"Discovered cache dir: '{out_dir}'. "
            f"Check that the folder contains files like '{{product_id}}__{{shade_id}}.jpg'."
        )
        st.stop()

    st.markdown('<div class="section-title">Matches</div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="large")
    for idx, it in enumerate(items):
        with cols[idx % 3]:
            _render_card_html(
                is_top=(idx == 0),
                brand=it["brand"],
                product=it["product"],
                shade=it["shade"],
                image_url=it["img"],
                chips_html=it["chips_html"],
            )

st.markdown("</div>", unsafe_allow_html=True)
