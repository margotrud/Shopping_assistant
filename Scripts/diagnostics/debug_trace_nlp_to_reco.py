# Scripts/debug_trace_nlp_to_reco.py
"""
DEBUG-ONLY SCRIPT (not used by app).

Goal:
    Trace EXACTLY the same pipeline as the app (no re-implementation):
        recommend_from_text(text, assets=...)  [gold path]
    plus: structured printing + JSON export, with extra diagnostics computed ONLY from
    returned data and existing functions (no custom scoring / no rebuilt pipeline).

Usage:
    python Scripts/debug_trace_nlp_to_reco.py --text "I want a pink lipstick" --topk 10
    python Scripts/debug_trace_nlp_to_reco.py \
      --compare "I want a pink lipstick" \
      --compare "I want a bright pink lipstick" \
      --compare "I want a really bright pink lipstick" \
      --topk 10 --export data/reports/debug_trace_pink_app.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------
# Infra helpers
# ----------------------------

def _repo_root() -> Path:
    # pythonProject/
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> Path:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _safe(obj: Any) -> Any:
    """Convert dataclasses/Enums/tuples into JSON-safe python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _safe(asdict(obj))
    if hasattr(obj, "value") and not isinstance(obj, (dict, list, tuple)):
        try:
            return obj.value
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(x) for x in obj]
    return str(obj)


def _load_assets_from_defaults(*, root: Path):
    from Shopping_assistant.io.assets import load_assets

    data = root / "data"
    enriched_csv = data / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    prototypes_csv = data / "enriched_data" / "color_prototypes_fused_lab.csv"
    assignments_csv = data / "enriched_data" / "color_cluster_assignments_fused.csv"
    calibration_json = data / "models" / "color_scoring_calibration.json"

    missing = [p for p in (enriched_csv, prototypes_csv, assignments_csv, calibration_json) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing default asset files:\n" + "\n".join(f"- {p}" for p in missing))

    return load_assets(
        enriched_csv=enriched_csv,
        prototypes_csv=prototypes_csv,
        assignments_csv=assignments_csv,
        calibration_json=calibration_json,
    )


def _print_header(title: str) -> None:
    line = "=" * max(12, len(title))
    print("\n" + line)
    print(title)
    print(line)


def _print_kv(d: Dict[str, Any], *, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{pad}{k}: {v:.6g}")
        else:
            print(f"{pad}{k}: {v}")


# ----------------------------
# Trace parsing from app debug logs (no pipeline rebuild)
# ----------------------------

_RE_LIKE_CLUSTER = re.compile(r"\blike_cluster_id\s*=\s*(\d+)\b")
_RE_BASE_CLUSTERS = re.compile(r"\bbase_clusters\(n=\d+\):\s*(\[[^\]]*\])")
_RE_FINAL_CLUSTERS = re.compile(r"\bfinal_clusters\(n=\d+\):\s*(\[[^\]]*\])")
_RE_CONSTRAINTS_BLOB = re.compile(r"\[Constraints blob\]\s*\n\s*('.*?')\s*$", re.DOTALL | re.MULTILINE)

def _parse_int_list(txt: str) -> List[int]:
    # expects something like "[18, 16, 19]"
    txt = txt.strip()
    if not (txt.startswith("[") and txt.endswith("]")):
        return []
    inner = txt[1:-1].strip()
    if not inner:
        return []
    out = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def _extract_app_debug(debug_log: str) -> Dict[str, Any]:
    like_cluster_id = None
    m = _RE_LIKE_CLUSTER.search(debug_log or "")
    if m:
        like_cluster_id = int(m.group(1))

    base_clusters = []
    m = _RE_BASE_CLUSTERS.search(debug_log or "")
    if m:
        base_clusters = _parse_int_list(m.group(1))

    final_clusters = []
    m = _RE_FINAL_CLUSTERS.search(debug_log or "")
    if m:
        final_clusters = _parse_int_list(m.group(1))

    # Constraints blob appears as a repr printed by recommend_from_text debug
    constraints_blob = None
    # We also accept the simple line: print(f"  {constraints_blob!r}")
    blob_lines = []
    for line in (debug_log or "").splitlines():
        s = line.strip()
        if s.startswith("'") and s.endswith("'") and ("<=" in s or ">=" in s) and ":" in s:
            blob_lines.append(s)
    if blob_lines:
        # last one is most likely the one printed under [Constraints blob]
        constraints_blob = blob_lines[-1]
        # remove quotes
        try:
            constraints_blob = eval(constraints_blob)  # noqa: S307 (debug-only; controlled string)
        except Exception:
            constraints_blob = constraints_blob.strip("'")

    return {
        "like_cluster_id": like_cluster_id,
        "base_clusters": base_clusters,
        "final_clusters": final_clusters,
        "constraints_blob": constraints_blob,
    }


# ----------------------------
# Tables / signatures
# ----------------------------

def _summarize_top(df: pd.DataFrame, *, topk: int) -> pd.DataFrame:
    """
    Does:
        Display TopK like app outputs, plus raw dims that explain "why" (no extra scoring).
    """
    base_cols = [
        "rank",
        "score",
        "score_total",
        "deltaE_norm",
        "constraint_penalty_norm",
        "cluster_id",
        "brand_name",
        "product_name",
        "shade_name",
        "chip_hex",
    ]
    dim_cols = [
        "light_hsl",
        "sat_hsl",
        "depth",
        "colorfulness",
        "L_lab",
        "C_lab",
        "a_lab",
        "b_lab",
    ]
    penalty_cols = [c for c in df.columns if c.startswith("penalty_")]

    cols = [c for c in base_cols if c in df.columns]
    cols += [c for c in dim_cols if c in df.columns]
    cols += penalty_cols

    out = df.head(int(topk)).loc[:, cols].copy()

    # prefer score_total if present; keep score for backward compatibility
    if "score_total" in out.columns and "score" in out.columns:
        out = out.drop(columns=["score"])

    for c in ["brand_name", "product_name", "shade_name"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.slice(0, 60)

    return out


def _top_signature(df: pd.DataFrame, *, k: int) -> List[Tuple[str, str]]:
    if df.empty:
        return []
    cols = [c for c in ("product_id", "shade_id") if c in df.columns]
    if len(cols) < 2:
        return []
    sub = df.head(int(k))
    return list(zip(sub[cols[0]].astype(str).tolist(), sub[cols[1]].astype(str).tolist()))


# ----------------------------
# App-faithful tracing
# ----------------------------

def trace_pipeline_app(
    text: str,
    *,
    assets,
    candidate_clusters_topn: int,
    topk: int,
    lambda_constraints: float,
    lambda_preference: float,
    force_global_pool: bool,
) -> Dict[str, Any]:
    """
    Does:
        Run the exact app pipeline by calling recommend_from_text().
        Optionally forces candidate_cluster_ids = all clusters (still via recommend_from_text).
    """
    import Shopping_assistant.reco.recommend as reco
    from Shopping_assistant.nlp.interpretation.preference import interpret_nlp

    # NLP diagnostics (existing function; does NOT rebuild scoring)
    nlp_res = interpret_nlp(text, debug=True)

    # Determine "all clusters" only from already-loaded assets (no scoring)
    all_clusters: Optional[List[int]] = None
    if force_global_pool:
        inv = assets.inventory
        if "cluster_id" in inv.columns:
            all_clusters = sorted({int(x) for x in pd.to_numeric(inv["cluster_id"], errors="coerce").dropna().astype(int)})
        else:
            all_clusters = None

    # Capture recommend_from_text debug output without re-implementing its logic
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        top_df = reco.recommend_from_text(
            text,
            assets=assets,
            like_cluster_id=None,
            candidate_cluster_ids=all_clusters,
            candidate_clusters_topn=int(candidate_clusters_topn),
            topk=int(topk),
            lambda_constraints=float(lambda_constraints),
            lambda_preference=float(lambda_preference),
            constraints_blob_override=None,
            thresholds_override=None,
            debug=True,  # critical: emits the internal thresholds/pools/blob chosen by app
        )
    debug_log = buf.getvalue()

    # Structured extraction from app debug (no pipeline rebuild)
    dbg = _extract_app_debug(debug_log)

    trace: Dict[str, Any] = {
        "input": {"text": text, "mode": "GLOBAL_FORCED" if force_global_pool else "APP_DEFAULT"},
        "imports": {"reco_module_file": getattr(reco, "__file__", None)},
        "nlp": {"diagnostics": _safe(getattr(nlp_res, "diagnostics", {})), "trace": _safe(getattr(nlp_res, "trace", {}))},
        "app_debug": {
            "captured_stdout": debug_log,
            "parsed": dbg,
        },
        "_top_df": top_df,
        "_top_table": _summarize_top(top_df, topk=int(topk)).to_dict(orient="records"),
    }
    return trace


def _print_trace(trace: Dict[str, Any], *, topk: int) -> None:
    text = trace["input"]["text"]
    mode = trace["input"]["mode"]
    _print_header(f"QUERY ({mode}): {text}")

    _print_header("IMPORTS")
    _print_kv(trace.get("imports", {}))

    _print_header("NLP: clauses / mentions / constraints")
    diag = (trace.get("nlp", {}) or {}).get("diagnostics", {}) or {}
    _print_kv(
        {
            "clauses": len(diag.get("clauses", [])),
            "mentions": len(diag.get("mentions", [])),
            "constraints": len(diag.get("constraints", [])),
        }
    )

    clauses = diag.get("clauses", []) or []
    if clauses:
        print("\n  clauses:")
        for c in clauses:
            cid = c.get("clause_id")
            pol = c.get("polarity")
            txt = (c.get("text") or "").strip().replace("\n", " ")[:120]
            print(f"    - #{cid} [{pol}] {txt}")

    mentions = diag.get("mentions", []) or []
    if mentions:
        print("\n  mentions:")
        for m in mentions:
            kind = m.get("kind")
            pol = m.get("polarity")
            raw = (m.get("raw") or "").strip()
            canon = (m.get("canonical") or "").strip()
            conf = m.get("confidence", None)

            kind_s = "None" if kind is None else str(kind)
            pol_s = "None" if pol is None else str(pol)
            conf_s = "None" if conf is None else f"{float(conf):.3f}"

            print(f"    - {kind_s:<8} {pol_s:<7} conf={conf_s:<6} raw='{raw}' canonical='{canon}'")

    tr = (trace.get("nlp", {}) or {}).get("trace", {}) or {}
    constraints_final = tr.get("constraints_final", []) or []
    if constraints_final:
        print("\n  constraints_final:")
        for c in constraints_final:
            axis = c.get("axis")
            direction = c.get("direction")
            strength = c.get("strength")
            conf = c.get("confidence")
            ev = c.get("evidence")
            print(f"    - {axis} {direction} {strength} conf={conf} ev='{ev}'")
    else:
        print("\n  constraints_final: []")

    _print_header("APP INTERNALS (CAPTURED)")
    parsed = ((trace.get("app_debug", {}) or {}).get("parsed", {}) or {})
    _print_kv(
        {
            "like_cluster_id": parsed.get("like_cluster_id"),
            "base_clusters": parsed.get("base_clusters"),
            "final_clusters": parsed.get("final_clusters"),
            "constraints_blob": parsed.get("constraints_blob"),
        }
    )

    _print_header(f"TOP-K ({mode})")
    df = trace["_top_df"]
    print(_summarize_top(df, topk=int(topk)).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="Single query text")
    ap.add_argument("--compare", type=str, action="append", default=[], help="Repeatable: multiple query texts to compare")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--candidate-topn", type=int, default=10)
    ap.add_argument("--lambda-constraints", type=float, default=2.0)
    ap.add_argument("--lambda-preference", type=float, default=0.0)
    ap.add_argument("--export", type=str, default=None, help="Write full JSON trace to this path")
    ap.add_argument(
        "--with-global",
        action="store_true",
        help="Also run the same app pipeline but forcing candidate_cluster_ids = all clusters (still via recommend_from_text).",
    )
    args = ap.parse_args()

    _ensure_src_on_path()
    root = _repo_root()
    assets = _load_assets_from_defaults(root=root)

    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.compare:
        texts.extend([t for t in args.compare if t])
    if not texts:
        texts = [
            "I want a red lipstick",
            "I want a bright red lipstick",
            "I want a really bright red lipstick",
        ]

    traces: List[Dict[str, Any]] = []

    for t in texts:
        tr_app = trace_pipeline_app(
            t,
            assets=assets,
            candidate_clusters_topn=int(args.candidate_topn),
            topk=int(args.topk),
            lambda_constraints=float(args.lambda_constraints),
            lambda_preference=float(args.lambda_preference),
            force_global_pool=False,
        )
        traces.append(tr_app)
        _print_trace(tr_app, topk=int(args.topk))

        if args.with_global:
            tr_glob = trace_pipeline_app(
                t,
                assets=assets,
                candidate_clusters_topn=int(args.candidate_topn),
                topk=int(args.topk),
                lambda_constraints=float(args.lambda_constraints),
                lambda_preference=float(args.lambda_preference),
                force_global_pool=True,
            )
            traces.append(tr_glob)
            _print_trace(tr_glob, topk=int(args.topk))

    # Compare signatures across sequential traces (within the same mode ordering)
    if len(traces) >= 2:
        _print_header("TOP-K SIGNATURE COMPARISON")
        sigs = []
        for tr in traces:
            text = tr["input"]["text"]
            mode = tr["input"]["mode"]
            df = tr["_top_df"]
            sig = _top_signature(df, k=int(args.topk))
            sigs.append((text, mode, sig))

        for i in range(len(sigs) - 1):
            a_text, a_mode, a_sig = sigs[i]
            b_text, b_mode, b_sig = sigs[i + 1]
            same = a_sig == b_sig
            print(f"\nPair {i+1}:")
            print(f"  A: ({a_mode}) {a_text!r}")
            print(f"  B: ({b_mode}) {b_text!r}")
            print(f"  same_top{args.topk}: {same}")

    if args.export:
        out_path = (root / args.export).resolve() if not os.path.isabs(args.export) else Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_payload = []
        for tr in traces:
            df = tr.pop("_top_df")
            slim = {k: v for k, v in tr.items()}
            # keep compact top table (already dict records)
            export_payload.append(_safe(slim))
            tr["_top_df"] = df  # restore in-memory

        out_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[export] wrote {out_path}")


if __name__ == "__main__":
    main()
