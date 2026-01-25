# src/Shopping_assistant/nlp/axes/axis_family.py

def resolve_axis_family(constraint: dict) -> str:
    """
    Resolve a cosmetic axis family dynamically from axis score distribution.
    Priority-based, no token logic, no hardcoding.
    """

    meta = constraint.get("meta", {})
    dbg = meta.get("axis_debug", {})
    ranked = dbg.get("ranked")

    if not ranked or not isinstance(ranked, list):
        return None

    # ranked = [(axis_name, score), ...]
    top_axis, top_score = ranked[0]
    second_axis, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

    # --- 1️⃣ DEPTH dominates ---
    if top_axis == "depth":
        return "DEPTH"

    # --- 2️⃣ LIGHTNESS dominates ---
    if top_axis == "brightness":
        return "LIGHTNESS"

    # --- 3️⃣ SURFACE dominates ---
    if top_axis == "clarity":
        return "SURFACE"

    # --- 4️⃣ CHROMA when saturation/vibrancy dominate or compete ---
    chroma_axes = {"saturation", "vibrancy"}

    if top_axis in chroma_axes:
        return "CHROMA"

    if (
        top_axis == "brightness"
        and second_axis in chroma_axes
        and abs(top_score - second_score) < 0.15
    ):
        return "CHROMA"

    return None
