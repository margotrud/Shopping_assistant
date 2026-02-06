__all__ = [
    "recommend_from_text",
    "resolve_anchor_from_text",
    "resolve_effective_anchor_from_text",
]

def __getattr__(name: str):
    if name in __all__:
        from .recommend import (
            recommend_from_text,
            resolve_anchor_from_text,
            resolve_effective_anchor_from_text,
        )
        return {
            "recommend_from_text": recommend_from_text,
            "resolve_anchor_from_text": resolve_anchor_from_text,
            "resolve_effective_anchor_from_text": resolve_effective_anchor_from_text,
        }[name]
    raise AttributeError(name)
