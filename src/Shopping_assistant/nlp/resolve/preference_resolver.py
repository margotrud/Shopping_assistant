# src/Shopping_assistant/nlp/resolve/preference_resolver.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from Shopping_assistant.nlp.schema import (
    Axis,
    Clause,
    Constraint,
    Mention,
    MentionKind,
    NLPResult,
    Polarity,
)


# -----------------------------
# Resolved (post-NLP) schema
# -----------------------------

@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """
    Does:
        Represent a resolved preference target (entity mention) with attached constraints.
    """
    mention: Mention
    constraints: Tuple[Constraint, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedConstraint:
    """
    Does:
        Represent a constraint kept at global scope when it cannot be safely attached.
    """
    constraint: Constraint


@dataclass(frozen=True, slots=True)
class ResolvedPreference:
    """
    Does:
        Provide a deterministic, scoring-ready aggregation of targets/dislikes/constraints.
    """
    text: str
    liked: Tuple[ResolvedTarget, ...] = ()
    disliked: Tuple[ResolvedTarget, ...] = ()
    global_constraints: Tuple[ResolvedConstraint, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Resolver
# -----------------------------

_PRIMARY_KINDS: Tuple[MentionKind, ...] = (MentionKind.COLOR, MentionKind.FINISH)


def resolve_preference(nlp: NLPResult) -> ResolvedPreference:
    """
    Does:
        Convert NLPResult (flat mentions/constraints) into resolved targets + scoped constraints.
    """
    clauses_by_id: Dict[int, Clause] = {c.clause_id: c for c in nlp.clauses}

    mentions_by_clause: Dict[int, List[Mention]] = {}
    for m in nlp.mentions:
        mentions_by_clause.setdefault(int(m.clause_id), []).append(m)

    constraints_by_clause: Dict[int, List[Constraint]] = {}
    for c in nlp.constraints:
        constraints_by_clause.setdefault(int(c.clause_id), []).append(c)

    liked_targets: List[ResolvedTarget] = []
    disliked_targets: List[ResolvedTarget] = []
    global_constraints: List[ResolvedConstraint] = []

    attached_constraints = 0
    unattached_constraints = 0

    # 1) Resolve clause-local attachment
    clause_ids = sorted(set(mentions_by_clause.keys()) | set(constraints_by_clause.keys()) | set(clauses_by_id.keys()))
    for clause_id in clause_ids:
        clause_mentions = mentions_by_clause.get(clause_id, [])
        clause_constraints = constraints_by_clause.get(clause_id, [])

        if not clause_mentions and clause_constraints:
            # No target candidates: keep constraints global
            for c in clause_constraints:
                global_constraints.append(ResolvedConstraint(c))
                unattached_constraints += 1
            continue

        # Split mentions by polarity first (deterministic)
        likes = [m for m in clause_mentions if m.polarity == Polarity.LIKE]
        dislikes = [m for m in clause_mentions if m.polarity == Polarity.DISLIKE]
        neutrals = [m for m in clause_mentions if m.polarity not in (Polarity.LIKE, Polarity.DISLIKE)]

        # Pick a single "primary" mention per polarity bucket for safe attachment
        like_primary = _pick_primary_target(likes) or _pick_primary_target(neutrals)
        dislike_primary = _pick_primary_target(dislikes)

        # Attach constraints only if we have exactly one plausible primary in the clause.
        # If both like_primary and dislike_primary exist, prefer like_primary attachment
        # and keep others global to avoid mis-scoping.
        if clause_constraints:
            attach_to = like_primary if like_primary is not None else dislike_primary
            if attach_to is not None:
                # Attach all clause constraints to the chosen target (minimal v0)
                # Scope refinement happens later (axis-based, connective-based, etc.)
                _add_target_with_constraints(
                    liked_targets if attach_to.polarity == Polarity.LIKE else disliked_targets,
                    attach_to,
                    clause_constraints,
                )
                attached_constraints += len(clause_constraints)
            else:
                for c in clause_constraints:
                    global_constraints.append(ResolvedConstraint(c))
                    unattached_constraints += 1

        # Ensure we also include targets even if no constraints were attached
        if like_primary is not None and not _has_target(liked_targets, like_primary):
            liked_targets.append(ResolvedTarget(mention=like_primary, constraints=()))
        if dislike_primary is not None and not _has_target(disliked_targets, dislike_primary):
            disliked_targets.append(ResolvedTarget(mention=dislike_primary, constraints=()))

    diagnostics: Dict[str, Any] = {
        "clauses": len(nlp.clauses),
        "mentions": len(nlp.mentions),
        "constraints": len(nlp.constraints),
        "attached_constraints": attached_constraints,
        "unattached_constraints": unattached_constraints,
        "liked_targets": len(liked_targets),
        "disliked_targets": len(disliked_targets),
    }

    return ResolvedPreference(
        text=nlp.text,
        liked=tuple(liked_targets),
        disliked=tuple(disliked_targets),
        global_constraints=tuple(global_constraints),
        diagnostics=diagnostics,
    )


# -----------------------------
# Helpers
# -----------------------------

def _pick_primary_target(cands: List[Mention]) -> Optional[Mention]:
    """
    Does:
        Return a single best target if unambiguous; otherwise None.
    """
    if not cands:
        return None

    # Prefer "primary" kinds first (color/finish), then highest confidence, then earliest span.
    def key(m: Mention) -> Tuple[int, float, int, int]:
        kind_rank = 0 if m.kind in _PRIMARY_KINDS else 1
        return (kind_rank, -float(m.confidence or 0.0), int(m.span.start), int(m.span.end))

    ordered = sorted(cands, key=key)
    best = ordered[0]

    # Ambiguity rule: if there is another candidate with same kind_rank and close confidence,
    # do not choose (prevents mis-attachment in multi-target clauses).
    if len(ordered) >= 2:
        b0 = key(ordered[0])
        b1 = key(ordered[1])
        same_kind_rank = b0[0] == b1[0]
        close_conf = abs(b0[1] - b1[1]) < 0.10  # because b* is negative confidence
        if same_kind_rank and close_conf:
            return None

    return best


def _has_target(targets: List[ResolvedTarget], mention: Mention) -> bool:
    """
    Does:
        Check if a mention is already present among resolved targets.
    """
    for t in targets:
        if t.mention.span == mention.span and t.mention.canonical == mention.canonical and t.mention.kind == mention.kind:
            return True
    return False


def _add_target_with_constraints(
    bucket: List[ResolvedTarget],
    mention: Mention,
    constraints: List[Constraint],
) -> None:
    """
    Does:
        Add or merge a target, extending its attached constraints deterministically.
    """
    for i, t in enumerate(bucket):
        if t.mention.span == mention.span and t.mention.canonical == mention.canonical and t.mention.kind == mention.kind:
            merged = tuple(list(t.constraints) + list(constraints))
            bucket[i] = ResolvedTarget(mention=t.mention, constraints=merged)
            return
    bucket.append(ResolvedTarget(mention=mention, constraints=tuple(constraints)))
