"""Phase-3 refinement hooks — wired into call sites but inert by default.

The research statement's Phase 3 (query decomposition, adaptive context-window
management, modality-aware retrieval weighting) depends on the Phase-2 ablation
findings, so these are deliberately left as identity / pass-through implementations.
Each is gated by a ``phase3.*`` flag in ``configs/rag_eval.yaml`` (all ``false`` now),
so enabling Phase 3 later is purely additive — no call site needs restructuring.
"""

from __future__ import annotations

from typing import List

from retrieval.base import RetrievedPair


def decompose_query(question: str, cfg: dict) -> List[str]:
    """Split a multi-part clinical question into independently-retrievable sub-queries.

    Phase-3 hook. Called by ``eval.runner`` when ``cfg['phase3']['query_decomposition']``
    is true. For now returns the question unchanged.
    """
    return [question]


def adaptive_window(
    pairs: List[RetrievedPair], question: str, cfg: dict
) -> List[RetrievedPair]:
    """Trim / expand retrieved reports to fit a dynamic token budget.

    Phase-3 hook. Called by ``ragcore.context_format.format_context`` when
    ``cfg['phase3']['adaptive_context_window']`` is true. For now a pass-through;
    the static ``eval.max_context_chars`` cap still applies downstream.
    """
    return pairs


def modality_weighting(query_meta: dict, cfg: dict) -> dict:
    """Return per-modality fusion weights for hybrid retrieval.

    Phase-3 hook. Called by ``retrieval.hybrid.HybridRetriever`` when
    ``cfg['phase3']['modality_aware_weighting']`` is true. For now returns the static
    weight from ``retrieval.modality_weight``.
    """
    w = float(cfg.get("retrieval", {}).get("modality_weight", 0.5))
    return {"visual": w, "text": 1.0 - w}
