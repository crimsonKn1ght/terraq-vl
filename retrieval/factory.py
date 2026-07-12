"""Map a retrieval-mode name to a concrete retriever (or ``None`` for the baseline).

Concrete retrievers are imported lazily so that merely importing this module (e.g. from
``rag_inference``) does not require faiss / sentence-transformers until a retriever is
actually constructed.
"""

from __future__ import annotations

from typing import Optional

from retrieval.base import BaseRetriever

# Modes recognised by the ablation driver. ``no_retrieval`` is the Phase-1 baseline.
MODES = ["no_retrieval", "dense_visual", "sparse_bm25", "hybrid"]


def build_retriever(mode: str, cfg: dict) -> Optional[BaseRetriever]:
    """Return a retriever for ``mode``, or ``None`` for the no-retrieval baseline."""
    if mode == "no_retrieval":
        return None
    if mode == "dense_visual":
        from retrieval.dense_visual import DenseVisualRetriever

        return DenseVisualRetriever.load(cfg)
    if mode == "sparse_bm25":
        from retrieval.sparse_bm25 import SparseBM25Retriever

        return SparseBM25Retriever.load(cfg)
    if mode == "hybrid":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever.load(cfg)
    raise ValueError(f"Unknown retrieval mode: {mode!r}. Expected one of {MODES}.")
