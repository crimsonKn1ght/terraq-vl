"""Cross-encoder reranker for the hybrid retriever.

Scores each (question, candidate-report) pair jointly and re-sorts, which is more precise
than the first-stage similarity used to assemble the candidate pool.
"""

from __future__ import annotations

import logging
from typing import List

from retrieval.base import RetrievedPair

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        logger.info("Loaded cross-encoder reranker %s", model_name)

    def rerank(
        self, query_text: str, candidates: List[RetrievedPair], top_k: int
    ) -> List[RetrievedPair]:
        if not candidates:
            return []
        scores = self.model.predict([(query_text, c.report) for c in candidates])
        for c, s in zip(candidates, scores):
            c.score = float(s)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
