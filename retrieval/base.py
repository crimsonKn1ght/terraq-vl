"""Retriever interface — the seam that makes the three ablation modes interchangeable.

Every retriever (dense-visual, sparse-BM25, hybrid) returns the same ``RetrievedPair``
list, so the prompt formatter (``ragcore.context_format``) and the eval runner are
completely agnostic to which retrieval modality produced the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievedPair:
    """A single retrieved image-report pair plus its retrieval score."""

    pair_id: str
    image_path: str
    report: str
    score: float
    meta: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Common interface for all retrieval modes used in the Phase-2 ablation."""

    @abstractmethod
    def retrieve(
        self,
        query_image_path: Optional[str],
        query_text: Optional[str],
        top_k: int,
    ) -> List[RetrievedPair]:
        """Return the ``top_k`` most relevant pairs for the given query.

        A retriever may use only one of the two query signals (e.g. the dense-visual
        retriever ignores ``query_text``; the BM25 retriever ignores the image).
        """

    @classmethod
    @abstractmethod
    def load(cls, cfg: dict) -> "BaseRetriever":
        """Construct a retriever from a parsed ``rag_eval.yaml`` config dict."""
