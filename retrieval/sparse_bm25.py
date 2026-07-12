"""Ablation (b): sparse BM25 retrieval over report text using the question as query."""

from __future__ import annotations

import os
from typing import List, Optional

from retrieval.base import BaseRetriever, RetrievedPair
from retrieval import store
from retrieval.bm25_index import load_bm25, tokenize


class SparseBM25Retriever(BaseRetriever):
    def __init__(self, bm25, pairs):
        self.bm25 = bm25
        self.pairs = pairs

    @classmethod
    def load(cls, cfg: dict) -> "SparseBM25Retriever":
        index_dir = cfg.get("retrieval", {}).get("index_dir", "./rag_index")
        bm25 = load_bm25(os.path.join(index_dir, store.BM25_FILE))
        pairs = store.read_pairs(index_dir)
        return cls(bm25, pairs)

    def retrieve(
        self,
        query_image_path: Optional[str],
        query_text: Optional[str],
        top_k: int,
    ) -> List[RetrievedPair]:
        if not query_text:
            return []
        scores = self.bm25.get_scores(tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [store.pair_from_row(self.pairs[i], scores[i]) for i in ranked]
