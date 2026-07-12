"""Ablation (c): hybrid retrieval — dense (text + visual) ∪ sparse, fused by weighted
reciprocal-rank fusion, then cross-encoder reranked.

Pipeline:
  1. dense text search (clinical SBERT over report vectors, text.faiss)
  2. dense visual search (pooled CLIP, visual.faiss)
  3. sparse BM25 over report text
  4. weighted reciprocal-rank fusion -> candidate pool
  5. cross-encoder rerank (question vs report) -> top_k

The visual-vs-text fusion weight comes from ``retrieval.modality_weight`` (the Phase-3
``modality_aware_weighting`` hook overrides it when enabled).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from retrieval.base import BaseRetriever, RetrievedPair
from retrieval import store
from retrieval.encoders import ClinicalTextEncoder, VisualEncoder
from retrieval.faiss_index import load_index, search
from retrieval.bm25_index import load_bm25, tokenize
from retrieval.reranker import CrossEncoderReranker
from ragcore import phase3_stubs

_RRF_K = 60  # standard reciprocal-rank-fusion constant


class HybridRetriever(BaseRetriever):
    def __init__(self, text_index, visual_index, bm25, pairs,
                 text_encoder, visual_encoder, reranker, cfg):
        self.text_index = text_index
        self.visual_index = visual_index
        self.bm25 = bm25
        self.pairs = pairs
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.reranker = reranker
        self.cfg = cfg
        r = cfg.get("retrieval", {})
        self.candidate_pool = int(r.get("candidate_pool", 30))
        self.modality_weight = float(r.get("modality_weight", 0.5))

    @classmethod
    def load(cls, cfg: dict) -> "HybridRetriever":
        r = cfg.get("retrieval", {})
        index_dir = r.get("index_dir", "./rag_index")
        device = cfg.get("model", {}).get("device", "cpu")

        text_index = load_index(os.path.join(index_dir, store.TEXT_INDEX))
        visual_index = load_index(os.path.join(index_dir, store.VISUAL_INDEX))
        bm25 = load_bm25(os.path.join(index_dir, store.BM25_FILE))
        pairs = store.read_pairs(index_dir)

        text_encoder = ClinicalTextEncoder(r.get("text_encoder_id"), device)
        visual_encoder = VisualEncoder(
            r.get("visual_encoder_id", "openai/clip-vit-large-patch14"), device
        )
        reranker = CrossEncoderReranker(r.get("cross_encoder_id"), device)
        return cls(text_index, visual_index, bm25, pairs,
                   text_encoder, visual_encoder, reranker, cfg)

    def _ranked_ids(self, index, qvec, k) -> List[int]:
        _, ids = search(index, qvec, k)
        return [int(i) for i in ids if i >= 0]

    def _fuse(self, ranked_lists: List[tuple]) -> Dict[int, float]:
        """Weighted reciprocal-rank fusion. ``ranked_lists`` is [(ids, weight), ...]."""
        fused: Dict[int, float] = {}
        for ids, weight in ranked_lists:
            for rank, doc_id in enumerate(ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (_RRF_K + rank + 1)
        return fused

    def retrieve(
        self,
        query_image_path: Optional[str],
        query_text: Optional[str],
        top_k: int,
    ) -> List[RetrievedPair]:
        if self.cfg.get("phase3", {}).get("modality_aware_weighting", False):
            weights = phase3_stubs.modality_weighting({}, self.cfg)
            w_visual, w_text = weights["visual"], weights["text"]
        else:
            w_visual, w_text = self.modality_weight, 1.0 - self.modality_weight

        ranked_lists = []
        if query_text:
            tvec = self.text_encoder.encode([query_text])[0]
            ranked_lists.append((self._ranked_ids(self.text_index, tvec, self.candidate_pool), w_text))
            bm25_scores = self.bm25.get_scores(tokenize(query_text))
            bm25_ids = sorted(range(len(bm25_scores)),
                              key=lambda i: bm25_scores[i], reverse=True)[: self.candidate_pool]
            ranked_lists.append((bm25_ids, w_text))
        if query_image_path:
            vvec = self.visual_encoder.encode_paths([query_image_path])[0]
            ranked_lists.append((self._ranked_ids(self.visual_index, vvec, self.candidate_pool), w_visual))

        fused = self._fuse(ranked_lists)
        if not fused:
            return []

        pool_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[: self.candidate_pool]
        candidates = [store.pair_from_row(self.pairs[i], fused[i]) for i in pool_ids]

        # Cross-encoder rerank needs text; fall back to fused order if no query text.
        if query_text:
            return self.reranker.rerank(query_text, candidates, top_k)
        return candidates[:top_k]
