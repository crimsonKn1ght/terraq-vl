"""Ablation (a): dense visual-only retrieval over pooled CLIP image embeddings."""

from __future__ import annotations

import os
from typing import List, Optional

from retrieval.base import BaseRetriever, RetrievedPair
from retrieval import store
from retrieval.encoders import VisualEncoder
from retrieval.faiss_index import load_index, search


class DenseVisualRetriever(BaseRetriever):
    def __init__(self, index, pairs, encoder: VisualEncoder):
        self.index = index
        self.pairs = pairs
        self.encoder = encoder

    @classmethod
    def load(cls, cfg: dict) -> "DenseVisualRetriever":
        r = cfg.get("retrieval", {})
        index_dir = r.get("index_dir", "./rag_index")
        device = cfg.get("model", {}).get("device", "cpu")
        index = load_index(os.path.join(index_dir, store.VISUAL_INDEX))
        pairs = store.read_pairs(index_dir)
        encoder = VisualEncoder(r.get("visual_encoder_id", "openai/clip-vit-large-patch14"), device)
        return cls(index, pairs, encoder)

    def retrieve(
        self,
        query_image_path: Optional[str],
        query_text: Optional[str],
        top_k: int,
    ) -> List[RetrievedPair]:
        if query_image_path is None:
            return []
        qvec = self.encoder.encode_paths([query_image_path])[0]
        scores, ids = search(self.index, qvec, top_k)
        return [store.pair_from_row(self.pairs[i], s) for s, i in zip(scores, ids) if i >= 0]
