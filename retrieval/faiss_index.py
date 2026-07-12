"""FAISS dense-index helpers.

``IndexFlatIP`` (exact inner product) is used: it needs no training, is deterministic, and
is correct at the sample/corpus scale here. Vectors must be L2-normalized so inner product
equals cosine similarity. The single ``build_flat_ip_index`` seam is where an
``IndexIVFFlat`` / ``IndexHNSWFlat`` would be swapped in for MIMIC-scale corpora.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_flat_ip_index(vectors: np.ndarray):
    import faiss

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index, path: str) -> None:
    import faiss

    faiss.write_index(index, path)


def load_index(path: str):
    import faiss

    return faiss.read_index(path)


def search(index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, row_ids), each shape (top_k,), for a single query vector."""
    q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
    top_k = min(top_k, index.ntotal)
    scores, ids = index.search(q, top_k)
    return scores[0], ids[0]
