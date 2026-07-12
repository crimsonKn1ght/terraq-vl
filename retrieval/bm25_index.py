"""BM25 sparse index over report text (the sparse-retrieval ablation).

Tokenization is deliberately a simple lowercase word split so there is no extra NLP
dependency beyond ``rank-bm25``.
"""

from __future__ import annotations

import pickle
import re
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(reports: List[str]):
    """Return (BM25Okapi, tokenized_corpus) built from report strings."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(r) for r in reports]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def save_bm25(bm25, tokenized: List[List[str]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)


def load_bm25(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"]
