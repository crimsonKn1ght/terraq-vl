"""Persistence for the row-id -> image-report-pair mapping shared by all indices.

``pairs.jsonl`` is written once by ``corpus/build_index.py`` in the exact order vectors
were added to the FAISS indices and the BM25 corpus, so a search result's integer row id
maps directly to line N of this file.
"""

from __future__ import annotations

import json
import os
from typing import List

from retrieval.base import RetrievedPair

PAIRS_FILE = "pairs.jsonl"
TEXT_INDEX = "text.faiss"
VISUAL_INDEX = "visual.faiss"
BM25_FILE = "bm25.pkl"


def write_pairs(records: List[dict], index_dir: str) -> str:
    """Persist pair metadata dicts (pair_id, image_path, report, meta) as JSONL."""
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, PAIRS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_pairs(index_dir: str) -> List[dict]:
    path = os.path.join(index_dir, PAIRS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build the index first: python -m corpus.build_index ..."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_from_row(row: dict, score: float) -> RetrievedPair:
    return RetrievedPair(
        pair_id=row["pair_id"],
        image_path=row["image_path"],
        report=row["report"],
        score=float(score),
        meta=row.get("meta", {}),
    )
