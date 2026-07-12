"""Exact-match accuracy for medical VQA, reported overall and by answer type.

Closed (yes/no) and open answers are aggregated separately because the proposal's
hallucination analysis cares about *where* the model fails, and closed vs open behave
very differently.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List

_ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (VQA convention)."""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in re.split(r"\s+", s) if t and t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, gold: str) -> bool:
    """True when the normalized prediction equals the normalized gold answer.

    Open-ended predictions are often a full sentence containing the answer, so we also
    accept the case where the normalized gold answer appears as a token span in the
    normalized prediction.
    """
    pred_n = normalize(prediction)
    gold_n = normalize(gold)
    if not gold_n:
        return pred_n == gold_n
    if pred_n == gold_n:
        return True
    # token-span containment (e.g. gold "cardiomegaly" inside a longer answer)
    return f" {gold_n} " in f" {pred_n} "


def aggregate_em(records: List[Dict]) -> Dict[str, float]:
    """Aggregate exact-match over per-sample records.

    Each record needs ``prediction``, ``answer`` and ``answer_type`` keys.
    """
    by_type: Dict[str, List[bool]] = {"closed": [], "open": []}
    all_hits: List[bool] = []
    for r in records:
        hit = exact_match(r["prediction"], r["answer"])
        all_hits.append(hit)
        by_type.setdefault(r.get("answer_type", "open"), []).append(hit)

    def acc(hits: List[bool]) -> float:
        return round(sum(hits) / len(hits), 4) if hits else 0.0

    return {
        "em_all": acc(all_hits),
        "em_closed": acc(by_type.get("closed", [])),
        "em_open": acc(by_type.get("open", [])),
        "n": len(all_hits),
        "n_closed": len(by_type.get("closed", [])),
        "n_open": len(by_type.get("open", [])),
    }
