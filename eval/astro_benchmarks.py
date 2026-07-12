"""Astronomy VQA benchmark, template-generated from Galaxy10 DECaLS labels.

Established astronomy VQA sets are scarce, so questions are synthesized from the verified
morphology labels (a standard way to bootstrap VQA benchmarks). Each image yields either:

* an **open** question ("What is the morphological classification?") -> answer is the class, or
* a **closed** yes/no question ("Is this galaxy best classified as <class>?"), balanced so
  half the closed answers are "yes" and half "no".

Uses the dataset's ``test`` split, disjoint from the ``train`` split used for the corpus, so
there is no retrieval-from-the-answer leakage. Returns ``eval.benchmarks.VQASample`` objects,
so the runner / metrics are identical to the medical path.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import List, Optional

from eval.benchmarks import VQASample
from corpus.galaxy_zoo import GALAXY10_CLASSES

logger = logging.getLogger(__name__)


def _make_qa(index: int, label: int):
    """Deterministically build (question, answer, answer_type) for a sample."""
    true_class = GALAXY10_CLASSES[label]
    if index % 2 == 0:
        return (
            "What is the morphological classification of this galaxy?",
            true_class,
            "open",
        )
    # closed yes/no, alternating the correct answer for balance
    if (index // 2) % 2 == 0:
        return (f"Is this galaxy best classified as {true_class.lower()}?", "yes", "closed")
    wrong_class = GALAXY10_CLASSES[(label + 5) % len(GALAXY10_CLASSES)]
    return (f"Is this galaxy best classified as {wrong_class.lower()}?", "no", "closed")


def load_galaxy_vqa(
    split: str = "test",
    n: Optional[int] = None,
    cache_dir: str = "./rag_cache",
    hf_id: str = "matthieulel/galaxy10_decals",
) -> List[VQASample]:
    from datasets import load_dataset

    logger.info("Building galaxy VQA from %s split=%s (cap=%s)", hf_id, split, n)
    ds = load_dataset(hf_id, split=split, streaming=True)
    rows = iter(ds)
    if n is not None:
        rows = itertools.islice(rows, n)

    img_dir = os.path.join(cache_dir, f"galaxy_vqa_{split}")
    os.makedirs(img_dir, exist_ok=True)

    samples: List[VQASample] = []
    for i, row in enumerate(rows):
        sample_id = f"galaxy_vqa_{split}_{i}"
        image_path = os.path.join(img_dir, f"{sample_id}.png")
        if not os.path.exists(image_path):
            row["image"].convert("RGB").save(image_path)
        question, answer, answer_type = _make_qa(i, int(row["label"]))
        samples.append(
            VQASample(
                sample_id=sample_id,
                image_path=image_path,
                question=question,
                answer=answer,
                answer_type=answer_type,
            )
        )
    logger.info("Built %d galaxy VQA samples", len(samples))
    return samples
