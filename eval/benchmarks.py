"""Medical VQA benchmark loaders (Phase 1): VQA-RAD and PathVQA.

Both are loaded via the Hugging Face ``datasets`` library (already a project dependency).
Images are cached to disk so each sample exposes an ``image_path`` usable by *both* the
VLM (``load_and_process_image``) and the dense-visual retriever (``encode_paths``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# HF dataset ids. Both expose columns: image (PIL), question (str), answer (str).
BENCHMARK_HF_IDS = {
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "path_vqa": "flaviagiammarino/path-vqa",
}

_CLOSED_ANSWERS = {"yes", "no"}


@dataclass
class VQASample:
    sample_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str  # "closed" (yes/no) | "open"


def _infer_answer_type(answer: str) -> str:
    return "closed" if answer.strip().lower() in _CLOSED_ANSWERS else "open"


def get_benchmark(
    name: str,
    split: str = "test",
    n: Optional[int] = None,
    cache_dir: str = "./rag_cache",
) -> List[VQASample]:
    """Load up to ``n`` samples of a benchmark, caching images under ``cache_dir``."""
    # Astronomy benchmark is generated, not a plain HF VQA set — dispatch separately.
    if name == "galaxy_vqa":
        from eval.astro_benchmarks import load_galaxy_vqa

        return load_galaxy_vqa(split=split, n=n, cache_dir=cache_dir)

    if name not in BENCHMARK_HF_IDS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Expected one of "
            f"{list(BENCHMARK_HF_IDS) + ['galaxy_vqa']}."
        )

    import itertools

    from datasets import load_dataset

    hf_id = BENCHMARK_HF_IDS[name]
    logger.info("Loading benchmark %s (%s) split=%s (cap=%s)", name, hf_id, split, n)

    # Stream + cap when n is set so we don't download the whole split for a small sample.
    if n is not None:
        ds = load_dataset(hf_id, split=split, streaming=True)
        rows = itertools.islice(iter(ds), n)
    else:
        rows = load_dataset(hf_id, split=split)

    img_dir = os.path.join(cache_dir, f"{name}_{split}")
    os.makedirs(img_dir, exist_ok=True)

    samples: List[VQASample] = []
    for i, row in enumerate(rows):
        sample_id = f"{name}_{split}_{i}"
        image_path = os.path.join(img_dir, f"{sample_id}.png")
        if not os.path.exists(image_path):
            row["image"].convert("RGB").save(image_path)
        answer = str(row["answer"])
        samples.append(
            VQASample(
                sample_id=sample_id,
                image_path=image_path,
                question=str(row["question"]),
                answer=answer,
                answer_type=_infer_answer_type(answer),
            )
        )

    logger.info("Loaded %d samples from %s", len(samples), name)
    return samples
