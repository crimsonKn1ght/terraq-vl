"""Astronomy corpus adapter: Galaxy10 DECaLS (galaxy morphology images + class labels).

Verified dataset: ``matthieulel/galaxy10_decals`` (splits train/test; columns ``image`` PNG,
``label`` int 0-9). The dataset has no free-text reports, so a descriptive "report" is
synthesized from the morphology class — enough text for the SBERT / BM25 / cross-encoder
retrieval signals. This mirrors how the medical adapters expose image-report pairs, so the
rest of the pipeline is unchanged.

Demonstrates the general recipe for porting the pipeline to a new domain: a new
``CorpusLoader`` that yields ``CorpusRecord(image_path, report, meta)``.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Iterator, Optional

from corpus.base import CorpusLoader, CorpusRecord

logger = logging.getLogger(__name__)

# Galaxy10 DECaLS index -> class name (from the dataset card).
GALAXY10_CLASSES = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# Short morphological descriptions, to give the text retrievers real signal.
GALAXY10_DESCRIPTIONS = {
    "Disturbed Galaxies": "An irregular galaxy with a distorted, asymmetric shape, often from gravitational interaction.",
    "Merging Galaxies": "Two or more galaxies colliding and merging, showing tidal tails and overlapping structure.",
    "Round Smooth Galaxies": "A smooth elliptical galaxy with a round, featureless light profile and no spiral arms.",
    "In-between Round Smooth Galaxies": "A smooth elliptical galaxy of intermediate, slightly flattened roundness.",
    "Cigar Shaped Smooth Galaxies": "A smooth elliptical galaxy with an elongated, cigar-like elliptical shape.",
    "Barred Spiral Galaxies": "A spiral galaxy with a central bar-shaped structure of stars from which spiral arms extend.",
    "Unbarred Tight Spiral Galaxies": "A spiral galaxy without a central bar, with tightly wound spiral arms.",
    "Unbarred Loose Spiral Galaxies": "A spiral galaxy without a central bar, with loosely wound, open spiral arms.",
    "Edge-on Galaxies without Bulge": "A disk galaxy seen edge-on as a thin line of light, lacking a prominent central bulge.",
    "Edge-on Galaxies with Bulge": "A disk galaxy seen edge-on with a prominent bright central bulge.",
}


def class_report(label: int) -> str:
    """Synthesize a descriptive report string for a morphology class index."""
    name = GALAXY10_CLASSES[label]
    desc = GALAXY10_DESCRIPTIONS.get(name, "")
    return f"Morphology: {name}. {desc}".strip()


class GalaxyZooLoader(CorpusLoader):
    name = "galaxy_zoo"

    def __init__(
        self,
        hf_id: str = "matthieulel/galaxy10_decals",
        split: str = "train",
        local_path: str = "./data/corpus_astro",
        max_pairs: Optional[int] = None,
        modality: str = "galaxy",
    ):
        self.hf_id = hf_id
        self.split = split
        self.max_pairs = max_pairs
        self.modality = modality
        self.image_dir = os.path.join(local_path, "galaxy_images")
        os.makedirs(self.image_dir, exist_ok=True)
        self._count = None

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GalaxyZooLoader":
        c = cfg.get("corpus", {})
        return cls(
            hf_id=c.get("hf_id", "matthieulel/galaxy10_decals"),
            split=c.get("split", "train"),
            local_path=c.get("local_path", "./data/corpus_astro"),
            max_pairs=c.get("max_pairs"),
            modality=c.get("modality", "galaxy"),
        )

    def __len__(self) -> int:
        return self.max_pairs if self.max_pairs is not None else (self._count or 0)

    def __iter__(self) -> Iterator[CorpusRecord]:
        from datasets import load_dataset

        logger.info("Streaming galaxy corpus %s split=%s (cap=%s)", self.hf_id, self.split, self.max_pairs)
        ds = load_dataset(self.hf_id, split=self.split, streaming=True)
        rows = iter(ds)
        if self.max_pairs is not None:
            rows = itertools.islice(rows, self.max_pairs)

        count = 0
        for i, row in enumerate(rows):
            pair_id = f"galaxy_{self.split}_{i}"
            image_path = os.path.join(self.image_dir, f"{pair_id}.png")
            if not os.path.exists(image_path):
                row["image"].convert("RGB").save(image_path)
            label = int(row["label"])
            yield CorpusRecord(
                pair_id=pair_id,
                image_path=image_path,
                report=class_report(label),
                meta={"modality": self.modality, "source": "galaxy10_decals",
                      "class": GALAXY10_CLASSES[label]},
            )
            count += 1
        self._count = count
