"""Synthetic image-report corpus for dependency-light smoke tests.

Generates a handful of distinct procedurally-drawn images plus templated chest-x-ray-like
reports. No downloads, no PHI — lets the full build_index -> retrieve -> evaluate path run
end-to-end before any real dataset is wired up.
"""

from __future__ import annotations

import os
import random
from typing import Iterator

from PIL import Image, ImageDraw

from corpus.base import CorpusLoader, CorpusRecord

_FINDINGS = [
    "Findings: bilateral lower-lobe airspace opacities consistent with pneumonia.",
    "Findings: cardiomegaly with an enlarged cardiac silhouette; lungs are clear.",
    "Findings: no acute cardiopulmonary abnormality; normal chest radiograph.",
    "Findings: small right-sided pleural effusion with blunting of the costophrenic angle.",
    "Findings: hyperinflated lungs with flattened diaphragms suggesting emphysema.",
    "Findings: left upper-lobe consolidation; no pneumothorax identified.",
    "Findings: interstitial edema with vascular congestion and Kerley B lines.",
    "Findings: right perihilar mass-like opacity; further evaluation recommended.",
]


class SyntheticLoader(CorpusLoader):
    def __init__(self, local_path: str, max_pairs: int = 10, modality: str = "chest_xray",
                 seed: int = 42):
        self.image_dir = os.path.join(local_path, "synthetic_images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.max_pairs = max_pairs
        self.modality = modality
        self.seed = seed

    def __len__(self) -> int:
        return self.max_pairs

    def _draw(self, idx: int, rng: random.Random) -> str:
        path = os.path.join(self.image_dir, f"synthetic_{idx}.png")
        if not os.path.exists(path):
            img = Image.new("RGB", (224, 224), color=(rng.randint(0, 60),) * 3)
            draw = ImageDraw.Draw(img)
            for _ in range(rng.randint(3, 7)):
                x0, y0 = rng.randint(0, 200), rng.randint(0, 200)
                x1, y1 = x0 + rng.randint(10, 60), y0 + rng.randint(10, 60)
                shade = rng.randint(80, 230)
                draw.ellipse([x0, y0, x1, y1], fill=(shade, shade, shade))
            img.save(path)
        return path

    def __iter__(self) -> Iterator[CorpusRecord]:
        rng = random.Random(self.seed)
        for i in range(self.max_pairs):
            report = _FINDINGS[i % len(_FINDINGS)]
            yield CorpusRecord(
                pair_id=f"synthetic_{i}",
                image_path=self._draw(i, rng),
                report=report,
                meta={"modality": self.modality, "source": "synthetic"},
            )
