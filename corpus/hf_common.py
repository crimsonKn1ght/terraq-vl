"""Shared base for Hugging Face image-report corpus adapters (IU-Xray, ROCO).

Streams from the Hub and caps at ``max_pairs`` so only the examples actually needed are
downloaded (important for large multi-shard corpora like ROCO). Caches PIL images to disk
and yields ``CorpusRecord``s. Column names are configurable so a different dataset revision
can be accommodated without code changes.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Iterator, Optional

from corpus.base import CorpusLoader, CorpusRecord

logger = logging.getLogger(__name__)


class HFImageReportLoader(CorpusLoader):
    name = "hf"

    def __init__(
        self,
        hf_id: str,
        split: str,
        image_column: str,
        report_column: str,
        local_path: str,
        max_pairs: Optional[int] = None,
        modality: str = "image",
        hf_config: Optional[str] = None,
    ):
        self.hf_id = hf_id
        self.split = split
        self.image_column = image_column
        self.report_column = report_column
        self.max_pairs = max_pairs
        self.modality = modality
        self.hf_config = hf_config
        self.image_dir = os.path.join(local_path, self.name + "_images")
        os.makedirs(self.image_dir, exist_ok=True)
        self._count = None

    def __len__(self) -> int:
        if self.max_pairs is not None:
            return self.max_pairs
        return self._count or 0

    def __iter__(self) -> Iterator[CorpusRecord]:
        from datasets import load_dataset

        logger.info("Streaming corpus %s split=%s (cap=%s)", self.hf_id, self.split, self.max_pairs)
        ds = load_dataset(self.hf_id, self.hf_config, split=self.split, streaming=True)
        rows = iter(ds)
        if self.max_pairs is not None:
            rows = itertools.islice(rows, self.max_pairs)

        count = 0
        for i, row in enumerate(rows):
            if self.image_column not in row:
                raise KeyError(
                    f"image_column {self.image_column!r} not in {list(row)}. "
                    "Set corpus.image_column in rag_eval.yaml."
                )
            if self.report_column not in row:
                raise KeyError(
                    f"report_column {self.report_column!r} not in {list(row)}. "
                    "Set corpus.report_column in rag_eval.yaml."
                )
            pair_id = f"{self.name}_{i}"
            image_path = os.path.join(self.image_dir, f"{pair_id}.png")
            if not os.path.exists(image_path):
                row[self.image_column].convert("RGB").save(image_path)
            report = row[self.report_column]
            if isinstance(report, (list, tuple)):
                report = " ".join(str(x) for x in report)
            yield CorpusRecord(
                pair_id=pair_id,
                image_path=image_path,
                report=str(report),
                meta={"modality": self.modality, "source": self.name},
            )
            count += 1
        self._count = count
