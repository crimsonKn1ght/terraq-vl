"""Common image-report corpus interface (MIMIC-CXR-ready).

Every concrete dataset adapter yields ``CorpusRecord`` objects with the same shape, so
``corpus/build_index.py`` is dataset-agnostic and a new modality (pathology,
ophthalmology) is just a new adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CorpusRecord:
    pair_id: str
    image_path: str   # local path to the image on disk
    report: str       # free-text report / findings / caption
    meta: dict = field(default_factory=dict)  # modality, source, study_id, ...


class CorpusLoader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[CorpusRecord]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
