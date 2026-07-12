"""MIMIC-CXR adapter — credentialed stub (PhysioNet access required).

This is the dataset named in the research statement. It requires a signed PhysioNet
data-use agreement, so it cannot be auto-downloaded. The class shape and the config keys
it expects are in place, so once a credentialed local dump is available it becomes the
active corpus by setting ``corpus.name: mimic_cxr`` and ``corpus.local_path`` to the dump.

Expected local layout (one common organization):
    <local_path>/files/<p..>/<patient>/<study>/<dicom_or_jpg>
    <local_path>/mimic-cxr-reports/...            # free-text reports
A real implementation parses the report sections (FINDINGS/IMPRESSION) and pairs each
study's frontal image with its report text.
"""

from __future__ import annotations

from typing import Iterator

from corpus.base import CorpusLoader, CorpusRecord


class MIMICCXRLoader(CorpusLoader):
    name = "mimic_cxr"

    def __init__(self, local_path: str, max_pairs=None, modality: str = "chest_xray"):
        self.local_path = local_path
        self.max_pairs = max_pairs
        self.modality = modality

    @classmethod
    def from_cfg(cls, cfg: dict) -> "MIMICCXRLoader":
        c = cfg.get("corpus", {})
        return cls(
            local_path=c.get("local_path", "./data/mimic_cxr"),
            max_pairs=c.get("max_pairs"),
            modality=c.get("modality", "chest_xray"),
        )

    def __len__(self) -> int:
        raise NotImplementedError(self._msg())

    def __iter__(self) -> Iterator[CorpusRecord]:
        raise NotImplementedError(self._msg())

    @staticmethod
    def _msg() -> str:
        return (
            "MIMIC-CXR requires PhysioNet credentialed access and is not auto-downloadable. "
            "Obtain access (https://physionet.org/content/mimic-cxr/), point "
            "corpus.local_path at the local dump, and implement report/image pairing here. "
            "Until then use corpus.name: iu_xray | roco | synthetic."
        )
