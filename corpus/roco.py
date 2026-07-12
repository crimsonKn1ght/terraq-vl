"""ROCO (Radiology Objects in COntext) image-caption adapter (open dataset).

ROCO pairs medical images with caption text; the caption is used as the ``report``.
Column names are configurable via ``rag_eval.yaml``.
"""

from __future__ import annotations

from corpus.hf_common import HFImageReportLoader


class ROCOLoader(HFImageReportLoader):
    name = "roco"

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ROCOLoader":
        c = cfg.get("corpus", {})
        return cls(
            hf_id=c.get("hf_id", "eltorio/ROCO-radiology"),  # verified: cols image, caption
            split=c.get("split", "train"),
            image_column=c.get("image_column", "image"),
            report_column=c.get("report_column", "caption"),
            local_path=c.get("local_path", "./data/corpus"),
            max_pairs=c.get("max_pairs"),
            modality=c.get("modality", "radiology"),
            hf_config=c.get("hf_config"),
        )
