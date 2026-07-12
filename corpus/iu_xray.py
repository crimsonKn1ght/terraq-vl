"""IU-Xray / OpenI chest-radiograph + report adapter (open dataset).

NOTE: IU-Xray HF mirrors are not standardized — there is no single canonical id, and the
one shipped as the default may not exist. Set ``corpus.hf_id`` (and, if needed,
``image_column`` / ``report_column``) in ``rag_eval.yaml`` to a mirror you have verified.
The ROCO adapter (``corpus.name: roco``) is a verified, working open radiology alternative.
"""

from __future__ import annotations

from corpus.hf_common import HFImageReportLoader


class IUXrayLoader(HFImageReportLoader):
    name = "iu_xray"

    @classmethod
    def from_cfg(cls, cfg: dict) -> "IUXrayLoader":
        c = cfg.get("corpus", {})
        return cls(
            # No verified canonical mirror — override corpus.hf_id with your own.
            hf_id=c.get("hf_id", "iu-xray-set-corpus.hf_id-in-rag_eval.yaml"),
            split=c.get("split", "train"),
            image_column=c.get("image_column", "image"),
            report_column=c.get("report_column", "report"),
            local_path=c.get("local_path", "./data/corpus"),
            max_pairs=c.get("max_pairs"),
            modality=c.get("modality", "chest_xray"),
            hf_config=c.get("hf_config"),
        )
