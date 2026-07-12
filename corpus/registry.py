"""Name -> corpus-loader factory, driven by ``corpus.name`` in ``rag_eval.yaml``."""

from __future__ import annotations

from corpus.base import CorpusLoader


def get_loader(cfg: dict) -> CorpusLoader:
    """Construct the corpus loader named by ``cfg['corpus']['name']``."""
    name = cfg.get("corpus", {}).get("name", "synthetic")

    if name == "synthetic":
        from corpus.synthetic import SyntheticLoader

        c = cfg.get("corpus", {})
        return SyntheticLoader(
            local_path=c.get("local_path", "./data/corpus"),
            max_pairs=c.get("max_pairs", 10),
            modality=c.get("modality", "chest_xray"),
        )
    if name == "iu_xray":
        from corpus.iu_xray import IUXrayLoader

        return IUXrayLoader.from_cfg(cfg)
    if name == "roco":
        from corpus.roco import ROCOLoader

        return ROCOLoader.from_cfg(cfg)
    if name == "mimic_cxr":
        from corpus.mimic_cxr import MIMICCXRLoader

        return MIMICCXRLoader.from_cfg(cfg)
    if name == "galaxy_zoo":
        from corpus.galaxy_zoo import GalaxyZooLoader

        return GalaxyZooLoader.from_cfg(cfg)

    raise ValueError(
        f"Unknown corpus {name!r}. Expected: synthetic | iu_xray | roco | mimic_cxr | galaxy_zoo."
    )
