"""Image-report corpus loaders for the retrieval index.

Open datasets (IU-Xray/OpenI, ROCO) are usable now; MIMIC-CXR is a credentialed stub
behind the same interface, so it drops in once PhysioNet access is granted.
"""

from corpus.base import CorpusLoader, CorpusRecord
from corpus.registry import get_loader

__all__ = ["CorpusLoader", "CorpusRecord", "get_loader"]
