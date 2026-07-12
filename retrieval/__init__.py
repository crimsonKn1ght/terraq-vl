"""Retrieval package: dense-visual, sparse-BM25, and hybrid retrievers for the
Phase-2 ablation, behind a common ``BaseRetriever`` interface."""

from retrieval.base import BaseRetriever, RetrievedPair
from retrieval.factory import build_retriever, MODES

__all__ = ["BaseRetriever", "RetrievedPair", "build_retriever", "MODES"]
