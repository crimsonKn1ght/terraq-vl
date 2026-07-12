"""Build the retrieval indices from a corpus.

    python -m corpus.build_index --config configs/rag_eval.yaml
    python -m corpus.build_index --config configs/rag_eval.yaml --synthetic --max_pairs 10

Writes to ``retrieval.index_dir``:
    text.faiss     dense clinical-SBERT report vectors
    visual.faiss   pooled CLIP image vectors (same vision tower as the VLM)
    bm25.pkl       sparse BM25 over report text
    pairs.jsonl    row-id -> {pair_id, image_path, report, meta}
"""

from __future__ import annotations

import argparse
import logging
import os

import yaml

from corpus.registry import get_loader
from retrieval import store
from retrieval.encoders import ClinicalTextEncoder, VisualEncoder
from retrieval.faiss_index import build_flat_ip_index, save_index
from retrieval.bm25_index import build_bm25, save_bm25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build(cfg: dict) -> None:
    r = cfg.get("retrieval", {})
    index_dir = r.get("index_dir", "./rag_index")
    device = cfg.get("model", {}).get("device", "cpu")
    os.makedirs(index_dir, exist_ok=True)

    loader = get_loader(cfg)
    records = list(loader)
    if not records:
        raise RuntimeError("Corpus is empty; nothing to index.")
    logger.info("Loaded %d corpus records", len(records))

    image_paths = [rec.image_path for rec in records]
    reports = [rec.report for rec in records]

    logger.info("Encoding %d images (visual) ...", len(image_paths))
    visual_encoder = VisualEncoder(r.get("visual_encoder_id", "openai/clip-vit-large-patch14"), device)
    visual_vecs = visual_encoder.encode_paths(image_paths)

    logger.info("Encoding %d reports (clinical SBERT) ...", len(reports))
    text_encoder = ClinicalTextEncoder(r.get("text_encoder_id"), device)
    text_vecs = text_encoder.encode(reports)

    save_index(build_flat_ip_index(text_vecs), os.path.join(index_dir, store.TEXT_INDEX))
    save_index(build_flat_ip_index(visual_vecs), os.path.join(index_dir, store.VISUAL_INDEX))

    bm25, tokenized = build_bm25(reports)
    save_bm25(bm25, tokenized, os.path.join(index_dir, store.BM25_FILE))

    store.write_pairs(
        [
            {
                "pair_id": rec.pair_id,
                "image_path": rec.image_path,
                "report": rec.report,
                "meta": rec.meta,
            }
            for rec in records
        ],
        index_dir,
    )
    logger.info("Index built in %s (text.faiss, visual.faiss, bm25.pkl, pairs.jsonl)", index_dir)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices from a corpus")
    parser.add_argument("--config", type=str, default="configs/rag_eval.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force a synthetic no-download corpus (smoke test).")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Override corpus.max_pairs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.synthetic:
        cfg.setdefault("corpus", {})["name"] = "synthetic"
        cfg["corpus"]["source"] = "synthetic"
    if args.max_pairs is not None:
        cfg.setdefault("corpus", {})["max_pairs"] = args.max_pairs

    build(cfg)


if __name__ == "__main__":
    main()
