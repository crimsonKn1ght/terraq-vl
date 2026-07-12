"""One-command runner for the medical-RAG pipeline: build the index, then run the ablation.

    # Full real run on a GPU box (downloads models + corpus on first use):
    python run_rag.py --config configs/rag_eval_gpu.yaml

    # No-download smoke test (synthetic corpus, CPU):
    python run_rag.py --config configs/rag_eval.yaml --synthetic --num_samples 5

    # Reuse an existing index, just (re)run the eval:
    python run_rag.py --config configs/rag_eval_gpu.yaml --skip-index

Outputs land in ``eval.output_dir`` (default ./rag_results): results_<mode>.json per mode
plus comparison.md / comparison.json.
"""

from __future__ import annotations

import argparse
import logging
import os

import yaml

from corpus.build_index import build
from eval.ablation import run_ablation

# Quieter HF cache warnings on Windows (symlink notice).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_rag")


def main():
    parser = argparse.ArgumentParser(description="Build index + run medical-RAG ablation")
    parser.add_argument("--config", type=str, default="configs/rag_eval_gpu.yaml")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Override eval.modes (subset of no_retrieval dense_visual sparse_bm25 hybrid).")
    parser.add_argument("--num_samples", type=int, default=None, help="Override eval.num_samples.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Override corpus.max_pairs.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use the synthetic no-download corpus (smoke test).")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip index building; reuse the existing retrieval.index_dir.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.synthetic:
        cfg.setdefault("corpus", {})["name"] = "synthetic"
        cfg["corpus"]["source"] = "synthetic"
    if args.max_pairs is not None:
        cfg.setdefault("corpus", {})["max_pairs"] = args.max_pairs
    if args.num_samples is not None:
        cfg.setdefault("eval", {})["num_samples"] = args.num_samples

    modes = args.modes or cfg.get("eval", {}).get("modes", ["no_retrieval", "hybrid"])

    if cfg.get("model", {}).get("checkpoint") is None:
        logger.warning(
            "model.checkpoint is null -> the connector is UNTRAINED. The pipeline will run "
            "end-to-end, but generated answers (and therefore EM/NLI) are not meaningful. "
            "Train the connector or point model.config at a trained VLM for real numbers."
        )

    if not args.skip_index:
        logger.info("=== Building retrieval index ===")
        build(cfg)
    else:
        logger.info("=== Skipping index build (reusing %s) ===",
                    cfg.get("retrieval", {}).get("index_dir"))

    logger.info("=== Running ablation: %s ===", modes)
    run_ablation(cfg, modes)
    out = cfg.get("eval", {}).get("output_dir", "./rag_results")
    logger.info("Done. See %s/comparison.md", out)


if __name__ == "__main__":
    main()
