"""Phase-2 ablation driver.

Loads the frozen VLM once, then evaluates the same benchmark under each retrieval mode
(``no_retrieval`` baseline + dense_visual / sparse_bm25 / hybrid), and writes a comparison
table isolating the contribution of each retrieval modality.

    python -m eval.ablation --config configs/rag_eval.yaml --modes no_retrieval hybrid --num_samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import yaml

from eval.benchmarks import get_benchmark
from eval.runner import run_eval, write_results
from ragcore.model_loader import load_vlm_from_cfg
from retrieval.factory import build_retriever, MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def write_comparison(all_results: List[dict], output_dir: str) -> str:
    """Write comparison.json + comparison.md (rows=modes, cols=EM/NLI)."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"mode": r["config_name"], **r["metrics"]} for r in all_results],
            f, indent=2,
        )

    header = (
        "| mode | EM (closed) | EM (open) | EM (all) | NLI-consistency | contradiction-rate |\n"
        "|------|-------------|-----------|----------|-----------------|--------------------|\n"
    )
    rows = []
    for r in all_results:
        m = r["metrics"]
        rows.append(
            "| {mode} | {c} | {o} | {a} | {nli} | {cr} |".format(
                mode=r["config_name"],
                c=m.get("em_closed", "-"),
                o=m.get("em_open", "-"),
                a=m.get("em_all", "-"),
                nli=m.get("nli_consistency", "-"),
                cr=m.get("contradiction_rate", "-"),
            )
        )
    md = "# Retrieval ablation\n\n" + header + "\n".join(rows) + "\n"
    md_path = os.path.join(output_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("\n%s", md)
    return md_path


def run_ablation(cfg: dict, modes: List[str]) -> None:
    eval_cfg = cfg.get("eval", {})
    output_dir = eval_cfg.get("output_dir", "./rag_results")

    model = load_vlm_from_cfg(cfg)
    samples = get_benchmark(
        eval_cfg.get("benchmark", "vqa_rad"),
        split=eval_cfg.get("split", "test"),
        n=eval_cfg.get("num_samples"),
    )

    all_results = []
    for mode in modes:
        logger.info("=== mode: %s ===", mode)
        retriever = build_retriever(mode, cfg)
        results = run_eval(model, retriever, samples, cfg, config_name=mode)
        write_results(results, output_dir)
        all_results.append(results)

    write_comparison(all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval ablation")
    parser.add_argument("--config", type=str, default="configs/rag_eval.yaml")
    parser.add_argument("--modes", nargs="+", default=None,
                        help=f"Subset of {MODES}. Defaults to eval.modes in the config.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override eval.num_samples.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    modes = args.modes or cfg.get("eval", {}).get("modes", ["no_retrieval", "hybrid"])
    if args.num_samples is not None:
        cfg.setdefault("eval", {})["num_samples"] = args.num_samples

    run_ablation(cfg, modes)


if __name__ == "__main__":
    main()
