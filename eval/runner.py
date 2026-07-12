"""Single-configuration evaluation runner.

Loops a benchmark under one retrieval mode (or the no-retrieval baseline), collects
predictions, scores exact-match + NLI factual-consistency, and writes a JSON result.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from eval.benchmarks import VQASample
from eval.metrics_em import aggregate_em
from eval.metrics_nli import NLIScorer, aggregate_nli
from rag_inference import rag_answer
from retrieval.base import BaseRetriever
from ragcore import phase3_stubs

logger = logging.getLogger(__name__)


def run_eval(
    model,
    retriever: Optional[BaseRetriever],
    samples: List[VQASample],
    cfg: dict,
    config_name: str = "run",
) -> Dict:
    """Evaluate ``model`` (+ optional ``retriever``) over ``samples``; return results."""
    eval_cfg = cfg.get("eval", {})
    metrics = eval_cfg.get("metrics", ["exact_match", "nli_consistency"])
    use_decomp = cfg.get("phase3", {}).get("query_decomposition", False)

    records: List[Dict] = []
    for s in tqdm(samples, desc=f"eval[{config_name}]"):
        retrieval_query = None
        if use_decomp:
            retrieval_query = " ".join(phase3_stubs.decompose_query(s.question, cfg))
        prediction = rag_answer(
            model, s.image_path, s.question, retriever, cfg,
            retrieval_query=retrieval_query,
        )
        records.append(
            {
                "sample_id": s.sample_id,
                "question": s.question,
                "answer": s.answer,
                "answer_type": s.answer_type,
                "prediction": prediction,
            }
        )

    summary: Dict[str, float] = {}
    if "exact_match" in metrics:
        summary.update(aggregate_em(records))
    if "nli_consistency" in metrics:
        scorer = NLIScorer(
            model_name=eval_cfg.get("nli_model_id", "roberta-large-mnli"),
            device=cfg.get("model", {}).get("device", "cpu"),
        )
        summary.update(aggregate_nli(records, scorer))

    results = {
        "config_name": config_name,
        "benchmark": eval_cfg.get("benchmark"),
        "num_samples": len(records),
        "metrics": summary,
        "per_sample": records,
    }
    logger.info("[%s] metrics: %s", config_name, summary)
    return results


def write_results(results: Dict, output_dir: str) -> str:
    """Write a results dict to ``<output_dir>/results_<config_name>.json``."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"results_{results['config_name']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)
    return out_path
