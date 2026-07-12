"""NLI-based factual-consistency scoring (Phase 1 hallucination metric).

The model's prediction is checked for entailment against the gold answer using a
natural-language-inference classifier. Consistency = P(entailment) - P(contradiction),
in [-1, 1]: high when the prediction is supported by the reference, negative when it
contradicts it (the signature of a hallucination).

Reuses the already-present ``transformers`` library via a text-classification pipeline,
so no extra dependency is needed. The NLI checkpoint is config-driven; a general MNLI
model is the default, but a MedNLI/SciNLI-tuned checkpoint should replace it for clinical
text (negation/hedging) — treat the absolute number as relative-across-modes.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class NLIScorer:
    def __init__(self, model_name: str = "roberta-large-mnli", device: str = "cpu"):
        from transformers import pipeline

        # device index: -1 = CPU, 0 = first CUDA device
        device_idx = 0 if str(device).startswith("cuda") else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # return scores for all labels
            device=device_idx,
        )
        logger.info("Loaded NLI model %s", model_name)

    def _label_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        # NLI models (e.g. roberta-large-mnli) cap at 512 tokens; long (premise, hypothesis) pairs
        # must be truncated or the position-embedding lookup indexes out of bounds (CUDA assert).
        out = self.pipe(
            {"text": premise, "text_pair": hypothesis}, truncation=True, max_length=512
        )
        # `out` is a list of {label, score} dicts (one classification, all labels).
        scores: Dict[str, float] = {}
        for item in out:
            label = item["label"].lower()
            if "entail" in label:
                scores["entailment"] = item["score"]
            elif "contradict" in label:
                scores["contradiction"] = item["score"]
            elif "neutral" in label:
                scores["neutral"] = item["score"]
        return scores

    def consistency(self, gold_answer: str, prediction: str) -> float:
        """Factual-consistency score for one (gold, prediction) pair, in [-1, 1]."""
        if not prediction.strip():
            return -1.0
        scores = self._label_scores(gold_answer, prediction)
        return float(scores.get("entailment", 0.0) - scores.get("contradiction", 0.0))


def aggregate_nli(records: List[Dict], scorer: NLIScorer) -> Dict[str, float]:
    """Mean factual-consistency over per-sample records (adds ``nli`` to each record)."""
    vals: List[float] = []
    for r in records:
        c = scorer.consistency(r["answer"], r["prediction"])
        r["nli"] = round(c, 4)
        vals.append(c)
    mean = round(sum(vals) / len(vals), 4) if vals else 0.0
    # fraction of answers that outright contradict the reference (hallucination rate proxy)
    contra_rate = round(sum(1 for v in vals if v < 0) / len(vals), 4) if vals else 0.0
    return {"nli_consistency": mean, "contradiction_rate": contra_rate}
