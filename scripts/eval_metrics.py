"""Generation-based held-out quality metrics for a checkpoint (the complement to the loss curve).

Loss/perplexity (see scripts/eval_loss_curve.py; perplexity = exp(loss)) measure teacher-forced
likelihood. This script instead *generates* an answer for each held-out question and scores it
against the gold answer several ways:

  - ROUGE-L F1 / token-F1  — lexical overlap with the reference (pure-python, no deps)
  - Exact match (overall / closed / open)  — reuses eval/metrics_em.py
  - NLI factual consistency + contradiction rate  — reuses eval/metrics_nli.py (the hallucination
    proxy: P(entail) - P(contradict); contradiction_rate is the fraction that outright contradict)
  - SBERT cosine  — semantic similarity via sentence-transformers (optional; already a dep)

Each test.json record is its own (question, gold-answer) pair, so we prompt with the record's actual
question and compare to its gold answer — a real held-out QA/caption eval.

Usage (from repo root, on a GPU):
    python scripts/eval_metrics.py \
        --config configs/finetune_astraq_vl_stage2.yaml \
        --checkpoint checkpoints/astraq-vl-stage2/checkpoint-2526 \
        --records-json datasets/astrollava_llava/test.json \
        --image-dir datasets/astrollava_llava/images \
        --num-samples 200 --out stage2_metrics

Writes stage2_metrics.json (aggregates) and stage2_metrics.per_sample.jsonl (per-record scores).
For a Stage-1 vs Stage-2 comparison, run twice (same --records-json/--num-samples/--seed) and diff.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import load_vlm, run_inference  # noqa: E402
from vlm_model.utils import IMAGE_TOKEN  # noqa: E402
from eval.metrics_em import aggregate_em, normalize  # noqa: E402


def _lcs_len(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    same = sum((Counter(p) & Counter(r)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def answer_type(gold: str) -> str:
    return "closed" if normalize(gold) in {"yes", "no"} else "open"


def load_qa_records(records_json: str) -> list:
    """Each record -> (image, question-without-<image>, gold answer)."""
    with open(records_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data:
        convs = r.get("conversations") or []
        if len(convs) < 2:
            continue
        question = convs[0]["value"].replace(IMAGE_TOKEN, "").strip()
        gold = convs[1]["value"].strip()
        if gold:
            out.append({"image": r["image"], "question": question, "answer": gold})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generation-based held-out metrics for a checkpoint.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Checkpoint dir (connector + optional lora/).")
    p.add_argument("--records-json", required=True, help="Held-out records, e.g. test.json.")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--num-samples", type=int, default=200, help="Records to score (0 = all).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval_metrics", help="Output stem (.json / .per_sample.jsonl).")
    p.add_argument("--no-nli", action="store_true", help="Skip NLI factual-consistency.")
    p.add_argument("--nli-model", default="roberta-large-mnli")
    p.add_argument("--no-semantic", action="store_true", help="Skip SBERT cosine.")
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_qa_records(args.records_json)
    if not records:
        raise SystemExit(f"No (question, answer) records in {args.records_json}")
    if args.num_samples and 0 < args.num_samples < len(records):
        records = random.Random(args.seed).sample(records, args.num_samples)
    print(f"Scoring {len(records)} held-out records from {args.checkpoint}")

    model = load_vlm(args.config, args.checkpoint, args.device)

    per_sample = []
    for i, rec in enumerate(records, 1):
        img_path = os.path.join(args.image_dir, rec["image"])
        try:
            pred = run_inference(
                model=model, image_path=img_path, prompt=rec["question"],
                max_new_tokens=args.max_new_tokens, temperature=0.0, device=args.device,
            )
        except Exception as exc:  # keep going; record the failure
            pred = f"<error: {exc}>"
        row = {
            "image": rec["image"], "prompt": rec["question"],
            "prediction": pred, "answer": rec["answer"],
            "answer_type": answer_type(rec["answer"]),
            "rougeL": round(rouge_l_f1(pred, rec["answer"]), 4),
            "token_f1": round(token_f1(pred, rec["answer"]), 4),
        }
        per_sample.append(row)
        if i % 25 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] generated")

    n = len(per_sample)

    def write_out(summary: dict) -> None:
        Path(f"{args.out}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with open(f"{args.out}.per_sample.jsonl", "w", encoding="utf-8") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": args.checkpoint,
        "records_json": args.records_json,
        "num_samples": n,
        "generation": {"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
        "lexical": {
            "rougeL_f1": round(sum(r["rougeL"] for r in per_sample) / n, 4),
            "token_f1": round(sum(r["token_f1"] for r in per_sample) / n, 4),
        },
        "exact_match": aggregate_em(per_sample),
    }
    # Persist the (expensive) generations + lexical/EM now, so a failure in the optional metrics
    # below can never throw away the generation work.
    write_out(summary)

    if not args.no_nli:
        try:
            from eval.metrics_nli import NLIScorer, aggregate_nli
            print(f"Scoring NLI factual-consistency with {args.nli_model} ...")
            summary["nli"] = aggregate_nli(per_sample, NLIScorer(args.nli_model, args.device))
            write_out(summary)
        except Exception as exc:  # noqa: BLE001 — never lose generations to an NLI failure
            print(f"NLI scoring failed ({exc}); continuing without it.", file=sys.stderr)

    if not args.no_semantic:
        try:
            from sentence_transformers import SentenceTransformer, util
            print(f"Scoring SBERT cosine with {args.sbert_model} ...")
            sbert = SentenceTransformer(args.sbert_model, device=args.device)
            preds = sbert.encode([r["prediction"] for r in per_sample], convert_to_tensor=True)
            refs = sbert.encode([r["answer"] for r in per_sample], convert_to_tensor=True)
            cos = util.cos_sim(preds, refs).diagonal()
            for r, c in zip(per_sample, cos.tolist()):
                r["sbert_cos"] = round(float(c), 4)
            summary["semantic"] = {"sbert_cosine": round(float(cos.mean()), 4)}
            write_out(summary)
        except Exception as exc:  # noqa: BLE001
            print(f"SBERT scoring failed ({exc}); continuing without it.", file=sys.stderr)

    write_out(summary)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}.json and {args.out}.per_sample.jsonl")


if __name__ == "__main__":
    main()
