"""Score an existing prediction JSONL against held-out references.

This is the offline companion to ``scripts/eval_metrics.py``. Use it when
predictions were already generated and you only want metrics.

Supported prediction keys:
  - prediction / response / output / generated_text
  - answer / reference / gold

If answers are not present in the prediction file, pass ``--records-json`` and
the script will match by image + prompt when possible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics_em import aggregate_em, normalize  # noqa: E402


IMAGE_TOKEN = "<image>"


PREDICTION_KEYS = ("prediction", "response", "output", "generated_text")
ANSWER_KEYS = ("answer", "reference", "gold", "ground_truth")
CAPTION_PROMPTS = {
    "describe this astronomical image",
    "describe this image",
    "caption this image",
}

INSTRUMENTS = (
    "alma",
    "atacama large millimeter/submillimeter array",
    "chandra",
    "gaia",
    "hubble",
    "hst",
    "herschel",
    "james webb",
    "jwst",
    "keck",
    "pan-starrs",
    "panstarrs",
    "sdss",
    "sloan digital sky survey",
    "spitzer",
    "subaru",
    "vla",
    "very large array",
    "vlt",
    "very large telescope",
    "wise",
    "xmm-newton",
)

CATALOG_PATTERNS = (
    r"\b(?:NGC|IC|UGC|PGC|ESO|ARP|ABELL|ACO|SH2|SH|BARNARD)\s*[-]?\s*\d+[A-Z]?\b",
    r"\bM\s*\d{1,3}[A-Z]?\b",
    r"\bMESSIER\s+\d{1,3}[A-Z]?\b",
    r"\b(?:2MASS|WISEA|SDSS|GAIA)\s+[A-Z0-9.+-]+\b",
)

MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:thousand|million|billion)?\s*"
    r"(?:light[-\s]?years?|ly|parsecs?|pc|kpc|mpc|gpc|au|astronomical units?|"
    r"arcseconds?|arcminutes?|degrees?|kelvin|k|solar masses|magnitude|mag)\b",
    re.IGNORECASE,
)
REDSHIFT_PATTERN = re.compile(r"\bz\s*[=~]\s*\d+(?:\.\d+)?\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b")


def _lcs_len(a: List[str], b: List[str]) -> int:
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


def clean_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.replace(IMAGE_TOKEN, " ")).strip()


def prompt_key(prompt: str) -> str:
    return normalize(clean_prompt(prompt))


def get_first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return None


def infer_record_type(row: Dict[str, Any], prompt: str) -> str:
    explicit = row.get("record_type") or row.get("task_type") or row.get("type")
    if explicit:
        value = str(explicit).lower()
        if "caption" in value:
            return "caption"
        if "qa" in value or "question" in value or "vqa" in value:
            return "qa"
    row_id = str(row.get("id", "")).lower()
    if "_qa" in row_id or row_id.endswith("qa"):
        return "qa"
    if prompt_key(prompt) in CAPTION_PROMPTS:
        return "caption"
    return "qa" if prompt else "caption"


def normalize_specific(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\bmessier\s+", "m", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\b(ngc|ic|ugc|pgc|eso|arp|abell|aco|sh2|sh|barnard|m)\s*-?\s*", r"\1", value)
    return value


def extract_specifics(text: str) -> set:
    found = set()
    upper = text.upper()
    for pattern in CATALOG_PATTERNS:
        for match in re.finditer(pattern, upper, flags=re.IGNORECASE):
            found.add(normalize_specific(match.group(0)))
    for pattern in (MEASUREMENT_PATTERN, REDSHIFT_PATTERN, YEAR_PATTERN):
        for match in pattern.finditer(text):
            found.add(normalize_specific(match.group(0)))
    lowered = text.lower()
    for name in INSTRUMENTS:
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            found.add(normalize_specific(name))
    return found


def specificity_row(prediction: str, answer: str) -> Dict[str, Any]:
    pred_specifics = extract_specifics(prediction)
    answer_specifics = extract_specifics(answer)
    unsupported = sorted(pred_specifics - answer_specifics)
    return {
        "pred_specifics": sorted(pred_specifics),
        "answer_specifics": sorted(answer_specifics),
        "unsupported_specifics": unsupported,
        "unsupported_specific_count": len(unsupported),
        "has_specificity_hallucination": bool(unsupported),
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def load_reference_maps(records_json: Optional[str]) -> Dict[str, Any]:
    maps: Dict[str, Any] = {
        "by_image_prompt": {},
        "caption_by_image": {},
        "unique_by_image": {},
    }
    if not records_json:
        return maps

    with open(records_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_image: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        convs = rec.get("conversations") or []
        if len(convs) < 2:
            continue
        image = str(rec.get("image", "")).strip()
        prompt = clean_prompt(str(convs[0].get("value", "")))
        answer = str(convs[1].get("value", "")).strip()
        if not image or not answer:
            continue
        record_type = "qa" if "_qa" in str(rec.get("id", "")).lower() else "caption"
        ref = {
            "id": str(rec.get("id", "")),
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "record_type": record_type,
        }
        maps["by_image_prompt"][(image, prompt_key(prompt))] = ref
        by_image[image].append(ref)
        if record_type == "caption":
            maps["caption_by_image"][image] = ref

    for image, refs in by_image.items():
        if len(refs) == 1:
            maps["unique_by_image"][image] = refs[0]
    return maps


def find_reference(row: Dict[str, Any], maps: Dict[str, Any], prompt: str) -> Optional[Dict[str, str]]:
    image = str(row.get("image", "")).strip()
    if image and prompt:
        ref = maps["by_image_prompt"].get((image, prompt_key(prompt)))
        if ref:
            return ref
    if image and prompt_key(prompt) in CAPTION_PROMPTS:
        ref = maps["caption_by_image"].get(image)
        if ref:
            return ref
    if image:
        return maps["unique_by_image"].get(image)
    return None


def aggregate_specificity(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "specificity_hallucination_rate": 0.0,
            "unsupported_specifics_per_record": 0.0,
            "records_with_pred_specifics": 0,
        }
    return {
        "specificity_hallucination_rate": round(
            sum(1 for r in records if r["has_specificity_hallucination"]) / len(records), 4
        ),
        "unsupported_specifics_per_record": round(
            sum(r["unsupported_specific_count"] for r in records) / len(records), 4
        ),
        "records_with_pred_specifics": sum(1 for r in records if r["pred_specifics"]),
    }


def aggregate_basic(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "lexical": {"rougeL_f1": 0.0, "token_f1": 0.0},
            "exact_match": aggregate_em([]),
            "specificity": aggregate_specificity([]),
        }
    return {
        "n": len(records),
        "lexical": {
            "rougeL_f1": round(sum(r["rougeL"] for r in records) / len(records), 4),
            "token_f1": round(sum(r["token_f1"] for r in records) / len(records), 4),
        },
        "exact_match": aggregate_em(records),
        "specificity": aggregate_specificity(records),
    }


def add_split_summaries(summary: Dict[str, Any], per_sample: List[Dict[str, Any]]) -> None:
    summary["splits"] = {}
    for split in ("caption", "qa"):
        rows = [r for r in per_sample if r["record_type"] == split]
        if rows:
            summary["splits"][split] = aggregate_basic(rows)


def write_outputs(out_stem: str, summary: Dict[str, Any], per_sample: List[Dict[str, Any]]) -> None:
    Path(f"{out_stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(f"{out_stem}.per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing VLM predictions against references.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file.")
    parser.add_argument(
        "--records-json",
        default=None,
        help="Optional held-out test.json, used when prediction rows do not include references.",
    )
    parser.add_argument("--label", default=None, help="Model label stored in the summary.")
    parser.add_argument("--out", default=None, help="Output stem. Defaults to <predictions>_metrics.")
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI factual-consistency.")
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--no-semantic", action="store_true", help="Skip SBERT cosine.")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cpu", help="Use cpu locally, or cuda on the GPU pod.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.predictions)
    refs = load_reference_maps(args.records_json)
    out_stem = args.out or str(Path(args.predictions).with_suffix("")) + "_metrics"

    per_sample = []
    skipped = 0
    for idx, row in enumerate(rows, 1):
        prediction = get_first(row, PREDICTION_KEYS)
        prompt = str(row.get("prompt") or row.get("question") or "").strip()
        answer = get_first(row, ANSWER_KEYS)
        ref = None if answer else find_reference(row, refs, prompt)
        if ref:
            answer = ref["answer"]
            prompt = prompt or ref["prompt"]
        if not prediction or not answer:
            skipped += 1
            continue

        record_type = ref["record_type"] if ref else infer_record_type(row, prompt)
        scored = {
            "index": idx,
            "image": row.get("image"),
            "prompt": clean_prompt(prompt),
            "prediction": prediction,
            "answer": answer,
            "record_type": record_type,
            "answer_type": answer_type(answer),
            "rougeL": round(rouge_l_f1(prediction, answer), 4),
            "token_f1": round(token_f1(prediction, answer), 4),
        }
        scored.update(specificity_row(prediction, answer))
        per_sample.append(scored)

    if not per_sample:
        raise SystemExit(
            "No scorable rows found. Make sure the JSONL has prediction+reference fields, "
            "or pass --records-json so references can be matched."
        )

    summary: Dict[str, Any] = {
        "label": args.label or Path(args.predictions).stem,
        "predictions": args.predictions,
        "records_json": args.records_json,
        "num_rows": len(rows),
        "num_scored": len(per_sample),
        "num_skipped": skipped,
        "overall": aggregate_basic(per_sample),
    }
    add_split_summaries(summary, per_sample)
    write_outputs(out_stem, summary, per_sample)

    if not args.no_nli:
        try:
            from eval.metrics_nli import NLIScorer, aggregate_nli

            print(f"Scoring NLI factual-consistency with {args.nli_model} on {args.device} ...")
            nli_scorer = NLIScorer(args.nli_model, args.device)
            summary["overall"]["nli"] = aggregate_nli(per_sample, nli_scorer)
            for split, split_summary in summary["splits"].items():
                split_rows = [r for r in per_sample if r["record_type"] == split]
                split_summary["nli"] = aggregate_nli(split_rows, nli_scorer)
            write_outputs(out_stem, summary, per_sample)
        except Exception as exc:  # noqa: BLE001
            print(f"NLI scoring failed ({exc}); continuing without it.", file=sys.stderr)

    if not args.no_semantic:
        try:
            from sentence_transformers import SentenceTransformer, util

            print(f"Scoring SBERT cosine with {args.sbert_model} on {args.device} ...")
            sbert = SentenceTransformer(args.sbert_model, device=args.device)
            preds = sbert.encode([r["prediction"] for r in per_sample], convert_to_tensor=True)
            answers = sbert.encode([r["answer"] for r in per_sample], convert_to_tensor=True)
            cos = util.cos_sim(preds, answers).diagonal().tolist()
            for row, score in zip(per_sample, cos):
                row["sbert_cos"] = round(float(score), 4)
            summary["overall"]["semantic"] = {
                "sbert_cosine": round(sum(cos) / len(cos), 4)
            }
            for split, split_summary in summary["splits"].items():
                split_scores = [
                    r["sbert_cos"] for r in per_sample if r["record_type"] == split and "sbert_cos" in r
                ]
                if split_scores:
                    split_summary["semantic"] = {
                        "sbert_cosine": round(sum(split_scores) / len(split_scores), 4)
                    }
            write_outputs(out_stem, summary, per_sample)
        except Exception as exc:  # noqa: BLE001
            print(f"SBERT scoring failed ({exc}); continuing without it.", file=sys.stderr)

    write_outputs(out_stem, summary, per_sample)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_stem}.json and {out_stem}.per_sample.jsonl")


if __name__ == "__main__":
    main()
