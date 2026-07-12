"""Compare multiple ``score_predictions.py`` or ``eval_metrics.py`` summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_path(data: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def metric(data: Dict[str, Any], new_path: str, old_path: Optional[str] = None) -> Optional[Any]:
    value = get_path(data, new_path)
    if value is not None:
        return value
    if old_path:
        return get_path(data, old_path)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def row_for(path: str, label: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "model": label,
        "n": metric(data, "overall.n", "num_samples"),
        "rougeL": metric(data, "overall.lexical.rougeL_f1", "lexical.rougeL_f1"),
        "token_f1": metric(data, "overall.lexical.token_f1", "lexical.token_f1"),
        "em_all": metric(data, "overall.exact_match.em_all", "exact_match.em_all"),
        "nli": metric(data, "overall.nli.nli_consistency", "nli.nli_consistency"),
        "contradiction": metric(data, "overall.nli.contradiction_rate", "nli.contradiction_rate"),
        "sbert": metric(data, "overall.semantic.sbert_cosine", "semantic.sbert_cosine"),
        "specificity_halluc": metric(data, "overall.specificity.specificity_hallucination_rate"),
        "unsupported_specifics": metric(data, "overall.specificity.unsupported_specifics_per_record"),
        "caption_n": metric(data, "splits.caption.n"),
        "caption_specificity_halluc": metric(
            data, "splits.caption.specificity.specificity_hallucination_rate"
        ),
        "qa_n": metric(data, "splits.qa.n"),
        "qa_specificity_halluc": metric(data, "splits.qa.specificity.specificity_hallucination_rate"),
    }


def split_row(data: Dict[str, Any], label: str, split: str) -> Dict[str, Any]:
    prefix = "overall" if split == "overall" else f"splits.{split}"
    return {
        "model": label,
        "split": split,
        "n": get_path(data, f"{prefix}.n"),
        "rougeL": get_path(data, f"{prefix}.lexical.rougeL_f1"),
        "token_f1": get_path(data, f"{prefix}.lexical.token_f1"),
        "em_all": get_path(data, f"{prefix}.exact_match.em_all"),
        "em_closed": get_path(data, f"{prefix}.exact_match.em_closed"),
        "em_open": get_path(data, f"{prefix}.exact_match.em_open"),
        "specificity_halluc": get_path(
            data, f"{prefix}.specificity.specificity_hallucination_rate"
        ),
        "unsupported_specifics": get_path(
            data, f"{prefix}.specificity.unsupported_specifics_per_record"
        ),
        "records_with_pred_specifics": get_path(
            data, f"{prefix}.specificity.records_with_pred_specifics"
        ),
        "sbert": get_path(data, f"{prefix}.semantic.sbert_cosine"),
        "nli": get_path(data, f"{prefix}.nli.nli_consistency"),
        "contradiction": get_path(data, f"{prefix}.nli.contradiction_rate"),
    }


def rows_for_splits(path: str, label: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [split_row(data, label, "overall")]
    for split in ("caption", "qa"):
        if get_path(data, f"splits.{split}"):
            rows.append(split_row(data, label, split))
    return rows


def write_markdown(rows: List[Dict[str, Any]], out_path: str) -> None:
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    columns = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact metrics comparison table.")
    parser.add_argument("summaries", nargs="+", help="Metric summary JSON files.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated model labels. Defaults to labels stored in JSON, then filenames.",
    )
    parser.add_argument("--out", default="metrics_comparison", help="Output stem.")
    parser.add_argument(
        "--split-rows",
        action="store_true",
        help="Write one row per model/split: overall, caption, and qa.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",")] if args.labels else []
    if labels and len(labels) != len(args.summaries):
        raise SystemExit("--labels must have the same number of entries as summary files")

    rows = []
    for idx, summary_path in enumerate(args.summaries):
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        label = labels[idx] if labels else data.get("label") or Path(summary_path).stem
        if args.split_rows:
            rows.extend(rows_for_splits(summary_path, label))
        else:
            rows.append(row_for(summary_path, label))

    Path(f"{args.out}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(rows, f"{args.out}.csv")
    write_markdown(rows, f"{args.out}.md")
    print(Path(f"{args.out}.md").read_text(encoding="utf-8"))
    print(f"Wrote {args.out}.json, {args.out}.csv, and {args.out}.md")


if __name__ == "__main__":
    main()
