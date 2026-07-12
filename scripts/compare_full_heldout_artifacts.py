"""Compare full-heldout metric JSON files and release ZIPs."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from compare_metrics import fmt, rows_for_splits


KNOWN_LABELS = {
    "stage1_ep1": "stage1_ep1",
    "stage1_ep2": "stage1_ep2",
    "stage1_ep3": "stage1_ep3",
    "stage2": "stage2",
    "qwen2_5_vl_7b": "qwen2_5_vl_7b",
    "astrollava_reference": "astrollava_reference",
}


def infer_label(path: str, data: Dict[str, Any] | None = None) -> str:
    if data and data.get("label"):
        return str(data["label"])
    parts = Path(path).parts
    for part in reversed(parts):
        if part in KNOWN_LABELS:
            return KNOWN_LABELS[part]
    stem = Path(path).stem
    return stem.replace("_metrics", "").replace("metrics_full_heldout", Path(path).parent.name)


def rows_from_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    label = infer_label(str(path), data)
    return rows_for_splits(str(path), label)


def zip_metric_entries(zip_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith("metrics_full_heldout.json"):
                continue
            if name.startswith("comparison/"):
                continue
            data = json.loads(zf.read(name).decode("utf-8"))
            label = infer_label(name, data)
            yield label, data


def split_row_from_data(data: Dict[str, Any], label: str, split: str) -> Dict[str, Any]:
    def get_path(path: str) -> Any:
        cur: Any = data
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    prefix = "overall" if split == "overall" else f"splits.{split}"
    return {
        "model": label,
        "split": split,
        "n": get_path(f"{prefix}.n"),
        "rougeL": get_path(f"{prefix}.lexical.rougeL_f1"),
        "token_f1": get_path(f"{prefix}.lexical.token_f1"),
        "em_all": get_path(f"{prefix}.exact_match.em_all"),
        "em_closed": get_path(f"{prefix}.exact_match.em_closed"),
        "em_open": get_path(f"{prefix}.exact_match.em_open"),
        "specificity_halluc": get_path(f"{prefix}.specificity.specificity_hallucination_rate"),
        "unsupported_specifics": get_path(f"{prefix}.specificity.unsupported_specifics_per_record"),
        "records_with_pred_specifics": get_path(f"{prefix}.specificity.records_with_pred_specifics"),
        "sbert": get_path(f"{prefix}.semantic.sbert_cosine"),
        "nli": get_path(f"{prefix}.nli.nli_consistency"),
        "contradiction": get_path(f"{prefix}.nli.contradiction_rate"),
    }


def rows_from_zip(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, data in zip_metric_entries(path):
        rows.append(split_row_from_data(data, label, "overall"))
        for split in ("caption", "qa"):
            if data.get("splits", {}).get(split):
                rows.append(split_row_from_data(data, label, split))
    return rows


def rows_from_artifact(path: str) -> List[Dict[str, Any]]:
    artifact = Path(path)
    if artifact.suffix.lower() == ".zip":
        return rows_from_zip(artifact)
    return rows_from_json(artifact)


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
    parser = argparse.ArgumentParser(description="Compare full-heldout metric JSONs or ZIPs.")
    parser.add_argument("artifacts", nargs="*", help="Metric JSON files or full-heldout ZIPs.")
    parser.add_argument("--stage1-zip", default=None)
    parser.add_argument("--stage2-zip", default=None)
    parser.add_argument("--qwen-zip", default=None)
    parser.add_argument("--astrollava-zip", default=None)
    parser.add_argument(
        "--out",
        default="eval_runs/full_heldout/comparison/all_models_full_heldout_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = list(args.artifacts)
    artifacts.extend(
        item
        for item in (args.stage1_zip, args.stage2_zip, args.qwen_zip, args.astrollava_zip)
        if item
    )
    if not artifacts:
        raise SystemExit("Pass at least one metric JSON or ZIP artifact.")

    rows: List[Dict[str, Any]] = []
    for artifact in artifacts:
        rows.extend(rows_from_artifact(artifact))
    if not rows:
        raise SystemExit("No metrics_full_heldout.json files found in supplied artifacts.")

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(rows, f"{out_stem}.csv")
    write_markdown(rows, f"{out_stem}.md")
    print(Path(f"{out_stem}.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_stem}.json, {out_stem}.csv, and {out_stem}.md")


if __name__ == "__main__":
    main()
