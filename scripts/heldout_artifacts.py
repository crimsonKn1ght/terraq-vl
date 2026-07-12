"""Utilities for offline full-heldout evaluation artifacts.

The evaluation packages can be supplied as ZIPs, directories, or direct JSONL
files. These helpers keep downstream analysis scripts independent from the
exact packaging layout used on the GPU pod.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PREDICTION_KEYS = ("prediction", "response", "output", "generated_text")
ANSWER_KEYS = ("answer", "reference", "gold", "ground_truth")

KNOWN_LABELS = {
    "stage1_ep1",
    "stage1_ep2",
    "stage1_ep3",
    "stage2",
    "qwen2_5_vl_7b",
    "astrollava_reference",
}

HIGHER_IS_BETTER = {
    "rougeL",
    "token_f1",
    "em_all",
    "sbert",
    "nli",
}
LOWER_IS_BETTER = {
    "specificity_halluc",
    "unsupported_specifics",
    "contradiction",
}


@dataclass
class ArtifactRows:
    label: str
    rows: List[Dict[str, Any]]
    source: str


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_key_text(value: Any) -> str:
    return normalize_space(value).lower()


def get_first(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def clean_prompt(prompt: Any) -> str:
    return normalize_space(str(prompt or "").replace("<image>", " "))


def record_type(row: Mapping[str, Any]) -> str:
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
    return "qa" if clean_prompt(row.get("prompt") or row.get("question")) else "caption"


def infer_label(path_name: str, data: Optional[Mapping[str, Any]] = None) -> str:
    if data and data.get("label"):
        return str(data["label"])

    parts: Sequence[str]
    if "/" in path_name:
        parts = PurePosixPath(path_name).parts
    else:
        parts = Path(path_name).parts

    for part in reversed(parts):
        if part in KNOWN_LABELS:
            return part

    if len(parts) >= 2:
        parent = parts[-2]
        if parent and parent not in {"comparison", "full_heldout", "eval_runs"}:
            return parent

    stem = Path(path_name).stem
    stem = stem.replace(".per_sample", "")
    stem = stem.replace("metrics_full_heldout", "")
    stem = stem.replace("predictions_full_heldout", "")
    return stem.strip("_-.") or "artifact"


def parse_artifact_spec(spec: str) -> Tuple[Optional[str], str]:
    if "=" not in spec:
        return None, spec
    label, path = spec.split("=", 1)
    label = label.strip()
    return (label or None), path.strip()


def read_jsonl_text(text: str, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{source}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def _load_jsonl_from_zip(
    path: Path,
    suffix: str,
    forced_label: Optional[str] = None,
) -> List[ArtifactRows]:
    artifacts: List[ArtifactRows] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.startswith("comparison/") or not name.endswith(suffix):
                continue
            label = forced_label or infer_label(name)
            rows = read_jsonl_text(zf.read(name).decode("utf-8"), f"{path}:{name}")
            artifacts.append(ArtifactRows(label=label, rows=rows, source=f"{path}:{name}"))
    return artifacts


def load_jsonl_artifacts(
    spec: str,
    suffix: str,
    forced_label: Optional[str] = None,
) -> List[ArtifactRows]:
    label_from_spec, raw_path = parse_artifact_spec(spec)
    label = forced_label or label_from_spec
    path = Path(raw_path)

    if path.is_dir():
        artifacts: List[ArtifactRows] = []
        for file_path in sorted(path.rglob(f"*{suffix}")):
            rows = read_jsonl_text(file_path.read_text(encoding="utf-8"), str(file_path))
            artifacts.append(
                ArtifactRows(label=label or infer_label(str(file_path)), rows=rows, source=str(file_path))
            )
        return artifacts

    if path.suffix.lower() == ".zip":
        return _load_jsonl_from_zip(path, suffix, label)

    rows = read_jsonl_text(path.read_text(encoding="utf-8"), str(path))
    return [ArtifactRows(label=label or infer_label(str(path)), rows=rows, source=str(path))]


def load_per_sample_artifacts(spec: str, forced_label: Optional[str] = None) -> List[ArtifactRows]:
    return load_jsonl_artifacts(spec, ".per_sample.jsonl", forced_label)


def load_prediction_artifacts(spec: str, forced_label: Optional[str] = None) -> List[ArtifactRows]:
    return load_jsonl_artifacts(spec, "predictions_full_heldout.jsonl", forced_label)


def answer_for_key(row: Mapping[str, Any]) -> str:
    return get_first(row, ANSWER_KEYS) or ""


def prompt_for_key(row: Mapping[str, Any]) -> str:
    return clean_prompt(row.get("prompt") or row.get("question") or "")


def row_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    image = normalize_key_text(row.get("image"))
    prompt = normalize_key_text(prompt_for_key(row))
    answer = normalize_key_text(answer_for_key(row))
    rtype = record_type(row)
    if image and prompt:
        return ("content", image, rtype, prompt, answer)
    if row.get("id") is not None:
        return ("id", normalize_key_text(row.get("id")))
    if row.get("index") is not None:
        return ("index", str(row.get("index")))
    return ("row", json.dumps(row, sort_keys=True, ensure_ascii=False))


def rows_by_key(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, ...], Mapping[str, Any]]:
    keyed: Dict[Tuple[str, ...], Mapping[str, Any]] = {}
    seen: Dict[Tuple[str, ...], int] = {}
    for row in rows:
        key = row_key(row)
        count = seen.get(key, 0)
        seen[key] = count + 1
        if count:
            key = (*key, "dup", str(count + 1))
        keyed[key] = row
    return keyed


def normalize_for_em(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [token for token in re.split(r"\s+", text) if token and token not in {"a", "an", "the"}]
    return " ".join(tokens)


def exact_match(prediction: str, answer: str) -> bool:
    pred_n = normalize_for_em(prediction)
    answer_n = normalize_for_em(answer)
    if not answer_n:
        return pred_n == answer_n
    return pred_n == answer_n or f" {answer_n} " in f" {pred_n} "


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(row: Mapping[str, Any], metric: str) -> Optional[float]:
    if metric == "rougeL":
        return as_float(row.get("rougeL"))
    if metric == "token_f1":
        return as_float(row.get("token_f1"))
    if metric == "em_all":
        prediction = get_first(row, PREDICTION_KEYS) or ""
        answer = get_first(row, ANSWER_KEYS) or ""
        return 1.0 if exact_match(prediction, answer) else 0.0
    if metric == "specificity_halluc":
        return as_float(row.get("has_specificity_hallucination"))
    if metric == "unsupported_specifics":
        return as_float(row.get("unsupported_specific_count"))
    if metric == "records_with_pred_specifics":
        specifics = row.get("pred_specifics") or []
        return 1.0 if specifics else 0.0
    if metric == "sbert":
        return as_float(row.get("sbert_cos") or row.get("sbert"))
    if metric == "nli":
        return as_float(row.get("nli"))
    if metric == "contradiction":
        value = as_float(row.get("contradiction"))
        if value is not None:
            return value
        nli = as_float(row.get("nli"))
        return None if nli is None else float(nli < 0)
    return as_float(row.get(metric))


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    return "unknown"


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
