"""Mine candidate qualitative examples from aligned full-heldout artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from heldout_artifacts import (
    ANSWER_KEYS,
    PREDICTION_KEYS,
    ArtifactRows,
    get_first,
    load_per_sample_artifacts,
    metric_value,
    prompt_for_key,
    record_type,
    row_key,
    rows_by_key,
    write_json,
)


def pick_artifact(spec: Optional[str], label: str) -> Optional[ArtifactRows]:
    if not spec:
        return None
    artifacts = load_per_sample_artifacts(spec)
    matches = [artifact for artifact in artifacts if artifact.label == label]
    if matches:
        return matches[0]
    if len(artifacts) == 1:
        return artifacts[0]
    loaded = ", ".join(artifact.label for artifact in artifacts)
    raise SystemExit(f"Could not pick label {label!r} from {spec}. Loaded: {loaded}")


def supported_specifics(row: Mapping[str, Any]) -> set:
    pred = set(row.get("pred_specifics") or [])
    answer = set(row.get("answer_specifics") or [])
    return pred & answer


def unsupported_count(row: Mapping[str, Any]) -> float:
    return metric_value(row, "unsupported_specifics") or 0.0


def hallucinated(row: Mapping[str, Any]) -> bool:
    return bool(metric_value(row, "specificity_halluc"))


def answer_text(row: Mapping[str, Any]) -> str:
    return get_first(row, ANSWER_KEYS) or ""


def prediction_text(row: Mapping[str, Any]) -> str:
    return get_first(row, PREDICTION_KEYS) or ""


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def common_items(*maps: Mapping[Tuple[str, ...], Mapping[str, Any]]) -> Iterable[Tuple[Tuple[str, ...], List[Mapping[str, Any]]]]:
    if not maps:
        return []
    keys = set(maps[0])
    for item in maps[1:]:
        keys &= set(item)
    return ((key, [item[key] for item in maps]) for key in sorted(keys))


def example_record(
    category: str,
    key: Tuple[str, ...],
    rows: Mapping[str, Mapping[str, Any]],
    rationale: str,
    text_limit: int,
) -> Dict[str, Any]:
    first = next(iter(rows.values()))
    return {
        "category": category,
        "key": list(key),
        "image": first.get("image"),
        "record_type": record_type(first),
        "prompt": prompt_for_key(first),
        "answer": answer_text(first),
        "rationale": rationale,
        "metrics": {
            label: {
                "rougeL": metric_value(row, "rougeL"),
                "token_f1": metric_value(row, "token_f1"),
                "specificity_halluc": metric_value(row, "specificity_halluc"),
                "unsupported_specifics": metric_value(row, "unsupported_specifics"),
                "sbert": metric_value(row, "sbert"),
                "nli": metric_value(row, "nli"),
                "contradiction": metric_value(row, "contradiction"),
            }
            for label, row in rows.items()
        },
        "predictions": {
            label: truncate(prediction_text(row), text_limit) for label, row in rows.items()
        },
    }


def top_n(scored: List[Tuple[float, Dict[str, Any]]], n: int) -> List[Dict[str, Any]]:
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:n]]


def mine_stage2_improves_stage1(
    stage2: Mapping[Tuple[str, ...], Mapping[str, Any]],
    stage1: Mapping[Tuple[str, ...], Mapping[str, Any]],
    n: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for key, (s2, s1) in common_items(stage2, stage1):
        token_delta = (metric_value(s2, "token_f1") or 0) - (metric_value(s1, "token_f1") or 0)
        rouge_delta = (metric_value(s2, "rougeL") or 0) - (metric_value(s1, "rougeL") or 0)
        halluc_delta = float(hallucinated(s1)) - float(hallucinated(s2))
        unsupported_delta = unsupported_count(s1) - unsupported_count(s2)
        if token_delta <= 0 and rouge_delta <= 0 and halluc_delta <= 0:
            continue
        score = token_delta + rouge_delta + 0.25 * halluc_delta + 0.05 * unsupported_delta
        rationale = (
            f"token_f1 delta={token_delta:.4f}, rougeL delta={rouge_delta:.4f}, "
            f"unsupported delta={unsupported_delta:.1f}"
        )
        scored.append(
            (
                score,
                example_record(
                    "stage2_improves_over_stage1",
                    key,
                    {"stage1_ep3": s1, "stage2": s2},
                    rationale,
                    text_limit,
                ),
            )
        )
    return top_n(scored, n)


def mine_stage2_beats_qwen_specificity(
    stage2: Mapping[Tuple[str, ...], Mapping[str, Any]],
    qwen: Mapping[Tuple[str, ...], Mapping[str, Any]],
    n: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for key, (s2, qw) in common_items(stage2, qwen):
        if not s2.get("answer_specifics"):
            continue
        s2_supported = len(supported_specifics(s2))
        qw_supported = len(supported_specifics(qw))
        token_delta = (metric_value(s2, "token_f1") or 0) - (metric_value(qw, "token_f1") or 0)
        if s2_supported <= qw_supported and token_delta <= 0:
            continue
        score = token_delta + 0.5 * (s2_supported - qw_supported)
        rationale = (
            f"supported reference specifics: stage2={s2_supported}, qwen={qw_supported}; "
            f"token_f1 delta={token_delta:.4f}"
        )
        scored.append(
            (
                score,
                example_record(
                    "stage2_beats_qwen_in_domain_specificity",
                    key,
                    {"qwen2_5_vl_7b": qw, "stage2": s2},
                    rationale,
                    text_limit,
                ),
            )
        )
    return top_n(scored, n)


def mine_qwen_safer(
    stage2: Mapping[Tuple[str, ...], Mapping[str, Any]],
    qwen: Mapping[Tuple[str, ...], Mapping[str, Any]],
    n: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for key, (s2, qw) in common_items(stage2, qwen):
        unsupported_delta = unsupported_count(s2) - unsupported_count(qw)
        contradiction_delta = (metric_value(s2, "contradiction") or 0) - (
            metric_value(qw, "contradiction") or 0
        )
        if unsupported_delta <= 0 and contradiction_delta <= 0:
            continue
        score = unsupported_delta + contradiction_delta
        rationale = (
            f"unsupported specifics delta stage2-qwen={unsupported_delta:.1f}, "
            f"contradiction delta={contradiction_delta:.1f}"
        )
        scored.append(
            (
                score,
                example_record(
                    "qwen_safer_or_more_conservative",
                    key,
                    {"qwen2_5_vl_7b": qw, "stage2": s2},
                    rationale,
                    text_limit,
                ),
            )
        )
    return top_n(scored, n)


def mine_stage2_hallucinates(
    stage2: Mapping[Tuple[str, ...], Mapping[str, Any]],
    n: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for key, s2 in stage2.items():
        if not hallucinated(s2):
            continue
        score = unsupported_count(s2) + (metric_value(s2, "token_f1") or 0)
        rationale = f"unsupported specifics={unsupported_count(s2):.0f}: {s2.get('unsupported_specifics')}"
        scored.append(
            (
                score,
                example_record(
                    "stage2_still_hallucinates",
                    key,
                    {"stage2": s2},
                    rationale,
                    text_limit,
                ),
            )
        )
    return top_n(scored, n)


def mine_astrollava_failures(
    stage2: Mapping[Tuple[str, ...], Mapping[str, Any]],
    astro: Mapping[Tuple[str, ...], Mapping[str, Any]],
    n: int,
    text_limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for key, (s2, ar) in common_items(stage2, astro):
        token_delta = (metric_value(s2, "token_f1") or 0) - (metric_value(ar, "token_f1") or 0)
        sbert_delta = (metric_value(s2, "sbert") or 0) - (metric_value(ar, "sbert") or 0)
        score = token_delta + sbert_delta + 0.1 * unsupported_count(ar)
        if score <= 0:
            continue
        rationale = f"stage2-token_f1 minus astrollava={token_delta:.4f}, sbert delta={sbert_delta:.4f}"
        scored.append(
            (
                score,
                example_record(
                    "astrollava_reference_failure_case",
                    key,
                    {"astrollava_reference": ar, "stage2": s2},
                    rationale,
                    text_limit,
                ),
            )
        )
    return top_n(scored, n)


def write_markdown(path: Path, categories: Mapping[str, List[Dict[str, Any]]]) -> None:
    lines = ["# Qualitative example candidates", ""]
    for category, rows in categories.items():
        lines.extend([f"## {category}", ""])
        if not rows:
            lines.extend(["_No candidates found._", ""])
            continue
        for idx, row in enumerate(rows, 1):
            lines.extend(
                [
                    f"### {idx}. {row.get('image')} ({row.get('record_type')})",
                    "",
                    f"Prompt: {row.get('prompt')}",
                    "",
                    f"Reference: {row.get('answer')}",
                    "",
                    f"Rationale: {row.get('rationale')}",
                    "",
                ]
            )
            for label, prediction in row["predictions"].items():
                metrics = row["metrics"].get(label, {})
                lines.extend(
                    [
                        f"**{label}** "
                        f"(token_f1={metrics.get('token_f1')}, unsupported={metrics.get('unsupported_specifics')})",
                        "",
                        prediction or "_empty prediction_",
                        "",
                    ]
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine qualitative examples from full-heldout artifacts.")
    parser.add_argument("--stage2", required=True, help="Stage-2 artifact ZIP, dir, or per-sample JSONL.")
    parser.add_argument("--stage1", default=None, help="Stage-1 artifact; defaults to label stage1_ep3.")
    parser.add_argument("--qwen", default=None, help="Qwen baseline artifact.")
    parser.add_argument("--astrollava", default=None, help="AstroLLaVA reference artifact.")
    parser.add_argument("--stage1-label", default="stage1_ep3")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--text-limit", type=int, default=700)
    parser.add_argument("--out", default="eval_runs/full_heldout/analysis/qualitative_examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2_artifact = pick_artifact(args.stage2, "stage2")
    if stage2_artifact is None:
        raise SystemExit("--stage2 is required.")

    stage2 = rows_by_key(stage2_artifact.rows)
    stage1_artifact = pick_artifact(args.stage1, args.stage1_label)
    qwen_artifact = pick_artifact(args.qwen, "qwen2_5_vl_7b")
    astro_artifact = pick_artifact(args.astrollava, "astrollava_reference")

    categories: Dict[str, List[Dict[str, Any]]] = {}
    if stage1_artifact:
        categories["Stage-2 improves over Stage-1"] = mine_stage2_improves_stage1(
            stage2, rows_by_key(stage1_artifact.rows), args.per_category, args.text_limit
        )
    if qwen_artifact:
        qwen = rows_by_key(qwen_artifact.rows)
        categories["Stage-2 beats Qwen on in-domain specificity"] = mine_stage2_beats_qwen_specificity(
            stage2, qwen, args.per_category, args.text_limit
        )
        categories["Qwen is safer/more conservative"] = mine_qwen_safer(
            stage2, qwen, args.per_category, args.text_limit
        )
    categories["Stage-2 still hallucinates"] = mine_stage2_hallucinates(
        stage2, args.per_category, args.text_limit
    )
    if astro_artifact:
        categories["AstroLLaVA reference failure cases"] = mine_astrollava_failures(
            stage2, rows_by_key(astro_artifact.rows), args.per_category, args.text_limit
        )

    out_stem = Path(args.out)
    write_json(Path(f"{out_stem}.json"), categories)
    write_markdown(Path(f"{out_stem}.md"), categories)
    print(Path(f"{out_stem}.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_stem}.json and {out_stem}.md")


if __name__ == "__main__":
    main()
