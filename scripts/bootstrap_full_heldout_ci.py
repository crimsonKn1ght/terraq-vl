"""Bootstrap confidence intervals for full-heldout model comparisons.

This script expects per-sample metric JSONL files, either directly or inside the
release ZIPs produced by the full-heldout evaluation scripts. It computes paired
bootstrap intervals over common records for Stage-2-vs-baseline comparisons.
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from heldout_artifacts import (
    ArtifactRows,
    load_per_sample_artifacts,
    metric_direction,
    metric_value,
    record_type,
    rows_by_key,
    write_csv,
    write_json,
    write_markdown_table,
)


DEFAULT_METRICS = [
    "rougeL",
    "token_f1",
    "em_all",
    "specificity_halluc",
    "unsupported_specifics",
    "records_with_pred_specifics",
    "sbert",
    "nli",
    "contradiction",
]
DEFAULT_SPLITS = ["overall", "caption", "qa"]
DEFAULT_COMPARISONS = [
    ("stage2", "stage1_ep3"),
    ("stage2", "qwen2_5_vl_7b"),
    ("stage2", "astrollava_reference"),
]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def bootstrap_ci_numpy(
    diffs: Sequence[float],
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> Optional[Tuple[float, float]]:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    means: List[float] = []
    chunk_size = 256
    for start in range(0, n_bootstrap, chunk_size):
        size = min(chunk_size, n_bootstrap - start)
        indices = rng.integers(0, arr.size, size=(size, arr.size))
        means.extend(arr[indices].mean(axis=1).tolist())
    means.sort()
    return (percentile(means, alpha / 2), percentile(means, 1 - alpha / 2))


def bootstrap_ci_fallback(
    diffs: Sequence[float],
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> Tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    if len(diffs) == 1:
        return (float(diffs[0]), float(diffs[0]))

    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_bootstrap):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (percentile(means, alpha / 2), percentile(means, 1 - alpha / 2))


def bootstrap_ci(
    diffs: Sequence[float],
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> Tuple[float, float]:
    numpy_result = bootstrap_ci_numpy(diffs, n_bootstrap, seed, alpha)
    if numpy_result is not None:
        return numpy_result
    return bootstrap_ci_fallback(diffs, n_bootstrap, seed, alpha)


def collect_artifacts(args: argparse.Namespace) -> Dict[str, ArtifactRows]:
    specs: List[str] = []
    for value in (args.stage1_zip, args.stage2_zip, args.qwen_zip, args.astrollava_zip):
        if value:
            specs.append(value)
    specs.extend(args.artifact or [])

    if not specs:
        raise SystemExit("Pass at least one artifact ZIP, directory, or per-sample JSONL file.")

    models: Dict[str, ArtifactRows] = {}
    for spec in specs:
        for artifact in load_per_sample_artifacts(spec):
            if artifact.label in models:
                raise SystemExit(
                    f"Duplicate label {artifact.label!r} from {artifact.source} "
                    f"and {models[artifact.label].source}. Use LABEL=PATH to disambiguate."
                )
            models[artifact.label] = artifact
    return models


def parse_comparisons(raw: Optional[List[str]], labels: Iterable[str]) -> List[Tuple[str, str]]:
    if raw:
        comparisons = []
        for item in raw:
            if ":" in item:
                target, baseline = item.split(":", 1)
            elif "," in item:
                target, baseline = item.split(",", 1)
            else:
                raise SystemExit(f"Invalid comparison {item!r}; use TARGET:BASELINE.")
            comparisons.append((target.strip(), baseline.strip()))
        return comparisons

    available = set(labels)
    return [(target, baseline) for target, baseline in DEFAULT_COMPARISONS if {target, baseline} <= available]


def split_keys(
    target_rows: Mapping[Tuple[str, ...], Mapping],
    baseline_rows: Mapping[Tuple[str, ...], Mapping],
    split: str,
) -> List[Tuple[str, ...]]:
    keys = sorted(set(target_rows) & set(baseline_rows))
    if split == "overall":
        return keys
    return [
        key
        for key in keys
        if record_type(target_rows[key]) == split and record_type(baseline_rows[key]) == split
    ]


def summarize_metric(
    target_label: str,
    baseline_label: str,
    target_rows: Mapping[Tuple[str, ...], Mapping],
    baseline_rows: Mapping[Tuple[str, ...], Mapping],
    split: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> Optional[Dict[str, object]]:
    baseline_values: List[float] = []
    target_values: List[float] = []
    diffs: List[float] = []
    for key in split_keys(target_rows, baseline_rows, split):
        target_value = metric_value(target_rows[key], metric)
        baseline_value = metric_value(baseline_rows[key], metric)
        if target_value is None or baseline_value is None:
            continue
        target_values.append(target_value)
        baseline_values.append(baseline_value)
        diffs.append(target_value - baseline_value)

    if not diffs:
        return None

    ci_low, ci_high = bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)
    diff = statistics.fmean(diffs)
    direction = metric_direction(metric)
    if direction == "lower":
        favors_target = diff < 0
    elif direction == "higher":
        favors_target = diff > 0
    else:
        favors_target = None

    return {
        "comparison": f"{target_label} vs {baseline_label}",
        "target": target_label,
        "baseline": baseline_label,
        "split": split,
        "metric": metric,
        "direction": direction,
        "n": len(diffs),
        "baseline_mean": statistics.fmean(baseline_values),
        "target_mean": statistics.fmean(target_values),
        "diff_target_minus_baseline": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "favors_target": favors_target,
    }


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap paired CIs for full-heldout metrics.")
    parser.add_argument("--artifact", action="append", help="Artifact as PATH or LABEL=PATH.")
    parser.add_argument("--stage1-zip", default=None)
    parser.add_argument("--stage2-zip", default=None)
    parser.add_argument("--qwen-zip", default=None)
    parser.add_argument("--astrollava-zip", default=None)
    parser.add_argument(
        "--comparison",
        action="append",
        help="Comparison as TARGET:BASELINE. Defaults to Stage-2 vs Stage-1 ep3/Qwen/AstroLLaVA.",
    )
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_runs/full_heldout/analysis/bootstrap_ci")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1.")
    if args.n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap must be positive.")

    models = collect_artifacts(args)
    comparisons = parse_comparisons(args.comparison, models.keys())
    if not comparisons:
        raise SystemExit(
            "No comparisons could be inferred. Pass --comparison TARGET:BASELINE "
            f"after loading labels: {', '.join(sorted(models))}"
        )

    keyed = {label: rows_by_key(artifact.rows) for label, artifact in models.items()}
    metrics = parse_csv_list(args.metrics)
    splits = parse_csv_list(args.splits)
    alpha = 1 - args.confidence

    results: List[Dict[str, object]] = []
    for target_label, baseline_label in comparisons:
        if target_label not in keyed:
            raise SystemExit(f"Missing target label {target_label!r}. Loaded: {', '.join(sorted(keyed))}")
        if baseline_label not in keyed:
            raise SystemExit(f"Missing baseline label {baseline_label!r}. Loaded: {', '.join(sorted(keyed))}")
        for split in splits:
            for metric in metrics:
                row = summarize_metric(
                    target_label=target_label,
                    baseline_label=baseline_label,
                    target_rows=keyed[target_label],
                    baseline_rows=keyed[baseline_label],
                    split=split,
                    metric=metric,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                    alpha=alpha,
                )
                if row:
                    results.append(row)

    out_stem = Path(args.out)
    write_json(Path(f"{out_stem}.json"), results)
    write_csv(Path(f"{out_stem}.csv"), results)
    write_markdown_table(Path(f"{out_stem}.md"), results)
    print(Path(f"{out_stem}.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_stem}.json, {out_stem}.csv, and {out_stem}.md")


if __name__ == "__main__":
    main()
