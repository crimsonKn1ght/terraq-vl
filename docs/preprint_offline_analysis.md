# Preprint offline analysis workflow

The full held-out generation work is complete. The remaining preprint-strengthening
steps can run from the packaged artifacts without keeping a GPU pod alive.

## Required artifacts

Use the ZIPs produced by the full-heldout scripts:

- `astraq-vl-stage1-full-heldout-eval-v1.zip`
- `astraq-vl-stage2-full-heldout-eval-v1.zip`
- `qwen2_5-vl-7b-full-heldout-eval-v1.zip`
- `astrollava-reference-full-heldout-eval-v1.zip`

Each ZIP should include predictions, `metrics_full_heldout.json`, and
`metrics_full_heldout.per_sample.jsonl`.

## One-command local run

After the ZIPs are in `eval_runs/full_heldout/`, run:

```powershell
.\scripts\run_local_offline_analysis.ps1
```

The script performs the preflight checks, omits `--records-json` automatically if
`datasets/astrollava_llava/test.json` is not present, and writes all analysis
outputs under `eval_runs/full_heldout/analysis/`.

## Bootstrap confidence intervals

Run paired bootstrap intervals on common held-out records:

```bash
python scripts/bootstrap_full_heldout_ci.py \
  --stage1-zip eval_runs/full_heldout/astraq-vl-stage1-full-heldout-eval-v1.zip \
  --stage2-zip eval_runs/full_heldout/astraq-vl-stage2-full-heldout-eval-v1.zip \
  --qwen-zip eval_runs/full_heldout/qwen2_5-vl-7b-full-heldout-eval-v1.zip \
  --astrollava-zip eval_runs/full_heldout/astrollava-reference-full-heldout-eval-v1.zip \
  --n-bootstrap 10000 \
  --out eval_runs/full_heldout/analysis/bootstrap_ci
```

Default comparisons are:

- `stage2` vs `stage1_ep3`
- `stage2` vs `qwen2_5_vl_7b`
- `stage2` vs `astrollava_reference`

The script writes JSON, CSV, and Markdown. Differences are reported as
`target - baseline`; for hallucination, unsupported-specific, and contradiction
metrics, negative differences favor Stage 2.

## AstroLLaVA skipped rows

Inspect unscored prediction rows before deciding whether to count them as failures
or document them as generation/scoring exclusions:

```bash
python scripts/inspect_skipped_rows.py \
  --artifact eval_runs/full_heldout/astrollava-reference-full-heldout-eval-v1.zip \
  --records-json datasets/astrollava_llava/test.json \
  --out eval_runs/full_heldout/analysis/astrollava_reference_skipped_rows
```

For the paper, report the exact count and reason distribution. If the rows are
empty/error predictions, the stricter analysis should count them as failures in a
sensitivity table.

## Qualitative examples

Mine candidates, then manually choose representative examples from the generated
Markdown:

```bash
python scripts/mine_qualitative_examples.py \
  --stage2 eval_runs/full_heldout/astraq-vl-stage2-full-heldout-eval-v1.zip \
  --stage1 eval_runs/full_heldout/astraq-vl-stage1-full-heldout-eval-v1.zip \
  --qwen eval_runs/full_heldout/qwen2_5-vl-7b-full-heldout-eval-v1.zip \
  --astrollava eval_runs/full_heldout/astrollava-reference-full-heldout-eval-v1.zip \
  --per-category 12 \
  --out eval_runs/full_heldout/analysis/qualitative_examples
```

The miner intentionally uses metric rules rather than hand-picking. Review the
candidates and keep examples that are faithful to the aggregate story:

- Stage 2 improves over Stage 1.
- Stage 2 beats Qwen on in-domain specificity.
- Qwen is safer or more conservative.
- Stage 2 still hallucinates.
- AstroLLaVA reference has failure cases under this scoring setup.

## Human or LLM-as-judge sample

Export a 100-200 record sample for external judgment:

```bash
python scripts/sample_judge_set.py \
  --artifact eval_runs/full_heldout/astraq-vl-stage2-full-heldout-eval-v1.zip \
  --artifact eval_runs/full_heldout/qwen2_5-vl-7b-full-heldout-eval-v1.zip \
  --artifact eval_runs/full_heldout/astrollava-reference-full-heldout-eval-v1.zip \
  --sample-size 150 \
  --out eval_runs/full_heldout/analysis/judge_sample
```

This writes a long-form CSV/JSONL plus a rubric covering faithfulness,
hallucination, visual relevance, and answer correctness. For a blind study,
shuffle or hide `model_label` before assigning examples.

## Paper framing

A conservative result statement is:

> Stage 2 improves in-domain held-out reference alignment and reduces unsupported
> specifics compared with Stage 1, while Qwen2.5-VL shows lower contradiction
> under the NLI proxy.

Treat the AstroLLaVA reference as a domain comparator, not a clean external
baseline, because of possible data-lineage overlap.
