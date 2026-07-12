# AstroLLaVA reference full held-out baseline

This baseline runs the original `UniverseTBD/AstroLLaVA` model on the same
3,271 held-out caption+QA records used for the Stage-1, Stage-2, and Qwen2.5-VL
full-heldout evaluations.

This is a domain reference baseline with possible split overlap because the
reference model was trained on the same AstroLLaVA data lineage. Treat it as a
domain comparator, while Qwen2.5-VL is the cleaner general external baseline.

## Branch

```bash
git switch releases/full-heldout-external-baselines
```

## Dependencies

The official AstroLLaVA repo is an adapted LLaVA package. Install it on the pod:

```bash
pip install -r requirements-astrollava-reference.txt
```

If its dependency pins conflict with the Qwen baseline environment, use a fresh
environment on the same pod/storage.

## Smoke run

```bash
python scripts/run_astrollava_reference_full_heldout_eval.py \
  --num-samples 5 \
  --no-nli \
  --no-semantic \
  --overwrite
```

## Full run

```bash
python scripts/run_astrollava_reference_full_heldout_eval.py \
  --records-json datasets/astrollava_llava/test.json \
  --image-dir datasets/astrollava_llava/images \
  --num-samples 0 \
  --resume \
  --package
```

The generator first tries the README-level `astrollava.AstroLLaVA` API. If that
is unavailable, it falls back to the official `llava.model.builder` loader.

## Outputs

Outputs are written under `eval_runs/full_heldout/astrollava_reference/`:

- `predictions_full_heldout.jsonl`
- `metrics_full_heldout.json`
- `metrics_full_heldout.per_sample.jsonl`

Comparison files are written under `eval_runs/full_heldout/comparison/`, and the
package is:

```text
eval_runs/full_heldout/astrollava-reference-full-heldout-eval-v1.zip
```

The ZIP contains `test.json`, predictions, metrics, split-row comparison files,
and `REPRODUCE_ASTROLLAVA_REFERENCE_FULL_HELDOUT.md`.

## All-model comparison

After Stage-1, Stage-2, Qwen2.5-VL, and AstroLLaVA reference zips are available:

```bash
python scripts/compare_full_heldout_artifacts.py \
  --stage1-zip astraq-vl-stage1-full-heldout-eval-v1.zip \
  --stage2-zip astraq-vl-stage2-full-heldout-eval-v1.zip \
  --qwen-zip qwen2_5-vl-7b-full-heldout-eval-v1.zip \
  --astrollava-zip astrollava-reference-full-heldout-eval-v1.zip \
  --out eval_runs/full_heldout/comparison/all_models_full_heldout_comparison
```
