# Qwen2.5-VL full held-out baseline

This baseline runs `Qwen/Qwen2.5-VL-7B-Instruct` on the same 3,271 held-out
caption+QA records used for the Stage-1 and Stage-2 full-heldout evaluations.

It is an external general VLM baseline, not an AstroLLaVA checkpoint and not a
model trained by this repository.

## Branch

```bash
git switch releases/full-heldout-external-baselines
```

## Dependencies

Install the normal project dependencies, then:

```bash
pip install -r requirements-qwen-vl.txt
```

If Qwen2.5-VL is not recognized by Transformers:

```bash
pip install git+https://github.com/huggingface/transformers accelerate
```

## Smoke run

```bash
python scripts/run_qwen_vl_full_heldout_eval.py \
  --num-samples 5 \
  --no-nli \
  --no-semantic \
  --overwrite
```

## Full run

```bash
python scripts/run_qwen_vl_full_heldout_eval.py \
  --records-json datasets/astrollava_llava/test.json \
  --image-dir datasets/astrollava_llava/images \
  --num-samples 0 \
  --resume \
  --package
```

## Outputs

Outputs are written under `eval_runs/full_heldout/qwen2_5_vl_7b/`:

- `predictions_full_heldout.jsonl`
- `metrics_full_heldout.json`
- `metrics_full_heldout.per_sample.jsonl`

Comparison files are written under `eval_runs/full_heldout/comparison/`, and the
package is:

```text
eval_runs/full_heldout/qwen2_5-vl-7b-full-heldout-eval-v1.zip
```

The ZIP contains `test.json`, predictions, metrics, split-row comparison files,
and `REPRODUCE_QWEN_VL_FULL_HELDOUT.md`.
