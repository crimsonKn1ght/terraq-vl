# Full held-out evaluation

This workflow scores every held-out AstroLLaVA record, not only one caption per image.
For the released split that means 3,271 caption+QA records over 591 unseen images.

## Branches

- Stage 1: `releases/stg1-v2.0-full-heldout`
- Stage 2: `releases/stg2-v1.0-full-heldout`

These are evaluation-release branches. They do not define new model weights.

## Dry run

```bash
python scripts/run_full_heldout_eval.py --stage stage1 --dry-run --no-train-if-missing
python scripts/run_full_heldout_eval.py --stage stage2 --dry-run --no-train-if-missing
```

To validate record extraction without loading a model:

```bash
python scripts/generate_heldout_records.py \
  --records-json datasets/astrollava_llava/test.json \
  --image-dir datasets/astrollava_llava/images \
  --num-samples 5 \
  --dry-run
```

## Full Stage-1 run

Download or place the Stage-1 checkpoints under `checkpoints/astraq-vl-stage1/`:

- `checkpoint-1300`
- `checkpoint-2500`
- `checkpoint-3789`

Then run:

```bash
python scripts/run_full_heldout_eval.py \
  --stage stage1 \
  --num-samples 0 \
  --resume \
  --package
```

If a checkpoint is missing, the script trains with `configs/pretrain_astraq_vl.yaml`
and then re-checks the expected checkpoint names. Use `--no-train-if-missing` for an
eval-only run that fails fast instead.

## Full Stage-2 run

Download or place the Stage-2 checkpoint under `checkpoints/astraq-vl-stage2/checkpoint-2526`.
The checkpoint must include both `connector.safetensors` and `lora/adapter_model.safetensors`.

```bash
python scripts/run_full_heldout_eval.py \
  --stage stage2 \
  --num-samples 0 \
  --resume \
  --package
```

If the Stage-2 checkpoint is missing, the script trains with
`configs/finetune_astraq_vl_stage2.yaml` and then re-checks `checkpoint-2526`.

## Outputs

Outputs are written under `eval_runs/full_heldout/`:

- `stage1_ep1/`, `stage1_ep2/`, `stage1_ep3/`, or `stage2/`
- `predictions_full_heldout.jsonl`
- `metrics_full_heldout.json`
- `metrics_full_heldout.per_sample.jsonl`
- `comparison/full_heldout_comparison.{json,csv,md}`
- `astraq-vl-stage1-full-heldout-eval-v1.zip` or `astraq-vl-stage2-full-heldout-eval-v1.zip`

The metrics are produced by `scripts/score_predictions.py`: ROUGE-L, token-F1,
exact match, specificity hallucination, NLI consistency, contradiction rate, and
SBERT cosine. Summaries include `overall`, `splits.caption`, and `splits.qa`.

## Release artifacts

The preprint-facing release ZIPs are:

| Artifact | SHA256 |
| --- | --- |
| `astraq-vl-stage1-full-heldout-eval-v1.zip` | `4B2B38C8F40B7345979F7D9E078453CE8FC23FBC87C77BC462CF5F7DD11DEEC1` |
| `astraq-vl-stage2-full-heldout-eval-v1.zip` | `9D940A77C4F072A00F925588069C6B1E8670891F5DFA1126489B914FA144CF99` |
| `qwen2_5-vl-7b-full-heldout-eval-v1.zip` | `602B6708D6F31898149942EEF85CEF87E9FE13874511F1E3BCB8E97FA98939C5` |
| `astrollava-reference-full-heldout-eval-v1.zip` | `F014B70908AA0EDCF27BECDECDF9940EF8F286935114F3F18B5B9B66965C92F5` |

Each ZIP contains the held-out split, predictions, aggregate metrics,
per-sample metrics, comparison rows when applicable, and a reproduction note.

## Overall comparison

| Model | n | ROUGE-L up | Token-F1 up | Exact match up | Specificity halluc. down | Pred. specifics / rec. | Unsupported / rec. down | Spec. precision up | SBERT up | NLI up | Contradiction down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-1 epoch 3 | 3271 | 0.3116 | 0.3672 | 0.0272 | 0.2372 | 0.4632 | 0.3733 | 0.1941 | 0.7150 | -0.1084 | 0.5671 |
| Stage-2 | 3271 | **0.3404** | **0.3948** | **0.0339** | 0.2284 | 0.4405 | 0.3369 | **0.2353** | **0.7313** | -0.0795 | 0.5436 |
| Qwen2.5-VL-7B | 3271 | 0.1610 | 0.2156 | 0.0003 | 0.2727 | 0.5812 | 0.4946 | 0.1489 | 0.6224 | **0.0281** | **0.1764** |
| AstroLLaVA reference | 3240 | 0.1759 | 0.1993 | 0.0000 | **0.1722** | 0.2423 | **0.2207** | 0.0892 | 0.4656 | 0.0101 | 0.5836 |

Interpretation:

- Stage-2 improves in-domain reference alignment over Stage-1 epoch 3.
- Stage-2 reduces unsupported specifics compared with Stage-1 while also improving the detected
  specificity precision proxy, so the gain is not only reduced verbosity.
- Qwen2.5-VL-7B is much stronger on the NLI contradiction proxy, so the comparison should not be
  framed as universal factual superiority for Stage-2.
- AstroLLaVA reference is a domain comparator with possible data-lineage overlap and 31 missing
  predictions in this scoring run.

## Paired bootstrap highlights

The paired bootstrap script compares common held-out records. For Stage-2 vs Stage-1 epoch 3 on the
overall split, the 95% intervals exclude zero for ROUGE-L, token-F1, exact match, unsupported
specifics per record, SBERT, NLI, and contradiction rate. The specificity hallucination-rate
interval overlaps zero, so describe that metric directionally.
