#!/usr/bin/env bash
# Package per-epoch release artifacts for the AstraQ-VL Stage-1 (held-out) run.
# Run from the repo root after training + held-out inference. Produces three zips:
#   astraq-vl-stage1-ep1.zip / -ep2.zip / -ep3.zip
# Each bundles: the epoch's checkpoint, its held-out predictions, the training config,
# the held-out test split, and a REPRODUCE.md provenance file.
set -euo pipefail

command -v zip >/dev/null || { echo "zip not found -> apt-get update && apt-get install -y zip"; exit 1; }

ROOT=$(pwd)
CKPT_DIR="checkpoints/astraq-vl-stage1"
CONFIG="configs/pretrain_astraq_vl.yaml"
TEST_JSON="datasets/astrollava_llava/test.json"

# epoch -> last saved step for that epoch (edit if your checkpoints differ)
declare -A CKPT=( [1]=1300 [2]=2500 [3]=3789 )

COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
VERS=$(python -c "import torch,transformers;print('torch',torch.__version__,'| transformers',transformers.__version__)")

for ep in 1 2 3; do
  step=${CKPT[$ep]}
  ckpt="$CKPT_DIR/checkpoint-$step"
  preds="predictions_test_ep$ep.jsonl"
  stage="release/ep$ep"

  [ -f "$ckpt/connector.safetensors" ] || { echo "MISSING: $ckpt/connector.safetensors"; exit 1; }
  [ -f "$preds" ] || { echo "MISSING: $preds (run held-out inference for epoch $ep first)"; exit 1; }
  [ -f "$TEST_JSON" ] || { echo "MISSING: $TEST_JSON"; exit 1; }

  loss=$(python -c "import json;print(round(json.load(open('$ckpt/meta.json'))['loss'],4))" 2>/dev/null || echo "n/a")

  rm -rf "$stage"; mkdir -p "$stage"
  cp -r "$ckpt" "$stage/"
  cp "$preds" "$CONFIG" "$TEST_JSON" "$stage/"

  cat > "$stage/REPRODUCE.md" <<EOF
# AstraQ-VL Stage-1 — epoch $ep (checkpoint-$step)

- Checkpoint: checkpoint-$step (epoch $ep representative, train loss $loss)
- Code: github.com/crimsonKn1ght/astraq-vl @ $COMMIT (branch $BRANCH)
- Base: vision openai/clip-vit-large-patch14 + LLM Qwen/Qwen2.5-1.5B-Instruct (connector-only, ~3.9M params)
- Versions: $VERS

## Build data (held-out test split; seeded => deterministic)
python scripts/build_astrollava_trainset.py --output-dir datasets/astrollava_llava --split train --overwrite --include-qa --max-image-size 384 --test-fraction 0.02 --seed 42
# -> train.json (29,151 imgs / 161,653 recs) + test.json (591 imgs / 3,271 recs), disjoint by image

## Train
python train.py --config configs/pretrain_astraq_vl.yaml   # effective batch 128, 3 epochs

## Held-out evaluation (produces this zip's predictions)
python scripts/batch_inference.py --config configs/pretrain_astraq_vl.yaml --checkpoint $ckpt --image-dir datasets/astrollava_llava/images --records-json datasets/astrollava_llava/test.json --num-samples 0 --temperature 0 --output $preds

## Contents
- checkpoint-$step/         connector.safetensors + training_state.pt (optimizer) + meta.json
- $preds                    captions on the 591 held-out images (response + reference)
- pretrain_astraq_vl.yaml  training / inference config
- test.json                 the held-out split (regenerate the images via the build command above)

Epoch checkpoints: ep1 = step ~1300 (~1 epoch), ep2 = step ~2500 (~2 epochs), ep3 = step 3789 (final).
Predictions are on a TRUE held-out set (these images were excluded from training). The connector
grounds on coarse visual structure (object class / morphology); fine specifics (catalog numbers,
instruments, dates) may be hallucinated — the Stage-1 ceiling. On held-out images, qualitative
caption quality improved ep1 < ep2 < ep3.
EOF

  ( cd "$stage" && zip -rq "$ROOT/astraq-vl-stage1-ep$ep.zip" . )
  echo "==> astraq-vl-stage1-ep$ep.zip  (checkpoint-$step, loss $loss, $preds)"
done

rm -rf release
echo
ls -lh "$ROOT"/astraq-vl-stage1-ep*.zip
