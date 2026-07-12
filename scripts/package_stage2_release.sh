#!/usr/bin/env bash
# Package the AstraQ-VL Stage-2 release bundle.
# Run from the repo root after training + held-out inference. Produces:
#   astraq-vl-stage2.zip
# Contents: checkpoint-2526/ (connector.safetensors + lora/), predictions_test_stage2.jsonl,
#           finetune_astraq_vl_stage2.yaml, test.json, REPRODUCE.md
set -euo pipefail

command -v zip >/dev/null || { echo "zip not found -> apt-get update && apt-get install -y zip"; exit 1; }

ROOT=$(pwd)
STEP=2526
CKPT_DIR="checkpoints/astraq-vl-stage2"
CKPT="$CKPT_DIR/checkpoint-$STEP"
CONFIG="configs/finetune_astraq_vl_stage2.yaml"
TEST_JSON="datasets/astrollava_llava/test.json"
PREDS="predictions_test_stage2.jsonl"
STAGE="release/stage2"

[ -f "$CKPT/connector.safetensors" ]          || { echo "MISSING: $CKPT/connector.safetensors (training not done?)"; exit 1; }
[ -f "$CKPT/lora/adapter_model.safetensors" ] || { echo "MISSING: $CKPT/lora/adapter_model.safetensors (LoRA not saved?)"; exit 1; }
[ -f "$PREDS" ]                               || { echo "MISSING: $PREDS — run held-out inference first:
  python scripts/batch_inference.py \\
    --config $CONFIG \\
    --checkpoint $CKPT \\
    --image-dir datasets/astrollava_llava/images \\
    --records-json $TEST_JSON \\
    --num-samples 0 --temperature 0 \\
    --output $PREDS"; exit 1; }
[ -f "$TEST_JSON" ] || { echo "MISSING: $TEST_JSON"; exit 1; }

COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
VERS=$(python -c "import torch,transformers,peft; print('torch',torch.__version__,'| transformers',transformers.__version__,'| peft',peft.__version__)")
loss=$(python -c "import json;print(round(json.load(open('$CKPT/meta.json'))['loss'],4))" 2>/dev/null || echo "n/a")

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -r "$CKPT" "$STAGE/"
cp "$PREDS" "$CONFIG" "$TEST_JSON" "$STAGE/"

cat > "$STAGE/REPRODUCE.md" <<EOF
# AstraQ-VL Stage-2 — checkpoint-$STEP (final, 1 epoch)

- Checkpoint: checkpoint-$STEP (train loss $loss)
- Code: github.com/crimsonKn1ght/astraq-vl @ $COMMIT (branch $BRANCH)
- Base: vision openai/clip-vit-large-patch14 + LLM Qwen/Qwen2.5-1.5B-Instruct
- Trainable: connector (~3.9M, warm-started from Stage-1 checkpoint-3789) + LoRA adapters (~18.5M, r=16, q/k/v/o/gate/up/down_proj)
- Versions: $VERS

## Prerequisites
Stage-1 connector checkpoint (checkpoint-3789 from grKnight/astraq-vl-stage1 ep3 bundle).

## Build data (same split as Stage-1; seeded => deterministic)
python scripts/build_astrollava_trainset.py \\
  --output-dir datasets/astrollava_llava --split train --overwrite \\
  --include-qa --max-image-size 384 --test-fraction 0.02 --seed 42
# -> train.json (29,151 imgs / 161,653 recs) + test.json (591 imgs / 3,271 recs), disjoint by image

## Train
python train.py --config configs/finetune_astraq_vl_stage2.yaml
# effective batch 64 (per-device 4 x grad-accum 16), 1 epoch, 2526 steps, gradient checkpointing

## Held-out evaluation (produces this zip's predictions)
python scripts/batch_inference.py \\
  --config configs/finetune_astraq_vl_stage2.yaml \\
  --checkpoint checkpoints/astraq-vl-stage2/checkpoint-$STEP \\
  --image-dir datasets/astrollava_llava/images \\
  --records-json datasets/astrollava_llava/test.json \\
  --num-samples 0 --temperature 0 \\
  --output $PREDS

## Inference (single image)
python inference.py \\
  --config configs/finetune_astraq_vl_stage2.yaml \\
  --checkpoint checkpoints/astraq-vl-stage2/checkpoint-$STEP \\
  --image your_image.jpg \\
  --prompt "What is this astronomical object and what is notable about it?" \\
  --temperature 0

## Contents
- checkpoint-$STEP/
    connector.safetensors           continued-trained MLP connector (~16 MB)
    lora/adapter_model.safetensors  trained LoRA adapters for Qwen2.5-1.5B (~72 MB)
    lora/adapter_config.json        LoRA config (r=16, alpha=32, target modules)
    training_state.pt               optimizer + scheduler state
    meta.json                       step + final loss
- $PREDS             captions on the 591 held-out images (response + reference)
- finetune_astraq_vl_stage2.yaml   training / inference config
- test.json          the held-out split (regenerate images via build command above)

Stage-2 improves on Stage-1 by fine-tuning the LLM (LoRA) jointly with the connector on the same
caption + QA data. Both the connector and LoRA are needed at inference time — pass the Stage-2 config
so the LoRA modules are built, then load this checkpoint; inference.py restores both automatically.
Compare predictions with Stage-1 predictions_test_ep3.jsonl to see the hallucination reduction.
EOF

( cd "$STAGE" && zip -rq "$ROOT/astraq-vl-stage2.zip" . )
echo "==> astraq-vl-stage2.zip  (checkpoint-$STEP, loss $loss)"
echo
rm -rf release
ls -lh "$ROOT/astraq-vl-stage2.zip"
