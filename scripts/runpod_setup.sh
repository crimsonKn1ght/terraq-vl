#!/usr/bin/env bash
# One-time setup on a RunPod pod (use the "RunPod PyTorch 2.x" template).
# Clone this repo into /workspace (a network volume) so downloads + checkpoints persist,
# then run from the repo root:
#
#     bash scripts/runpod_setup.sh            # full AstraQ-VL build from AstroLLaVA_convos (train + 2% held-out test)
#     bash scripts/runpod_setup.sh 50         # smoke test with 50 samples first
#
set -euo pipefail

# Persist HF model/dataset downloads on the network volume (survives pod restarts).
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"

MAX_SAMPLES="${1:-}"
BUILD_ARGS=(--output-dir datasets/astrollava_llava --split train --overwrite --include-qa --max-image-size 384 --test-fraction 0.02)
if [ -n "$MAX_SAMPLES" ]; then
  BUILD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "==> HF_HOME=$HF_HOME"
echo "==> Installing Python dependencies"
pip install --no-cache-dir -r requirements.txt

echo "==> nvidia-smi"
nvidia-smi || echo "WARNING: no GPU visible — pick a GPU pod."

echo "==> Building the AstraQ-VL training set from AstroLLaVA_convos (real astronomy image-text)"
python scripts/build_astrollava_trainset.py "${BUILD_ARGS[@]}"

echo
echo "==> Setup complete."
echo "    Train data:    datasets/astrollava_llava/train.json"
echo "    Held-out test: datasets/astrollava_llava/test.json"
echo "    Next:          bash scripts/runpod_train.sh"
