#!/usr/bin/env bash
# One-time setup on a RunPod pod (use the "RunPod PyTorch 2.x" template).
# Clone this repo into /workspace (a roomy volume) so downloads + checkpoints persist AND stay off
# the small container disk, then run from the repo root:
#
#     bash scripts/runpod_setup.sh            # full VRSBench build (train + ~1% val + ~1% held-out test)
#     bash scripts/runpod_setup.sh 50         # smoke test with 50 source images first
#
# Disk-frugal by default: images are stored downscaled (384 px long side) and the ~8.4 GB
# Images_train.zip is deleted right after extraction (--cleanup-zip). Net on-disk data is ~10 GB
# (Qwen2.5-3B ~6 GB + CLIP ~1.7 GB + extracted images ~1.5 GB). NOTE: because the zip is removed,
# re-running re-downloads it, so prefer going straight to the full build if disk is tight.
#
# Legacy astronomy path (the original backbone): build with scripts/build_astrollava_trainset.py
# and train a configs/*astraq*.yaml config instead.
set -euo pipefail

# Persist HF model/dataset downloads on the volume (survives pod restarts) and keep them off the
# small container disk. Make sure /workspace is a roomy volume; check the "Disk" report below.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"

MAX_SAMPLES="${1:-}"
# Hold out 2% of images, split evenly into val.json (in-training validation loss) and test.json
# (final held-out eval, never trained/selected on). --val-fraction only re-partitions the held-out
# side, so train.json is identical to a plain --test-fraction 0.02 build.
BUILD_ARGS=(--output-dir datasets/vrsbench_llava --split train --overwrite \
            --test-fraction 0.02 --val-fraction 0.5 --max-image-size 384 --cleanup-zip)
if [ -n "$MAX_SAMPLES" ]; then
  BUILD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "==> HF_HOME=$HF_HOME"
echo "==> Disk (target mounts) BEFORE build:"
df -h "$HF_HOME" "$PWD" 2>/dev/null || df -h
echo "    If the mount holding HF_HOME / this repo is the small container disk (~30 GB), create the"
echo "    pod with a >=50 GB network/volume disk at /workspace and clone there: the 8.4 GB image"
echo "    archive + 6 GB model will not fit on a 30 GB container disk alongside the base image."

echo "==> Installing Python dependencies"
pip install --no-cache-dir -r requirements.txt

echo "==> nvidia-smi"
nvidia-smi || echo "WARNING: no GPU visible; pick a GPU pod."

echo "==> Building the VRSBench training set (remote-sensing image -> caption + VQA)"
echo "    First run downloads VRSBench_train.json (~65 MB) + Images_train.zip (~8.4 GB); the zip is"
echo "    removed after extraction to reclaim disk."
python scripts/build_vrsbench_trainset.py "${BUILD_ARGS[@]}"

echo
echo "==> Disk (target mounts) AFTER build:"
df -h "$HF_HOME" "$PWD" 2>/dev/null || df -h
echo "==> Setup complete."
echo "    Train data:    datasets/vrsbench_llava/train.json"
echo "    Validation:    datasets/vrsbench_llava/val.json   (in-training validation loss)"
echo "    Held-out test: datasets/vrsbench_llava/test.json  (final eval only)"
echo "    Next:          bash scripts/runpod_train.sh                                       # Stage 1"
echo "                   bash scripts/runpod_train.sh configs/finetune_vrsbench_stage2.yaml # Stage 2"
