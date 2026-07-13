#!/usr/bin/env bash
# Train on the VRSBench set. Run after scripts/runpod_setup.sh.
#
#     bash scripts/runpod_train.sh                                        # Stage 1: configs/pretrain_vrsbench.yaml
#     bash scripts/runpod_train.sh configs/finetune_vrsbench_stage2.yaml  # Stage 2: connector + LoRA
#     bash scripts/runpod_train.sh configs/my_config.yaml                 # any custom config
#     bash scripts/runpod_train.sh configs/pretrain_vrsbench.yaml --resume checkpoints/vrsbench-stage1/checkpoint-2300
#
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# Reduce CUDA fragmentation OOMs (the large-vocab lm_head spikes allocations).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${1:-configs/pretrain_vrsbench.yaml}"
if [ "$#" -gt 0 ]; then shift; fi   # drop the config; forward any remaining args (e.g. --resume <ckpt>)

echo "==> Training with $CONFIG (HF_HOME=$HF_HOME)${*:+  extra: $*}"
python train.py --config "$CONFIG" "$@"

echo
echo "==> Training complete. Checkpoints are under ./checkpoints/ (see the config's output_dir,"
echo "    e.g. ./checkpoints/vrsbench-stage1/ for Stage 1)."
echo "    IMPORTANT: copy them off the pod BEFORE terminating (or keep them on /workspace):"
echo "      runpodctl send checkpoints/vrsbench-stage1     # then 'runpodctl receive <code>' on your laptop"
