#!/usr/bin/env bash
# Train the connector on the astronomy set. Run after scripts/runpod_setup.sh.
#
#     bash scripts/runpod_train.sh                          # uses configs/pretrain_astro.yaml
#     bash scripts/runpod_train.sh configs/my_config.yaml   # custom config
#
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# Reduce CUDA fragmentation OOMs (the large-vocab lm_head spikes allocations).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${1:-configs/pretrain_astraq_vl.yaml}"

echo "==> Training with $CONFIG (HF_HOME=$HF_HOME)"
python train.py --config "$CONFIG"

echo
echo "==> Training complete. Trained connector is in ./checkpoints/astraq-vl-stage1/"
echo "    IMPORTANT: copy it off the pod BEFORE terminating, e.g. from your laptop:"
echo "      runpodctl receive <code>        # after running 'runpodctl send checkpoints/astraq-vl-stage1' on the pod"
echo "    (Or keep it: it already lives on the /workspace network volume.)"
