#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B}"
HF_CKPT="${HF_CKPT:-/root/Qwen2.5-3B}"
TORCH_DIST="${TORCH_DIST:-/root/Qwen2.5-3B_torch_dist}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
SLIME_DIR="${SLIME_DIR:-/root/slime}"

if [ ! -d "${HF_CKPT}" ] || [ -z "$(ls -A "${HF_CKPT}" 2>/dev/null)" ]; then
    hf download "${MODEL_ID}" --local-dir "${HF_CKPT}"
fi

cd "${SLIME_DIR}"
source scripts/models/qwen2.5-3B.sh
PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${TORCH_DIST}"
