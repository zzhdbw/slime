#!/bin/bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/root/slime}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet}"

cd "${SLIME_DIR}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_GPUS
export PROMPT_DATA

bash examples/search-r1/run_qwen2.5_3B.sh
