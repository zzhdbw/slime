#!/bin/bash
set -euo pipefail

INDEX_DIR="${INDEX_DIR:-/root/slime/Index}"
INDEX_FILE="${INDEX_FILE:-${INDEX_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${INDEX_DIR}/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_GPUS="${RETRIEVER_GPUS:-0}"
TOPK="${TOPK:-3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SLIME_DIR="${SLIME_DIR:-/root/slime}"

export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPUS}"
export HF_ENDPOINT

python "${SLIME_DIR}/examples/search-r1/local_dense_retriever/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk "${TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --faiss_gpu
