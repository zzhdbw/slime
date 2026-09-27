#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-/mnt/afs/models/Qwen/Qwen2.5-3B}"
IMAGE="${IMAGE:-slimerl/slime:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-slime-search-r1}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker start "${CONTAINER_NAME}" >/dev/null
else
    docker run --gpus all -dit \
        --name "${CONTAINER_NAME}" \
        --ipc=host --shm-size=16g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "${REPO_ROOT}:/root/slime" \
        -v "${MODEL_DIR}:/root/Qwen2.5-3B" \
        "${IMAGE}" /bin/bash
fi

echo "Container '${CONTAINER_NAME}' is ready."
echo "Enter it with: docker exec -it ${CONTAINER_NAME} bash"
