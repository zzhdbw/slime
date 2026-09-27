#!/bin/bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/root/slime/Search-R1}"
INDEX_DIR="${INDEX_DIR:-/root/slime/Index}"
SEARCH_R1_REPO="${SEARCH_R1_REPO:-https://github.com/PeterGriffinJin/Search-R1.git}"
SLIME_DIR="${SLIME_DIR:-/root/slime}"
DATA_SOURCES="${DATA_SOURCES:-nq,hotpotqa}"
TEST_DATA_SOURCES="${TEST_DATA_SOURCES:-nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle}"

if [ ! -d "${WORK_DIR}/.git" ]; then
    git clone "${SEARCH_R1_REPO}" "${WORK_DIR}"
fi

cd "${WORK_DIR}"
LOCAL_DIR="${WORK_DIR}/data/nq_hotpotqa_train"

python scripts/data_process/qa_search_train_merge.py \
    --local_dir "${LOCAL_DIR}" \
    --data_sources "${DATA_SOURCES}"

if [ "${PROCESS_TEST_DATA:-1}" = "1" ]; then
    python scripts/data_process/qa_search_test_merge.py \
        --local_dir "${LOCAL_DIR}" \
        --data_sources "${TEST_DATA_SOURCES}"
fi

mkdir -p "${INDEX_DIR}"
python "${SLIME_DIR}/examples/search-r1/local_dense_retriever/download.py" --save_path "${INDEX_DIR}"
cat "${INDEX_DIR}"/part_* > "${INDEX_DIR}/e5_Flat.index"
if [ -f "${INDEX_DIR}/wiki-18.jsonl.gz" ]; then
    gzip -d -f "${INDEX_DIR}/wiki-18.jsonl.gz"
fi
