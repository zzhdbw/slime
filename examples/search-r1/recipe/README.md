# Search-R1 end-to-end recipe

This recipe packages the local steps used to reproduce the Search-R1 Qwen2.5-3B experiment with slime:

1. Start the slime Docker container.
2. Convert the Hugging Face Qwen2.5-3B checkpoint to Megatron format.
3. Prepare the Search-R1 training data and local dense retrieval index.
4. Start the local retrieval server.
5. Launch the Search-R1 training run.

## Prerequisites

- An NVIDIA GPU host with Docker and the NVIDIA container toolkit.
- A Qwen2.5-3B Hugging Face checkpoint directory. The default is `/mnt/afs/models/Qwen/Qwen2.5-3B`; override it with `MODEL_DIR`.
- 4 GPUs for training and 1 GPU for retrieval. If you only have 4 GPUs, set `RETRIEVER_GPUS` to one of the training GPUs so the processes share it.
- The Search-R1 data preparation dependencies from `../README.md` installed in the container.

## Step 0: start the container (host)

```bash
bash examples/search-r1/recipe/00_run_docker.sh
docker exec -it slime-search-r1 bash
```

Set `MODEL_DIR` if your checkpoint is somewhere else:

```bash
MODEL_DIR=/path/to/Qwen2.5-3B bash examples/search-r1/recipe/00_run_docker.sh
```

## Step 1: build the Megatron checkpoint (container)

```bash
cd /root/slime
bash examples/search-r1/recipe/01_build_model.sh
```

This downloads Qwen2.5-3B when needed and writes `/root/Qwen2.5-3B_torch_dist`.

## Step 2: prepare Search-R1 data and index (container)

```bash
bash examples/search-r1/recipe/02_build_search.sh
```

This clones `Search-R1` to `/root/slime/Search-R1`, prepares the NQ + HotpotQA training data, downloads the wiki-18 index parts, and builds `/root/slime/Index/e5_Flat.index`.

## Step 3: start the retrieval server (container, separate terminal)

```bash
bash examples/search-r1/recipe/03_start_retrieval_server.sh
```

The server listens on `http://127.0.0.1:8000/retrieve` and stays in the foreground.

## Step 4: train (container, first terminal)

```bash
bash examples/search-r1/recipe/04_run_train.sh
```

The wrapper exports the recipe paths and GPU count, then runs `examples/search-r1/run_qwen2.5_3B.sh`.

## Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `MODEL_DIR` | `/mnt/afs/models/Qwen/Qwen2.5-3B` | `00_run_docker.sh` |
| `HF_CKPT` | `/root/Qwen2.5-3B` | `01_build_model.sh` |
| `TORCH_DIST` | `/root/Qwen2.5-3B_torch_dist` | `01_build_model.sh` |
| `WORK_DIR` | `/root/slime/Search-R1` | `02_build_search.sh` |
| `INDEX_DIR` | `/root/slime/Index` | `02_build_search.sh`, `03_start_retrieval_server.sh` |
| `TRAIN_GPUS` | `0,1,2,3` | `04_run_train.sh` |
| `NUM_GPUS` | `4` | `04_run_train.sh`; overrides both counts below |
| `RAY_NUM_GPUS` | `NUM_GPUS` or `8` | `run_qwen2.5_3B.sh` |
| `TRAIN_GPUS_PER_NODE` | `NUM_GPUS` or `4` | `run_qwen2.5_3B.sh` |
| `RETRIEVER_GPUS` | `0` | `03_start_retrieval_server.sh` |
| `TOPK` | `3` | `03_start_retrieval_server.sh` |

## Notes

- `/root/slime/Index/` and `/root/slime/Search-R1/` are generated/local artifacts and are ignored by git.
- `run_qwen2.5_3B.sh` remains the single source of truth for training hyperparameters; the recipe wrapper only supplies environment-specific paths and GPU counts.
