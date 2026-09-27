# Search-R1 端到端配方

这个目录打包了使用 slime 复现 Search-R1 Qwen2.5-3B 本地检索实验所需的步骤：

1. 启动 slime Docker 容器。
2. 将 Hugging Face Qwen2.5-3B checkpoint 转换为 Megatron 格式。
3. 准备 Search-R1 训练数据和本地稠密检索索引。
4. 启动本地检索服务。
5. 启动 Search-R1 训练。

## 前置条件

- 带有 Docker 和 NVIDIA Container Toolkit 的 GPU 机器。
- Qwen2.5-3B Hugging Face checkpoint 目录，默认是 `/mnt/afs/models/Qwen/Qwen2.5-3B`，可用 `MODEL_DIR` 覆盖。
- 建议 4 张卡训练、1 张卡跑检索；如果只有 4 张卡，可把 `RETRIEVER_GPUS` 设为其中一张训练卡以共享 GPU。
- 容器中已按 `../README.md` 安装 Search-R1 数据准备所需依赖。

## 步骤 0：在宿主机启动容器

```bash
bash examples/search-r1/recipe/00_run_docker.sh
docker exec -it slime-search-r1 bash
```

如果 checkpoint 在其他路径：

```bash
MODEL_DIR=/path/to/Qwen2.5-3B bash examples/search-r1/recipe/00_run_docker.sh
```

## 步骤 1：构建 Megatron checkpoint（容器内）

```bash
cd /root/slime
bash examples/search-r1/recipe/01_build_model.sh
```

脚本会在需要时下载 Qwen2.5-3B，并生成 `/root/Qwen2.5-3B_torch_dist`。

## 步骤 2：准备 Search-R1 数据和索引（容器内）

```bash
bash examples/search-r1/recipe/02_build_search.sh
```

脚本会把 `Search-R1` 克隆到 `/root/slime/Search-R1`，处理 NQ + HotpotQA 训练数据，下载 wiki-18 索引分片，并生成 `/root/slime/Index/e5_Flat.index`。

## 步骤 3：启动检索服务（容器内，另开终端）

```bash
bash examples/search-r1/recipe/03_start_retrieval_server.sh
```

服务监听 `http://127.0.0.1:8000/retrieve`，并在前台运行。

## 步骤 4：训练（容器内，第一个终端）

```bash
bash examples/search-r1/recipe/04_run_train.sh
```

这个 wrapper 会导出配方需要的路径和 GPU 数量，然后调用 `examples/search-r1/run_qwen2.5_3B.sh`。

## 环境变量

| 变量 | 默认值 | 使用脚本 |
| --- | --- | --- |
| `MODEL_DIR` | `/mnt/afs/models/Qwen/Qwen2.5-3B` | `00_run_docker.sh` |
| `HF_CKPT` | `/root/Qwen2.5-3B` | `01_build_model.sh` |
| `TORCH_DIST` | `/root/Qwen2.5-3B_torch_dist` | `01_build_model.sh` |
| `WORK_DIR` | `/root/slime/Search-R1` | `02_build_search.sh` |
| `INDEX_DIR` | `/root/slime/Index` | `02_build_search.sh`、`03_start_retrieval_server.sh` |
| `TRAIN_GPUS` | `0,1,2,3` | `04_run_train.sh` |
| `NUM_GPUS` | `4` | `04_run_train.sh`；同时覆盖下面两个数量 |
| `RAY_NUM_GPUS` | `NUM_GPUS` 或 `8` | `run_qwen2.5_3B.sh` |
| `TRAIN_GPUS_PER_NODE` | `NUM_GPUS` 或 `4` | `run_qwen2.5_3B.sh` |
| `RETRIEVER_GPUS` | `0` | `03_start_retrieval_server.sh` |
| `TOPK` | `3` | `03_start_retrieval_server.sh` |

## 说明

- `/root/slime/Index/` 和 `/root/slime/Search-R1/` 是生成/本地克隆产物，已加入 git ignore。
- 训练超参数仍然以 `run_qwen2.5_3B.sh` 为准，配方脚本只负责提供环境相关路径和 GPU 数量。
