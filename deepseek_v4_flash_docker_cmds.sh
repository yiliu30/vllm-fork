#!/usr/bin/env bash
set -euo pipefail

HOST_GPUS="${HOST_GPUS:-2,3}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ds-precompiled-smoke}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/yiliu7/workspace/vllm}"
STORAGE_DIR="${STORAGE_DIR:-/storage}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
MODEL_PATH="${MODEL_PATH:-/storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash/}"
LOG_FILE="${LOG_FILE:-/workspace/vllm/logs/ds_sweep/server_turbo_funnel_dense_docker_20260801.log}"

cat <<EOF
# 1. Create the Docker container on the host.
docker run -d --name ${CONTAINER_NAME} \\
  --gpus '"device=${HOST_GPUS}"' \\
  --shm-size 64g \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -v ${WORKSPACE_DIR}:/workspace/vllm \\
  -v ${STORAGE_DIR}:/storage \\
  -v ${HF_CACHE_DIR}:/root/.cache/huggingface \\
  -v ${UV_CACHE_DIR}:/root/.cache/uv \\
  -w /workspace/vllm \\
  ${IMAGE} \\
  sleep infinity

# 2. Launch the working DeepSeek-V4-Flash server inside the container.
docker exec ${CONTAINER_NAME} bash -lc '
  cd /workspace/vllm &&
  mkdir -p logs/ds_sweep &&
  source /opt/vllm-precompiled-venv/bin/activate &&
  export VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo \
         VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense \
         FLASHINFER_DISABLE_VERSION_CHECK=1 \
         NCCL_IB_DISABLE=1 \
         NCCL_P2P_DISABLE=1 \
         NCCL_SHM_DISABLE=1 \
         TORCH_NCCL_BLOCKING_WAIT=1 \
         VLLM_DISABLE_PYNCCL=1 \
         VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
         VLLM_DEEP_GEMM_WARMUP=skip \
         VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm_flashinfer_autotune \
         CUDA_VISIBLE_DEVICES=0,1 &&
  nohup /opt/vllm-precompiled-venv/bin/vllm serve ${MODEL_PATH} \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.75 \
    --kernel-config.enable_flashinfer_autotune=False \
    --kernel-config.enable_jit_warmup=False \
    --kernel-config.enable_cutedsl_warmup=False \
    --disable-custom-all-reduce \
    --enforce-eager \
    --port 8000 > ${LOG_FILE} 2>&1 &
'
EOF
