#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VLLM_BIN="${SCRIPT_DIR}/.venv/bin/vllm"
MODEL_PATH="${MODEL_PATH:-/storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash/}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/ds_sweep}"
PORT="${PORT:-8000}"
CONFIG="${1:-turbo_funnel_dense}"

usage() {
  cat <<'EOF'
Usage: run_ds_server_2gpu.sh [baseline|turbo_funnel_dense]

Configs:
  baseline             Launch without sparse-indexer funnel env vars.
  turbo_funnel_dense   Launch with turbo funnel mode and funnel_dense backend.
EOF
}

case "${CONFIG}" in
  baseline)
    CONFIG_DESC="baseline (sparse-indexer funnel env vars unset)"
    ENV_ARGS=(
      -u VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE
      -u VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND
    )
    ;;
  turbo_funnel_dense)
    CONFIG_DESC="turbo funnel mode with funnel_dense backend"
    ENV_ARGS=(
      "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo"
      "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense"
    )
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown config: ${CONFIG}" >&2
    usage >&2
    exit 1
    ;;
esac

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_${CONFIG}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting DeepSeek-V4-Flash server"
echo "Config: ${CONFIG_DESC}"
echo "Port: ${PORT}"
echo "Log file: ${LOG_FILE}"

SERVE_CMD=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --trust-remote-code
  --kv-cache-dtype fp8
  --block-size 256
  --enable-expert-parallel
  --tensor-parallel-size 2
  --attention_config.use_fp4_indexer_cache=True
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --gpu-memory-utilization 0.9
  --kernel-config.enable_flashinfer_autotune=False
  --port "${PORT}"
)

exec env \
  "${ENV_ARGS[@]}" \
  VLLM_DEEP_GEMM_WARMUP=skip \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
  PYTHONPATH="${SCRIPT_DIR}/../funnel-topk:${PYTHONPATH:-}" \
  VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$HOME/.cache/vllm_flashinfer_autotune}" \
  "${SERVE_CMD[@]}"
