#!/bin/bash
# Serve DeepSeek-V4-Flash with vLLM

source /home/yiliu7/workspace/venvs/vllm/bin/activate
MODEL=/storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash
PORT=8081

export CUDA_HOME=/home/yiliu7/workspace/venvs/vllm/lib/python3.10/site-packages/nvidia/cu13
VLLM_MXFP4_SF_ROUND_SHIFT=21 \
SAFETENSORS_FAST_GPU=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve "$MODEL" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend cutlass \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1048576 \
  --port "$PORT" \
  2>&1 | tee deepseek_flash_gsm8k_serve_s21_$(date +%Y%m%d_%H%M%S).log
