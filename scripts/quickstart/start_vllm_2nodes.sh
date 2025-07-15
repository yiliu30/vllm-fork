#!/bin/bash

export MODEL_PATH=/mnt/disk9/hf_models/models--moonshotai--Kimi-K2-Instruct/

max_num_seqs=32
max_num_batched_tokens=4096
block_size=128
VLLM_GPU_MEMORY_UTILIZATION=0.6

python -m vllm.entrypoints.openai.api_server \
    --host 192.168.1.101 \
    --port 8688 \
    --model $MODEL_PATH \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_inc \
    --use-v2-block-manager \
    --num_scheduler_steps 1\
    --block-size $block_size \
    --max-model-len $max_num_batched_tokens \
    --distributed_executor_backend ray \
    --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --trust_remote_code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1