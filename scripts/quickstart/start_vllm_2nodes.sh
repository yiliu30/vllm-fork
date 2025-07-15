#!/bin/bash



export MODEL_PATH=/mnt/disk9/hf_models/Kimi-K2-Instruct-G2/
# export MODEL_PATH=/mnt/disk7/yiliu4/DeepSeek-R1-0528-G2-2nd/
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=true
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_DEDUP_LOGS=0

max_num_seqs=32
max_num_batched_tokens=4096
block_size=128
VLLM_GPU_MEMORY_UTILIZATION=0.6

PT_HPU_LAZY_MODE=1 \
VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_MOE_N_SLICE=1 \
    python -m vllm.entrypoints.openai.api_server \
    --port 8688 \
    --model $MODEL_PATH \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps 1\
    --block-size $block_size \
    --max-model-len $max_num_batched_tokens \
    --distributed_executor_backend ray \
    --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --trust_remote_code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1