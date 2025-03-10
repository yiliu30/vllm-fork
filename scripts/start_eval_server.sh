#!/bin/bash

model_path="/mnt/disk5/hf_models/DeepSeek-R1-BF16"


tp_parrallel=8
bs=4
in_len=1024
out_len=4
model_max_len=2048
multi_step=1
total_len=$((in_len + out_len))
ep_size=8
moe_n_slice=8
gpu_utils=0.5
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / 128 - 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128 + 128))
 
#model="/data/DeepSeek-R1-G2/"
#tokenizer="/data/DeepSeek-R1-G2/"
 
model=$model_path
tokenizer=$model_path
 
#model="/data/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2"
#tokenizer="/data/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2"
 
model_name="DeepSeek-R1"

export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache,True,16384
#VLLM_SKIP_WARMUP=1 \
#export VLLM_DECODE_BS_BUCKET_STEP=16
 
QUANT_CONFIG=inc_quant_with_fp8kv_one_node_config.json \
VLLM_SKIP_WARMUP="true" \
VLLM_GRAPH_PROMPT_RATIO="0.5" \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MOE_N_SLICE=${moe_n_slice} \
VLLM_EP_SIZE=${ep_size} \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=${bs} \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${total_len} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=64 \
VLLM_DECODE_BLOCK_BUCKET_STEP=64 \
VLLM_DECODE_BLOCK_BUCKET_MAX=512 \
python -m vllm.entrypoints.openai.api_server \
    --host localhost \
    --port 8080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step}\
    --max-num-batched-tokens ${model_max_len} \
    --max-model-len ${model_max_len} \
    --distributed_executor_backend mp \
    --gpu_memory_utilization ${gpu_utils} \
    --trust_remote_code \
    --quantization inc \
    --weights_load_device cpu \
    --kv_cache_dtype fp8_inc  2>&1 | tee eval_logs/${log_name}_serving_lm_eval.log



