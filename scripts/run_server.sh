#!/bin/bash
tp_parrallel=8
bs=8
in_len=5120
out_len=1024
multi_step=1
total_len=$((in_len + out_len))
ep_size=8
moe_n_slice=8
gpu_utils=0.82
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128 + 128))

#model="/data/DeepSeek-R1-G2/"
#tokenizer="/data/DeepSeek-R1-G2/"
model="/data/DeepSeek-R1-Dynamic-full-FP8"
tokenizer="/data/DeepSeek-R1-Dynamic-full-FP8"
model_name="DeepSeek-R1"

export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache,True,16384
#VLLM_SKIP_WARMUP=1 \
#export VLLM_DECODE_BS_BUCKET_STEP=16

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
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
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
    --max-model-len 8192 \
    --distributed_executor_backend ray \
    --gpu_memory_utilization ${gpu_utils} \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))

until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Uvicorn running on" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}

