#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

pkill -9 python
ray stop --force
sleep 2

model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2
model_path=/mnt/disk5/yiliu4/DeepSeek-R1-G2-dynamic
model_path=/mnt/disk5/yiliu4/DeepSeek-R1-G2-static
model_path=/dev/shm/DeepSeek-R1-G2-static/
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/

#model_path=/dev/shm/Llama-3.1-70B
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=256
input_min=1
input_max=16384
output_max=16384
block_size=128

max_model_len=8192
max_num_batched_tokens=8192
max_num_seqs=256
input_min=1
input_max=8192
output_max=8192

tp_size=4
pp_size=2

log_name=pp_server_max_8k_len_${max_model_len}_max_num_seqs_${max_num_seqs}_max_num_batched_tokens_${max_num_batched_tokens}_$(date +%F-%H-%M-%S).txt
log_dir="pp_fp8_mla_420"
mkdir -p $log_dir
log_file="${log_dir}/${log_name}"

# export VLLM_PROFILER_ENABLED=true
# export VLLM_DEVICE_PROFILER_ENABLED=true
#export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0
#export VLLM_DISABLE_TENSOR_CACHE=0
#export VLLM_HPU_COM_MARK_STEP=0

# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1


# PP Support
export VLLM_EP_SIZE=$tp_size

if [ "$pp_size" -gt 1 ]; then
    export VLLM_PP_LAYER_PARTITION="32,29"
fi

export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_DELAYED_SAMPLING="false"

### For 1.21 ###
export VLLM_FORCE_STATIC_MOE=true
export PT_HPU_LAZY_MODE=1

### FP8 KV & FP8 MLA ###
export VLLM_USE_FP8_MATMUL="true"
export VLLM_USE_SINGLE_TENSOR_CACHE="1"
VLLM_KV_CACHE_DTYPE="auto"
VLLM_KV_CACHE_DTYPE="fp8_inc"

### DEBUG ###
export GRAPH_VISUALIZATION=1
# export ENABLE_EXPERIMENTAL_FLAGS=1
# export ENABLE_GVD=1 
export VLLM_SKIP_WARMUP=true
export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1
export PT_HPU_METRICS_GC_DETAILS=1


export RAY_DEDUP_LOGS="1"
# VLLM_ENFORCE_EAGER="true"
if [ "$VLLM_ENFORCE_EAGER" = "true" ]; then
    ENFORCE_EAGER_FLAG="--enforce-eager"
else
    ENFORCE_EAGER_FLAG=""
fi

# export TMP_DEBUG="1"
if [ "$TMP_DEBUG" = "1" ]; then
    export VLLM_PP_LAYER_PARTITION="8,8"
fi


# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0

# params
unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

# cache
export PT_HPU_RECIPE_CACHE_CONFIG=/data/8k_cache,false,${max_model_len}

#set_bucketing

# !!!!!!!!!!!!!!!!!!!! set bucketing !!!!!!!!!!!!!
prompt_bs_min=1
prompt_bs_step=$(( $max_num_seqs > 32 ? 32 : $max_num_seqs ))
prompt_bs_max=$(( $max_num_seqs > 64 ? 64 : $max_num_seqs ))
export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

prompt_seq_step=128
prompt_seq_min=128
prompt_seq_max=$max_num_batched_tokens
export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

decode_bs_min=1
decode_bs_step=$(( $max_num_seqs > 32 ? 32 : $max_num_seqs ))
decode_bs_max=$max_num_seqs
export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

decode_block_min=128
decode_block_step=128
block_size=128
decode_block_max=$(( ((max_num_seqs * max_model_len / block_size) > 128) ? (max_num_seqs * max_model_len / block_size) : 128 ))
export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}



echo " environments are reseted "

env | grep VLLM



# VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
# VLLM_PROFILE_EXECUTE_MODEL_PROMPT=1 \
# HABANA_PROFILE=1 \
# HABANA_PROFILE_WRITE_HLTV=1 \
# VLLM_PROFILE_EXECUTE_MODEL_DECODE_STEPS=5 \
# HABANA_PROF_CONFIG=/root/.habana/prof_config.json \
python3 -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8688 \
    --block-size $block_size \
    --model $model_path \
    --device hpu \
    --dtype bfloat16 \
    --tensor-parallel-size $tp_size \
    --pipeline-parallel-size $pp_size \
    --trust-remote-code  \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend ray \
    --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --kv_cache_dtype $VLLM_KV_CACHE_DTYPE  2>&1 | tee  $log_file 

    
curl -X POST http://127.0.0.1:8688/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk2/hf_models/DeepSeek-R1-G2-static/",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 100,
           "temperature": 0.7,
           "top_p": 1.0
         }'
