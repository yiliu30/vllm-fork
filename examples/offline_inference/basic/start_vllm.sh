#! /bin/bash

# VLLM_HPU_LOG_HPU_GRAPH=1 VLLM_DISABLE_INPUT_QDQ=0  bash start_vllm.sh --dummy-run
# VLLM_HPU_LOG_HPU_GRAPH=1 VLLM_DISABLE_INPUT_QDQ=0  bash start_vllm.sh --skip-warmup

model_path=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/
model_path=/software/users/yiliu4/deepseek-ai/DeepSeek-R1-MXFP8-OFFLINE/
model_path=/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP8-RTN
v2_model_path=/software/users/yiliu4/HF_HOME/Yi30/Yi30/DeepSeek-V2-Lite-MXFP8-llmc
mxfp4_model_path=/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP4-RTN
tp_size=8

num_samples=128
task_name="mmlu_pro_math,mmlu_pro_biology"
task_name="humaneval"
batch_size=32


# set -x

# Default values for arguments
USE_FP8_KV=false
USE_NATIVE_SCALING=true
# Default value for dummy run
USE_DUMMY_RUN=false
SKIP_WARMUP=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --fp8_kv)
            USE_FP8_KV=true
            ;;
        --dummy-run)
            USE_DUMMY_RUN=true
            ;;
        --skip-warmup)
            SKIP_WARMUP=true
            ;;
        --ds-v2)
            model_path=$v2_model_path
            tp_size=2
            ;;
        --ds-mxfp4)
            model_path=$mxfp4_model_path
            export VLLM_MXFP4_PREUNPACK_WEIGHTS=1
            export VLLM_USE_MXFP4_CT_EMULATIONS=1
            export VLLM_INPUT_QUICK_QDQ=1
            export USE_CT_UNPACK=1
            ;;
        --disable_native_scaling)
            USE_NATIVE_SCALING=false
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--fp8_kv] [--disable_native_scaling]"
            exit 1
            ;;
    esac
done

# Debugging: Print the values of the variables
echo "USE_FP8_KV=$USE_FP8_KV"
echo "USE_NATIVE_SCALING=$USE_NATIVE_SCALING"

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
# source "$BASH_DIR"/utils.sh

# ray stop --force

export PT_HPU_LAZY_MODE=1
# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

# export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=$tp_size

block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_DELAYED_SAMPLING="true"
#export VLLM_MOE_SLICE_LENGTH=20480

# params
CONST_LEN=4096
max_model_len=$CONST_LEN
max_num_batched_tokens=$CONST_LEN
max_num_seqs=32
input_min=1
input_max=$CONST_LEN
output_max=$CONST_LEN

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX


# export PT_HPU_RECIPE_CACHE_CONFIG=/data/16k_cache,false,16384
#set_bucketing

####### INC WOQ ReQuant Start #######

export VLLM_USE_STATIC_MOE_HPU=1
export VLLM_HPU_FORCE_CHANNEL_FP8=0
# unset PT_HPU_RECIPE_CACHE_CONFIG
# export VLLM_MOE_N_SLICE=1
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
# export VLLM_REQUANT_FP8_INC=1
# export VLLM_ENABLE_RUNTIME_DEQUANT=1
# export VLLM_HPU_MARK_SCALES_AS_CONST=false

# if $USE_NATIVE_SCALING; then
#     echo "Using naive scaling"
#     export INC_FORCE_NAIVE_SCALING=1
# else
#     echo "Disabling naive scaling"
#     export INC_FORCE_NAIVE_SCALING=0
# fi


KV_CACHE_DTYPE="auto"
# # Check if FP8 KVCache is enabled
# if $USE_FP8_KV; then
#     echo "Using FP8 KVCache"
#     export QUANT_CONFIG="./scripts/quant_configs/inc_quant_per_channel_with_fp8kv_config.json"
#     KV_CACHE_DTYPE="fp8_inc"
# else
#     echo "Using BF16 KV"
#     export QUANT_CONFIG="./scripts/quant_configs/inc_quant_per_channel_bf16kv.json"
#     KV_CACHE_DTYPE="auto"
# fi
####### INC WOQ ReQuant End   #######

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

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="vllm_server_${timestamp}.log"
# export VLLM_SKIP_WARMUP=True
# IF SKIP_WARMUP
if [ "$SKIP_WARMUP" = true ]; then
    echo "Skipping warmup"
    export VLLM_SKIP_WARMUP=True
else
    echo "Not skipping warmup"
    export VLLM_SKIP_WARMUP=False
fi




mkdir -p benchmark_logs
# Construct the command
CMD="python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8688 \
    --block-size 128 \
    --model $model_path \
    --device hpu \
    --dtype bfloat16 \
    --tensor-parallel-size $tp_size \
    --trust-remote-code \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend ray \
    --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --disable-log-requests \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --enable-expert-parallel"

# Add dummy run option if selected
if [ "$USE_DUMMY_RUN" = true ]; then
    CMD="$CMD --load-format dummy"
fi

# Execute the command and log output
$CMD 2>&1 | tee benchmark_logs/${LOG_FILE}_serving.log &
    pid=$(($!-1))
    

# Wait for server to start
n=0
ready=false
until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${LOG_FILE}_serving.log; then
        break
    fi
    sleep 6s
done
sleep 10s
echo "Server started with PID: ${pid}"



#===========================================================
# RUN BENCHMARK
#===========================================================
export no_proxy=localhost,127.0.0.1


model_base_name=$(basename $model_path)

EVAL_LOG_NAME="mxfp8_${model_base_name}_lm_eval_output_${task_name}_bs${batch_size}__${timestamp}"

echo "Running lm_eval with model: ${model_path}, task: ${task_name}, batch size: ${batch_size}, num samples: ${num_samples}"

start_time=$(date +%s)

HF_ALLOW_CODE_EVAL=1 \
lm_eval --model local-completions \
    --tasks "$task_name" \
    --model_args model=${model_path},base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=1 \
    --batch_size 32  \
    --confirm_run_unsafe_code \
    --log_samples \
    --output_path "benchmark_logs/$EVAL_LOG_NAME" \
    2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"



end_time=$(date +%s)
echo "Benchmark completed in $((end_time - start_time)) seconds"

# Clean up
echo "Stopping vLLM server"
kill ${pid}
echo "Script execution completed"
sleep 10
