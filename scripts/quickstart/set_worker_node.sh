#! /bin/bash

# set -x
# set IP address of worker node
export VLLM_HOST_IP=10.239.129.40
# set NIC interface name of worker IP address
export GLOO_SOCKET_IFNAME=enx6c1ff7012f4d

#warmup cache folder
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k_1k_20k_16k,false,32768


####### INC WOQ ReQuant Start #######
USE_FP8_KV=false
USE_NATIVE_SCALING=true

model_path=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/
model_path=/mnt/disk7/yiliu4/DeepSeek-R1-0528-G2-2nd
# unset PT_HPU_RECIPE_CACHE_CONFIG
export VLLM_MOE_N_SLICE=1
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_REQUANT_FP8_INC=1
export VLLM_ENABLE_RUNTIME_DEQUANT=1
export VLLM_HPU_MARK_SCALES_AS_CONST=false

if $USE_NATIVE_SCALING; then
    echo "Using naive scaling"
    export INC_FORCE_NAIVE_SCALING=1
else
    echo "Disabling naive scaling"
    export INC_FORCE_NAIVE_SCALING=0
fi

# Check if FP8 KVCache is enabled
if $USE_FP8_KV; then
    echo "Using FP8 KVCache"
    export QUANT_CONFIG="./scripts/quant_configs/inc_quant_per_channel_with_fp8kv_config.json"
    KV_CACHE_DTYPE="fp8_inc"
else
    echo "Using BF16 KV"
    export QUANT_CONFIG="/mnt/disk3/yiliu4/vllm-fork/scripts/quant_configs/inc_quant_per_channel_bf16kv.json"
    KV_CACHE_DTYPE="auto"
fi
####### INC WOQ ReQuant End   #######

# vllm parameters
# max_num_batched_tokens=32768
# max_num_seqs=512
# input_min=768
# input_max=20480
# output_max=16896

CONST_LEN=16384
CONST_LEN=67584
max_model_len=$CONST_LEN
max_num_batched_tokens=$CONST_LEN
max_num_seqs=32
input_min=1
input_max=$CONST_LEN
output_max=$CONST_LEN

export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_DELAYED_SAMPLING="true"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0


BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

ray stop --force


# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

export VLLM_MOE_N_SLICE=1
export VLLM_EP_SIZE=16

block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0

#export VLLM_SKIP_WARMUP=true

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

echo " environments are reseted "

env | grep VLLM