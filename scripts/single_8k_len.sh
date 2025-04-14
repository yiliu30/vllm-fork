#! /bin/bash

# set -x

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

export VLLM_MOE_N_SLICE=8
# The VLLM_MOE_N_SLICE of vllm static dmoe on G2 is 8
# 
export VLLM_EP_SIZE=8
# export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_DELAYED_SAMPLING="true"

block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0
# params

model_path=/data/youlei/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static
model_path=/mnt/disk6/yiliu4/DeepSeek-R1-G2-static
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
max_model_len=8192
max_num_batched_tokens=8192
max_num_seqs=32
input_min=1
input_max=8192
output_max=8192

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

# ! False means NOT CLEARING the cache when initializing.........
# export PT_HPU_RECIPE_CACHE_CONFIG=/data/16k_cache,false,16384

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


# ####### For INC WOQ  #######
# inc_quant_with_fp8kv_config.json: FP8KV
# inc_quant_fp8kv_pts_scalar_fp8_mla.json: FP8KV + FP8 MLA
# inc_quant_per_channel_bf16kv.json : BF16KV

export QUANT_CONFIG="inc_quant_fp8kv_pts_scalar_fp8_mla.json"
export VLLM_REQUANT_FP8_INC=1
export VLLM_ENABLE_RUNTIME_DEQUANT=1
export VLLM_MOE_N_SLICE=1


export VLLM_SKIP_WARMUP=true

# default false, algin with example
export VLLM_MLA_DISABLE_REQUANTIZATION=1


echo " environments are reseted "

env | grep VLLM

echo "model path is $model_path"

export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/DeepSeek-R1-G2/"


GRAPH_VISUALIZATION=1 \
python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8688 \
    --block-size 128 \
    --model $model_path \
    --device hpu \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --trust-remote-code  \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend mp \
    --gpu_memory_utilization 0.9 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1  \
    --kv_cache_dtype "fp8_inc" 2>&1 | tee ./g2_perf_logs/server.1.20.1.fp8kv.fp8mla.txt
