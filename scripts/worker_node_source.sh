#! /bin/bash
# set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh
ray stop --force
# DO NOT change unless you fully undersand its purpose
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export HCCL_OVER_OFI=1
export HCCL_GAUDI_DIRECT=1
export HCCL_SOCKET_IFNAME=enx6c1ff7012f4d
export LIBFABRIC_ROOT=/opt/habanalabs/libfabric-1.22.0
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/habanalabs/libfabric-1.22.0/lib:/usr/lib/habanalabs
export GLOO_SOCKET_IFNAME=enx6c1ff7012f4d
export VLLM_HOST_IP=10.239.129.40
export HABANA_VISIBLE_DEVICES="ALL"
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=16
export VLLM_TP_SIZE=16
export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache,True,16384
export VLLM_SKIP_WARMUP="true"
export VLLM_LOGGING_LEVEL="DEBUG"
block_size=128
# DO NOT change ends...
# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.98
export VLLM_GRAPH_RESERVED_MEM=0.35
export VLLM_GRAPH_PROMPT_RATIO=0

# INC
unset QUANT_CONFIG

# params
# max_num_batched_tokens=2048
# max_num_seqs=1024
# input_min=1024
# input_max=4096
# output_max=1024

# Fot prepare
max_num_batched_tokens=2048
max_num_seqs=1024
input_min=1024
input_max=1024
output_max=32

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX
set_bucketing
echo " environments are reseted "
env | grep VLLM
