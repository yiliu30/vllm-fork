#!/bin/bash
#########################################################
# vLLM Benchmark Script for Qwen3
# 
# This script runs a vLLM server with specific configurations
# and benchmarks it using the sonnet dataset.
#########################################################

#===========================================================
# CONFIGURATION PARAMETERS
#===========================================================
# /mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/
# bash bench_llama4.sh --model /path/to/model --tp_size 16
# bash bench_llama4.sh --model /mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/  --tp_size 8

pkill -9 python

if [ $# -gt 0 ] && [ "$1" == "--model" ]; then
    model=$2
else
    model="/mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=8
fi

model_name=$(basename ${model})

# Model Configuration
tokenizer=$model

# Hardware Configuration
moe_n_slice=1         # MoE groups
gpu_utils=0.95        # GPU memory utilization

# Request Configuration
max_model_len=9216    # Max model len
request_rate="inf"    # Request rate (inf = unlimited)
multi_step=1          # Number of scheduler steps


#===========================================================
# START the LOOP
#===========================================================

tp_parallel=$tp_size
# req_in_out_list=(512_1024_1024 192_5120_1024)
#req_in_out_list=(192_5120_1024)
req_in_out_list=(16_1024_1024)
req_in_out_list=(128_1024_1024)
req_in_out_list=(512_1024_1024)
req_in_out_list=(256_2048_2048)
req_in_out_list=(128_2048_2048)

for req_in_out in "${req_in_out_list[@]}"; do
    # Token Length Configuration
    bs=$(echo "$req_in_out" | awk -F'_' '{ print $1 }')
    in_len=$(echo "$req_in_out" | awk -F'_' '{ print $2 }')
    out_len=$(echo "$req_in_out" | awk -F'_' '{ print $3 }')

    num_prompts=$((bs * 3)) 
    # Expert parallelism size
    ep_size=${tp_parallel}

    #===========================================================
    # DERIVED PARAMETERS
    #===========================================================

    # Calculate and align total length
    # Calculate aligned lengths for buckets
    in_len_aligned=$(((in_len + 127) / 128 * 128))
    prompt_seq_max=$((in_len * 1125 / 1000))
    prompt_seq_max=$(((prompt_seq_max + 127) / 128 * 128))

    total_len=$((prompt_seq_max + out_len))
    if [ $((total_len % 128)) -ne 0 ]; then
        echo 'Rounding up total length to multiple of 128'
        total_len=$(((total_len / 128 + 1) * 128))
    fi

    total_len_aligned=$(((total_len + 127) / 128 * 128))

    decode_total_len=$((total_len + 128))
    decode_total_len_aligned=$(((decode_total_len + 127) / 128 * 128))

    # Calculate bucket sizes
    VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
    VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
    VLLM_DECODE_BLOCK_BUCKET_MAX=$((decode_total_len_aligned * bs / 128))
    VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

    #===========================================================
    # LOG CONFIGURATION
    #===========================================================

    # Create a descriptive log name based on parameters
    log_name="${model_name}-gaudi3-tp${tp_parallel}-ep${ep_size}-moe${moe_n_slice}-ms${multi_step}_np${num_prompts}_rr${request_rate}_bs${bs}_i${in_len}_o${out_len}_len${total_len}"

    # Create log directory
    mkdir -p benchmark_logs

    #===========================================================
    # START vLLM SERVER
    #===========================================================

    echo "Starting vLLM server with the following configuration:"
    echo "- Model: ${model_name}"
    echo "- Tensor Parallel Size: ${tp_parallel}"
    echo "- Expert Parallel Size: ${ep_size}"
    echo "- Batch Size: ${bs}"
    echo "- Input Length: ${in_len}"
    echo "- Output Length: ${out_len}"
    echo "- Total Length: ${total_len}"
    warmup_cache_path=/mnt/disk7/yiliu4/llama4_${model_name}_${max_model_len}
    mkdir -p ${warmup_cache_path}
    echo "Warmup cache path: ${warmup_cache_path}"
    PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},false,16384 \
    PT_HPU_LAZY_MODE=1 \
    VLLM_PROMPT_BS_BUCKET_MIN=1 \
    VLLM_PROMPT_BS_BUCKET_MAX=8 \
    VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
    VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max} \
    VLLM_DECODE_BS_BUCKET_MIN=${bs} \
    VLLM_DECODE_BS_BUCKET_MAX=${bs} \
    VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
    VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
    VLLM_DECODE_BLOCK_BUCKET_STEP=128 \
    VLLM_DELAYED_SAMPLING=true \
    HABANA_VISIBLE_DEVICES="ALL" \
    VLLM_EP_SIZE=${ep_size} \
    PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
    PT_HPU_WEIGHT_SHARING=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --port 18080 \
        --model ${model} \
        --load-format safetensors \
        --config-format hf \
        --tensor-parallel-size ${tp_parallel} \
        --max-num-seqs ${bs} \
        --disable-log-requests \
        --dtype bfloat16 \
        --use-v2-block-manager \
        --use-padding-aware-scheduling \
        --num_scheduler_steps ${multi_step} \
        --max-model-len $((total_len_aligned)) \
        --max-num-batched-tokens $((total_len_aligned * 4)) \
        --distributed_executor_backend mp \
        --gpu_memory_utilization ${gpu_utils} \
        --enable-expert-parallel \
        2>&1 | tee benchmark_logs/${log_name}_serving.log &
    pid=$(($!-1))
        #  --trust-remote-code false    --enforce-eager \

    # Wait for server to start
    n=0
    ready=false
    until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
        n=$((n+1))
        if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
            break
        fi
        sleep 5s
    done
    sleep 10s
    echo "Server started with PID: ${pid}"

    #===========================================================
    # RUN BENCHMARK
    #===========================================================

    echo "Starting benchmark with Sonnet dataset"
    max_concurrency_client=${bs}
    start_time=$(date +%s)
    timestamp=$(date +%Y%m%d_%H%M%S)

    python3 /mnt/disk3/yiliu4/vllm-fork/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model ${model} \
        --tokenizer ${tokenizer} \
        --dataset-name random \
        --request-rate ${request_rate} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        --num-prompts ${num_prompts} \
        --port 18080 \
        --random-input-len ${in_len} \
        --random-output-len ${out_len} \
        --max-concurrency ${max_concurrency_client} \
        --save-result 2>&1 | tee benchmark_logs/${log_name}_benchmark_${timestamp}.log


    end_time=$(date +%s)
    echo "Benchmark completed in $((end_time - start_time)) seconds"

    # Clean up
    echo "Stopping vLLM server"
    kill ${pid}
    echo "Script execution completed"
    sleep 10
done
