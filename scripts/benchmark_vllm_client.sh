#!/bin/bash

model_path=/data/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static
model_path=/mnt/disk6/yiliu4/DeepSeek-R1-G2-static
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/

ip_addr=127.0.0.1
port=8688

log_dir="single_16k_bf16kv_skip_mark_disable_matix_sweep_retest417_len_ratio_1"
# Create folder if needed


test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency, num prompts: $local_num_prompts"

    log_name=inc_benchmark_serving_DeekSeek-R1_cardnumber_16_datatype_bfloat16_random_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max_concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore_eos \
    --save-result --result-filename ${log_name}.json 2>&1 | tee ./${log_dir}/${log_name}.txt
}

### no warmup
test_benchmark_serving_range 1024 1024 1 3 0.8
test_benchmark_serving_range 1024 1024 16 96 0.8
test_benchmark_serving_range 1024 1024 32 96 0.8
test_benchmark_serving_range 1024 1024 64 192 0.8
test_benchmark_serving_range 1024 1024 128 512 0.8

test_benchmark_serving_range 2048 2048 1 3 0.8
test_benchmark_serving_range 2048 2048 16 96 0.8

test_benchmark_serving_range 8192 1024 32 96 0.8


# test_benchmark_serving_range 2048 2048 32 96 0.8
# test_benchmark_serving_range 2048 2048 64 192 0.8

# test_benchmark_serving_range 8192 1024 1 3 0.8
# test_benchmark_serving_range 8192 1024 16 96 0.8
# test_benchmark_serving_range 8192 1024 32 96 0.8


# test_benchmark_serving_range 1024 1024 32 96 1
# test_benchmark_serving_range 1024 1024 32 96 1
# test_benchmark_serving_range 1024 1024 64 192 1
# test_benchmark_serving_range 1024 1024 128 512 1

# test_benchmark_serving_range 2048 2048 1 3 1
# test_benchmark_serving_range 2048 2048 32 96 1
# test_benchmark_serving_range 2048 2048 64 192 1
# test_benchmark_serving_range 2048 2048 128 512 1


# test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 1024 1024 32 96 1
# test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 1024 1024 32 96 1
# test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 1024 1024 32 96 1


# test_benchmark_serving_range 1024 1024 64 192 1

# test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 1024 1024 64 128 0.8
# test_benchmark_serving_range 1024 1024 32 96 0.8

# test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 1024 1024 64 192 0.8
# test_benchmark_serving_range 1024 1024 128 256 0.8
# test_benchmark_serving_range 1024 1024 256 512 0.8

# test_benchmark_serving_range 2048 2048 32 96 0.8
# test_benchmark_serving_range 2048 2048 64 128 0.8
# test_benchmark_serving_range 2048 2048 128 256 0.8
# test_benchmark_serving_range 2048 2048 256 512 0.8

# test_benchmark_serving_range 4096 1024 16 128 1
# test_benchmark_serving_range 4096 1024 32 128 1
# test_benchmark_serving_range 4096 1024 64 128 1
# test_benchmark_serving_range 1024 4096 16 128 1
# test_benchmark_serving_range 1024 4096 32 128 1
# test_benchmark_serving_range 1024 4096 64 128 1
# test_benchmark_serving_range 2048 2048 32 96 1
# test_benchmark_serving_range 2048 2048 32 96 1
