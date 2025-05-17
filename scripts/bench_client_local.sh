#!/bin/bash

model_path=/mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static
ip_addr=127.0.0.1
port=8989

    #    test_benchmark_serving 2048 2048 32 96 0.8 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static


test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name=./scripts/pp_mem_crash/benchmark_serving_DeekSeek-R1_cardnumber_16_datatype_bfloat16_random_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    python3 ./benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max_concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore_eos \
    --save-result --result-filename ${log_name}.json 

}




# test_benchmark_serving_range 1024 1024 1 3 1
# # test_benchmark_serving_range 1024 1024 32 96 1
# test_benchmark_serving_range 2048 2048 64 96 0.8
# test_benchmark_serving_range 2048 2048 32 96 0.8
# test_benchmark_serving_range 2048 2048 48 96 0.8
# kv_cache_dtype,pp_size,tp_size,comm_backend,partiton,max_model_len,input_tokens,output_tokens,num_prompt,max_concurrency,mean_ttft,mean_tpot,total_throughput,output_throughput
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,96,32,1759.19,97.99,611.09,304.56
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,108,36,2149.09,108.52,624.57,311.57
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,120,40,2274.55,111.18,672.40,334.46
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,132,44,2358.76,115.37,714.37,355.30
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,144,48,2414.06,119.67,759.47,376.92
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,192,64,2898.30,135.33,872.54,436.07
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,240,80,3457.78,159.42,934.80,466.36
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,288,96,3562.10,171.75,1044.56,522.30
# fp8_inc,2,4,gloo,"32,29",16384,1024,1024,384,128,4731.28,417.19,520.15,260.97
# fp8_inc,2,4,gloo,"32,29",16384,2048,2048,3,1,765.36,32.56,58.41,30.35
# fp8_inc,2,4,gloo,"32,29",16384,2048,2048,48,16,2305.32,75.23,400.35,200.54
# fp8_inc,2,4,gloo,"32,29",16384,2048,2048,96,32,607.52,0.00,1478.82,0.00

record_mem() {
    sleep 5
    curl -X POST http://127.0.0.1:8989/start_profile
}


# Define configurations as a list
configs=(
    # "1024 1024 32 96 0.8"
    # "1024 1024 36 108 0.8"
    # "1024 1024 40 120 0.8"
    # "1024 1024 44 132 0.8"
    # "1024 1024 48 144 0.8"
    # "1024 1024 64 192 0.8"
    # "1024 1024 80 240 0.8"
    # "1024 1024 96 288 0.8"
    # "1024 1024 128 384 0.8"
    # "2048 2048 1 3 0.8"
    # "2048 2048 16 48 0.8"
    # "2048 2048 32 96 0.8"
    # "2048 2048 48 192 0.8"
    "128 1024 1 3 0.8"
    "128 1024 16 48 0.8"
    "128 1024 32 96 0.8"
    "128 1024 64 192 0.8"
    "1024 1024 1 3 0.8" 
    "1024 1024 16 48 0.8"
    "1024 1024 32 96 0.8"
    "1024 1024 64 192 0.8"
    "2048 2048 1 3 0.8"
    "2048 2048 16 48 0.8"
    "2048 2048 32 96 0.8"
    "2048 2048 64 192 0.8"
    "6144 1024 1 3 0.8"
    "6144 1024 16 48 0.8"
    "6144 1024 32 96 0.8"
    "14336 1024 1 3 0.8"
    "14336 1024 4 12 0.8"
    "14336 1024 8 24 0.8"
)

record_mem

for i in {1..3}; do
    echo "Iteration $i"
    # Iterate through each configuration and test
    for config in "${configs[@]}"; do
        echo "Running test_benchmark_serving_range with config: $config"
        test_benchmark_serving_range $config
        record_mem
    done
done
