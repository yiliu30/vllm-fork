#!/bin/bash
model_path=/mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/
ip_addr=127.0.0.1
port=18080


test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name=benchmark_serving_DeekSeek-R1_cardnumber_16_datatype_bfloat16_random_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    python3 /mnt/disk3/yiliu4/vllm-fork/benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max_concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore_eos \
    --save-result --result-filename ${log_name}.json 

}


# test_benchmark_serving_range 1024 1024 1 3 0
# test_benchmark_serving_range 1024 1024 32 96 0
test_benchmark_serving_range 1024 1024 128 256 0

# 100%|███████████████████████████████████| 96/96 [01:24<00:00,  1.13it/s]
# ============ Serving Benchmark Result ============
# Successful requests:                     96        
# Benchmark duration (s):                  84.78     
# Total input tokens:                      98208     
# Total generated tokens:                  98304     
# Request throughput (req/s):              1.13      
# Output token throughput (tok/s):         1159.47   
# Total Token throughput (tok/s):          2317.81   
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          12552.00  
# Median TTFT (ms):                        14779.71  
# P99 TTFT (ms):                           15260.55  
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          13.07     
# Median TPOT (ms):                        13.06     
# P99 TPOT (ms):                           13.36     
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           13.07     
# Median ITL (ms):                         13.00     
# P99 ITL (ms):                            13.64     
# ==================================================