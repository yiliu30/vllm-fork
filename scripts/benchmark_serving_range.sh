#! /bin/bash

model_path=/mnt/disk5/yiliu4/DeepSeek-R1-G2-dynamic
model_path=/mnt/disk5/yiliu4/DeepSeek-R1-G2-static
model_path=/dev/shm/DeepSeek-R1-G2-static/

host=127.0.0.1
port=8688
input=1024
output=1024
ratio=0.8
num_prompts=96
request_rate=inf
max_concurrency=16

model_name=$( echo $model_path | awk -F/ '{print $NF}' )
echo "Benchmarking ${model_path} for vllm server '${host}:${port}' with input_max=${input}, output_max=${output}, ratio=${ratio}, num_prompts=${num_prompts}, request_rate=${request_rate}, max_concurrency=${max_concurrency}"

log_name=benchmark_serving_static_dmoe_inc_fp8mla_fp8kv_${model_name}_random_in-${input}_out-${output}_ratio-${ratio}_rate-${request_rate}_prompts-${num_prompts}_${max_concurrency}_$(date +%F-%H-%M-%S)

# VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
VLLM_PROFILE_EXECUTE_MODEL_PROMPT=1 \
HABANA_PROFILE=1 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE_STEPS=5 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
HABANA_PROF_CONFIG=/root/.habana/prof_config.json \
python ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $model_path \
    --trust-remote-code \
    --host $host \
    --port $port \
    --dataset-name random \
    --random-input-len $input \
    --random-output-len $output \
    --random-range-ratio $ratio \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --seed 0 \
    --save-result \
    --max-concurrency $max_concurrency \
    --result-filename "${log_name}".json \
    --ignore-eos \
    |& tee "${log_name}".log 2>&1
