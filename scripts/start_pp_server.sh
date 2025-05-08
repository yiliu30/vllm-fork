# # Usage: benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE COMM_BACKEND [PP_LAYER_PARTITION]

# Example: bash benchmark_server_param.sh 1 8192 128 4 2 hccl 
ray stop 
pkill -9 python


export PT_HPU_LAZY_MODE=1
MODEL_PATH="/mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static"

# bash benchmark_server_param.sh 1 8192 16 4 2 hccl 32,29 fp8_inc 2>&1 | tee  ./pp_profiling_logs/benchmark_server.log
# bash benchmark_server_param.sh 1 8192 16 4 2 hccl 32,29 auto 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  ./pp_profiling_logs/benchmark_server.log

export VLLM_TORCH_PROFILER_DIR="./test_profiler_pp"
bash benchmark_server_param.sh 1 8192 16 4 2 hccl 5,3 auto 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  ./pp_profiling_logs/benchmark_server.log


curl -X POST http://127.0.0.1:8989/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 100,
           "temperature": 0.7,
           "top_p": 1.0
         }'

curl -X POST http://127.0.0.1:8989/start_profile

# test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${MAX_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
#     | tee -a ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log
