# # Usage: benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE COMM_BACKEND [PP_LAYER_PARTITION]

# Example: bash benchmark_server_param.sh 1 8192 128 4 2 hccl 
ray stop 
pkill -9 python


export PT_HPU_LAZY_MODE=1
MODEL_PATH="/mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static"


timestamp=$(date +%Y%m%d_%H%M%S)
profile_folder="a_profile_results_${timestamp}"
torch_profile_folder="a_torch_profile_results_${timestamp}"
day=$(date +%Y-%m-%d)
logdir="./pp_logs/${day}"
mkdir -p $logdir
LOG_FILE="./${logdir}/benchmark_server_${timestamp}.log"
export VLLM_TORCH_PROFILER_DIR=$torch_profile_folder
# export VLLM_SKIP_WARMUP=true
# bash benchmark_server_param.sh 1 8192 16 4 2 hccl 32,29 fp8_inc 2>&1 | tee  ./pp_profiling_logs/benchmark_server.log
# bash benchmark_server_param.sh 1 8192 16 4 2 hccl 32,29 fp8_inc 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE

# benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE COMM_BACKEND [PP_LAYER_PARTITION]
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL="DEBUG"
# # export VLLM_TRACE_FUNCTION=1
# export VLLM_TRACE_FUNCTION_DIR="./pp_global_3/vllm_func_trace_${timestamp}"

export VLLM_TORCH_PROFILER_DIR="dummy_dir"

##############
# For trace mem
export PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT=1
export PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE=0
export PT_HPU_EAGER_PIPELINE_ENABLE=0
export PT_HPU_DISABLE_ASYNC_COLLECTIVE=1
export PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD=0
export PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD=0
export PT_HPU_ENABLE_COMPILE_THREAD=0
export PT_HPU_ENABLE_EXECUTION_THREAD=0
export PT_HPU_LAZY_ACC_PAR_MODE=0
export PT_HPU_SYNC_LAUNCH=1

export LOG_FILE_SIZE=1048576000
export PT_TOWL_LOG_ENABLE=1
export HABANA_LOGS=.habana_logs-pp-mem-22-20
# clean the HABANA_LOGS if exists
if [ -d "$HABANA_LOGS" ]; then
    rm -rf $HABANA_LOGS
fi
####################

# bash start_global_pp_local.sh 1 8192 16 4 2 hccl 5,3 auto 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
# bash start_global_pp_local.sh 1 8192 16 4 2 hccl 5,3 auto 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
# bash start_global_pp_local.sh 1 16384 16 4 2 hccl 32,29 auto 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
# bash start_global_pp_local.sh 1 16384 16 4 2 hccl 5,3 auto 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
bash start_global_pp_local.sh 1 16384 16 4 2 hccl 2,2 fp8_inc 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
# bash start_global_pp_local.sh 1 16384 16 4 2 hccl 32,29 fp8_inc 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE
# bash start_global_pp_local.sh 1 8192 16 4 2 gloo 32,29 auto 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  $LOG_FILE

# export VLLM_TORCH_PROFILER_DIR="./test_profiler_pp"
# bash benchmark_server_param.sh 1 8192 16 4 2 hccl 5,3 auto 127.0.0.1 8989 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static  2>&1 | tee  ./pp_profiling_logs/benchmark_server.log

# source benchmark_client_param.sh
#    test_benchmark_serving 2048 2048 32 96 0.8 127.0.0.1 8988 /mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static


curl -X POST http://127.0.0.1:8688/start_profile

curl -X POST http://127.0.0.1:8989/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk2/hf_models/changwa1/DeepSeek-R1-G2-static",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 20,
           "temperature": 0.7,
           "top_p": 1.0
         }'

curl -X POST http://127.0.0.1:8988/start_profile


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

curl -X POST http://127.0.0.1:8688/start_profile

curl -X POST http://127.0.0.1:8688/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 20,
           "temperature": 0.7,
           "top_p": 1.0
         }'

# test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${MAX_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
#     | tee -a ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log
