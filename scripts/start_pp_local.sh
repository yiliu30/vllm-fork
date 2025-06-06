# # benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE COMM_BACKEND 
# bash start_pp_server_local_g4.sh 1 16384 128 2 4 hccl 17,14,1515 fp8_inc 127.0.0.1 8688  /mnt/disk2/hf_models/DeepSeek-R1-G2-static
# # start_pp_server_local_g4.sh 1 16384 128 4 2 hccl 29,32 127.0.0.1 8688 /data5/yiliu4/deepseek-ai/DeepSeek-R1-G2

ray stop --force
pkill -9 python



export RAY_DEDUP_LOGS=0
export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1
export PT_HPU_METRICS_GC_DETAILS=1
export VLLM_DUMP_STEP_MEM=1
export VLLM_FAKE_SEND_RECV=0
export VLLM_REPLACE_SEND_RECV_WITH_ALL_REDUCE=0

export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="pp_server.${timestamp}.log"
# bash benchmark_server_param.sh \
#     1 16384 128 4 2 hccl 32,29 fp8_inc \
#     false false 127.0.0.1 8688  /mnt2/models/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2 2>&1 | tee $LOG_FILE


# /mnt/disk6/yiliu4/DeepSeek-R1-G2-static

timestamp=$(date +%Y%m%d_%H%M%S)
# ### Profiling ###
export HABANA_PROFILE_WRITE_HLTV=1 
export HABANA_PROFILE=1
# hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on -invoc csv,hltv -merged csv,hltv
hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on
hl-prof-config --gaudi2

export GRAPH_VISUALIZATION=1

# Is it a prompt phase or decode phase? 
# What batch size?
# What size of input length (in case of prompt) or number of blocks allocated in PagedAttention (decode)?
# Do you want to collect a trace with HPU graphs (t== true, f=false)?
# For example to profile decode with batch size 256 and 512 and HPU graphs, then the flag is:

#  VLLM_PT_PROFILE=decode_256_512_t
# PROFILE_PHASE="prompt"  # or "decode"
# PROFILE_PHASE="decode"
# PROFILE_BATCH_SIZE=16
# PROFILE_INPUT_LENGTH=128
# export VLLM_PT_PROFILE="${PROFILE_PHASE}_${PROFILE_BATCH_SIZE}_${PROFILE_INPUT_LENGTH}_t"
timestamp=$(date +%Y%m%d_%H%M%S)
VLLM_PT_PROFILE="n"
hl_prof_out_dir="pp_prof_hlv_${VLLM_PT_PROFILE}_${timestamp}"
hl-prof-config -o $hl_prof_out_dir

torch_prof_out_dir="pp_prof_torch_${VLLM_PT_PROFILE}_${timestamp}"

# curl -X POST http://localhost:8688/start_profile
#  bash scripts/quickstart/benchmark_vllm_client.sh 
#  curl -X POST http://localhost:8688/stop_profile
export VLLM_TORCH_PROFILER_DIR=$torch_prof_out_dir
export VLLM_ENGINE_PROFILER_ENABLED=1
export VLLM_ENGINE_PROFILER_WARMUP_STEPS=4100
export VLLM_ENGINE_PROFILER_STEPS=2
export VLLM_ENGINE_PROFILER_REPEAT=1

bash benchmark_server_param.sh \
    1 16384 128 4 2 hccl 4,4 fp8_inc \
    false false 127.0.0.1 8688  /mnt/disk6/yiliu4/DeepSeek-R1-G2-static 2>&1 | tee $LOG_FILE

# bash benchmark_server_param.sh \
#     1 16384 128 1 4 hccl "20,20,20,20" fp8_inc \
#     false false 127.0.0.1 8688  /mnt2/models/meta-llama/Llama-3.1-70B  2>&1 | tee $LOG_FILE

# bash benchmark_server_param.sh \
#     1 16384 128 2 4 hccl 17,14,15,15 fp8_inc \
#     true false 127.0.0.1 8688  /mnt2/models/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2 2>&1 | tee $LOG_FILE


curl -X POST http://127.0.0.1:8688/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk2/hf_models/DeepSeek-R1-G2-static",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 100,
           "temperature": 0.7,
           "top_p": 1.0
         }'
         
curl -X POST http://127.0.0.1:8688/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk6/yiliu4/DeepSeek-R1-G2-static",
           "prompt": "Solve the following math problem step by step: What is 25 + 37?",
           "max_tokens": 100,
           "temperature": 0.7,
           "top_p": 1.0
         }'
         
