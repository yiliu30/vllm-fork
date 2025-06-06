timestamp=$(date +%Y%m%d-%H%M%S)
LOG_FILE="./_pd_logs/prefill.${timestamp}.log"
TP_SIZE=4
export MOONCAKE_CONFIG_PATH=./mooncake.json
export PT_HPU_LAZY_MODE=1
export VLLM_USE_V1=0
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_SKIP_WARMUP=true
export VLLM_LOGGING_LEVEL=DEBUG

model_path="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/"
# model_path="/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct"
python3 -m vllm.entrypoints.openai.api_server \
    --model  $model_path \
    --port 8100 \
    -tp $TP_SIZE \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' 2>&1 | tee $LOG_FILE