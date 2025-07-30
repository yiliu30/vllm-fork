DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
# DEFAULT_MODEL_PATH="/mnt/disk7/yiliu4/DeepSeek-R1-0528-G2-2nd"
# DEFAULT_MODEL_PATH=/mnt/disk9/hf_models/Kimi-K2-Instruct-G2/
# DEFAULT_MODEL_PATH=/mnt/disk8/yiliu7/moonshotai/Kimi-K2-Instruct
DEFAULT_MODEL_PATH=/mnt/disk5/Kimi-K2-Instruct-G2/
DEFAULT_MODEL_PATH=/models/DeepSeek-R1-0528-G2/
FP8_MODEL_PATH=$DEFAULT_MODEL_PATH
# FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

QUANT_CONFIG_FILE="/mnt/disk3/yiliu4/vllm-fork/scripts/quant_configs/inc_measure_with_fp8kv_config.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.512.${timestamp}.log"

# remove ./scripts/nc_workspace_measure_kvache if needed
if [ -e ./scripts/nc_workspace_measure_kvache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

# Usage of this script:
# bash ./scripts/run_inc_calib.sh --wd 16 --prompts 512


# Default value for WORLD_SIZE
WORLD_SIZE=16
NUM_PROMPTS=512
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wd)
            WORLD_SIZE=$2
            shift 2
            ;;
        --prompts)
            NUM_PROMPTS=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export RAY_DEDUP_LOGS=0

export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

VLLM_LOGGING_LEVEL=DEBUG \
PT_HPU_LAZY_MODE=1 \
VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_MAX=1 \
VLLM_REQUANT_FP8_INC=1 \
VLLM_MOE_N_SLICE=1 \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python scripts/run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 1024 \
    --max_num_seqs 16 \
    --nprompts $NUM_PROMPTS \
    --max_model_len 16192 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --dataset pile 2>&1 | tee $LOG_FILE
    
    
    
# local-completions (model=/models/DeepSeek-R1-0528-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=16,), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
# |  Tasks  |Version|  Filter   |n-shot|Metric|   |Value|   |Stderr|
# |---------|------:|-----------|-----:|------|---|----:|---|-----:|
# |humaneval|      1|create_test|     0|pass@1|   |  0.5|±  |0.0392|
    
# ts batched requests with varying total sequence lengths.
# 2025-07-28:14:50:18,024 INFO     [lm_eval.models.api_models:115] Using max length 2048 - 1
# 2025-07-28:14:50:18,025 INFO     [lm_eval.models.api_models:118] Concurrent requests are disabled. To enable concurrent requests, set `num_concurrent` > 1.
# 2025-07-28:14:50:18,025 INFO     [lm_eval.models.api_models:133] Using tokenizer huggingface
# 2025-07-28:14:50:30,986 INFO     [lm_eval.api.task:420] Building contexts for gsm8k on rank 0...
# 100%|██████████████| 1319/1319 [00:03<00:00, 387.62it/s]
# 2025-07-28:14:50:34,405 INFO     [lm_eval.evaluator:517] Running generate_until requests
# Requesting API: 100%|█| 1319/1319 [12:40<00:00,  1.73it/
# 2025-07-28:15:03:18,974 INFO     [lm_eval.loggers.evaluation_tracker:209] Saving results aggregated
# 2025-07-28:15:03:18,977 INFO     [lm_eval.loggers.evaluation_tracker:290] Saving per-sample results for: gsm8k
# local-completions (model=/models/DeepSeek-R1-0528-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=16,), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
# |     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|