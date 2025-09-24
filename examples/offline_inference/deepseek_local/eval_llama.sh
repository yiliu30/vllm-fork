DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-3.3-70B-Instruct-FP8_STATIC-0915-G2"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916-G2"
DEFAULT_MODEL_PATH="/mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/"
DEFAULT_MODEL_PATH="/mnt/disk5/meta-llama/Llama-4-Maverick-17B-128E-Instruct"
DEFAULT_MODEL_PATH="/mnt/disk5/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-G2"
# DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916"
# DEFAULT_MODEL_PATH=/mnt/disk5/Qwen3-30B-A3B-FP8-G2/
# DEFAULT_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2/"
# DEFAULT_MODEL_PATH="/mnt/disk6/yiliu4/deepseek-ai/DeepSeek-R1-0528"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# export CALC_SCALE_WITH_CGUID=1
# export QUANT_CONFIG="./quant_configs/dynamic_quant_config.json"
# export QUANT_CONFIG="./quant_configs/unit_quant_config.json"
export VLLM_SKIP_WARMUP=true

WORLD_SIZE=1
WORLD_SIZE=8

# if 1, set VLLM_NUM_LAYERS to 4
if [ $WORLD_SIZE -eq 1 ]; then
    export VLLM_NUM_LAYERS=4
else
    unset VLLM_NUM_LAYERS
fi

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dynamic_quant.inc.${timestamp}.log"

export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1
export PT_HPU_METRICS_GC_DETAILS=1

export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

export VLLM_HPU_CONVERT_TO_FP8UZ=0
# VLLM_HPU_FORCE_CHANNEL_FP8=1 \

# export VLLM_HPU_CONVERT_TO_FP8UZ=1
# export VLLM_HPU_FORCE_CHANNEL_FP8=0

# # export QUANT_CONFIG=./quant_configs/inc_measure.json
# export QUANT_CONFIG=./quant_configs/unit_quant_config.json 
# export  QUANT_CONFIG="./quant_configs/inc_quant.json"
# export  QUANT_CONFIG="./quant_configs/inc_quant_naive.json"
# export  QUANT_CONFIG="./quant_configs/inc_quant_fp8kv.json"
# VLLM_HPU_CONVERT_TO_FP8UZ=1 \
VLLM_SUPPORT_MOE_CHUNK=true \
PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --max_model_len 512 2>&1 | tee $LOG_FILE
    # --fp8_inc \
    

# model_path=${FP8_MODEL_PATH}
# model_name=$(basename ${model_path})
# timestamp=$(date +"%Y%m%d_%H%M%S")
# output_dir="${model_name}-tp${tp_size}-gsm8k-acc-${timestamp}"
# #limit=None
# tp_size=${WORLD_SIZE}
# mkdir -p ${output_dir}

# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True" \
#   --tasks gsm8k --batch_size 128 --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt
  
# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \
# lm_eval --model vllm-vlm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,max_images=1" \
#   --tasks mmmu_val \
#   --apply_chat_template \
#   --batch_size 128 --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt
  
  
#   |             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |--------------------------------|------:|------|------|------|---|-----:|---|-----:|
# |mmmu_val                        |      0|none  |      |acc   |↑  |0.5211|±  |0.0160|
# | - Art and Design               |      0|none  |      |acc   |↑  |0.6000|±  |0.0437|
# | - Business                     |      0|none  |      |acc   |↑  |0.6067|±  |0.0404|
# | - Health and Medicine          |      0|none  |      |acc   |↑  |0.5533|±  |0.0402|
# | - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7083|±  |0.0409|
# | - Science                      |      0|none  |      |acc   |↑  |0.4000|±  |0.0393|
# | - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3714|±  |0.0332|


  
#   vllm (pretrained=/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916-G2,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9007|±  |0.0082|
# |     |       |strict-match    |     5|exact_match|↑  |0.8848|±  |0.0088|

# vllm-vlm (pretrained=/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916-G2,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |-----|------:|------|-----:|------|---|-----:|---|-----:|
# |Art  |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|


# vllm-vlm (pretrained=/mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,max_images=1), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
# |mmmu_val                               |      0|none  |      |acc   |↑  |0.5444|±  |0.0160|
# | - Art and Design                      |      0|none  |      |acc   |↑  |0.6167|±  |0.0422|
# |  - Art                                |      0|none  |     0|acc   |↑  |0.7000|±  |0.0851|
# |  - Art Theory                         |      0|none  |     0|acc   |↑  |0.8000|±  |0.0743|
# |  - Design                             |      0|none  |     0|acc   |↑  |0.6333|±  |0.0895|
# |  - Music                              |      0|none  |     0|acc   |↑  |0.3333|±  |0.0875|
# | - Business                            |      0|none  |      |acc   |↑  |0.6000|±  |0.0393|
# |  - Accounting                         |      0|none  |     0|acc   |↑  |0.6000|±  |0.0910|
# |  - Economics                          |      0|none  |     0|acc   |↑  |0.8000|±  |0.0743|
# |  - Finance                            |      0|none  |     0|acc   |↑  |0.5000|±  |0.0928|
# |  - Manage                             |      0|none  |     0|acc   |↑  |0.4333|±  |0.0920|
# |  - Marketing                          |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|
# | - Health and Medicine                 |      0|none  |      |acc   |↑  |0.6133|±  |0.0392|
# |  - Basic Medical Science              |      0|none  |     0|acc   |↑  |0.6333|±  |0.0895|
# |  - Clinical Medicine                  |      0|none  |     0|acc   |↑  |0.5000|±  |0.0928|
# |  - Diagnostics and Laboratory Medicine|      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
# |  - Pharmacy                           |      0|none  |     0|acc   |↑  |0.6667|±  |0.0875|
# |  - Public Health                      |      0|none  |     0|acc   |↑  |0.8000|±  |0.0743|
# | - Humanities and Social Science       |      0|none  |      |acc   |↑  |0.7083|±  |0.0414|
# |  - History                            |      0|none  |     0|acc   |↑  |0.7667|±  |0.0785|
# |  - Literature                         |      0|none  |     0|acc   |↑  |0.7000|±  |0.0851|
# |  - Psychology                         |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
# |  - Sociology                          |      0|none  |     0|acc   |↑  |0.8000|±  |0.0743|
# | - Science                             |      0|none  |      |acc   |↑  |0.4467|±  |0.0409|
# |  - Biology                            |      0|none  |     0|acc   |↑  |0.4000|±  |0.0910|
# |  - Chemistry                          |      0|none  |     0|acc   |↑  |0.4000|±  |0.0910|
# |  - Geography                          |      0|none  |     0|acc   |↑  |0.4000|±  |0.0910|
# |  - Math                               |      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
# |  - Physics                            |      0|none  |     0|acc   |↑  |0.5667|±  |0.0920|
# | - Tech and Engineering                |      0|none  |      |acc   |↑  |0.3905|±  |0.0335|
# |  - Agriculture                        |      0|none  |     0|acc   |↑  |0.5333|±  |0.0926|
# |  - Architecture and Engineering       |      0|none  |     0|acc   |↑  |0.2000|±  |0.0743|
# |  - Computer Science                   |      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
# |  - Electronics                        |      0|none  |     0|acc   |↑  |0.3333|±  |0.0875|
# |  - Energy and Power                   |      0|none  |     0|acc   |↑  |0.3667|±  |0.0895|
# |  - Materials                          |      0|none  |     0|acc   |↑  |0.4667|±  |0.0926|
# |  - Mechanical Engineering             |      0|none  |     0|acc   |↑  |0.3667|±  |0.0895|

# |             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |--------------------------------|------:|------|------|------|---|-----:|---|-----:|
# |mmmu_val                        |      0|none  |      |acc   |↑  |0.5444|±  |0.0160|
# | - Art and Design               |      0|none  |      |acc   |↑  |0.6167|±  |0.0422|
# | - Business                     |      0|none  |      |acc   |↑  |0.6000|±  |0.0393|
# | - Health and Medicine          |      0|none  |      |acc   |↑  |0.6133|±  |0.0392|
# | - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7083|±  |0.0414|
# | - Science                      |      0|none  |      |acc   |↑  |0.4467|±  |0.0409|
# | - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3905|±  |0.0335|

