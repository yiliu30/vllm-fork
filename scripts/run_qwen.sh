ray stop --force
pkill -9 python

export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 

export OFFICIAL_MODEL="/mnt/disk5/qwen3/Qwen3-30B-A3B-250425"
export OFFICIAL_MODEL="/mnt/disk5/Qwen3-30B-A3B-250425"

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1
#############################
# Qwen
#############################
# remove it ?
export VLLM_DYNAMIC_MOE_MIN_TOKENS=0

# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1 
      
# QUANT_CONFIG=./scripts/inc_measure_v2.json \
# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --max_model_len 2048 \
#     --inc \
#     --dataset pile \
#     --nprompts 128

# QUANT_CONFIG=./scripts/inc_quant_v2.json \
# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --inc \
#     --fp8_kv_cache

QUANT_CONFIG=./scripts/inc_quant_v2.json \
python ./scripts/run_lm_eval_local.py \
    --model ${OFFICIAL_MODEL} \
    --tokenizer ${OFFICIAL_MODEL} \
    --task gsm8k \
    --batch_size 16 \
    --limit 128 \
    --inc \
    --fp8_kv_cache

# python ./scripts/run_lm_eval_local.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --task gsm8k \
#     --batch_size 16 \
#     --limit 128

# DEBUG 04-28 07:23:31 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|████████████| 128/128 [00:57<00:00,  2.23it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8750|±  |0.0293|
# |     |       |strict-match    |     5|exact_match|↑  |0.9219|±  |0.0238|

# INC pile-128
# WARNING 04-28 07:43:17 [hpu_model_runner.py:1013] Configuration: ('prompt', 16, 896) was not warmed-up!
# DEBUG 04-28 07:45:18 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|██████████████████████| 128/128 [15:50<00:00,  7.43s/it]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8516|±  |0.0315|
# |     |       |strict-match    |     5|exact_match|↑  |0.8906|±  |0.0277|

#############################
# Qwen End
#############################

