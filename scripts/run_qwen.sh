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

# BF16
# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1 

# # Calibration
# QUANT_CONFIG=./scripts/inc_measure_v2.json \
# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --max_model_len 2048 \
#     --inc 

# Quantization
# QUANT_CONFIG=./scripts/inc_quant_v2.json \
# python ./scripts/run_example_tp_qwen.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --inc \
#     --fp8_kv_cache


# # Evaluation
# VLLM_PROMPT_SEQ_BUCKET_MIN=2048 \
# VLLM_PROMPT_SEQ_BUCKET_STEP=2048 \
# VLLM_PROMPT_SEQ_BUCKET_MAX=2048 \
# QUANT_CONFIG=./scripts/inc_quant_v2.json \
# python ./scripts/run_lm_eval_local.py \
#     --model ${OFFICIAL_MODEL} \
#     --tokenizer ${OFFICIAL_MODEL} \
#     --task gsm8k \
#     --batch_size 16 \
#     --inc \
#     --fp8_kv_cache 2>&1 | tee ./qwen_logs/gsm8k.pile.inc_quant_v2.g2.428.log
    
    
VLLM_PROMPT_SEQ_BUCKET_MIN=2048 \
VLLM_PROMPT_SEQ_BUCKET_STEP=2048 \
VLLM_PROMPT_SEQ_BUCKET_MAX=2048 \
python ./scripts/run_lm_eval_local.py \
    --model ${OFFICIAL_MODEL} \
    --tokenizer ${OFFICIAL_MODEL} \
    --task gsm8k \
    --batch_size 16  2>&1 | tee ./qwen_logs/gsm8k.pile.BF16.g2.428.log


# REPORT OUT
# BF16
# Running generate_until requests: 100%|██████████| 1319/1319 [06:35<00:00,  3.33it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
# |     |       |strict-match    |     5|exact_match|↑  |0.8961|±  |0.0084|


# - FP8 FULL
# Running generate_until requests: 100%|██████████| 1319/1319 [17:29<00:00,  1.26it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8484|±  |0.0099|
# |     |       |strict-match    |     5|exact_match|↑  |0.8779|±  |0.0090|

#=---------------------------------------------------------------------------=

# BF16
# DEBUG 04-28 07:23:31 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|████████████| 128/128 [00:57<00:00,  2.23it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8750|±  |0.0293|
# |     |       |strict-match    |     5|exact_match|↑  |0.9219|±  |0.0238|

# BF16
# Running generate_until requests: 100%|██████████| 1319/1319 [06:35<00:00,  3.33it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
# |     |       |strict-match    |     5|exact_match|↑  |0.8961|±  |0.0084|

# INC pile-128
# WARNING 04-28 07:43:17 [hpu_model_runner.py:1013] Configuration: ('prompt', 16, 896) was not warmed-up!
# DEBUG 04-28 07:45:18 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|██████████████████████| 128/128 [15:50<00:00,  7.43s/it]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8516|±  |0.0315|
# |     |       |strict-match    |     5|exact_match|↑  |0.8906|±  |0.0277|


# Running generate_until requests:  76%|███████████████████████████████████████▍            | 97/128 [07:51<00:56,  1.82s/it]DEBUG 04-28 09:40:26 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|███████████████████████████████████████████████████| 128/128 [07:58<00:00,  3.74s/it]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8750|±  |0.0293|
# |     |       |strict-match    |     5|exact_match|↑  |0.8906|±  |0.0277|


# - FP8 FULL
# Running generate_until requests: 100%|█████████▉| 1313/131Running generate_until requests: 100%|██████████| 1319/1319 [17:29<00:00,  1.26it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8484|±  |0.0099|
# |     |       |strict-match    |     5|exact_match|↑  |0.8779|±  |0.0090|



#############################
# Qwen End
#############################

