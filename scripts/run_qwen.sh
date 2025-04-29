pkill -9 python

export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

export OFFICIAL_MODEL="/mnt/disk5/Qwen3-30B-A3B-250425"

#############################
# Qwen
#############################
# FIXME: (Yi) Enable the static MoE path 
export VLLM_DYNAMIC_MOE_MIN_TOKENS=0



#!/bin/bash

set -e

MODE=$1  # First argument
MODEL=$2 # Second argument (model name)
TOKENIZER=$3 # Third argument (tokenizer name)

if [ -z "$TOKENIZER" ]; then
  TOKENIZER=$MODEL
fi

if [ -z "$MODE" ] || [ -z "$MODEL" ] || [ -z "$TOKENIZER" ]; then
  echo "Usage: $0 {bf16|calib|quant|eval} <model> <tokenizer>"
  exit 1
fi

COMMON_ARGS="--model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size 1 --ep_size 1"

if [ "$MODE" == "bf16" ]; then
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS

elif [ "$MODE" == "calib" ]; then
  QUANT_CONFIG=./scripts/inc_measure_v2.json \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 512

elif [ "$MODE" == "quant" ]; then
  QUANT_CONFIG=./scripts/inc_quant_v2.json \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --fp8_kv_cache

elif [ "$MODE" == "eval" ]; then
  VLLM_PROMPT_SEQ_BUCKET_MIN=2048 \
  VLLM_PROMPT_SEQ_BUCKET_STEP=2048 \
  VLLM_PROMPT_SEQ_BUCKET_MAX=2048 \
  QUANT_CONFIG=./scripts/inc_quant_v2.json \
  python ./scripts/run_lm_eval_local.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --task gsm8k \
    --batch_size 16 \
    --inc \
    --fp8_kv_cache 2>&1 | tee ./gsm8k.pile.inc_quant_v2.g2.428.log

else
  echo "Unknown mode: $MODE"
  echo "Valid modes are: bf16, calib, quant, eval"
  exit 1
fi

