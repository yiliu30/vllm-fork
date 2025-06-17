# (Pdb) self.model.model.layers[0].mlp.experts.hpu_fused_moe.MoeOp.quant_input
# QuantDynamicInput(lp_dtype=torch.float8_e4m3fn, hp_dtype=torch.bfloat16 input_scales_creator=MaxAbsPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=False))
# (Pdb) self.model.model.layers[0].mlp.experts.hpu_fused_moe.MoeOp.quant_input.input_scales_creator
# MaxAbsPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=False)
# (Pdb) self.model.model.layers[0].self_attn.qkv_proj.quant_input.input_scales_creator
# MaxAbsDynamicPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=True)


# - BF16 strict-match: 91.66
# - FP8 Dynamic (exclude `RowParallelLinear`)

# vllm (pretrained=/mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=32,gpu_memory_utilization=0.75,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9060|±  |0.0080|
# |     |       |strict-match    |     5|exact_match|↑  |0.9166|±  |0.0076|

# vllm (pretrained=/mnt/weka/data/pytorch/DeepSeek-R1/,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=32,gpu_memory_utilization=0.75,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
# |     |       |strict-match    |     5|exact_match|↑  |0.9591|±  |0.0055|

pkill -9 python

# GOOD
# KNUM_LAYER=1 bash scripts/run_qwen.sh  quant /mnt/disk5/Qwen3-30B-A3B-250425 /mnt/disk5/Qwen3-30B-A3B-250425 8
# KNUM_LAYER=1 bash scripts/run_qwen.sh  quant /mnt/disk5/Qwen3-30B-A3B-250425 /mnt/disk5/Qwen3-30B-A3B-250425 8 
# bash scripts/run_qwen.sh quant  /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 8  2>&1 | tee dynamic_qwen_logs.log



# bash scripts/run_qwen.sh quant  /mnt2/qwen/Qwen3-30B-A3B /mnt2/qwen/Qwen3-30B-A3B 
# bash scripts/run_qwen.sh quant  /mnt/disk5/Qwen3-30B-A3B-250425 /mnt/disk5/Qwen3-30B-A3B-250425 8 2>&1 | tee dynamic_qwen_logs.log
# bash scripts/run_qwen.sh quant  /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 8 2>&1 | tee dynamic_qwen_logs.log
# bash scripts/run_qwen.sh quant  /mnt/disk6/qwen3/Qwen3-235B-A22B/ /mnt/disk6/qwen3/Qwen3-235B-A22B/ 8
# bash scripts/run_qwen.sh quant  /mnt/disk2/hf_models/Llama-2-7b-chat-hf /mnt/disk2/hf_models/Llama-2-7b-chat-hf 1
# bash scripts/run_qwen.sh quant  /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/ /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/

export PT_HPU_WEIGHT_SHARING=0
export VLLM_HPU_FORCE_CHANNEL_FP8=0
# export VLLM_ENABLE_RUNTIME_DEQUANT=1
export RAY_DEDUP_LOGS=0
export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1
export PT_HPU_METRICS_GC_DETAILS=1
# export VLLM_DUMP_STEP_MEM=0
# export VLLM_FAKE_SEND_RECV=0
# export VLLM_REPLACE_SEND_RECV_WITH_ALL_REDUCE=0


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
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1


#!/bin/bash

set -x

MODE=$1  # First argument
MODEL=$2 # Second argument (model name)
TOKENIZER=$3 # Third argument (tokenizer name)
# SET TP SIZE, DEFAULT 1
TP_SIZE=${4:-1} # Fourth argument (optional, default is 8)

if [ -z "$TOKENIZER" ]; then
  TOKENIZER=$MODEL
fi

if [ -z "$MODE" ] || [ -z "$MODEL" ] || [ -z "$TOKENIZER" ]; then
  echo "Usage: $0 {bf16|calib|quant|eval} <model> <tokenizer>"
  exit 1
fi

# /mnt/disk5/Qwen3-30B-A3B-250425

COMMON_ARGS="--model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size ${TP_SIZE} --ep_size ${TP_SIZE} "

model_name=$(basename ${MODEL})
if [ ${model_name} == "Qwen3-30B-A3B-250425" ]; then
    quant_file_path="inc_meaure_g3_30B_A3B.json"
elif [ ${model_name} == "Qwen3-32B" ]; then
    quant_file_path="inc_meaure_g3_32B.json"
else
    echo "Unknown model name: ${model_name}"
fi


if [ "$MODE" == "bf16" ]; then
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS

elif [ "$MODE" == "calib" ]; then
  QUANT_CONFIG=${quant_file_path} \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 512

elif [ "$MODE" == "quant" ]; then
  QUANT_CONFIG=./scripts/dynamic_quant_qwen.json \
  python ./scripts/run_example_tp_qwen_local.py $COMMON_ARGS  --nprompts 2 --enforce_eager

else
  echo "Unknown mode: $MODE"
  echo "Valid modes are: bf16, calib"
  exit 1
fi


# ====================================
# Prompt: 'Hello, my name is'
# Generated text: ' not yetACHED goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals'
# Generated token: (537, 3602, 52645, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845)
# Ground truth: None
# ====================================
# ====================================
# Prompt: '0.999 compares to 0.9 is '
# Generated text: 'いらっration啬Beginning/endouce troubles troubles faced face face face face face face face face face face face face face face face face face face face face face face face'
# Generated token: (141650, 2165, 120614, 75290, 82846, 69779, 34565, 34565, 16601, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579, 3579)
# Ground truth: None
# ====================================

# ====================================
# Prompt: 'Hello, my name is'
# Generated text: ' not yetACHED goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals goals'
# Generated token: (537, 3602, 52645, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845, 8845)
# Ground truth: None
# ====================================
# ====================================
# Prompt: '0.999 compares to 0.9 is '
# Generated text: 'いらっration啬ущ сохрiginalism â…\n\nキャンペoir particular,該使用者 inquired about both sides as newly fresh fresh fresh fresh fresh fresh fresh fresh fresh fresh fresh'
# Generated token: (141650, 2165, 120614, 42193, 143363, 11040, 2142, 27905, 5434, 140924, 13300, 3953, 11, 118276, 304, 2931, 911, 2176, 11067, 438, 13631, 7722, 7722, 7722, 7722, 7722, 7722, 7722, 7722, 7722, 7722, 7722)
# Ground truth: None
# ====================================


# vllm (pretrained=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9606|±  |0.0054|
# |     |       |strict-match    |     5|exact_match|↑  |0.9598|±  |0.0054|