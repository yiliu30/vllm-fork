# (Pdb) self.model.model.layers[0].mlp.experts.hpu_fused_moe.MoeOp.quant_input
# QuantDynamicInput(lp_dtype=torch.float8_e4m3fn, hp_dtype=torch.bfloat16 input_scales_creator=MaxAbsPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=False))
# (Pdb) self.model.model.layers[0].mlp.experts.hpu_fused_moe.MoeOp.quant_input.input_scales_creator
# MaxAbsPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=False)
# (Pdb) self.model.model.layers[0].self_attn.qkv_proj.quant_input.input_scales_creator
# MaxAbsDynamicPcs(round_scale_method=ScaleToPow2, params={'hp_dtype': torch.bfloat16, 'lp_dtype': torch.float8_e4m3fn}, fullscale=240.0, scale=None, is_dynamic=True)


pkill -9 python
# bash scripts/run_qwen.sh quant  /mnt2/qwen/Qwen3-30B-A3B /mnt2/qwen/Qwen3-30B-A3B 

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

model_name=$(basename ${MODEL})
if [ ${model_name} == "Qwen3-30B-A3B" ]; then
    quant_file_path="inc_meaure_g3_30B_A3B.json"
elif [ ${model_name} == "Qwen3-32B" ]; then
    quant_file_path="inc_meaure_g3_32B.json"
else
    echo "Unknown model name: ${model_name}"
    exit 1
fi


if [ "$MODE" == "bf16" ]; then
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS

elif [ "$MODE" == "calib" ]; then
  QUANT_CONFIG=${quant_file_path} \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 512

elif [ "$MODE" == "quant" ]; then
  QUANT_CONFIG=./scripts/dynamic_quant_qwen.json \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 2 --enforce_eager

else
  echo "Unknown mode: $MODE"
  echo "Valid modes are: bf16, calib"
  exit 1
fi

