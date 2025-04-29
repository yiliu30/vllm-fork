# !/bin/bash
set +x

if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model_path=$2
else
    model_path="/mnt/weka/llm/qwen3_pre_release/Qwen3-30B-A3B-250425/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 
export VLLM_LOGGING_LEVEL=DEBUG

model_name=$(basename ${model_path})
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${model_name}-tp${tp_size}-gsm8k-acc-${timestamp}"
#limit=None

mkdir -p ${output_dir}
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True" \
  --tasks gsm8k --batch_size 'auto' --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt