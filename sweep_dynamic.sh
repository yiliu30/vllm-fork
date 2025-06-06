
# KNUM_LAYER=1 bash scripts/run_qwen.sh  quant /mnt/disk5/Qwen3-30B-A3B-250425 /mnt/disk5/Qwen3-30B-A3B-250425 1 2>&1 | tee dynamic_qwen_logs_ep1.log
# KNUM_LAYER=1 bash scripts/run_qwen.sh  quant /mnt/disk5/Qwen3-30B-A3B-250425 /mnt/disk5/Qwen3-30B-A3B-250425 8  2>&1 | tee dynamic_qwen_logs_ep8.log
# bash scripts/run_qwen.sh quant  /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207 8  2>&1 | tee dynamic_deepseek_logs.log




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

output_dir=".vllm_dynamic_logs"
mkdir -p ${output_dir}
tp_size=8
model_path="/mnt/disk5/Qwen3-30B-A3B-250425"
model_path="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/"
# model_path="/mnt/disk5/Qwen3-30B-A3B-FP8"

# QUANT_CONFIG=${quant_file_path} \
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${output_dir}/dynamic_qwen_logs_${timestamp}.log"

export VLLM_EP_SIZE=8

export QUANT_CONFIG=./scripts/dynamic_quant_qwen.json
VLLM_HPU_FORCE_CHANNEL_FP8=0 \
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
HABANA_VISIBLE_DEVICES="ALL" \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True" \
  --tasks gsm8k --batch_size 16 --log_samples --output_path ${output_dir}  --show_config 2>&1 | tee ${LOG_FILE}
  
# 
# vllm (pretrained=/mnt/disk5/Qwen3-30B-A3B-250425,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True), gen_kwargs: (None), limit: 128.0, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9375|±  |0.0215|
# |     |       |strict-match    |     5|exact_match|↑  |0.9375|±  |0.0215|


# vllm (pretrained=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True), gen_kwargs: (None), limit: 256.0, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9688|±  |0.0109|
# |     |       |strict-match    |     5|exact_match|↑  |0.9688|±  |0.0109|

# verified accuracy @chendi
# ** Deepseek **
# vllm (pretrained=/mnt/weka/data/pytorch/DeepSeek-R1/,tensor_parallel_size=8,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=4096,use_v2_block_manager=True,dtype=bfloat16,enable_expert_parallel=True,max_num_seqs=128), gen_kwargs: (None), limit: 256.0, num_fewshot: 5, batch_size: 128
# Tasks	Version	Filter	n-shot	Metric		Value		Stderr
# gsm8k	3	flexible-extract	5	exact_match	↑	0.9688	±	0.0109
# strict-match	5	exact_match	↑	0.9688	±	0.0109
# ** Qwen3-30B **
# vllm (pretrained=/mnt/weka/llm/Qwen3-30B-A3B-FP8/,tensor_parallel_size=8,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=4096,use_v2_block_manager=True,dtype=bfloat16,enable_expert_parallel=True,max_num_seqs=128), gen_kwargs: (None), limit: 256.0, num_fewshot: 5, batch_size: 128

# Tasks	Version	Filter	n-shot	Metric		Value		Stderr
# gsm8k	3	flexible-extract	5	exact_match	↑	0.8828	±	0.0201
# strict-match	5	exact_match	↑	0.9297	±	0.0160
# ** llama4-Maverick **
# vllm (pretrained=/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct-FP8/,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=1024,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=False,enable_expert_parallel=True), gen_kwargs: (None), limit: 0.1, num_fewshot: 0, batch_size: 128

# mmlu_pro	2	custom-extract		exact_match	↑	0.8089	±	0.0109