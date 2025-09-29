export VLLM_LOGGING_LEVEL=DEBUG
timestamp=$(date +%Y%m%d-%H%M%S)
log_file=server.$timestamp.log
model_path=/home/yiliu7/models/deepseek-ai/DeepSeek-R1
model_path=/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantization_non_uniform/Llama-3.2-1B-Instruct-NVFP4-FP8-Dynamic
model_path="/data5/yliu7/HF_HOME/qwen_moe_skip_lm_head"
# model_path=/data5/yliu7/HF_HOME/ByteDance-Seed/Seed-OSS-36B-Instruct
model_path=/data5/yliu7/HF_HOME/GLM-4.5-Air-w8afp8-llmc/GLM-4.5-Air-w8afp8
# model_path=/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-NVFPP_B16/
# model_path=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-20b-BF16-MXFP8/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-120b-BF16-unsloth-MXFP8
tp_size=4
ep_size=2

VLLM_USE_STATIC_MOE_HPU=1 \
    vllm serve $model_path \
    --max-model-len 8192 \
    --tensor-parallel-size $tp_size \
    --max-num-batched-tokens  8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --port 8088 \
    --enable-expert-parallel \
    --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","level":3}' \
    --trust-remote-code  2>&1 | tee $log_file


# VLLM_USE_MXFP4_CT_EMULATIONS=1 \
# VLLM_USE_NVFP4_CT_EMULATIONS=1 \
# VLLM_DISABLE_INPUT_QDQ=1 \
# VLLM_TORCH_PROFILER_DIR="./qwen_decode_bs32" \
# VLLM_ENGINE_PROFILER_WARMUP_STEP=3 \
# VLLM_ENGINE_PROFILER_STEPS=5 \
# VLLM_ENGINE_PROFILER_REPEAT=1 \
# VLLM_USE_V1=0 \
# VLLM_W8A8_QDQ=1 VLLM_W8A8_STATIC_MOE=1  \
# VLLM_USE_STANDALONE_COMPILE=0 VLLM_WORKER_MULTIPROC_METHOD=spawn  vllm serve $model_path \
#     --max-model-len 8192 \
#     --tensor-parallel-size $tp_size \
#     --max-num-batched-tokens  8192 \
#     --max-num-seqs 64 \
#     --gpu-memory-utilization 0.8 \
#     --dtype bfloat16 \
#     --port 8088 \
#     --enable-expert-parallel \
#     --trust-remote-code  2>&1 | tee $log_file

# VLLM_USE_V1=0 \
# VLLM_W8A8_QDQ=1 VLLM_W8A8_STATIC_MOE=1  \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_seqs=256,gpu_memory_utilization=0.65,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,enable_expert_parallel=True" \
#   --tasks gsm8k --batch_size 128 

# /data5/yliu7/HF_HOME/GLM-4.5-Air-w8afp8-llmc/GLM-4.5-Air-w8afp8
# VLLM_USE_V1=0 \
# VLLM_W8A8_QDQ=1 VLLM_W8A8_STATIC_MOE=1  \
# VLLM_USE_STANDALONE_COMPILE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn  vllm serve $model_path \
#     --max-model-len 8192 \
#     --tensor-parallel-size $tp_size \
#     --max-num-batched-tokens  16384 \
#     --max-num-seqs 256 \
#     --cuda-graph-sizes 1 4 32 64 256 \
#     --gpu-memory-utilization 0.8 \
#     --dtype bfloat16 \
#     --port 8688 \
#     --enable-expert-parallel \
#     --trust-remote-code  2>&1 | tee $log_file


# tp_size=1
# ep_size=1
# VLLM_USE_STANDALONE_COMPILE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn  vllm serve $model_path \
#     --port 8687 \
#     --enable-auto-tool-choice \
#     --tool-call-parser seed_oss \
#     --trust-remote-code \
#     --max-model-len 8192 \
#     --max-num-seqs 16 \
#     --chat-template ${model_path}/chat_template.jinja \
#     --tensor-parallel-size $tp_size \
#     --dtype bfloat16  2>&1 | tee $log_file
