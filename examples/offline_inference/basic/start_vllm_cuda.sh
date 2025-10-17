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
model_path=/data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/
# model_path=/data5/yliu7/HF_HOME/Yi30/unsloth-gpt-oss-20b-BF16-MXFP4
model_path="/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/"

tp_size=2
ep_size=$tp_size
#  OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 



#  OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 

# # !!! no-enable-prefix-caching !!!
# # NOT use EP for W4A8 official model
# # MUST USE EP for MXFP4 Modular MoE model
# 
# #  Support Matrix
# | Model Type         | Prefix Caching | Expert Parallel |
# |--------------------|----------------|-----------------|
# | W4A8(Official)     | No             | No              |
# | BF16               | No             | No              |
# | MXFP4(Modular MoE) | No             | Yes             |


# # W4A8 B200
# # OPENAI_API_KEY=None  python -m gpt_oss.evals --model /storage/yiliu7/openai/gpt-oss-20b/ --eval aime25 --n-threads 128   --reasoning-effort  low  --output_dir ./w4a4-res/
# # OPENAI_API_KEY=None  python -m gpt_oss.evals --model /storage/yiliu7/openai/gpt-oss-120b/ --eval aime25 --n-threads 128   --reasoning-effort  low  --output_dir ./w4a4-res/
# model_path=/storage/yiliu7/openai/gpt-oss-20b/
# model_path=/storage/yiliu7/openai/gpt-oss-120b/
# VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1  \
#     vllm serve $model_path \
#     --tensor-parallel-size $tp_size \
#     --max-model-len 131072 \
#     --max-num-batched-tokens 10240 \
#     --max-num-seqs 128 \
#     --gpu-memory-utilization 0.85 \
#     --no-enable-prefix-caching \
#     --trust-remote-code  2>&1 | tee $log_file

# # BF16 A100/B200
# model_path="/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16/"
# # VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true \
# # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#accuracy-evaluation-panels
# #  OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 
# PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
#     vllm serve $model_path \
#     --max-model-len 131072 \
#     --max-num-batched-tokens 10240 \
#     --tensor-parallel-size $tp_size \
#     --max-num-seqs 256 \
#     --gpu-memory-utilization 0.6 \
#     --dtype bfloat16 \
#     --port 8099 \
#     --no-enable-prefix-caching \
#     --trust-remote-code  2>&1 | tee $log_file


# W4A4 A100/B200
model_path="/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/"
model_path=/data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/
model_path=/storage/yiliu7/unsloth/gpt-oss-120b-BF16-ar-MXFP4
tp_size=4
# VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true \
# OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 
# OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 
# OPENAI_API_KEY=None  python -m gpt_oss.evals --model /storage/yiliu7/unsloth/gpt-oss-120b-BF16-ar-MXFP4 --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low  --output_dir ./w4a4-res/
PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
VLLM_ENABLE_AR_EXT=1 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_AR_MXFP4_MODULAR_MOE=1 \
    vllm serve $model_path \
    --max-model-len 131072 \
    --max-num-batched-tokens 10240 \
    --tensor-parallel-size $tp_size \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --port 8099 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --trust-remote-code  2>&1 | tee $log_file


# ==-----------------------------------------------------------------==
# END
# ==-----------------------------------------------------------------==

# naive moe
# PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_ENABLE_STATIC_MOE=1 \
#     vllm serve $model_path \
#     --max-model-len 8192 \
#     --tensor-parallel-size $tp_size \
#     --max-num-batched-tokens  8192 \
#     --max-num-seqs 32 \
#     --gpu-memory-utilization 0.7 \
#     --dtype bfloat16 \
#     --port 8099 \
#     --enable-expert-parallel \
#     --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","level":3}' \
#     --trust-remote-code  2>&1 | tee $log_file


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
