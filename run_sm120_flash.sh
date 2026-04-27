source .venv/bin/activate

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# SM120 reference attention controls (required)
export VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION=1
export VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE=256
export VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE=128


# export VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION=1
export VLLM_SM120_TRITON_MLA=1
    # /home/yiliu7/workspace/yi-dashboard/scripts/trace_gen.py \
# source 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     vllm serve \
#     --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
#     --kv-cache-dtype fp8 \
#     -tp 4 \
#    --max-model-len 8192 \
#    --max-num-batched-tokens 32768 \
#      --gpu-memory-utilization 0.8 \
#     --block-size 256 \
#     --enable-expert-parallel  2>&1 | tee vllm_serve.log
        # --max-num-batched-tokens 32768 \
    #   --max-num-seqs 128
    #     --max-num-batched-tokens 32768 \
    # --tokenizer-mode deepseek_v4 

    #  --max-num-batched-tokens 32768 
    # --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \

# /home/yiliu7/workspace/yi-dashboard/scripts/trace_gen.py
    # examples/basic/offline_inference/generate.py \

VLLM_SM120_DISABLE_DEEPGEMM=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
    examples/basic/offline_inference/generate.py \
    --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
    -tp 4  --kv-cache-dtype fp8 \
    --max-model-len 2084 --gpu-memory-utilization 0.8 \
    --enforce-eager

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
#     /home/yiliu7/workspace/yi-dashboard/scripts/trace_gen.py \
#     --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
#     -tp 4  --kv-cache-dtype fp8 \
#     --max-model-len 2084 --gpu-memory-utilization 0.8 \
#     --enforce-eager

# VLLM_SM120_DISABLE_DEEPGEMM=1 \
# VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NUM_HIDDEN_LAYERS=4 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
#       examples/basic/offline_inference/generate.py  \
#     --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
#     -tp 1  --kv-cache-dtype fp8 \
#     --max-model-len 2084 --gpu-memory-utilization 0.8 \
#     --enforce-eager

# VLLM_SM120_DISABLE_DEEPGEMM=1 \
# VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NUM_HIDDEN_LAYERS=4 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
#      /home/yiliu7/workspace/yi-dashboard/scripts/trace_gen.py  \
#     --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
#     -tp 1  --kv-cache-dtype fp8 \
#     --max-model-len 2084 --gpu-memory-utilization 0.8 \
#     --enforce-eager