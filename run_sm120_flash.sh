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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
    examples/basic/offline_inference/generate.py \
    --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
    -tp 4 --enforce-eager --kv-cache-dtype fp8 \
    --max-model-len 2084 --gpu-memory-utilization 0.8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
#     examples/basic/offline_inference/generate.py \
#     --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
#     -tp 4 --enforce-eager --kv-cache-dtype fp8 \
#     --max-model-len 2084 --gpu-memory-utilization 0.8