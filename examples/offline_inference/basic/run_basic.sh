# PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
# PYTHONPATH=/home/yiliu7/workspace/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
#     python basic_local_2.py --tp 1 -e
PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
PYTHONPATH=/home/yiliu7/workspace/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
VLLM_ENABLE_AR_EXT=1 \
VLLM_ENABLE_STATIC_MOE=1 \
VLLM_AR_MXFP4_MODULAR_MOE=0 \
    python basic_local_2.py --tp 1 -e --model_path /data5/yliu7/HF_HOME/Yi30/gpt-oss-20b-BF16-MXFP8