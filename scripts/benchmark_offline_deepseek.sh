#!/bin/bash
total_prompts=224
bs=224 # batch_size is dynamic, this setting throttles the max batch size
in_len=4000 # input length is dynamic, this tell warmup the max input len
out_len=10 # if not set fixed_out_len, the output_len will be dynamic
total_len=$((in_len + out_len))
tp_parallel=8
 
dataset="random"

log_prefix="331-default-inc-moe-op"

out_len_aligned=$((out_len + 127 / 128 * 128))
in_len_aligned=$(((in_len + 127) / 128 * 128))
total_len_aligned=$((in_len_aligned + out_len_aligned))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))
model="/data/models/DeepSeek-R1-static/"
tokenizer="/data/models/DeepSeek-R1-static/"
model="/data/models/DeepSeek-R1/"
tokenizer="/data/models/DeepSeek-R1/"
model_name="DeepSeek-R1"

#VLLM_TORCH_PROFILER_DIR=/workspace/vllm/vllm/pt_profiling/mtp/vllm_profile \
#VLLM_PROFILER_ENABLED=true \
#HABANA_PROFILE=1 HABANA_PROFILE_WRITE_HLTV=1 \
VLLM_REQUANT_FP8_INC=1 \
QUANT_CONFIG=inc_quant_bf16_flat_pa_mla_with_fp8kv_config.json \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE_STEPS=5 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
HABANA_PROF_CONFIG=scripts/profile_api_trace_analyzer.json \
VLLM_DELAYED_SAMPLING=true \
VLLM_MOE_N_SLICE=1 \
VLLM_EP_SIZE=8 \
VLLM_SKIP_WARMUP=true \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=16 \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len_aligned} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
python3 ../benchmarks/benchmark_throughput.py \
    --model ${model} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --backend vllm \
    --num-prompts ${total_prompts} \
    --tensor-parallel-size  ${tp_parallel} \
    --speculative_draft_tensor_parallel_size ${tp_parallel} \
    --max_model_len 4096 \
    --input-len ${in_len} \
    --output-len ${out_len} \
    --trust-remote-code \
    --distributed_executor_backend mp \
    --kv_cache_dtype fp8_inc \
    --gpu-memory-util 0.95 2>&1 | tee bench_logs_331/${log_prefix}offline-throughput-${log_prefix}-mtp-nocontpa-bs${bs}-in${in_len}-out${out_len}-tp${tp_parallel}.log

    #--num_speculative_tokens 3 \
    #    --enable-expert-parallel \