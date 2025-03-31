# FIXME: (Yi) remove it before merge
#!/bin/bash
tp_parrallel=8
in_len=1024
out_len=1024
multi_step=1
total_len=$((in_len + out_len))
# if total_len is not multiple of 128, round up to the next multiple of 128
if [ $((total_len % 128)) -ne 0 ]; then
    echo 'round up for 128'
    total_len=$(((total_len / 128 +  1) * 128 ))
fi
ep_size=8
moe_n_slice=1
gpu_utils=0.92
bs=448
num_prompts=448
request_rate=inf
log_name="[prof-331-inc-maxabs_hw-const-moe-pmodule.fp8_fused_weights_scalars-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"

VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128 + 128))
# model="/data/models/DeepSeek-R1-static/"
# tokenizer="/data/models/DeepSeek-R1-static/"
# model_name="DeepSeek-R1-static"

model="/data/models/DeepSeek-R1/"
tokenizer="/data/models/DeepSeek-R1/"
model_name="DeepSeek-R1"

# VLLM_SKIP_WARMUP=true \
# VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
# HABANA_PROF_CONFIG=./profile_api_trace_analyzer.json \
# VLLM_USE_MATMUL_V1=1 \
QUANT_CONFIG="inc_quant_bf16_flat_pa_mla_with_fp8kv_config.json" \
VLLM_REQUANT_FP8_INC=1 \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_DELAYED_SAMPLING=true \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MOE_N_SLICE=${moe_n_slice} \
VLLM_EP_SIZE=${ep_size} \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=16 \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
python -m vllm.entrypoints.openai.api_server \
    --port 8080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step}\
    --max-model-len 4096 \
    --distributed_executor_backend mp \
    --gpu_memory_utilization ${gpu_utils} \
    --kv_cache_dtype "fp8_inc" \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}

hl-smi -l > tee benchmark_logs/${log_name}_smi.log &
hl_pid=$(($!-1))


start_time=$(date +%s)
echo "Start to benchmark"
python ../benchmarks/benchmark_serving.py --backend vllm --model ${model} --tokenizer ${tokenizer} --dataset-name sonnet --dataset-path ../benchmarks/sonnet.txt --request-rate ${request_rate} --num-prompts ${num_prompts} --port 8080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee benchmark_logs/${log_name}_run1.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

# start_time=$(date +%s)
# echo "Start to benchmark"
# python benchmarks/benchmark_serving.py --backend vllm --model ${model} --tokenizer ${tokenizer} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate} --num-prompts ${num_prompts} --port 8080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee benchmark_logs/${log_name}_run2.log
# end_time=$(date +%s)
# echo "Time elapsed: $((end_time - start_time))s"

# sleep 10

kill ${pid}
kill ${hl_pid}
#--backend openai-chat --endpoint "v1/chat/completions"