#  curl http://127.0.0.1:8088/metrics

export no_proxy="localhost, 127.0.0.1, ::1"
task_name=gsm8k
batch_size=16
# LIMIT=128
timestamp=$(date +%Y%m%d_%H%M%S)
EVAL_LOG_NAME="eval_${task_name}_${timestamp}"
max_length=8192
max_gen_toks=6144

mkdir -p benchmark_logs
model_path=/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantization_non_uniform/Llama-3.2-1B-Instruct-NVFP4-FP8-Dynamic
model_path="/data5/yliu7/HF_HOME/qwen_moe_skip_lm_head"
model_path=/data5/yliu7/HF_HOME/ByteDance-Seed/Seed-OSS-36B-Instruct
model_path=/data5/yliu7/HF_HOME/GLM-4.5-Air-w8afp8-llmc/GLM-4.5-Air-w8afp8
# model_path=/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-NVFPP_B16/
# model_path=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-20b-BF16-MXFP8/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-120b-BF16-unsloth-MXFP8
port=8088
HF_ALLOW_CODE_EVAL=1 \
lm_eval --model local-completions \
    --tasks $task_name \
    --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=1,max_length=${max_length},max_gen_toks=${max_gen_toks} \
    --batch_size ${batch_size}  \
    --gen_kwargs="max_gen_toks=${max_gen_toks}" \
    --confirm_run_unsafe_code \
    --log_samples \
    --limit 64 \
    --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \
    --output_path "benchmark_logs/$EVAL_LOG_NAME" \
    2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"




# curl -X POST http://127.0.0.1:8088/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/data5/yliu7/HF_HOME/Yi30/gpt-oss-120b-BF16-unsloth-MXFP8",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'


# batch_size=4
# lm_eval --model local-completions \
#     --tasks $task_name \
#     --model_args model=${model_path},base_url=http://127.0.0.1:8687/v1/completions,max_concurrent=1,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size ${batch_size}  \
#     --gen_kwargs="max_length=${max_length},max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --limit 16 \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"

# curl -X POST http://127.0.0.1:8687/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "seed_oss",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'



# SKIP INPUT QDQ  
# INFO:lm_eval.api.task:Building contexts for gsm8k on rank 0...
# 100%|██████████| 1319/1319 [00:03<00:00, 363.97it/s]
# INFO:lm_eval.evaluator:Running generate_until requests
# Requesting API: 100%|██████████| 1319/1319 [35:48<00:00,  1.63s/it]  
# INFO:lm_eval.loggers.evaluation_tracker:Saving results aggregated
# INFO:lm_eval.loggers.evaluation_tracker:Saving per-sample results for: gsm8k
# local-completions (model=/data5/yliu7/HF_HOME/qwen_moe_skip_lm_head,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=1,max_length=8192,max_gen_toks=2048), gen_kwargs: (max_length=8192,max_gen_toks=2048), limit: None, num_fewshot: None, batch_size: 256
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9212|±  |0.0074|
# |     |       |strict-match    |     5|exact_match|↑  |0.9219|±  |0.0074|


# dq + qdq input + skip quant lm-head
# local-completions (model=/data5/yliu7/HF_HOME/qwen_moe_skip_lm_head,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=1,max_length=8192,max_gen_toks=2048), gen_kwargs: (max_length=8192,max_gen_toks=2048), limit: None, num_fewshot: None, batch_size: 256
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9249|±  |0.0073|
# |     |       |strict-match    |     5|exact_match|↑  |0.9249|±  |0.0073|