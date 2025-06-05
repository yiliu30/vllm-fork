/software/users/yiliu4/local_dockers/vllm-pd.tar

model_path="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/"
model_path="/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct"

curl -s http://localhost:8088/v1/completions -H "Content-Type: application/json" -d '{
    "model": "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct",
    "prompt": "Explain the significance of the moon landing in 1969.",
    "max_tokens": 100
}'

export PATH=$PATH:/home/yiliu4/.local/bin

mooncake_master --enable_gc true --port 50001

lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/,base_url=http://127.0.0.1:8000/v1/completions,max_concurrent=1 \
    --batch_size 16  \
    --log_samples \
    --output_path ./lm_eval_output_gsm8k_bs1__



local-completions (model=/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/,base_url=http://127.0.0.1:8000/v1/completions,max_concurrent=1), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5049|±  |0.0138|
|     |       |strict-match    |     5|exact_match|↑  |0.5027|±  |0.0138|



- rdma
local-completions (model=/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/,base_url=http://127.0.0.1:8000/v1/completions,max_concurrent=1), gen_kwargs: (None), limit: 64.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4219|±  |0.0622|
|     |       |strict-match    |     5|exact_match|↑  |0.4219|±  |0.0622|