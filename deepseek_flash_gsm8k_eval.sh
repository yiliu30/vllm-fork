# w/ xin + 0.75
#     "model_name": "DeepSeek-V4-Flash",
    # "score": 0.9659,

# 

#!/bin/bash
# Run GSM8K evaluation against DeepSeek-V4-Flash served by vLLM
source /home/yiliu7/workspace/venvs/vllm/bin/activate
MODEL=/storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash
PORT=8081

# Wait for server to be ready
echo "Checking server health..."
until curl -s "http://127.0.0.1:${PORT}/v1/models" | python -m json.tool > /dev/null 2>&1; do
  echo "Waiting for server on port ${PORT}..."
  sleep 5
done
echo "Server is ready."

evalscope eval \
  --model "$MODEL" \
  --eval-type openai_api \
  --api-key EMPTY \
  --datasets mmlu_pro \
  --eval-batch-size 128 \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  2>&1 | tee deepseek_flash_gsm8k_eval.log
