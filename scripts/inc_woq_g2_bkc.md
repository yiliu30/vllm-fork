```bash
# Install vllm
git clone https://github.com/yiliu30/vllm-fork.git
git checkout inc-r1-g2
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;


# Install INC
pip install git+https://github.com/intel/neural-compressor.git@r1-woq

# Get calibration file
huggingface-cli download Yi30/inc-woq-default-pile-one-cache-408  --local-dir ./scripts/nc_workspace_measure_kvache


# Benchmark
cd ./scripts
# Update model_path to "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
bash single_8k_len.sh
```

```bash

curl -X POST http://127.0.0.1:8688/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/mnt/disk2/hf_models/DeepSeek-R1-G2/",
           "prompt": "Hi, the result of 9 + 9.11",
           "max_tokens": 16,
           "temperature": 0.7,
           "top_p": 1.0
         }'
         
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=/mnt/disk2/hf_models/DeepSeek-R1-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=16 \
    --batch_size 16 \
    --log_samples \
    --output_path ./lm_eval_output_gsm8k_new_disable_VLLM_MLA_PERFORM_MATRIX_ABSORPTION_bs16
    
 HF_ALLOW_CODE_EVAL=1 lm_eval \
    --model local-completions \
    --tasks humaneval \
    --model_args model=/mnt/disk2/hf_models/DeepSeek-R1-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=4 \
    --batch_size 4 --confirm_run_unsafe_code \
    --log_samples \
    --output_path ./lm_eval_output_humaneval_full_disable_VLLM_MLA_PERFORM_MATRIX_ABSORPTION_bs4


 HF_ALLOW_CODE_EVAL=1 lm_eval \
    --model local-completions \
    --tasks humaneval,gsm8k \
    --model_args model=/mnt/disk2/hf_models/DeepSeek-R1-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=1 \
    --batch_size 8 --confirm_run_unsafe_code \
    --log_samples \
    --output_path ./lm_eval_output_humaneval_and_gsm8k_full_bs8_fp8kv.pts

 HF_ALLOW_CODE_EVAL=1 lm_eval \
    --model local-completions \
    --tasks humaneval,gsm8k \
    --model_args model=/mnt/disk2/hf_models/DeepSeek-R1-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=8 \
    --batch_size 8 --confirm_run_unsafe_code \
    --log_samples \
    --output_path ./lm_eval_output_humaneval_and_gsm8k_full_bs8_max_concurrent8_fp8kv.pts
```