uv env: /dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python
model: /dev/shm/.tmp_yi/Qwen/Qwen3-8B
auto-round: /dev/shm/.tmp_yi/workspace/auto-round
target:
    - quantize Qwen/Qwen3-8B, mlp 2 bits, attention 4 bits, iters=0,
    - load the quantized model in vllm, for 2 bits, using humming kernel
gpu: 2,3
evluation: lm-eval gsm8k
requirements: save as auto-round format and load it in vllm through quantization/inc
