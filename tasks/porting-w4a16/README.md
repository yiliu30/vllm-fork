Gobal:
Add w4a16 xpu support for inc.py

Context:
The INC w4a16 xpu support was removed due the deprecated of IPEX, please check the apply_ipex_quant_layer in inc.py for more details.
In current version, we need call the kernel from the vllm-xpu-kernels, as the wNa16 did here.
https://github.com/vllm-project/vllm/pull/33973


Source: 
inc.py: /home/yiliu7/workspace/vllm/vllm/model_executor/layers/quantization/inc.py

e2e example:
python3 examples/basic/offline_inference/generate.py   --model Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound  --block-size 64 --enforce-eager --max-model-len 4096

python env: /home/yiliu7/workspace/vllm/.venv/bin/python

