Goal:
Using uv create a new envs and setup the vllm (w/ vllm-xpu-kernel) in this node.

vllm: /home/yiliu7/workspace/vllm
vllm-xpu-kernels: /home/yiliu7/workspace/vllm-xpu-kernels
test:   python3 examples/offline_inference/basic/generate.py --model superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4  --block-size 64 --enforce-eager

Requires:
    - We'd better not use docker, but if need please comfirm before process
Note:
If any proxy issue: try use http://proxy.ims.intel.com:911

Ref:
- https://github.com/vllm-project/vllm/pull/33973/
- https://github.com/vllm-project/vllm/issues/33214