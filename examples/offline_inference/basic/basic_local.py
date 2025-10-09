#  VLLM_USE_V1=0  VLLM_USE_NVFP4_CT_EMULATIONS=1  p basic_local.py 
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_USE_V1"] = "0"  # Use v2 API
# os.environ["VLLM_USE_NVFP4_CT_EMULATIONS"] = "1"  # Use v2 API
# os.environ["VLLM_USE_MXFP4_CT_EMULATIONS"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Use v2 API

seed = 0
import random
random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=20)


model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantization_w4a4_fp4/Llama-3.2-1B-Instruct-NVFP4"
model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-FP8"
model="/data5/yliu7/HF_HOME/DeepSeek-V2-Lite-MXFP8"
model="/data5/zww/fast_config/Llama-3.2-1B-Instruct-w8g32"
model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-MXFP4"
# USE_CT_UNPACK=1  VLLM_DISABLE_INPUT_QDQ=1  VLLM_USE_MXFP4_CT_EMULATIONS=1  p basic_local.py 
model="/data5/yliu7/HF_HOME/DeepSeek-V2-Lite-MXFP4"
# model="/data5/zww/fast_config/DeepSeek-V2-Lite-w8g32/"
# model="/data5/zww/fast_config/rtn_lite/DeepSeek-V2-Lite-w8g32/"
# model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-R1-bf16-NVFP4"
model="/data5/wzy/vLLM-mxfp4/mnt/Meta-Llama-3.1-8B-Instruct-MXFP4"
model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
model="/data0/saved_model/zai-org/GLM-4.5-Air/1/GLM-4.5-Air-w8afp8"
model="/data6/qwen_fp8/"
# <｜begin▁of▁sentence｜>Hello my name is
# I am a 28 year old male and I am currently living in the United States.
if "deepseek" in model.lower():
    os.environ["VLLM_USE_STATIC_MOE_HPU"] = "1"
# model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
# model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-MXFP8"
def main(args):
    # Create an LLM.
    tp_size = args.tp 
    kwargs = {}
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    if args.ep > 1:
        kwargs["enable_expert_parallel"] = True
        os.environ["VLLM_EP_SIZE"] = f"{args.ep}"
    llm = LLM(
        # model="facebook/opt-125m"
        # model="/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/",
        model=model,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.85,
        **kwargs,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run basic inference.")
    parser.add_argument("--model_path", type=str, default=model,
                        help="Path to the model directory.")
    # tp size
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    # ep size
    parser.add_argument("--ep", type=int, default=1, help="Pipeline parallel size.")
    args = parser.parse_args()
    main(args)


# p basic_local.py
# # tp 2 only
# p basic_local.py --tp 2
# # tp 2 ep 2
# p basic_local.py --tp 2 --ep 2