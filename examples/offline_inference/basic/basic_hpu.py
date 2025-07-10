# SPDX-License-Identifier: Apache-2.0

model_path = "/models/Qwen3-32B"
model_path = "/models/DeepSeek-R1-Distill-Qwen-7B"
model_path= "/mnt/disk3/yiliu4/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic"
model_path  = "/software/users/yiliu4/HF_HOME/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic"

model_path  = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.2-1B-Instruct-NVFP4-llm-compressor"
model_path = "/mnt/disk3/yiliu4/Yi30/DeepSeek-V2-Lite-NVFP4-llm-compressor"

model_path = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.3-70B-Instruct-NVFP4-llmc"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/Yi30/Llama-3.2-1B-Instruct-MXFP8-llmc"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/Yi30/Llama-3.3-70B-Instruct-MXFP8-llmc"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/Yi30/DeepSeek-V2-Lite-MXFP8-llmc"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/DeepSeek-R1-bf16-MXFP8-4L-llmc/"
model_path = "/software/users/yiliu4/deepseek-ai/DeepSeek-V2-Lite-MXFP8-OFFLINE"
model_path = "/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-V2-Lite-MXFP8-RTN"
model_path = "/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-V2-Lite-MXFP8-autoround"

# model_path = "/software/users/yiliu4/deepseek-ai/DeepSeek-R1-MXFP8-OFFLINE"
# model_path = "/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP8-RTN-tiny"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.2-1B-Instruct-MXFP4-llmc"

model_path = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.3-70B-Instruct-MXFP4-llmc"
model_path = "/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP4-RTN"
model_name = model_path.split("/")[-1]
model_path = "/software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-V2-Lite-MXFP4-autoround"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/DeepSeek-V2-Lite-NVFP4-llm-compressor/"
# model_path  = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.2-1B-Instruct-NVFP4-llm-compressor"
import os

os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["HABANA_VISIBLE_DEVICES"] = "All"
os.environ["HABANA_VISIBLE_MODULES"] = "0,1,2,3,4,5,6,7"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_HPU_FORCE_CHANNEL_FP8"] = "0"
os.environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"] = "1"
os.environ["VLLM_DELAYED_SAMPLING"] = "true"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["VLLM_MLA_PERFORM_MATRIX_ABSORPTION"] = "0"


if "DeepSeek" in model_path:
    # os.environ["VLLM_DISABLE_INPUT_QDQ"] = "1"
    os.environ["VLLM_USE_STATIC_MOE_HPU"] = "1"

# os.environ["HABANA_LOGS"] = "./habana_logs"
# os.environ["LOG_LEVEL_ALL"] = "0"

# os.environ["GRAPH_VISUALIZATION"] = "1"
os.environ["PT_HPU_LAZY_MODE"] = "1"
# os.environ["VLLM_SKIP_WARMUP"] = "true"
# os.environ["VLLM_PROFILER_ENABLED"] = "true"
# os.environ["QUANT_CONFIG"] = f"quantization/{model_name}/maxabs_quant_g2.json"

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


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)


def main(args):
    # Create an LLM.
    # kv_cache_dtype="fp8_inc",
    tp_size = args.tp 
    kwargs = {}
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    if args.ep > 1:
        kwargs["enable_expert_parallel"] = True
        os.environ["VLLM_EP_SIZE"] = f"{args.ep}"
    #load-format dummy"
    if args.warmup:
        os.environ["VLLM_SKIP_WARMUP"] = "false"
        kwargs["load_format"] = "dummy"
    else:
        os.environ["VLLM_SKIP_WARMUP"] = "true"
        # kwargs["load_format"] = "none"
        
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ] * args.batch_size
    # Create a sampling params object.
    max_model_len = 2048
    model_path = args.model_path
    llm = LLM(
        model=model_path,
        #   quantization="inc",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.65,
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
        token_ids = output.outputs[0].token_ids
        cumulative_logprob = output.outputs[0].cumulative_logprob
        logprobs = output.outputs[0].logprobs
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print(f"Token IDs: {token_ids}")
        if cumulative_logprob is not None:
            print(f"Cumulative Logprob: {cumulative_logprob}")
        if logprobs is not None:
            print(f"Logprobs:  {logprobs}")
        print("-" * 60)

    import time
    start_time = time.time()
    if args.profile:
        print("Starting profiling for second inference...")
        llm.start_profile()
    
    outputs = llm.generate(prompts, sampling_params)
    if args.profile:
        print("Stopping profiling for second inference...")
        llm.stop_profile()
    end_time = time.time()
    print(f"Time taken for second inference: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run basic HPU inference.")
    parser.add_argument("--model_path", type=str, default=model_path,
                        help="Path to the model directory.")
    # tp size
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    # ep size
    parser.add_argument("--ep", type=int, default=1, help="Pipeline parallel size.")
    # run profile
    parser.add_argument("--profile", action="store_true", help="Run with profiling enabled.")
    # warmup
    parser.add_argument("--warmup", action="store_true", help="Run with warmup enabled.")
    # batch size
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for inference.")
    args = parser.parse_args()
    main(args)


# VLLM_MXFP4_PREUNPACK_WEIGHTS=1  VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --model_path  /software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP4-RTN  --tp 8 --ep 8
# VLLM_MXFP4_PREUNPACK_WEIGHTS=1  VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --tp 2 --ep 2
# VLLM_MXFP4_PREUNPACK_WEIGHTS=1  VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --model_path  /software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP4-RTN  --tp 8 --ep 8
# VLLM_MXFP4_PREUNPACK_WEIGHTS=1  VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --model_path  /software/users/yiliu4/HF_HOME/weiweiz1/DeepSeek-R1-MXFP4-RTN  --tp 8 --ep 8 --warmup
# VLLM_MXFP4_PREUNPACK_WEIGHTS=1  VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --tp 2 --ep 2