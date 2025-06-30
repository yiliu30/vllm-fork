# SPDX-License-Identifier: Apache-2.0

model_path = "/models/Qwen3-32B"
model_path = "/models/DeepSeek-R1-Distill-Qwen-7B"
model_path= "/mnt/disk3/yiliu4/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic"
model_path  = "/software/users/yiliu4/HF_HOME/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic"
model_path  = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.2-1B-Instruct-NVFP4-llm-compressor"
model_path  = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.2-1B-Instruct-NVFP4-llm-compressor"
model_path = "/mnt/disk3/yiliu4/Yi30/DeepSeek-V2-Lite-NVFP4-llm-compressor"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/DeepSeek-V2-Lite-NVFP4-llm-compressor/"
model_path = "/software/users/yiliu4/HF_HOME/Yi30/Llama-3.3-70B-Instruct-NVFP4-llmc"
model_name = model_path.split("/")[-1]

import os

os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["HABANA_VISIBLE_DEVICES"] = "All"
os.environ["HABANA_VISIBLE_MODULES"] = "0,1,2,3,4,5,6,7"

if "DeepSeek" in model_path:
    os.environ["VLLM_DISABLE_INPUT_QDQ"] = "1"
    os.environ["VLLM_USE_STATIC_MOE_HPU"] = "1"

# os.environ["HABANA_LOGS"] = "./habana_logs"
# os.environ["LOG_LEVEL_ALL"] = "0"

# os.environ["GRAPH_VISUALIZATION"] = "1"
os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["VLLM_SKIP_WARMUP"] = "true"
# os.environ["VLLM_PROFILER_ENABLED"] = "true"
# os.environ["QUANT_CONFIG"] = f"quantization/{model_name}/maxabs_quant_g2.json"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


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

    llm = LLM(
        model=model_path,
        #   quantization="inc",
        max_num_batched_tokens=1024,
        max_model_len=1024,
        # enforce_eager=True,
        trust_remote_code=True,
    dtype="bfloat16",
        tensor_parallel_size=tp_size,
        **kwargs
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
    import time
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
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
    args = parser.parse_args()
    main(args)


# INFO 06-26 17:17:55 [llm_engine.py:439] init engine (profile, create kv cache, warmup model) took 53.52 seconds
# Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 197.22it/s]
# Processed prompts:   0%|                                                       | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54973) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54971) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54975) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54969) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54972) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54970) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54974) WARNING 06-26 17:17:55 [hpu_model_runner.py:1230] Configuration: ('prompt', 4, 128, 0) was not warmed-up!
# (VllmWorkerProcess pid=54974) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54969) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54971) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54973) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54970) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54975) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# (VllmWorkerProcess pid=54972) WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# WARNING 06-26 17:18:01 [hpu_model_runner.py:1230] Configuration: ('decode', 4, 1, 128) was not warmed-up!
# Processed prompts: 100%|██████████████████████████████████████████████| 4/4 [09:06<00:00, 136.71s/it, est. speed input: 0.05 toks/s, output: 0.12 toks/s]

# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello, my name is'
# Output:    ' Tony, I am a Software Engineer at Google. I have been working on the'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' the head of state and head of government of the United States. The president is'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' always a good idea, but there are many other destinations that are worth visiting as'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    " here, and it's more powerful than ever. But with great power comes great"
# ------------------------------------------------------------
# Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 1055.37it/s]
# Processed prompts: 100%|██████████████████████████████████████████████| 4/4 [09:35<00:00, 143.78s/it, est. speed input: 0.05 toks/s, output: 0.11 toks/s]
# Time taken for second inference: 575.12 seconds
# INFO 06-26 17:36:37 [multiproc_worker_utils.py:139] Terminating local vLLM worker processes
# (VllmWorkerProcess pid=54969) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54973) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54971) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54970) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54972) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54975) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# (VllmWorkerProcess pid=54974) INFO 06-26 17:36:37 [multiproc_worker_utils.py:261] Worker exiting
# <p22> yiliu4@yiliu4-63gd-g3-l-vm:basic$ /usr/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
#   warnings.warn('resource_tracker: There appear to be %d '

# <p22> yiliu4@yiliu4-63gd-g3-l-vm:basic$ p basic_hpu.py --tp 8