# SPDX-License-Identifier: Apache-2.0

model_path = "/models/Qwen3-32B"
model_path = "/models/DeepSeek-R1-Distill-Qwen-7B"
model_path= "/mnt/disk3/yiliu4/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic"
model_name = model_path.split("/")[-1]

import os
# os.environ["HABANA_LOGS"] = "./habana_logs"
# os.environ["LOG_LEVEL_ALL"] = "0"

# os.environ["GRAPH_VISUALIZATION"] = "1"
os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["VLLM_PROFILER_ENABLED"] = "true"
# os.environ["QUANT_CONFIG"] = f"quantization/{model_name}/maxabs_quant_g2.json"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    # kv_cache_dtype="fp8_inc", 
    llm = LLM(model=model_path, 
            #   quantization="inc", 
              max_num_batched_tokens=1024, max_model_len=1024,
              enforce_eager=True,
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
    main()
