# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
import os

os.environ["VLLM_LOG_LEVEL"] = "INFO"  # Set log level to INFO for more detailed output
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# VLLM_USE_TRTLLM_ATTENTION=1   VLLM_USE_TRTLLM_DECODE_ATTENTION=1   VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1   VLLM_USE_FLASHINFER_MXFP4_MOE=1  python examples/offline_inference/basic/basic.py 
def main():
    # Create an LLM.
    model = "/home/yiliu7/models/openai/gpt-oss-20b"
    model = "/home/yiliu7/models/openai/gpt-oss-120b"
    llm = LLM(
        enforce_eager=True,  # Enable eager mode for faster inference
        model=model,
        gpu_memory_utilization=0.65,  # Set GPU memory utilization to 80%
        dtype="bfloat16",
    
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
