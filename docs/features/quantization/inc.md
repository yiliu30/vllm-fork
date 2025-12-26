# Intel Neural Compressor


[Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) is an open-source toolkit for model compression, with a strong focus on quantization. It supports both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) across a variety of tasks and architectures, including:
  - Large Language Models (LLMs)
  - Vision-Language Models (VLMs)
  - Diffusion models

Under the INC umbrella, [AutoRound](https://github.com/intel/auto-round) is Intel’s advanced quantization algorithm designed specifically for transformer and large language models. It is designed to produce highly efficient **INT2, INT3, INT4, INT8, MXFP4, NVFP4, and GGUF** quantized models, striking a balance between accuracy and inference performance. For a deeper introduction to AutoRound, see the [AutoRound step-by-step guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md).


INC provides multiple ways to quantize models and deploy them on Intel GPU, CPU, and Gaudi. For **Intel CPUs and GPUs**, we currently recommend using **AutoRound directly**.

!!! note
    Intel Gaudi quantization support (such as `W8A8`, `W4A16`) has been migrated to [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi). For details, see the [vLLM-Gaudi quantization documentation](https://docs.vllm.ai/projects/gaudi/en/latest/configuration/quantization/quantization.html).


Key Features:

✅ **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** are supported

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for quick quantization with slight accuracy loss

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for **10+ backends**


## Installation

```bash
uv pip install auto-round
```

## Quantizing a model

For VLMs, please change to `auto-round-mllm` in CLI usage and `AutoRoundMLLM` in API usage.

### CLI usage

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --format "gguf:q4_k_m" \
    --output_dir ./tmp_autoround
```

### API usage

??? code

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from auto_round import AutoRound

    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bits, group_size, sym = 4, 128, True
    autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

    # the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
    # autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

    # 2-3X speedup, slight accuracy drop at W4G128
    # autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

    output_dir = "./tmp_autoround"
    # format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
    autoround.quantize_and_save(output_dir, format="auto_round")
    ```

## Running a quantized model with vLLM

Here is some example code to run auto-round format in vLLM:

??? code

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
    model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
    llm = LLM(model=model_name)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

## Acknowledgement

Special thanks to open-source low precision libraries such as AutoGPTQ, AutoAWQ, GPTQModel, Triton, Marlin, and
ExLLaMAV2 for providing low-precision CUDA kernels, which are leveraged in AutoRound.
