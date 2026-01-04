# Intel Quantization Support

[AutoRound](https://github.com/intel/auto-round) is Intel’s advanced quantization algorithm designed for transformer and large language models. It produces highly efficient **INT2, INT3, INT4, INT8, MXFP8, MXFP4, NVFP4**, and **GGUF** quantized models, balancing accuracy and inference performance. AutoRound is also part of the [Intel Neural Compressor](https://github.com/intel/neural-compressor). For a deeper introduction, see the [AutoRound step-by-step guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md).



## Key Features

✅ **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** are supported

✅ **10+ vision-language models (VLMs)** are supported

✅ **Per-layer mixed-bit quantization** for fine-grained control

✅ **RTN (Round-To-Nearest) mode** for quick quantization with slight accuracy loss

✅ **Multiple quantization recipes**: best, base, and light

✅ Advanced utilities such as immediate packing and support for **10+ backends**

On Intel platforms, AutoRound recipes are being enabled progressively by format and hardware; currently, the `wNa16` recipe is supported on Intel GPUs and Intel CPUs (weight-only, N-bit weights with 16-bit activations).

## Installation

```bash
uv pip install auto-round
```

## Quantizing a model

For VLMs, please change to `auto-round-mllm` in CLI usage and `AutoRoundMLLM` in API usage.

### Quantize with CLI

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```


### Quantize with Python API

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

## Deploying AutoRound Quantized Models in vLLM

```python
from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
    model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
    llm = LLM(model=model_name, enforce_eager=True, max_model_len=4192)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
```

!!! note
     To deploy `wNa16` quantized models on Intel GPU/CPU, please add `enforce_eager=True` to the LLM initialization for now.