"""Test the AutoRound model using V0 engine and compare with unquantized baseline.

Goals:
1. Test the AutoRound model with VLLM_USE_V1=0 (V0 engine) to see if the
   issue is V1-specific.
2. Check if embed_tokens weights are correctly handled (model has bf16 embeddings
   but vLLM may load with fp16 params_dtype).
3. Compare with the unquantized Qwen2-0.5B-Instruct model as a baseline.
"""

import os
import sys
import torch

# Force V0 engine
os.environ["VLLM_USE_V1"] = "0"
# Enable debug logging for INC
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

sys.path.insert(0, "/home/yiliu7/workspace/vllm")


def check_model_weights():
    """Inspect raw checkpoint weights to understand dtype/value ranges."""
    from safetensors.torch import load_file

    print("=" * 70)
    print("STEP 1: Inspect raw checkpoint weights")
    print("=" * 70)

    model_path = (
        "/home/yiliu7/.cache/huggingface/hub/"
        "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
        "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
    )
    tensors = load_file(model_path)

    # Check embedding layer
    if "model.embed_tokens.weight" in tensors:
        emb = tensors["model.embed_tokens.weight"]
        print(f"embed_tokens.weight: shape={emb.shape}, dtype={emb.dtype}")
        print(f"  range: [{emb.min().item():.4f}, {emb.max().item():.4f}]")
        print(f"  mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
        print(f"  has_nan={emb.isnan().any().item()}, has_inf={emb.isinf().any().item()}")
    else:
        print("embed_tokens.weight NOT found in checkpoint!")

    # Check lm_head (might be tied)
    if "lm_head.weight" in tensors:
        lmh = tensors["lm_head.weight"]
        print(f"\nlm_head.weight: shape={lmh.shape}, dtype={lmh.dtype}")
    else:
        print("\nlm_head.weight NOT in checkpoint (likely tied to embed_tokens)")

    # Check first layer's quantized weights
    layer0_prefix = "model.layers.0.self_attn.q_proj"
    qw = tensors[f"{layer0_prefix}.qweight"]
    sc = tensors[f"{layer0_prefix}.scales"]
    print(f"\nLayer 0 q_proj:")
    print(f"  qweight: {qw.shape} {qw.dtype}")
    print(f"  scales:  {sc.shape} {sc.dtype}, range=[{sc.min().item():.6f}, {sc.max().item():.6f}]")

    # Check layer norm weights
    ln = tensors.get("model.layers.0.input_layernorm.weight")
    if ln is not None:
        print(f"\nLayer 0 input_layernorm.weight: {ln.shape} {ln.dtype}")

    # Print all tensor names and dtypes
    print("\nAll tensor dtypes summary:")
    dtype_counts = {}
    for name, t in tensors.items():
        key = f"{t.dtype}"
        if key not in dtype_counts:
            dtype_counts[key] = []
        dtype_counts[key].append(name)
    for dtype, names in dtype_counts.items():
        print(f"  {dtype}: {len(names)} tensors")
        for n in names[:3]:
            print(f"    {n}: {tensors[n].shape}")
        if len(names) > 3:
            print(f"    ... and {len(names)-3} more")

    return tensors


def test_v0_autoround():
    """Test the AutoRound model with V0 engine."""
    print("\n" + "=" * 70)
    print("STEP 2: Test AutoRound model with V0 engine")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model="Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",
            enforce_eager=True,
            max_model_len=256,
            gpu_memory_utilization=0.5,
        )

        prompts = [
            "The capital of France is",
            "1 + 1 = ",
            "Hello, my name is",
        ]

        params = SamplingParams(max_tokens=30, temperature=0)
        outputs = llm.generate(prompts, params)

        print("\nAutoRound model outputs (V0 engine):")
        for i, output in enumerate(outputs):
            print(f"  Prompt: {prompts[i]!r}")
            print(f"  Output: {output.outputs[0].text!r}")
            print()

        # Clean up
        del llm
        torch.xpu.empty_cache() if hasattr(torch, 'xpu') else None

        return True
    except Exception as e:
        print(f"ERROR with V0 engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unquantized_baseline():
    """Test unquantized Qwen2-0.5B-Instruct as a baseline."""
    print("\n" + "=" * 70)
    print("STEP 3: Test unquantized Qwen2-0.5B-Instruct as baseline")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model="Qwen/Qwen2-0.5B-Instruct",
            enforce_eager=True,
            max_model_len=256,
            gpu_memory_utilization=0.5,
        )

        prompts = [
            "The capital of France is",
            "1 + 1 = ",
            "Hello, my name is",
        ]

        params = SamplingParams(max_tokens=30, temperature=0)
        outputs = llm.generate(prompts, params)

        print("\nUnquantized model outputs (V0 engine):")
        for i, output in enumerate(outputs):
            print(f"  Prompt: {prompts[i]!r}")
            print(f"  Output: {output.outputs[0].text!r}")
            print()

        del llm
        torch.xpu.empty_cache() if hasattr(torch, 'xpu') else None

        return True
    except Exception as e:
        print(f"ERROR with unquantized model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing AutoRound model path diagnostics")
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1', 'not set')}")
    print(f"Python: {sys.executable}")
    print()

    # Step 1: Inspect raw weights
    tensors = check_model_weights()

    # Step 2: Test with V0 engine
    print("\n\n>>> Testing AutoRound model with V0 engine <<<\n")
    v0_ok = test_v0_autoround()

    # Step 3: Test unquantized baseline
    print("\n\n>>> Testing unquantized baseline <<<\n")
    baseline_ok = test_unquantized_baseline()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"V0 AutoRound: {'OK' if v0_ok else 'FAILED'}")
    print(f"Unquantized baseline: {'OK' if baseline_ok else 'FAILED'}")
