"""Minimal debug: Load one layer from checkpoint, repack, run XPU kernel,
compare with CPU dequant reference. Then also test full vLLM forward for
a single token to check where divergence starts."""

import torch
from safetensors.torch import load_file

# Register the XPU ops
import vllm._xpu_ops  # noqa: F401

MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)

WEIGHT_BITS = 4
PACK_FACTOR = 8
GROUP_SIZE = 128
DEVICE = "xpu"


def awq_dequant_cpu(qweight, scales, qzeros, weight_bits=4, group_size=128):
    """CPU reference dequant for AWQ format weights."""
    pack_factor = 32 // weight_bits
    mask = (1 << weight_bits) - 1
    in_size, out_packed = qweight.shape
    out_size = out_packed * pack_factor

    # Unpack qweight: [in, out_packed] -> [in, out]
    unpacked_w = torch.zeros((in_size, out_size), dtype=torch.float32)
    for i in range(pack_factor):
        unpacked_w[:, i::pack_factor] = ((qweight >> (weight_bits * i)) & mask).float()

    # Unpack qzeros: [ngroups, out_packed] -> [ngroups, out]
    ngroups = qzeros.shape[0]
    unpacked_z = torch.zeros((ngroups, out_size), dtype=torch.float32)
    for i in range(pack_factor):
        unpacked_z[:, i::pack_factor] = ((qzeros >> (weight_bits * i)) & mask).float()

    # Dequantize: w_float = (w_int - zero) * scale
    scales_f = scales.float()
    dequant = torch.zeros((in_size, out_size), dtype=torch.float32)
    for g in range(ngroups):
        row_start = g * group_size
        row_end = min(row_start + group_size, in_size)
        w_slice = unpacked_w[row_start:row_end, :]
        z_slice = unpacked_z[g:g+1, :]
        s_slice = scales_f[g:g+1, :]
        dequant[row_start:row_end, :] = (w_slice - z_slice) * s_slice

    return dequant


def repack_awq_to_ct(qweight_awq, weight_bits=4):
    """Repack AWQ [in, out_packed] -> CT [out, in_packed]."""
    pack_factor = 32 // weight_bits
    mask = (1 << weight_bits) - 1
    in_size, out_packed = qweight_awq.shape
    out_size = out_packed * pack_factor

    # Unpack: [in, out_packed] -> [in, out]
    unpacked = torch.zeros((in_size, out_size), dtype=torch.int32,
                           device=qweight_awq.device)
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (qweight_awq >> (weight_bits * i)) & mask

    # Transpose: [in, out] -> [out, in]
    unpacked = unpacked.t().contiguous()

    # Repack along input dim: [out, in] -> [out, in_packed]
    in_packed = in_size // pack_factor
    repacked = torch.zeros((out_size, in_packed), dtype=torch.int32,
                           device=qweight_awq.device)
    for i in range(pack_factor):
        repacked |= (unpacked[:, i::pack_factor] & mask) << (weight_bits * i)

    return repacked


def test_single_layer():
    """Test a single layer: load from checkpoint, repack, run kernel, compare."""
    print("=" * 70)
    print("TEST 1: Single layer (q_proj) - checkpoint vs kernel")
    print("=" * 70)

    tensors = load_file(MODEL_PATH)

    layer_name = "model.layers.0.self_attn.q_proj"
    qw = tensors[f"{layer_name}.qweight"]  # [896, 112] int32
    sc = tensors[f"{layer_name}.scales"]    # [7, 896] fp16
    qz = tensors[f"{layer_name}.qzeros"]   # [7, 112] int32
    bias = tensors[f"{layer_name}.bias"]    # [896] fp16

    print(f"qweight: {qw.shape} {qw.dtype}")
    print(f"scales:  {sc.shape} {sc.dtype}")
    print(f"qzeros:  {qz.shape} {qz.dtype}")
    print(f"bias:    {bias.shape} {bias.dtype}")

    # CPU reference
    dequant_w = awq_dequant_cpu(qw, sc, qz)  # [896, 896] float32

    torch.manual_seed(42)
    test_x = torch.randn(1, 896, dtype=torch.float16)
    ref_out = (test_x.float() @ dequant_w).half() + bias  # [1, 896]
    print(f"\nCPU reference output: shape={ref_out.shape}, "
          f"sum={ref_out.sum().item():.4f}, "
          f"first5={ref_out[0,:5].tolist()}")

    # XPU kernel
    qw_xpu = qw.to(DEVICE)
    sc_xpu = sc.to(DEVICE)
    qz_scalar = torch.tensor([8], dtype=torch.int8, device=DEVICE)
    test_x_xpu = test_x.to(DEVICE)

    # Repack AWQ -> CT
    qw_ct = repack_awq_to_ct(qw_xpu)
    print(f"\nRepacked: {qw_ct.shape} (CT layout)")

    # Run kernel: expects (input, q_weight, bias, scales, qzeros, group_size, g_idx)
    # q_weight should be passed as .t() to get [in_packed, out]
    xpu_out = torch.ops._xpu_C.int4_gemm_w4a16(
        test_x_xpu, qw_ct.t(), None, sc_xpu, qz_scalar, GROUP_SIZE, None
    )
    xpu_out = xpu_out + bias.to(DEVICE)
    xpu_out_cpu = xpu_out.cpu()
    print(f"XPU kernel output:   shape={xpu_out_cpu.shape}, "
          f"sum={xpu_out_cpu.sum().item():.4f}, "
          f"first5={xpu_out_cpu[0,:5].tolist()}")

    cos = torch.nn.functional.cosine_similarity(
        ref_out.float().flatten(), xpu_out_cpu.float().flatten(), dim=0
    ).item()
    print(f"Cosine similarity: {cos:.6f}")

    return cos


def test_fused_qkv():
    """Test fused QKV layer: simulate what vLLM does during weight loading."""
    print("\n" + "=" * 70)
    print("TEST 2: Fused QKV layer - simulate vLLM weight loading + repack")
    print("=" * 70)

    tensors = load_file(MODEL_PATH)
    layer = "model.layers.0.self_attn"

    # Load individual Q, K, V weights
    q_qw = tensors[f"{layer}.q_proj.qweight"]  # [896, 112]
    k_qw = tensors[f"{layer}.k_proj.qweight"]  # [896, 16]
    v_qw = tensors[f"{layer}.v_proj.qweight"]  # [896, 16]

    q_sc = tensors[f"{layer}.q_proj.scales"]    # [7, 896]
    k_sc = tensors[f"{layer}.k_proj.scales"]    # [7, 128]
    v_sc = tensors[f"{layer}.v_proj.scales"]    # [7, 128]

    q_qz = tensors[f"{layer}.q_proj.qzeros"]   # [7, 112]
    k_qz = tensors[f"{layer}.k_proj.qzeros"]   # [7, 16]
    v_qz = tensors[f"{layer}.v_proj.qzeros"]   # [7, 16]

    q_bias = tensors[f"{layer}.q_proj.bias"]
    k_bias = tensors[f"{layer}.k_proj.bias"]
    v_bias = tensors[f"{layer}.v_proj.bias"]

    print(f"Q qweight: {q_qw.shape}, K qweight: {k_qw.shape}, V qweight: {v_qw.shape}")

    # Simulate vLLM fused weight concatenation (along output dim=1 for packed)
    fused_qw = torch.cat([q_qw, k_qw, v_qw], dim=1)  # [896, 144]
    fused_sc = torch.cat([q_sc, k_sc, v_sc], dim=1)    # [7, 1152]
    fused_qz = torch.cat([q_qz, k_qz, v_qz], dim=1)   # [7, 144]
    fused_bias = torch.cat([q_bias, k_bias, v_bias])     # [1152]

    print(f"Fused qweight: {fused_qw.shape}")
    print(f"Fused scales:  {fused_sc.shape}")

    # CPU reference
    dequant_w = awq_dequant_cpu(fused_qw, fused_sc, fused_qz)
    torch.manual_seed(42)
    test_x = torch.randn(1, 896, dtype=torch.float16)
    ref_out = (test_x.float() @ dequant_w).half() + fused_bias
    print(f"\nCPU reference: shape={ref_out.shape}, sum={ref_out.sum().item():.4f}")

    # XPU kernel with repacking
    fused_qw_xpu = fused_qw.to(DEVICE)
    fused_sc_xpu = fused_sc.to(DEVICE)
    qz_scalar = torch.tensor([8], dtype=torch.int8, device=DEVICE)
    test_x_xpu = test_x.to(DEVICE)

    qw_ct = repack_awq_to_ct(fused_qw_xpu)
    xpu_out = torch.ops._xpu_C.int4_gemm_w4a16(
        test_x_xpu, qw_ct.t(), None, fused_sc_xpu, qz_scalar, GROUP_SIZE, None
    )
    xpu_out = xpu_out + fused_bias.to(DEVICE)
    xpu_out_cpu = xpu_out.cpu()
    print(f"XPU kernel:    shape={xpu_out_cpu.shape}, sum={xpu_out_cpu.sum().item():.4f}")

    cos = torch.nn.functional.cosine_similarity(
        ref_out.float().flatten(), xpu_out_cpu.float().flatten(), dim=0
    ).item()
    print(f"Cosine similarity: {cos:.6f}")

    return cos


def test_vllm_loaded_weights():
    """Hook into vLLM to capture weights after loading but before/after repack."""
    print("\n" + "=" * 70)
    print("TEST 3: vLLM weight loading - capture and verify")
    print("=" * 70)

    import sys
    sys.path.insert(0, "/home/yiliu7/workspace/vllm")

    # Store captured weights
    captured = {}

    # Monkey-patch process_weights_after_loading to capture pre-repack state
    from vllm.model_executor.layers.quantization.inc import INCXPULinearMethod
    original_process = INCXPULinearMethod.process_weights_after_loading

    def patched_process(self, layer):
        # Capture pre-repack state
        prefix = getattr(layer, '_prefix', 'unknown')
        if 'layers.0.self_attn.qkv_proj' in str(id(layer)) or True:
            # Just capture first few layers
            key = f"pre_{id(layer)}"
            captured[key] = {
                'qweight_pre': layer.qweight.data.clone().cpu(),
                'scales': layer.scales.data.clone().cpu(),
                'qzeros': layer.qzeros.data.clone().cpu(),
            }

        # Call original
        original_process(self, layer)

        # Capture post-repack state
        captured[f"post_{id(layer)}"] = {
            'qweight_post': layer.qweight.data.clone().cpu(),
            'scales_post': layer.scales.data.clone().cpu(),
        }

    INCXPULinearMethod.process_weights_after_loading = patched_process

    # Now load the model via vLLM
    from vllm import LLM

    llm = LLM(
        model="Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",
        block_size=64,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=0.5,
    )

    # Check how many layers were captured
    pre_keys = [k for k in captured if k.startswith("pre_")]
    post_keys = [k for k in captured if k.startswith("post_")]
    print(f"Captured {len(pre_keys)} pre-repack, {len(post_keys)} post-repack states")

    # Now compare the first captured layer's pre-repack qweight
    # with the raw checkpoint
    if pre_keys:
        tensors = load_file(MODEL_PATH)
        first_pre = captured[pre_keys[0]]
        vllm_qw = first_pre['qweight_pre']
        vllm_sc = first_pre['scales']

        # We don't know which layer this is, but let's check shapes
        print(f"\nFirst captured layer:")
        print(f"  qweight shape: {vllm_qw.shape}")
        print(f"  scales shape:  {vllm_sc.shape}")

        # If it's a fused QKV, shape would be [896, 144]
        # If it's a single layer, shape would be [896, 112] etc.
        if vllm_qw.shape == torch.Size([896, 144]):
            print("  Looks like fused QKV (896x144 = [896, (896+128+128)/8])")
            # Compare with manual fusion from checkpoint
            q_qw = tensors["model.layers.0.self_attn.q_proj.qweight"]
            k_qw = tensors["model.layers.0.self_attn.k_proj.qweight"]
            v_qw = tensors["model.layers.0.self_attn.v_proj.qweight"]
            manual_fused = torch.cat([q_qw, k_qw, v_qw], dim=1)
            match = torch.equal(vllm_qw, manual_fused)
            print(f"  Matches manual concatenation: {match}")
            if not match:
                diff = (vllm_qw != manual_fused).sum().item()
                print(f"  Number of differing int32 elements: {diff} / {vllm_qw.numel()}")
                # Find first difference
                diff_idx = (vllm_qw != manual_fused).nonzero()
                if len(diff_idx) > 0:
                    r, c = diff_idx[0].tolist()
                    print(f"  First diff at [{r},{c}]: "
                          f"vLLM={vllm_qw[r,c].item():08x}, "
                          f"checkpoint={manual_fused[r,c].item():08x}")

    # Now test inference
    print("\n--- Running inference ---")
    from vllm import SamplingParams
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=20, temperature=0)
    )
    print(f"Output: {outputs[0].outputs[0].text!r}")

    # Restore
    INCXPULinearMethod.process_weights_after_loading = original_process

    return captured


if __name__ == "__main__":
    print("Running minimal debug tests...\n")

    cos1 = test_single_layer()
    cos2 = test_fused_qkv()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single layer cosine: {cos1:.6f}")
    print(f"Fused QKV cosine:    {cos2:.6f}")

    if cos1 > 0.99 and cos2 > 0.99:
        print("\nKernel tests PASS - running vLLM e2e test...")
        test_vllm_loaded_weights()
    else:
        print("\nKernel tests FAIL - fix kernel first!")
