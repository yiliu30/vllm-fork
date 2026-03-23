"""Compare our dequant-from-CT-repacked with direct dequant from AWQ checkpoint."""
import torch
from safetensors.torch import load_file

MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)

WEIGHT_BITS = 4
PACK_FACTOR = 8
GROUP_SIZE = 128

tensors = load_file(MODEL_PATH)
qw_awq = tensors["model.layers.0.self_attn.q_proj.qweight"]  # [896, 112]
sc = tensors["model.layers.0.self_attn.q_proj.scales"]        # [7, 896]

# Method 1: Direct AWQ dequant (known correct from transformers test)
mask = (1 << WEIGHT_BITS) - 1
in_size, out_packed = qw_awq.shape
out_size = out_packed * PACK_FACTOR

unpacked_awq = torch.zeros((in_size, out_size), dtype=torch.float32)
for i in range(PACK_FACTOR):
    unpacked_awq[:, i::PACK_FACTOR] = ((qw_awq >> (WEIGHT_BITS * i)) & mask).float()

# Dequant: [in, out] with scales [num_groups, out]
num_groups = in_size // GROUP_SIZE
dequant_awq = torch.zeros((in_size, out_size), dtype=torch.float32)
for g in range(num_groups):
    row_start = g * GROUP_SIZE
    row_end = row_start + GROUP_SIZE
    dequant_awq[row_start:row_end, :] = (
        (unpacked_awq[row_start:row_end, :] - 8.0) * sc[g, :].float().unsqueeze(0)
    )

print(f"Method 1 (AWQ direct): dequant shape={dequant_awq.shape}")
print(f"  sum={dequant_awq.sum():.4f}, first5={dequant_awq[0, :5].tolist()}")

# Method 2: Repack AWQ->CT, then dequant from CT (what our apply() does)
# Step 1: Repack
unpacked_for_repack = torch.zeros((in_size, out_size), dtype=torch.int32)
for i in range(PACK_FACTOR):
    unpacked_for_repack[:, i::PACK_FACTOR] = (qw_awq >> (WEIGHT_BITS * i)) & mask
unpacked_for_repack = unpacked_for_repack.t().contiguous()  # [out, in]

in_packed = in_size // PACK_FACTOR
repacked_ct = torch.zeros((out_size, in_packed), dtype=torch.int32)
for i in range(PACK_FACTOR):
    repacked_ct |= (unpacked_for_repack[:, i::PACK_FACTOR] & mask) << (WEIGHT_BITS * i)

print(f"\nRepacked CT: {repacked_ct.shape}")

# Step 2: Dequant from CT layout (matching the apply() code)
unpacked_ct = torch.zeros((out_size, in_size), dtype=torch.float32)
for i in range(PACK_FACTOR):
    unpacked_ct[:, i::PACK_FACTOR] = (
        (repacked_ct >> (WEIGHT_BITS * i)) & mask
    ).float()

# unpacked_ct is [out, in], scales is [num_groups, out]
dequant_ct = torch.zeros((out_size, in_size), dtype=torch.float32)
for g in range(num_groups):
    col_start = g * GROUP_SIZE
    col_end = col_start + GROUP_SIZE
    # unpacked_ct[:, col_start:col_end] is [out, group_size]
    # scales[g, :] is [out]
    dequant_ct[:, col_start:col_end] = (
        (unpacked_ct[:, col_start:col_end] - 8.0)
        * sc[g, :].float().unsqueeze(1)
    )

print(f"Method 2 (CT repack): dequant shape={dequant_ct.shape}")
print(f"  sum={dequant_ct.sum():.4f}, first5={dequant_ct[0, :5].tolist()}")

# Compare: dequant_awq is [in, out], dequant_ct is [out, in]
# So dequant_ct.t() should equal dequant_awq
diff = (dequant_awq - dequant_ct.t()).abs()
print(f"\nComparison (dequant_awq vs dequant_ct.t()):")
print(f"  max diff: {diff.max():.6f}")
print(f"  mean diff: {diff.mean():.6f}")

# Test matmul
torch.manual_seed(42)
x = torch.randn(1, in_size, dtype=torch.float16)

out_awq = (x.float() @ dequant_awq).half()
out_ct = (x.float() @ dequant_ct.t()).half()

print(f"\nMatmul test:")
print(f"  AWQ out: first5={out_awq[0, :5].tolist()}")
print(f"  CT  out: first5={out_ct[0, :5].tolist()}")

cos = torch.nn.functional.cosine_similarity(
    out_awq.float().flatten(), out_ct.float().flatten(), dim=0
).item()
print(f"  cosine: {cos:.6f}")
