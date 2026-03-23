"""Compare dumped vLLM weights with raw checkpoint weights."""
import torch
from safetensors.torch import load_file

MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)
DUMP_DIR = "/home/yiliu7/workspace/vllm/tasks/porting-w4a16/dumped_weights"

tensors = load_file(MODEL_PATH)

# Layer 1 = fused QKV [896, 144]
print("=" * 70)
print("Layer 1: Fused QKV (qkv_proj)")
print("=" * 70)
vllm_qw = torch.load(f"{DUMP_DIR}/qweight_1.pt")
vllm_sc = torch.load(f"{DUMP_DIR}/scales_1.pt")
vllm_qz = torch.load(f"{DUMP_DIR}/qzeros_1.pt")

# Manual fusion from checkpoint
q_qw = tensors["model.layers.0.self_attn.q_proj.qweight"]
k_qw = tensors["model.layers.0.self_attn.k_proj.qweight"]
v_qw = tensors["model.layers.0.self_attn.v_proj.qweight"]
expected_qw = torch.cat([q_qw, k_qw, v_qw], dim=1)

q_sc = tensors["model.layers.0.self_attn.q_proj.scales"]
k_sc = tensors["model.layers.0.self_attn.k_proj.scales"]
v_sc = tensors["model.layers.0.self_attn.v_proj.scales"]
expected_sc = torch.cat([q_sc, k_sc, v_sc], dim=1)

q_qz = tensors["model.layers.0.self_attn.q_proj.qzeros"]
k_qz = tensors["model.layers.0.self_attn.k_proj.qzeros"]
v_qz = tensors["model.layers.0.self_attn.v_proj.qzeros"]
expected_qz = torch.cat([q_qz, k_qz, v_qz], dim=1)

print(f"vLLM qweight:     {vllm_qw.shape} {vllm_qw.dtype}")
print(f"Expected qweight: {expected_qw.shape} {expected_qw.dtype}")
qw_match = torch.equal(vllm_qw, expected_qw)
print(f"qweight MATCH: {qw_match}")
if not qw_match:
    diff_mask = (vllm_qw != expected_qw)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_qw.numel()}")
    # Find first difference
    idx = diff_mask.nonzero()
    if len(idx) > 0:
        r, c = idx[0].tolist()
        print(f"  First diff at [{r},{c}]: vLLM={hex(vllm_qw[r,c].item())} vs expected={hex(expected_qw[r,c].item())}")
    # Check each section
    q_end = q_qw.shape[1]  # 112
    k_end = q_end + k_qw.shape[1]  # 128
    v_end = k_end + v_qw.shape[1]  # 144
    print(f"  Q section [0:{q_end}] match: {torch.equal(vllm_qw[:, :q_end], expected_qw[:, :q_end])}")
    print(f"  K section [{q_end}:{k_end}] match: {torch.equal(vllm_qw[:, q_end:k_end], expected_qw[:, q_end:k_end])}")
    print(f"  V section [{k_end}:{v_end}] match: {torch.equal(vllm_qw[:, k_end:v_end], expected_qw[:, k_end:v_end])}")

print(f"\nscales MATCH: {torch.equal(vllm_sc, expected_sc)}")
if not torch.equal(vllm_sc, expected_sc):
    diff_mask = (vllm_sc != expected_sc)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_sc.numel()}")
    idx = diff_mask.nonzero()
    if len(idx) > 0:
        r, c = idx[0].tolist()
        print(f"  First diff at [{r},{c}]: vLLM={vllm_sc[r,c].item()} vs expected={expected_sc[r,c].item()}")

print(f"qzeros MATCH: {torch.equal(vllm_qz, expected_qz)}")

# Layer 2 = o_proj [896, 112]
print("\n" + "=" * 70)
print("Layer 2: o_proj")
print("=" * 70)
vllm_qw2 = torch.load(f"{DUMP_DIR}/qweight_2.pt")
vllm_sc2 = torch.load(f"{DUMP_DIR}/scales_2.pt")
expected_qw2 = tensors["model.layers.0.self_attn.o_proj.qweight"]
expected_sc2 = tensors["model.layers.0.self_attn.o_proj.scales"]
print(f"qweight MATCH: {torch.equal(vllm_qw2, expected_qw2)}")
print(f"scales MATCH: {torch.equal(vllm_sc2, expected_sc2)}")
if not torch.equal(vllm_sc2, expected_sc2):
    diff_mask = (vllm_sc2 != expected_sc2)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_sc2.numel()}")

# Layer 3 = gate_up_proj [896, 1216]
print("\n" + "=" * 70)
print("Layer 3: gate_up_proj")
print("=" * 70)
vllm_qw3 = torch.load(f"{DUMP_DIR}/qweight_3.pt")
vllm_sc3 = torch.load(f"{DUMP_DIR}/scales_3.pt")
gate_qw = tensors["model.layers.0.mlp.gate_proj.qweight"]
up_qw = tensors["model.layers.0.mlp.up_proj.qweight"]
expected_qw3 = torch.cat([gate_qw, up_qw], dim=1)
gate_sc = tensors["model.layers.0.mlp.gate_proj.scales"]
up_sc = tensors["model.layers.0.mlp.up_proj.scales"]
expected_sc3 = torch.cat([gate_sc, up_sc], dim=1)
print(f"qweight MATCH: {torch.equal(vllm_qw3, expected_qw3)}")
print(f"scales MATCH: {torch.equal(vllm_sc3, expected_sc3)}")
if not torch.equal(vllm_qw3, expected_qw3):
    diff_mask = (vllm_qw3 != expected_qw3)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_qw3.numel()}")
if not torch.equal(vllm_sc3, expected_sc3):
    diff_mask = (vllm_sc3 != expected_sc3)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_sc3.numel()}")

# Layer 4 = down_proj [4864, 112]
print("\n" + "=" * 70)
print("Layer 4: down_proj")
print("=" * 70)
vllm_qw4 = torch.load(f"{DUMP_DIR}/qweight_4.pt")
vllm_sc4 = torch.load(f"{DUMP_DIR}/scales_4.pt")
expected_qw4 = tensors["model.layers.0.mlp.down_proj.qweight"]
expected_sc4 = tensors["model.layers.0.mlp.down_proj.scales"]
print(f"qweight MATCH: {torch.equal(vllm_qw4, expected_qw4)}")
print(f"scales MATCH: {torch.equal(vllm_sc4, expected_sc4)}")
if not torch.equal(vllm_sc4, expected_sc4):
    diff_mask = (vllm_sc4 != expected_sc4)
    num_diff = diff_mask.sum().item()
    print(f"  Number of differences: {num_diff} / {vllm_sc4.numel()}")

# Full matmul test on layer 1 (QKV)
print("\n" + "=" * 70)
print("Matmul test on dumped layer 1 vs checkpoint")
print("=" * 70)
torch.manual_seed(42)
test_x = torch.randn(1, 896, dtype=torch.float16)

# AWQ dequant from checkpoint
mask = 0xF
in_size = 896
for name, qw, sc in [("checkpoint", expected_qw, expected_sc),
                       ("vllm_dump", vllm_qw, vllm_sc)]:
    out_packed = qw.shape[1]
    out_size = out_packed * 8
    unpacked = torch.zeros((in_size, out_size), dtype=torch.float32)
    for i in range(8):
        unpacked[:, i::8] = ((qw >> (4 * i)) & mask).float()
    num_groups = in_size // 128
    dequant = torch.zeros_like(unpacked)
    for g in range(num_groups):
        rs = g * 128
        re = rs + 128
        dequant[rs:re, :] = (unpacked[rs:re, :] - 8.0) * sc[g, :].float().unsqueeze(0)
    out = (test_x.float() @ dequant).half()
    print(f"{name}: out sum={out.sum().item():.4f}, first5={out[0,:5].tolist()}")
