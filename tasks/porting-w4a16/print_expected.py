"""Print the expected first4 hex values for layer 0 QKV from raw checkpoint."""
from safetensors.torch import load_file
import torch

MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)

tensors = load_file(MODEL_PATH)

# Layer 0 QKV
for proj in ['q_proj', 'k_proj', 'v_proj']:
    name = f"model.layers.0.self_attn.{proj}"
    qw = tensors[f"{name}.qweight"]
    sc = tensors[f"{name}.scales"]
    print(f"{proj}:")
    print(f"  qweight {qw.shape}: first4 = {[hex(x) for x in qw[0, :4].tolist()]}")
    print(f"  scales  {sc.shape}: first4 = {sc[0, :4].tolist()}")
    print()

# Fused = concat along dim=1
q_qw = tensors["model.layers.0.self_attn.q_proj.qweight"]
k_qw = tensors["model.layers.0.self_attn.k_proj.qweight"]
v_qw = tensors["model.layers.0.self_attn.v_proj.qweight"]
fused_qw = torch.cat([q_qw, k_qw, v_qw], dim=1)  # [896, 144]

q_sc = tensors["model.layers.0.self_attn.q_proj.scales"]
k_sc = tensors["model.layers.0.self_attn.k_proj.scales"]
v_sc = tensors["model.layers.0.self_attn.v_proj.scales"]
fused_sc = torch.cat([q_sc, k_sc, v_sc], dim=1)  # [7, 1152]

print("Fused QKV (expected in vLLM):")
print(f"  qweight {fused_qw.shape}: first4 = {[hex(x) for x in fused_qw[0, :4].tolist()]}")
print(f"  scales  {fused_sc.shape}: first4 = {fused_sc[0, :4].tolist()}")

# Layer 0 MLP gate_up_proj (fused gate + up)
gate_qw = tensors["model.layers.0.mlp.gate_proj.qweight"]
up_qw = tensors["model.layers.0.mlp.up_proj.qweight"]
gate_sc = tensors["model.layers.0.mlp.gate_proj.scales"]
up_sc = tensors["model.layers.0.mlp.up_proj.scales"]

print(f"\ngate_proj qweight {gate_qw.shape}: first4 = {[hex(x) for x in gate_qw[0, :4].tolist()]}")
print(f"up_proj   qweight {up_qw.shape}: first4 = {[hex(x) for x in up_qw[0, :4].tolist()]}")

fused_gate_up = torch.cat([gate_qw, up_qw], dim=1)
fused_gate_up_sc = torch.cat([gate_sc, up_sc], dim=1)
print(f"Fused gate_up {fused_gate_up.shape}: first4 = {[hex(x) for x in fused_gate_up[0, :4].tolist()]}")
print(f"Fused scales  {fused_gate_up_sc.shape}: first4 = {fused_gate_up_sc[0, :4].tolist()}")

# down_proj
down_qw = tensors["model.layers.0.mlp.down_proj.qweight"]
down_sc = tensors["model.layers.0.mlp.down_proj.scales"]
print(f"\ndown_proj qweight {down_qw.shape}: first4 = {[hex(x) for x in down_qw[0, :4].tolist()]}")
print(f"down_proj scales  {down_sc.shape}: first4 = {down_sc[0, :4].tolist()}")
