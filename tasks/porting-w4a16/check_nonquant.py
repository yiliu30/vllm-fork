"""Check non-quantized weights: embeddings, layernorms, biases.
Compare vLLM's loaded model against the raw checkpoint."""
import torch
from safetensors.torch import load_file

MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)

tensors = load_file(MODEL_PATH)

# List all non-quantized tensors (not qweight/qzeros)
print("Non-quantized tensors in checkpoint:")
non_quant = {}
for name, t in sorted(tensors.items()):
    if 'qweight' not in name and 'qzeros' not in name:
        non_quant[name] = t
        if 'layers.0' in name or 'embed' in name or 'norm' in name or 'lm_head' in name:
            print(f"  {name}: {t.shape} {t.dtype}")

# Check if any biases exist
print("\nBias tensors:")
for name, t in sorted(tensors.items()):
    if 'bias' in name and 'layers.0' in name:
        print(f"  {name}: {t.shape} {t.dtype} sum={t.sum().item():.6f}")

# Check embed_tokens
embed = tensors.get("model.embed_tokens.weight")
if embed is not None:
    print(f"\nembed_tokens: {embed.shape} {embed.dtype}")
    print(f"  first5: {embed[0, :5].tolist()}")
    print(f"  sum: {embed.sum().item():.4f}")

# Check if lm_head exists or is tied
lm_head = tensors.get("lm_head.weight")
print(f"\nlm_head.weight exists: {lm_head is not None}")

# Check layer norms
for name in ["model.layers.0.input_layernorm.weight",
             "model.layers.0.post_attention_layernorm.weight",
             "model.norm.weight"]:
    t = tensors.get(name)
    if t is not None:
        print(f"\n{name}: {t.shape} {t.dtype}")
        print(f"  mean={t.mean().item():.6f} std={t.std().item():.6f}")

# Check scales - are they all fp16?
print("\nScale tensor dtypes:")
scale_dtypes = set()
for name, t in tensors.items():
    if 'scales' in name:
        scale_dtypes.add(t.dtype)
print(f"  All scale dtypes: {scale_dtypes}")
