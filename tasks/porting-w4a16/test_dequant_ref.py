"""Test: replace INCXPULinearMethod.apply with a CPU dequant reference
to see if the issue is in the kernel or elsewhere."""

import torch
from torch.nn.parameter import Parameter

# Monkey-patch BEFORE importing vLLM
from vllm.model_executor.layers.quantization import inc as inc_module

OriginalClass = inc_module.INCXPULinearMethod

# Override apply to use CPU dequant reference instead of XPU kernel
_orig_apply = OriginalClass.apply

def debug_apply(self, layer, x, bias=None):
    """Dequantize weights on-the-fly and do a regular matmul."""
    # layer.qweight is in CT layout [out, in_packed] after repacking
    qweight = layer.qweight.data  # [out, in_packed]
    scales = layer.scales.data    # [num_groups, out]

    mask = (1 << self.weight_bits) - 1
    out_size, in_packed = qweight.shape
    in_size = in_packed * self.pack_factor

    # Unpack CT layout: [out, in_packed] -> [out, in]
    unpacked = torch.zeros((out_size, in_size), dtype=torch.float32,
                           device=qweight.device)
    for i in range(self.pack_factor):
        unpacked[:, i::self.pack_factor] = (
            (qweight >> (self.weight_bits * i)) & mask
        ).float()

    # Dequantize: w = (w_int - 8) * scale
    # scales is [num_groups, out], we need it as [out, num_groups]
    group_size = self.group_size
    num_groups = in_size // group_size

    dequant = torch.zeros_like(unpacked)
    for g in range(num_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        # unpacked[:, col_start:col_end] is [out, group_size]
        # scales[g, :] is [out]
        dequant[:, col_start:col_end] = (
            (unpacked[:, col_start:col_end] - 8.0) * scales[g, :].float().unsqueeze(1)
        )

    # dequant is [out, in] float32
    # x is [*, in] fp16
    # output = x @ dequant.t() -> [*, out]
    reshaped_x = x.reshape(-1, x.shape[-1]).float()
    out = (reshaped_x @ dequant.t()).to(x.dtype)

    if bias is not None:
        out = out + bias

    out_shape = x.shape[:-1] + (out_size,)
    return out.reshape(out_shape)


OriginalClass.apply = debug_apply

from vllm import LLM, SamplingParams

llm = LLM(
    model='Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound',
    block_size=64,
    enforce_eager=True,
    max_model_len=256,
    gpu_memory_utilization=0.5,
)

out = llm.generate(
    ['The capital of France is'],
    SamplingParams(max_tokens=20, temperature=0),
)
print('OUTPUT:', repr(out[0].outputs[0].text))
print('TOKEN_IDS:', out[0].outputs[0].token_ids)
