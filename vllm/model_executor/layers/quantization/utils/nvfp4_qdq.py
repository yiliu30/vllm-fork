import os
import gc
import functools
import habana_frameworks.torch.internal.bridge_config as bc

os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"

# os.environ['HABANA_PROFILE'] = '1'
# os.environ['GRAPH_VISUALIZATION'] = '1'
import torch.distributed as dist
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.internal.bridge_config as bc

# from vllm_hpu_extension.profiler import (HabanaMemoryProfiler, format_bytes)


from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

from habana_frameworks.torch.hpu import wrap_in_hpu_graph

import os

os.environ["PT_HPU_LAZY_MODE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as ht
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import time
import argparse
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import (
    dequant_mx_fp8,
    quant_mx_fp8,
)


kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
# kE2M1ToFloat = torch.tensor( [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float8_e4m3fn)

from typing import Dict, Optional, Tuple


# reference: : https://github.com/vllm-project/vllm/pull/16362
def unpack_fp4_from_uint8(
    a: torch.Tensor,
    m: int,
    n: int,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four corresond to a consecutive
    fp4 value). The bits represent an index, which are mapped to an fp4 value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8, f"Ex got{a.dtype}"

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    # breakpoint()
    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)


# >>>>>>>>>>>>>>>>>>

SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_BF16, MBITS_BF16 = 8, 7
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2
from torchao.prototype.mx_formats.mx_tensor import to_dtype as ao_to_dtype

import os

_USE_CT_UNPACK = os.getenv("USE_CT_UNPACK", "0").lower() in ("1", "true", "yes")

from enum import Enum, auto
from typing import Callable, Dict, Union

import torch
import os
from torchao.prototype.mx_formats.constants import (
    F4_E2M1_MAX,
    F8E4M3_MAX,
)


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-tensor amax to per-tensor scale.
    Used to scale fp32 scales down to fp8 scales

    Args:
        amax: Per-tensor amax tensor

    Returns:
        torch.Tensor: Per-tensor scale tensor
    """
    return torch.clamp(amax / F8E4M3_MAX, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float32
    )


E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA
from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)


def fp4_121_positive(
    x: torch.Tensor, stochastic_rounding: bool = False
) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0

    return (
        step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)
    )


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    # x[(x >= 0.0) & (x <= 0.25)] = 0.0
    # x[(x > 0.25) & (x < 0.75)] = 0.5
    # x[(x >= 0.75) & (x <= 1.25)] = 1.0
    # x[(x > 1.25) & (x < 1.75)] = 1.5
    # x[(x >= 1.75) & (x <= 2.5)] = 2.0
    # x[(x > 2.5) & (x < 3.5)] = 3.0
    # x[(x >= 3.5) & (x <= 5.0)] = 4.0
    # x[x > 5.0] = 6.0
    x = fp4_121_positive(x)
    return x * sign


def nvfp4_quantize(
    data_hp: torch.Tensor,
    block_size: int = 16,
    per_tensor_scale: Optional[torch.Tensor] = None,
    do_pack: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVIDIA FP4 quantization with UE4M3 scales.

    Implements the NVIDIA algorithm for quantizing tensors to FP4 format
    with unsigned E4M3 (UE4M3) scales.

    Args:
        data_hp: High precision input tensor (bfloat16 or float32)
        block_size: Block size for quantization (must be 16)
        per_tensor_amax: Optional pre-computed absolute maximum for calibration.
            If provided, uses per-tensor scaling. If None, uses block-wise scaling only.

    Returns:
        tuple: A tuple containing:
            - total_scale_fp8: Blockwise scales in float8_e4m3fn format
            - per_tensor_scale: Global per-tensor scale if per_tensor_amax provided, else None
            - data_lp: Packed FP4 data (2 values per byte)

    Raises:
        AssertionError: If input dtype is not supported, tensor size is not
            divisible by block_size, tensor is not contiguous, or block_size != 16
    """
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} not supported"
    assert (
        data_hp.size(-1) % block_size == 0
    ), "K dim must be divisible by block_size"
    assert data_hp.is_contiguous(), "Only support contiguous data for now"
    assert block_size == 16, "NVFP4 requires block_size=16"

    orig_shape = data_hp.shape
    # Convert to float32 early for consistent precision with Triton implementation
    data_hp = data_hp.float().reshape(orig_shape[0], -1, block_size)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1)
    # These scales are currently in fp32, we are going to `quantize` them to e4m3
    block_scale = max_abs / F4_E2M1_MAX

    out_scales = None
    if per_tensor_scale is None:
        # We are doing single level scaling
        block_scale_fp8 = torch.clamp(
            block_scale, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        block_scale_fp32 = block_scale_fp8.to(torch.float32)
        data_scaled = data_hp / block_scale_fp32.unsqueeze(-1)
        out_scales = block_scale_fp8
    else:
        # We are doing two level scaling,
        # This will likely be calibrated but
        # we want the per_tensor_scale ~= amax of the block_scale_fp32
        block_scale_fp32 = block_scale.to(torch.float32)
        # Quantize the blockwise scales w/ the per_tensor_scale
        scaled_block_scales = block_scale_fp32 / per_tensor_scale
        scaled_block_scales_fp8 = torch.clamp(
            scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
        # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
        # To apply to data
        total_scale = per_tensor_scale * scaled_block_scales_fp32
        data_scaled = data_hp / total_scale.unsqueeze(-1)
        out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)
    # data_lp = f32_to_f4_unpacked(data_scaled)
    # # TODO: NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
    # # data_lp = pack_uint4(data_lp).view(torch.float4_e2m1fn_x2)
    # data_lp = pack_uint4(data_lp)
    data_lp = cast_to_fp4(data_scaled)
    if not do_pack:
        return out_scales, data_lp
    data_lp = pack_fp4_to_uint8(data_lp)

    return out_scales, data_lp


def to_nvfp4(x, do_pack=True):
    tensor_amax = torch.max(torch.abs(x))
    per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
    out_scales, data_lp = nvfp4_quantize(
        data_hp=x,
        block_size=16,
        per_tensor_scale=per_tensor_scale,
        do_pack=do_pack,
    )
    return data_lp, out_scales, 1.0/per_tensor_scale


def dequant_nvfp4(
    data_lp,
    out_scales,
    per_tensor_scale,
    original_dtype=torch.bfloat16,
    packed=True,
):
    scale_fp = out_scales.to(original_dtype) / per_tensor_scale.to(
        original_dtype
    )
    if packed:
        m, half_n = data_lp.shape
        n = half_n * 2
        data_hp = unpack_fp4_from_uint8(
            data_lp, m, half_n * 2, dtype=original_dtype
        )
    else:
        data_hp = data_lp.to(original_dtype)

    m, n = data_hp.shape
    data_hp = data_hp.reshape(m, -1, 16)
    scale_fp = scale_fp.reshape(m, -1, 1)
    data_hp = data_hp * scale_fp
    data_hp = data_hp.reshape(m, n)
    return data_hp


def unpacked_nvfp4_to_fp8(data_lp):
    assert data_lp.dtype == torch.uint8, f"Expect uint8, got {data_lp.dtype}"
    m, half_n = data_lp.shape
    n = half_n * 2
    data_hp = unpack_fp4_from_uint8(data_lp, m, n, dtype=torch.float8_e4m3fn)
    return data_hp


def qdq_nvfp4(x):
    data_lp, x_scale, x_global_scale = to_nvfp4(x, do_pack=False)
    x_dq = dequant_nvfp4(
        data_lp,
        x_scale,
        x_global_scale,
        original_dtype=x.dtype,
        packed=False,
    )
    return x_dq


class NVFP4Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_packed = torch.nn.Parameter(
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(out_features // 16, dtype=torch.uint8),
            requires_grad=False,
        )
        weight_global_scale = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_parameter("weight_global_scale", weight_global_scale)

    def nvfp4_emulations(self):
        hp_weight = dequant_nvfp4(
            self.weight_packed.data,
            self.weight_scale.data,
            self.weight_global_scale.data,
            original_dtype=torch.bfloat16,
        )
        return hp_weight

    def forward(self, x):
        hp_weight = self.nvfp4_emulations()
        x = qdq_nvfp4(x)
        return x @ hp_weight.t()

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear):
        in_features = linear.in_features
        out_features = linear.out_features
        nvfp4_linear = cls(in_features, out_features).to("hpu")
        # convert weight to nvfp4
        data_lp, weight_scale, weight_global_scale = to_nvfp4(
            x=linear.weight.clone()
        )
        nvfp4_linear.weight_packed.data = data_lp
        nvfp4_linear.weight_scale.data = weight_scale
        nvfp4_linear.weight_global_scale.data = weight_global_scale
        return nvfp4_linear


class NVFP4LinearUnpacked(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_unpacked = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(out_features // 16, dtype=torch.uint8),
            requires_grad=False,
        )
        weight_global_scale = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_parameter("weight_global_scale", weight_global_scale)

    def nvfp4_emulations(self):
        hp_weight = dequant_nvfp4(
            self.weight_unpacked.data,
            self.weight_scale.data,
            self.weight_global_scale.data,
            original_dtype=torch.bfloat16,
            packed=False,
        )
        return hp_weight

    def forward(self, x):
        hp_weight = self.nvfp4_emulations()
        x = qdq_nvfp4(x)
        return x @ hp_weight.t()

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear):
        in_features = linear.in_features
        out_features = linear.out_features
        nvfp4_linear = cls(in_features, out_features).to("hpu")
        # convert weight to nvfp4
        data_lp, weight_scale, weight_global_scale = to_nvfp4(
            x=linear.weight.clone(), do_pack=True
        )
        data_lp_unpacked = unpacked_nvfp4_to_fp8(data_lp)
        nvfp4_linear.weight_unpacked.data = data_lp_unpacked
        nvfp4_linear.weight_scale.data = weight_scale
        nvfp4_linear.weight_global_scale.data = weight_global_scale
        return nvfp4_linear


def test_qdq(args):
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    cpu_tensor = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16
    )
    hpu_tensor = cpu_tensor.to("hpu")
    # scale_e8m0_biased, quant_tensor = quant_mx_fp8(hpu_tensor)
    # dequant_tensor = dequant_mx_fp8(quant_tensor, scale_e8m0_biased, block_size=32)
    # print(f"quant_tensor shape: {quant_tensor.shape}, dtype: {quant_tensor.dtype}")
    dequant_tensor = quant_dequant_mxfp4(hpu_tensor)

    diff = torch.abs(hpu_tensor - dequant_tensor)
    print(f"diff: max: {diff.max()}, min: {diff.min()}, mean: {diff.mean()}")
    ht.hpu.synchronize()


def time_fn(func, times, warmup=50):
    torch.hpu.synchronize()
    for _ in range(warmup):
        func()
    torch.hpu.synchronize()
    gc.collect()

    start = time.time()
    for _ in range(times):
        func()
    torch.hpu.synchronize()
    end = time.time()
    return (end - start) / times


def test_linear(args):
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    batch_size = args.bs
    cpu_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    hpu_tensor = cpu_tensor.to("hpu")
    linear = torch.nn.Linear(
        in_features=hidden_size, out_features=intermediate_size, bias=False
    ).to("hpu")
    nvfp4_linear = NVFP4Linear.from_linear(linear)
    nvfp4_linear = nvfp4_linear.to("hpu")
    nvfp4_unpacked_linear = NVFP4LinearUnpacked.from_linear(linear)
    nvfp4_unpacked_linear = nvfp4_unpacked_linear.to("hpu")

    linear = wrap_in_hpu_graph(linear)
    nvfp4_linear = wrap_in_hpu_graph(nvfp4_linear)
    nvfp4_unpacked_linear = wrap_in_hpu_graph(nvfp4_unpacked_linear)

    out_ref = linear(hpu_tensor)
    out1 = nvfp4_linear(hpu_tensor)
    out2 = nvfp4_unpacked_linear(hpu_tensor)
    print(
        f"diff ref vs nvfp4: max: {torch.abs(out_ref - out1).max()}, min: {torch.abs(out_ref - out1).min()}, mean: {torch.abs(out_ref - out1).mean()}"
    )
    print(
        f"diff ref vs nvfp4 unpacked: max: {torch.abs(out_ref - out2).max()}, min: {torch.abs(out_ref - out2).min()}, mean: {torch.abs(out_ref - out2).mean()}"
    )
    print(
        f"diff nvfp4 vs nvfp4 unpacked: max: {torch.abs(out1 - out2).max()}, min: {torch.abs(out1 - out2).min()}, mean: {torch.abs(out1 - out2).mean()}"
    )
    torch.hpu.synchronize()
    latency0 = time_fn(lambda: linear(hpu_tensor), args.bench_steps)
    latency1 = time_fn(lambda: nvfp4_linear(hpu_tensor), args.bench_steps)
    latency2 = time_fn(
        lambda: nvfp4_unpacked_linear(hpu_tensor), args.bench_steps
    )
    print(
        f"latency linear: {latency0:.6f} s, nvfp4_linear: {latency1:.6f} s, nvfp4_unpacked_linear: {latency2:.6f} s"
    )

    print(
        f"speed up nvfp4_linear: {latency0 / latency1:.2f}x, nvfp4_unpacked_linear: {latency0 / latency2:.2f}x"
    )
    print(
        f"speed up nvfp4_unpacked_linear vs nvfp4_linear: {latency1 / latency2:.2f}x"
    )
    exit()
    profile_steps = args.profile_steps
    warmup_steps = args.warmup_steps
    if profile_steps > 0:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.HPU,
        ]
        schedule = torch.profiler.schedule(
            wait=0, warmup=warmup_steps, active=profile_steps, repeat=1
        )
        print(f"Profiling steps {profile_steps} with warmup {warmup_steps}")
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./profile_mxfp4/"
            ),
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            for i in range(warmup_steps + profile_steps):
                out_ref = linear(hpu_tensor)
                out1 = nvfp4_linear(hpu_tensor)
                out2 = nvfp4_unpacked_linear(hpu_tensor)
                # result2 = dist.all_reduce(result2, op=dist.ReduceOp.SUM)
                ht.hpu.synchronize()
                profiler.step()
        profiler.stop()
        ht.hpu.synchronize()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--bs", default=32, type=int)
#     parser.add_argument("--hidden_size", "-H", default=7168, type=int)
#     parser.add_argument("--intermediate_size", "-I", default=2048, type=int)
#     parser.add_argument("--num_total_experts", "-E", default=256, type=int)
#     parser.add_argument("--ep_size", "-P", default=8, type=int)
#     parser.add_argument("--topk", "-K", default=8, type=int)
#     parser.add_argument("--warmup_steps", default=5, type=int)
#     parser.add_argument("--bench_steps", "-S", default=1000, type=int)
#     parser.add_argument("--profile_steps", default=0, type=int)
#     args = parser.parse_args()

#     test_linear(args)
