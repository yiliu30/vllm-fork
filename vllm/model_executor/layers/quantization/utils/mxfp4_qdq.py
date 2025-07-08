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
from torchao.prototype.mx_formats.mx_tensor import (
    # to_mx,
    # to_dtype,
    ScaleCalculationMode,
    get_fp_scale,
    # _to_mx_rceil,
)
from torchao.prototype.mx_formats.mx_tensor import to_dtype as ao_to_dtype

import os

_USE_CT_UNPACK = os.getenv("USE_CT_UNPACK", "0").lower() in ("1", "true", "yes")

from enum import Enum, auto
from typing import Callable, Dict, Union

import torch
import os
from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.constants import (
    BF16_EXP_BIAS,
    BLOCK_SIZE_DEFAULT,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_FP4_E2M1,
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F4_E2M1_MAX,
    F4_E2M1_MAX_POW2,
    F6_E2M3_MAX,
    F6_E2M3_MAX_POW2,
    F6_E3M2_MAX,
    F6_E3M2_MAX_POW2,
    F8E4M3_MAX,
    F8E4M3_MAX_POW2,
    F8E5M2_MAX,
    F8E5M2_MAX_POW2,
    F32_EXP_BIAS,
    F32_MIN_NORMAL,
    SUPPORTED_ELEM_DTYPES,
)
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f6_e2m3_unpacked_to_f32,
    f6_e3m2_unpacked_to_f32,
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    pack_uint4,
    pack_uint6,
    triton_f4_to_scaled_bf16,
    triton_f6_e2m3_to_scaled_bf16,
    triton_f6_e3m2_to_scaled_bf16,
    unpack_uint4,
)


def _to_mx_rceil(
    data_hp: torch.Tensor,
    max_abs: torch.Tensor,
    max_pos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A prototype implementation of MXFP scale factor derivation method described in
    https://docs.nvidia.com/cuda/cublas/#d-block-quantization

    For Nvidia GPU with Blackwell+ architecture, the scale factor derivation method
    could be accelerated by the `cvt.rp.satfinite.ue8m0x2.f32` instruction.

    Args:
        data_hp: High precision data.
        max_abs: Maximum absolute value for data_hp along specified dimension/block_size.
        max_pos: The maximum value of the low precision data type.

    Returns:
        exponent: The biased exponent with dtype E8M0 in uint8 container.
        data_lp: The targeted low precision data, in high precision container
            (requires cast to low precision data type).
    """
    descale = max_abs / max_pos
    # breakpoint()
    # TODO: nan/inf needs to be set for any value
    # of nan/inf in input not just amax.
    exponent = torch.where(
        torch.isnan(descale),
        0xFF,  # Handle biased exponent for nan
        # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            )
            + E8M0_EXPONENT_BIAS
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        exponent == 0,
        1.0,
        torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
    )

    # scale and saturated cast the data elements to max of target dtype
    data_lp = torch.clamp(
        data_hp * descale_fp.unsqueeze(1), min=-1 * max_pos, max=max_pos
    )
    return exponent, data_lp


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    pack_fp6: bool = False,
):
    """
    Takes a high precision tensor and converts to MX scale and raw data, in
    naive layout (scale and raw data are separate tensors).
    """

    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.numel() % block_size == 0, "unsupported"
    assert data_hp.is_contiguous(), "unsupported"
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    # breakpoint()
    max_abs = torch.amax(torch.abs(data_hp), 1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
        mbits = MBITS_F8_E4M3
        max_pos = F8E4M3_MAX
    elif elem_dtype == torch.float8_e5m2:
        target_max_pow2 = F8E5M2_MAX_POW2
        mbits = MBITS_F8_E5M2
        max_pos = F8E5M2_MAX
    elif elem_dtype == DTYPE_FP6_E2M3:
        target_max_pow2 = F6_E2M3_MAX_POW2
        mbits = MBITS_F6_E2M3
        max_pos = F6_E2M3_MAX
    elif elem_dtype == DTYPE_FP6_E3M2:
        target_max_pow2 = F6_E3M2_MAX_POW2
        mbits = MBITS_F6_E3M2
        max_pos = F6_E3M2_MAX
    elif elem_dtype == DTYPE_FP4_E2M1:
        target_max_pow2 = F4_E2M1_MAX_POW2
        mbits = MBITS_F4_E2M1
        max_pos = F4_E2M1_MAX
    else:
        raise AssertionError("unsupported element dtype")
    # breakpoint()
    if scaling_mode == ScaleCalculationMode.RCEIL:
        scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)
        # breakpoint()
    else:
        if data_hp.dtype is torch.float32:
            hp_int_dtype = torch.int32
            hp_mbits = MBITS_F32
            hp_ebits = EBITS_F32
            hp_exp_bias = F32_EXP_BIAS
        else:
            assert data_hp.dtype is torch.bfloat16
            hp_int_dtype = torch.int16
            hp_mbits = MBITS_BF16
            hp_ebits = EBITS_BF16
            hp_exp_bias = BF16_EXP_BIAS

        # rounding before calculating the largest power of 2
        # X = 2^(floor(log2(rounding(max_abs(v)))-max_exp))
        if scaling_mode == ScaleCalculationMode.EVEN:
            nan_mask = torch.isnan(max_abs)
            max_abs = max_abs.view(hp_int_dtype)
            val_to_add = 1 << (hp_mbits - mbits - 1)
            mask = ((1 << (hp_ebits + SBITS)) - 1) << hp_mbits
            max_abs = (max_abs + val_to_add) & mask
            max_abs = max_abs.view(data_hp.dtype)
            max_abs[nan_mask] = torch.tensor(
                float("nan"), device=max_abs.device, dtype=max_abs.dtype
            )

        # Calculate the scale for different modes
        max_abs_int32 = (max_abs + eps).view(hp_int_dtype)
        extracted_pow2 = (
            (max_abs_int32 >> hp_mbits) & 0b11111111
        ) - hp_exp_bias

        if scaling_mode in (
            ScaleCalculationMode.FLOOR,
            ScaleCalculationMode.EVEN,
        ):
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        elif scaling_mode == ScaleCalculationMode.CEIL:
            # round up: add one to scale if the mantissa is larger than 0
            # 0x7FFFFF is equal to 23 ones
            mantissa_gt_one = (max_abs_int32 & 0x7FFFFF) > 0
            extracted_pow2 += mantissa_gt_one
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        else:
            raise AssertionError("unsupported scaling calculation mode")

        # Clamp to exponents that can be represented in e8m0
        # add one to positive range to capture NaNs
        scale_e8m0_unbiased = torch.clamp(
            scale_e8m0_unbiased,
            min=-E8M0_EXPONENT_BIAS,
            max=E8M0_EXPONENT_BIAS + 1,
        )

        # Create the biased e8m0 representation and cast it to 8 bits
        scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
        scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

        # Conversion to torch.uint8 sets NaN values to 0, fix this by
        # explicitly setting known NaN values to 255
        scale_e8m0_biased = torch.where(
            torch.isnan(max_abs),
            E8M0_EXPONENT_NAN_VAL,
            scale_e8m0_biased,
        )

        # For now, calculate the scale in floating point.
        scale_fp32 = (scale_e8m0_biased.to(torch.int32) << MBITS_F32).view(
            torch.float32
        )

        # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
        # float32 denormal range. For now, manually adjust the fp scale. This is
        # relevant if all of the incoming block values are zeroes.
        # See https://github.com/pytorch/pytorch/issues/125557 for details.
        # Note: it would be more correct to set the minimum to 2**-127, but this
        # does not work in triton either as it looks like subnormal value handling
        # has some gaps.  So, for now just set to the minimum normal value.
        scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

        # scale and saturated cast the data elements to max of target dtype
        data_lp = data_hp / scale_fp32.unsqueeze(1)

        if (
            elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not torch._dynamo.is_compiling()
        ):
            # As of 20250317, the Pytorch eager mode cast to `torch.float8_e4m3fn`
            # is unsaturated. This cast is saturated in triton. If we are compute bound,
            # we see a speedup if we remove this redundant clamp if we are compiling
            # to triton.
            # TODO(#1912): make the saturated cast work in eager mode and remove this
            # workaround.
            data_lp = torch.clamp(data_lp, min=-1 * max_pos, max=max_pos)

    # cast to target dtype
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_lp = data_lp.to(elem_dtype)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_lp = f32_to_f6_e2m3_unpacked(data_lp)
        if pack_fp6:
            orig_shape = [*orig_shape[:-1], 3 * orig_shape[-1] // 4]
            data_lp = pack_uint6(data_lp)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_lp = f32_to_f6_e3m2_unpacked(data_lp)
        if pack_fp6:
            orig_shape = [*orig_shape[:-1], 3 * orig_shape[-1] // 4]
            data_lp = pack_uint6(data_lp)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP4_E2M1:
        # can't reshape at the end without handling it in the packing code,
        # punt until later since we'll need to rethink the torch.compile
        # approach for fp4x2 in any case
        data_lp = data_lp.reshape(orig_shape)
        # data_lp = f32_to_f4_unpacked(data_lp)
        # orig_shape = [*orig_shape[:-1], orig_shape[-1] // 2]
        #  data_lp = pack_uint4(data_lp)
        from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import (
            pack_fp4_to_uint8,
        )

        data_lp = pack_fp4_to_uint8(data_lp)
    else:
        raise AssertionError("unsupported")

    # scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased, data_lp


def dequant_mxfp4_to_fp8(data_lp, scale_e8m0):
    data_fp8, scale_float = to_dtype(
        data_lp=data_lp,
        scale_e8m0=scale_e8m0,
        elem_dtype="fp4_e2m1",
        block_size=32,
        # target_dtype=x.dtype,
        target_dtype=torch.float8_e4m3fn,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
        scale_dtype=torch.bfloat16,
        return_scale=True,
    )
    return data_fp8, scale_float


def mxfp4_fp8_weight_to_bf16(weight_fp8, scale_bf16):
    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, 32)
    assert (
        weight_fp8.shape[0] == scale_bf16.shape[0]
    ), f"shape mismatch: {weight_fp8.shape} vs {scale_bf16.shape}"
    # TODO use cast_from_fp8_v2  ?
    dequant_weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_bf16
    dequant_weight_bf16 = dequant_weight_bf16.reshape(origin_shape)
    return dequant_weight_bf16


def mxfp4_gemm_with_unpacked_weight(x, weigth_fp8, weight_scale_bf16):
    from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (
        qdq_mxfp4,
    )

    x = qdq_mxfp4(x)
    # dequantize weight
    w_dq = mxfp4_fp8_weight_to_bf16(weigth_fp8, weight_scale_bf16)
    # matmul
    out = torch.matmul(x, w_dq.t())
    return out


def mxfp4_gemm_packed_weight(x, weigth_uint8, weight_scale_uint8):
    from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (
        qdq_mxfp4,
    )

    x = qdq_mxfp4(x)
    # dequantize weight
    w_dq = to_dtype(
        data_lp=weigth_uint8,
        scale_e8m0=weight_scale_uint8,
        elem_dtype="fp4_e2m1",
        block_size=32,
        target_dtype=x.dtype,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
        scale_dtype=x.dtype,
        return_scale=False,
    )
    # matmul
    out = torch.matmul(x, w_dq.t())
    return out


class MXFP4Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MXFP4Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_packed = torch.nn.Parameter(
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(out_features // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def forward(self, x):
        return mxfp4_gemm_packed_weight(
            x, self.weight_packed, self.weight_scale
        )

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear):
        in_features = linear.in_features
        out_features = linear.out_features
        mxfp4_linear = cls(in_features, out_features).to("hpu")
        # convert weight to mxfp4
        scale_e8m0_biased, data_lp = to_mx(
            data_hp=linear.weight.clone(),
            elem_dtype=DTYPE_FP4_E2M1,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
            pack_fp6=False,
        )
        mxfp4_linear.weight_packed.data = data_lp
        mxfp4_linear.weight_scale.data = scale_e8m0_biased
        return mxfp4_linear


class MXFP4LinearUnpacked(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_unpacked = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(out_features // 32, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x):
        return mxfp4_gemm_with_unpacked_weight(x, self.weight_unpacked, self.weight_scale)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear):
        in_features = linear.in_features
        out_features = linear.out_features
        mxfp4_linear = cls(in_features, out_features).to("hpu")
        # convert weight to mxfp4
        scale_e8m0_biased, data_lp = to_mx(
            data_hp=linear.weight,
            elem_dtype=DTYPE_FP4_E2M1,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
            pack_fp6=False,
        )
        weight_unpacked, scale_bf16 = dequant_mxfp4_to_fp8(
            data_lp, scale_e8m0_biased
        )
        mxfp4_linear.weight_unpacked.data = weight_unpacked
        mxfp4_linear.weight_scale.data = scale_bf16
        return mxfp4_linear


def to_dtype(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
    use_fp4_custom_triton_dequant_kernel,
    pack_fp6,
    scale_dtype=None,
    return_scale=False,
):
    orig_shape = data_lp.shape
    is_transposed = not data_lp.is_contiguous()
    # if the underlying data is transposed, convert to row major before
    # unpacking and unscaling
    if is_transposed:
        data_lp = data_lp.t()
        assert data_lp.is_contiguous()
        orig_shape = (orig_shape[1], orig_shape[0])

    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_hp = data_lp.to(target_dtype)
    elif elem_dtype == DTYPE_FP4_E2M1:
        # fp4
        if 1 or _USE_CT_UNPACK:
            # from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import unpack_fp4_from_uint8

            m, half_n = data_lp.shape
            n = half_n * 2
            data_hp = unpack_fp4_from_uint8(data_lp, m, n, dtype=target_dtype)
        else:
            f4_unpacked = unpack_uint4(data_lp)
            # for now we only have a cast to f32
            # TODO(future PR): add cast directly to bf16
            f32 = f4_unpacked_to_f32(f4_unpacked)
            data_hp = f32.to(target_dtype)
        # manually adjust shape to account for the unpacking
        # TODO(future PR): clean up the shape code and remove the hack
        # below
        orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)
    else:
        raise AssertionError("unsupported")

    data_hp = data_hp.reshape(-1, block_size)
    # Get scale
    if scale_dtype is None:
        scale_dtype = target_dtype
    s_fp = get_fp_scale(scale_e8m0).reshape(-1, 1).to(scale_dtype)
    if return_scale:
        return data_hp.reshape(orig_shape), s_fp
        # when inference:
        # data_hp: m, n
        # s_fp: m * n // block_size, 1
        # data_hp.reshape(-1, block_size).mul(s_fp).reshape(orig_shape)
    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(orig_shape)

    # if we converted to row-major before unscaling convert back
    if is_transposed:
        data_hp = data_hp.t()

    return data_hp


def quant_dequant_mxfp4(
    x: torch.Tensor,
):
    group_size = 32

    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mx(
        data_hp=x,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        pack_fp6=False,
    )
    # breakpoint()

    # dequantize input
    x_dq_fp8, scale1 = to_dtype(
        data_lp=x_q,
        scale_e8m0=input_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        # target_dtype=x.dtype,
        target_dtype=torch.float8_e4m3fn,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )
    x_dq, scale2 = to_dtype(
        data_lp=x_q,
        scale_e8m0=input_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
        # target_dtype=torch.float8_e4m3fn,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )
    # breakpoint()
    diff = (x_dq_fp8.to(x_dq.dtype) - x_dq).abs().max()
    assert diff < 1e-5, f"dequantization error: {diff}"
    # breakpoint()
    return x_dq


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
    mxfp4_linear = MXFP4Linear.from_linear(linear)
    mxfp4_linear = mxfp4_linear.to("hpu")
    mxfp4_unpacked_linear = MXFP4LinearUnpacked.from_linear(linear)
    mxfp4_unpacked_linear = mxfp4_unpacked_linear.to("hpu")

    linear = wrap_in_hpu_graph(linear)
    mxfp4_linear = wrap_in_hpu_graph(mxfp4_linear)
    mxfp4_unpacked_linear = wrap_in_hpu_graph(mxfp4_unpacked_linear)

    out_ref = linear(hpu_tensor)
    out1 = mxfp4_linear(hpu_tensor)
    out2 = mxfp4_unpacked_linear(hpu_tensor)
    print(
        f"diff ref vs mxfp4: max: {torch.abs(out_ref - out1).max()}, min: {torch.abs(out_ref - out1).min()}, mean: {torch.abs(out_ref - out1).mean()}"
    )
    print(
        f"diff ref vs mxfp4 unpacked: max: {torch.abs(out_ref - out2).max()}, min: {torch.abs(out_ref - out2).min()}, mean: {torch.abs(out_ref - out2).mean()}"
    )
    print(
        f"diff mxfp4 vs mxfp4 unpacked: max: {torch.abs(out1 - out2).max()}, min: {torch.abs(out1 - out2).min()}, mean: {torch.abs(out1 - out2).mean()}"
    )
    torch.hpu.synchronize()
    latency0 = time_fn(lambda: linear(hpu_tensor), args.bench_steps)
    latency1 = time_fn(lambda: mxfp4_linear(hpu_tensor), args.bench_steps)
    latency2 = time_fn(
        lambda: mxfp4_unpacked_linear(hpu_tensor), args.bench_steps
    )
    print(
        f"latency linear: {latency0:.6f} s, mxfp4_linear: {latency1:.6f} s, mxfp4_unpacked_linear: {latency2:.6f} s"
    )
    print(
        f"speed up mxfp4_linear: {latency0 / latency1:.2f}x, mxfp4_unpacked_linear: {latency0 / latency2:.2f}x"
    )
    print(
        f"speed up mxfp4_unpacked_linear vs mxfp4_linear: {latency1 / latency2:.2f}x"
    )

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
                out1 = mxfp4_linear(hpu_tensor)
                out2 = mxfp4_unpacked_linear(hpu_tensor)
                # result2 = dist.all_reduce(result2, op=dist.ReduceOp.SUM)
                ht.hpu.synchronize()
                profiler.step()
        profiler.stop()
        ht.hpu.synchronize()

