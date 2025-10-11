from typing import Dict, Union, Optional
import torch
import vllm.envs as envs

from compressed_tensors.quantization import FP4_E2M1_DATA
from torchao.prototype.mx_formats.mx_tensor import (
    _to_mx_rceil,
    get_fp_scale,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.constants import (
    BF16_EXP_BIAS,
    DTYPE_FP4_E2M1,
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F4_E2M1_MAX,
    F4_E2M1_MAX_POW2,
    F32_EXP_BIAS,
    F32_MIN_NORMAL,
)

SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_BF16, MBITS_BF16 = 8, 7
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2

FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]

# Module-level device tensor cache
_DEVICE_E2M1_TENSORS = {}

# Constants for FP4 values (E2M1 format)
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def get_e2m1_tensor(device):
    """Get device-specific E2M1 lookup tensor, creating it if needed."""
    device_str = str(device)
    if device_str not in _DEVICE_E2M1_TENSORS:
        _DEVICE_E2M1_TENSORS[device_str] = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    return _DEVICE_E2M1_TENSORS[device_str]


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    device = x.device

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = get_e2m1_tensor(x.device)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_diff_x = torch.abs(abs_x.unsqueeze(-1) - kE2M1)  # [m, n, 8]
    abs_indices = torch.argmin(abs_diff_x, dim=-1)  # [m, n]

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x).to(torch.long) << 3)

    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)

    # Handle odd length by padding if necessary
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])

    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)

    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    return packed.reshape(m, n // 2)


def unpack_fp4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four correspond to a
    consecutive fp4 value). The bits represent an index, which are mapped to an fp4
    value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8, f"expected uint8, got {a.dtype}"

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
    kE2M1 = get_e2m1_tensor(a.device)

    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
):
    """
    Based on the original implementation in torchao.prototype.mx_formats.mx_tensor.to_mx()

    Modifications:
    - Replaced [torchao.prototype.mx_formats.custom_cast.pack_uint4()]
      with [compressed_tensors.compressors.quantized_compressors.nvfp4_quantized.pack_fp4_to_uint8()]

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
    assert elem_dtype == "fp4_e2m1", "unsupported"

    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), 1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == DTYPE_FP4_E2M1:
        target_max_pow2 = F4_E2M1_MAX_POW2
        mbits = MBITS_F4_E2M1
        max_pos = F4_E2M1_MAX
    else:
        raise AssertionError("unsupported element dtype")

    if scaling_mode == ScaleCalculationMode.RCEIL:
        scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)
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
            max_abs[nan_mask] = torch.tensor(float("nan"), device=max_abs.device, dtype=max_abs.dtype)

        # Calculate the scale for different modes
        max_abs_int32 = (max_abs + eps).view(hp_int_dtype)
        extracted_pow2 = ((max_abs_int32 >> hp_mbits) & 0b11111111) - hp_exp_bias

        if scaling_mode in (ScaleCalculationMode.FLOOR, ScaleCalculationMode.EVEN):
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
        scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1)

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
        scale_fp32 = (scale_e8m0_biased.to(torch.int32) << MBITS_F32).view(torch.float32)

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

        if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not torch._dynamo.is_compiling():
            # As of 20250317, the Pytorch eager mode cast to `torch.float8_e4m3fn`
            # is unsaturated. This cast is saturated in triton. If we are compute bound,
            # we see a speedup if we remove this redundant clamp if we are compiling
            # to triton.
            # TODO(#1912): make the saturated cast work in eager mode and remove this
            # workaround.
            data_lp = torch.clamp(data_lp, min=-1 * max_pos, max=max_pos)

    # cast to target dtype
    if elem_dtype == DTYPE_FP4_E2M1:
        data_lp = data_lp.reshape(orig_shape)
        orig_shape = [*orig_shape[:-1], orig_shape[-1] // 2]
        data_lp = FP4_E2M1_DATA.cast_to_fp4(data_lp)
        data_lp = pack_fp4_to_uint8(data_lp)

    else:
        raise AssertionError("unsupported")

    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    return scale_e8m0_biased, data_lp


def to_dtype(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
    scale_dtype=None,
    return_scale=False,
):
    orig_shape = data_lp.shape
    last_dim = orig_shape[-1]
    data_lp = data_lp.reshape(-1, last_dim)
    result_shape = orig_shape[:-1] + (last_dim * 2,)
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
        # from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import unpack_fp4_from_uint8

        m, half_n = data_lp.shape
        n = half_n * 2
        data_hp = unpack_fp4_from_uint8(data_lp, m, n, dtype=target_dtype)

        # manually adjust shape to account for the unpacking
        # TODO(future PR): clean up the shape code and remove the hack
        # below
        # orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)
        orig_shape = result_shape
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


def to_dtype_cuda(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
):
    """
    Based on the original implementation in torchao.prototype.mx_formats.mx_tensor.dtype()

    Modifications:
    - Replaced [torchao.prototype.mx_formats.custom_cast.unpack_uint4()] with
      [compressed_tensors.compressors.quantized_compressors.nvfp4_quantized.unpack_fp4_from_uint8()]
    """
    orig_shape = data_lp.shape
    is_transposed = not data_lp.is_contiguous()
    # if the underlying data is transposed, convert to row major before
    # unpacking and unscaling
    if is_transposed:
        data_lp = data_lp.t()
        assert data_lp.is_contiguous()
        orig_shape = (orig_shape[1], orig_shape[0])

    if elem_dtype == DTYPE_FP4_E2M1:
        m, n = data_lp.shape
        f4_unpacked = unpack_fp4_from_uint8(data_lp, m, n * 2)
        data_hp = f4_unpacked.to(target_dtype)
        orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)
    else:
        raise AssertionError("unsupported")

    data_hp = data_hp.reshape(-1, block_size)
    s_fp = get_fp_scale(scale_e8m0).reshape(-1, 1).to(target_dtype)
    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(orig_shape)

    # if we converted to row-major before unscaling convert back
    if is_transposed:
        data_hp = data_hp.t()

    return data_hp


def run_mxfp4_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    group_size = 32
    if envs.VLLM_AR_MXFP4_DISABLE_INPUT_QDQ:
        x_dq = x
    else:
        # quantize input to (FP4 and interleaved block scale)
        input_scale, x_q = to_mx(
            data_hp=x,
            elem_dtype="fp4_e2m1",
            block_size=group_size,
            scaling_mode=ScaleCalculationMode.RCEIL,
        )

        # dequantize input
        x_dq = to_dtype(
            data_lp=x_q,
            scale_e8m0=input_scale,
            elem_dtype="fp4_e2m1",
            block_size=group_size,
            target_dtype=x.dtype,
        )

    # dequantize weight
    w_dq = to_dtype(
        data_lp=weight,
        scale_e8m0=weight_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    if bias is not None:
        out += bias
    return out


def dequant_mxfp4_to_fp8(data_lp, scale_e8m0):
    data_fp8, scale_float = to_dtype(
        data_lp=data_lp,
        scale_e8m0=scale_e8m0,
        elem_dtype="fp4_e2m1",
        block_size=32,
        # target_dtype=x.dtype,
        target_dtype=torch.float8_e4m3fn,
        # use_fp4_custom_triton_dequant_kernel=False,
        # pack_fp6=False,
        scale_dtype=torch.bfloat16,
        return_scale=True,
    )
    return data_fp8, scale_float


def mxfp4_fp8_weight_to_bf16(weight_fp8, scale_bf16):
    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, 32)
    assert weight_fp8.shape[0] == scale_bf16.shape[0], f"shape mismatch: {weight_fp8.shape} vs {scale_bf16.shape}"
    # TODO use cast_from_fp8_v2  ?
    dequant_weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_bf16
    dequant_weight_bf16 = dequant_weight_bf16.reshape(origin_shape)
    return dequant_weight_bf16


def mxfp4_gemm_with_unpacked_weight(x, weigth_fp8, weight_scale_bf16, bias=None):
    # from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (
    #     qdq_mxfp4,
    # )
    # from vllm.model_executor.layers.quantization.utils import mxfp4_utils
    # x = mxfp4_utils.quant_dequant_mxfp4(x)
    x = qdq_mxfp4(x)

    # dequantize weight
    w_dq = mxfp4_fp8_weight_to_bf16(weigth_fp8, weight_scale_bf16)
    # matmul
    out = torch.matmul(x, w_dq.t())
    if bias is not None:
        out += bias
    return out


# ==------------------
#


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

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def ue5m3(x: torch.Tensor) -> torch.Tensor:
    # NOTE: Assume that array values are in [0, 114688]. (14*2**13 = 114688)
    mask = x <= 2 ** (-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask

    x_1 = torch.round(x_1 / 2 ** (-17)) * (2 ** (-17))

    e = torch.floor(torch.log2(x_2)) - 3
    s = 2**e
    x_2 = torch.round(x_2 / s) * s

    return x_1 * mask + x_2 * (~mask)


FP8_E4M3_MAX = 240.0
FP8_E4M3_MAX = 448.0


def fp4_121_scaled_even_rounding(
    x: torch.Tensor, stochastic_rounding: bool = False, scale_format: str = "e8m0"
) -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    assert scale_format == "e8m0", f"Unsupported scale format: {scale_format}"
    if scale_format == "e8m0":
        # scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
        amax_x = x_abs.max(dim=-1, keepdim=True)[0]
        scale_tmp = torch.floor(torch.log2(amax_x)) - 2.0
        scale_clamp = torch.clamp(scale_tmp, min=-127, max=127)
        scale = torch.pow(2.0, scale_clamp)
    else:
        raise NotImplementedError(f"Unsupported scale format: {scale_format}")

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)

    x_fp4_abs = fp4_121_positive(x_abs / scale, stochastic_rounding) * scale
    return sign * x_fp4_abs


fp4_121_scaled = fp4_121_scaled_even_rounding


# https://github.com/Anonymous1252022/fp4-all-the-way/blob/main/experimental/fp4.py
def qdq_mxfp4(
    x: torch.Tensor,
    stochastic_rounding: bool = False,
    dim: int = -1,
    format: str = "fp4_e2m1",
    block_size: int = 32,
    scale_format: str = "e8m0",
    grid: bool = False,
) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(-1, block_size)

    x = fp4_121_scaled(x, stochastic_rounding, scale_format)

    x = x.reshape(shape)

    return x
