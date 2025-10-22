from typing import Union, Optional
import torch

from compressed_tensors.quantization import FP4_E2M1_DATA
from .utils import get_fp_scale, _to_mx_rceil

BF16_EXP_BIAS = 127
DTYPE_FP4_E2M1 = "fp4_e2m1"
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

F4_E2M1_MAX = 6.0
F4_E2M1_MAX_POW2 = 2  # 4
F32_EXP_BIAS = 127

F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
FP8_E4M3_MAX = 448.0

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
        _DEVICE_E2M1_TENSORS[device_str] = torch.tensor(
            _E2M1_VALUES, dtype=torch.float32, device=device
        )
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


def to_mxfp4_rceil(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
):


    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.numel() % block_size == 0, f"data size must be multiple of block_size={block_size}"
    assert data_hp.is_contiguous(), f"data must be contiguous, got {data_hp.stride()}"

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

    max_pos = F4_E2M1_MAX
    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)


    data_lp = data_lp.reshape(orig_shape)
    orig_shape = [*orig_shape[:-1], orig_shape[-1] // 2]
    data_lp = FP4_E2M1_DATA.cast_to_fp4(data_lp)
    data_lp = pack_fp4_to_uint8(data_lp)


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
    assert data_lp.is_contiguous, f"Data must be contiguous, got {data_lp.stride()}"

    assert elem_dtype == "fp4_e2m1", f"Expected 'fp4_e2m1', got {elem_dtype}"

    m, half_n = data_lp.shape
    n = half_n * 2
    data_hp = unpack_fp4_from_uint8(data_lp, m, n, dtype=target_dtype)

    data_hp = data_hp.reshape(-1, block_size)

    if scale_dtype is None:
        scale_dtype = target_dtype
    s_fp = get_fp_scale(scale_e8m0).reshape(-1, 1).to(scale_dtype)
    if return_scale:
        return data_hp.reshape(result_shape), s_fp

    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(result_shape)

    return data_hp


def run_mxfp4_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    group_size = 32
    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mxfp4_rceil(
        data_hp=x,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
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
        target_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.bfloat16,
        return_scale=True,
    )
    return data_fp8, scale_float


def mxfp4_fp8_weight_to_bf16(weight_fp8, scale_bf16):
    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, 32)
    assert weight_fp8.shape[0] == scale_bf16.shape[0], (
        f"shape mismatch: {weight_fp8.shape} vs {scale_bf16.shape}"
    )
    dequant_weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_bf16
    dequant_weight_bf16 = dequant_weight_bf16.reshape(origin_shape)
    return dequant_weight_bf16


def mxfp4_gemm_with_unpacked_weight(x, weigth_fp8, weight_scale_bf16, bias=None):
    x = qdq_mxfp4(x)

    # dequantize weight
    w_dq = mxfp4_fp8_weight_to_bf16(weigth_fp8, weight_scale_bf16)
    # matmul
    out = torch.matmul(x, w_dq.t())
    if bias is not None:
        out += bias
    return out


def fp4_121_positive(x: torch.Tensor) -> torch.Tensor:
    step1 = torch.round(2.0 * x) / 2.0
    step2 = torch.round(x)
    step3 = 2.0 * torch.round(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def fp4_121_scaled_even_rounding(x: torch.Tensor) -> torch.Tensor:
    sign = x.sign()
    x_abs = x.abs()
    amax_x = x_abs.max(dim=-1, keepdim=True)[0]
    scale_tmp = torch.floor(torch.log2(amax_x)) - 2.0
    scale_clamp = torch.clamp(scale_tmp, min=-127, max=127)
    scale = torch.pow(2.0, scale_clamp)

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)

    x_fp4_abs = fp4_121_positive(x_abs / scale) * scale
    return sign * x_fp4_abs



# https://github.com/Anonymous1252022/fp4-all-the-way/blob/main/experimental/fp4.py
def qdq_mxfp4(
    x: torch.Tensor,
) -> torch.Tensor:
    block_size = 32
    shape = x.shape
    x = x.reshape(-1, block_size)

    x = fp4_121_scaled_even_rounding(x)

    x = x.reshape(shape)

    return x
