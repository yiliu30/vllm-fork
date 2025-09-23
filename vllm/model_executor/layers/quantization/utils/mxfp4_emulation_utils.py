from typing import Dict, Union
import torch
import vllm.envs as envs

from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import unpack_fp4_from_uint8, pack_fp4_to_uint8
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
            max_abs[nan_mask] = torch.tensor(
                float("nan"), device=max_abs.device, dtype=max_abs.dtype
            )

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
        scale_e8m0_unbiased = torch.clamp(
            scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1
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
        f4_unpacked = unpack_fp4_from_uint8(data_lp, m, n*2)
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
):
    group_size = 32
    if envs.VLLM_DISABLE_INPUT_QDQ:
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
    return out
