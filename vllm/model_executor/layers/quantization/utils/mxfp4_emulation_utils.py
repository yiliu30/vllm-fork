import torch
import vllm.envs as envs

import torchao
from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import unpack_fp4_from_uint8, pack_fp4_to_uint8
from torchao.prototype.mx_formats.mx_tensor import (
    get_fp_scale,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.custom_cast import (
    f32_to_f4_unpacked, 
    f4_unpacked_to_f32, 
    triton_f4_to_scaled_bf16, 
    unpack_uint4,
    pack_uint4
)
from torchao.prototype.mx_formats.constants import DTYPE_FP4

SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_BF16, MBITS_BF16 = 8, 7
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


def to_dtype(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
    use_fp4_custom_triton_dequant_kernel,
    pack_fp6,
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

    if elem_dtype == DTYPE_FP4:
        if use_fp4_custom_triton_dequant_kernel:
            data_hp_rescaled = triton_f4_to_scaled_bf16(
                data_lp,
                scale_e8m0,
                block_size,
            )
            if is_transposed:
                data_hp_rescaled = data_hp_rescaled.t()
            return data_hp_rescaled.to(target_dtype)
        else:
            # fp4
            m, n = data_lp.shape
            f4_unpacked = unpack_fp4_from_uint8(data_lp, m, n*2)
            # f4_unpacked = unpack_uint4(data_lp)
            # for now we only have a cast to f32
            # TODO(future PR): add cast directly to bf16
            # f32 = f4_unpacked_to_f32(f4_unpacked)
            data_hp = f4_unpacked.to(target_dtype)
            # manually adjust shape to account for the unpacking
            # TODO(future PR): clean up the shape code and remove the hack
            # below
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
        input_scale, x_q = torchao.prototype.mx_formats.mx_tensor.to_mx(
            data_hp=x,
            elem_dtype="fp4_e2m1",
            block_size=group_size,
            scaling_mode=ScaleCalculationMode.RCEIL,
            pack_fp6=False,
        )

        # dequantize input
        x_dq = torchao.prototype.mx_formats.mx_tensor.to_dtype(
            data_lp=x_q,
            scale_e8m0=input_scale,
            elem_dtype="fp4_e2m1",
            block_size=group_size,
            target_dtype=x.dtype,
            use_fp4_custom_triton_dequant_kernel=False,
            pack_fp6=False,
        )

    # dequantize weight
    w_dq = to_dtype(
        data_lp=weight,
        scale_e8m0=weight_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
        use_fp4_custom_triton_dequant_kernel=False,
        pack_fp6=False,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out
