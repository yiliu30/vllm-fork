import torch
import vllm.envs as envs
from compressed_tensors.quantization.utils.nvfpp_helper import (
    unpack_weight,
    qdq_nvfpp,
)

from compressed_tensors.quantization.utils.nvfpp_helper import (
    float_to_nvfpp,
    nvfpp_to_float,
)


def dq_nvfpp4(
    data_lp,
    scale_uint8,
    block_size=32,
    target_dtype=torch.float32,
    data_packed=False,
):
    fp_scale = nvfpp_to_float(scale_uint8).to(target_dtype)
    # FIXME: refine the logic of `data_packed` and `envs.VLLM_PRE_UNPACK_FP4_WEIGHTS`
    if envs.VLLM_PRE_UNPACK_FP4_WEIGHTS:
        data_lp_unpacked = data_lp.data.to(target_dtype)
    else:
        raise AssertionError("data_packed should be True if not pre-unpacked")
    orig_shape = data_lp_unpacked.shape
    dequant_value = data_lp_unpacked.reshape(-1, block_size)
    fp_scale = fp_scale.reshape(-1, 1)
    dequant_value = dequant_value * fp_scale
    return dequant_value.reshape(orig_shape)


def run_nvfpp_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size,
    weight_packed=False,
):
    # !!!NOTE: This func not handle the bias
    if envs.VLLM_DISABLE_INPUT_QDQ:
        x_dq = x
    else:
        x_dq = qdq_nvfpp(x, group_size=group_size)
        x_dq = x_dq.to(x.dtype)

    # dequantize weight
    w_dq = dq_nvfpp4(
        data_lp=weight,
        scale_uint8=weight_scale,
        block_size=group_size,
        target_dtype=x.dtype,
        data_packed=weight_packed,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out
