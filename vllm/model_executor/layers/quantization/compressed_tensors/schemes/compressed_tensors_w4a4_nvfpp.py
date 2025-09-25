# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    cutlass_scaled_mm_supports_fp4,
    scaled_fp4_quant,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

# from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (  # noqa: E501
#     run_nvfpp_emulations)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4NVFPP4"]


class CompressedTensorsW4A4NVFPP4(CompressedTensorsScheme):
    def __init__(self, group_size: int = 32):
        self.group_size = group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        # weight_global_scale = PerTensorScaleParameter(
        #     data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        #     weight_loader=weight_loader)
        # layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                # dtype=torch.uint8,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        # input_global_scale = PerTensorScaleParameter(
        #     data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        #     weight_loader=weight_loader)
        # layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        if envs.VLLM_PRE_UNPACK_FP4_WEIGHTS:
            weight_unpacked = unpack_weight(layer.weight_packed.data)
            del layer.weight_packed
            layer.register_parameter(
                "weight_packed",
                Parameter(weight_unpacked, requires_grad=False),
            )
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # breakpoint()
        if envs.VLLM_USE_MXFP4_CT_EMULATIONS:
            out = run_nvfpp_emulations(
                x=x,
                weight=layer.weight_packed,
                weight_scale=layer.weight_scale,
                group_size=self.group_size,
            )
            if bias is not None:
                out = out + bias
            return out

        # output_dtype = x.dtype
        # output_shape = [x.shape[0], layer.weight.shape[0]]

        # # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        # x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_global_scale)

        # out = cutlass_scaled_fp4_mm(x_fp4, layer.weight, x_blockscale,
        #                             layer.weight_scale_swizzled,
        #                             1 / layer.alpha, output_dtype)
        # if bias is not None:
        #     out = out + bias
        # return out.view(*output_shape)

from compressed_tensors.quantization.utils.nvfpp_helper import unpack_weight

def dq_nvfpp4(
    data_lp,
    scale_uint8,
    block_size=32,
    target_dtype=torch.float32,
):
    from compressed_tensors.quantization.utils.nvfpp_helper import (
        float_to_nvfpp,
        nvfpp_to_float,
    )

    fp_scale = nvfpp_to_float(scale_uint8).to(target_dtype)
    if envs.VLLM_PRE_UNPACK_FP4_WEIGHTS:
        data_lp_unpacked = data_lp.data.to(target_dtype)
    else:
        data_lp_unpacked = unpack_weight(data_lp.data, dtype=target_dtype)
    orig_shape = data_lp_unpacked.shape
    dequant_value = data_lp_unpacked.reshape(-1, block_size)
    fp_scale = fp_scale.reshape(-1, 1)
    dequant_value = dequant_value * fp_scale
    return dequant_value.reshape(orig_shape)


def run_nvfpp_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size=32,
):
    if envs.VLLM_DISABLE_INPUT_QDQ:
        x_dq = x
    else:
        from compressed_tensors.quantization.utils.nvfpp_helper import qdq_nvfpp

        x_dq = qdq_nvfpp(x, group_size=group_size)
        x_dq = x_dq.to(x.dtype)

    # dequantize weight
    w_dq = dq_nvfpp4(
        data_lp=weight,
        scale_uint8=weight_scale,
        block_size=group_size,
        target_dtype=x.dtype,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out
