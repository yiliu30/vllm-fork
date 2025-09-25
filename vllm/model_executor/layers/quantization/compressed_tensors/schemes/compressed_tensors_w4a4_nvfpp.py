# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)


from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform
from compressed_tensors.quantization.utils.nvfpp_helper import (
    unpack_weight,
    qdq_nvfpp,
)
from vllm.model_executor.layers.quantization.compressed_tensors.nvfpp_utils import (
    dq_nvfpp4,
    run_nvfpp_emulations,
)

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
