# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (  # noqa: E501
    run_mxfp4_emulations,
    run_mxfp4_emulations_unpacked_weight,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4MXFp4"]


class CompressedTensorsW4A4MXFp4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_USE_MXFP4_CT_EMULATIONS:
            return 80
        return 100

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

    def process_weights_after_loading(self, layer) -> None:
        if envs.VLLM_MXFP4_PREUNPACK_WEIGHTS:
            logger.debug(f"start processing weights for {getattr(layer, 'prefix', 'unknown')}")
            from torchao.prototype.mx_formats.mx_tensor import (
                to_mx,
                to_dtype,
                ScaleCalculationMode,
            )
            from vllm.model_executor.layers.quantization.utils.mxfp4_qdq import dequant_mxfp4_to_fp8

            weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(
                data_lp=layer.weight_packed.data,
                scale_e8m0=layer.weight_scale.data,
            )
            del layer.weight_packed
            del layer.weight_scale
            layer.register_parameter(
                "weight_unpacked",
                torch.nn.Parameter(
                    weight_fp8,
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "weight_scale",
                torch.nn.Parameter(
                    scale_bf16,
                    requires_grad=False,
                ),
            )
                
            torch.hpu.synchronize()
        else:
            pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if envs.VLLM_USE_MXFP4_CT_EMULATIONS:
            if envs.VLLM_MXFP4_PREUNPACK_WEIGHTS:
                from vllm.model_executor.layers.quantization.utils.mxfp4_qdq import (
                    mxfp4_gemm_with_unpacked_weight,
                )

                out = mxfp4_gemm_with_unpacked_weight(
                    x=x,
                    weigth_fp8=layer.weight_unpacked.data,
                    weight_scale_bf16=layer.weight_scale.data,
                )
                return out
            out = run_mxfp4_emulations(
                x=x, weight=layer.weight_packed, weight_scale=layer.weight_scale
            )
            if bias is not None:
                out = out + bias
            return out
        else:
            raise NotImplementedError(
                "CompressedTensorsW4A4MXFp4 is not supported on this platform or configuration."
            )
