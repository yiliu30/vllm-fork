# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsW8A8MXFp8"]

from .quant_dquant_utils import get_fp_scale, dequant_mx_fp8, quant_mx_fp8


class CompressedTensorsW8A8MXFp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        # self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        # FIXME: (Yi) correct the minimum capability
        return 80

    def process_weights_after_loading(self, layer) -> None:
        return

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        # maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.strategy == QuantizationStrategy.TENSOR_GROUP:
            # Per Group Weight Scale
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition // self.group_size,
                    dtype=torch.uint8, # E8M0 for MXFP8 scale
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )   
        else:
            raise NotImplementedError(f"Strategy {self.strategy} is not supported for W8A8-MXFp8")

        # min requirement for fp8 kernels
        # weight_scale[:] = torch.finfo(torch.float32).min
        # weight_scale.fill_(torch.finfo(torch.float32).min)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # dequant weight
        weight = layer.weight
        weight_scale = layer.weight_scale
        dequnat_weight = dequant_mx_fp8(
            weight_fp8=weight.data,
            scale_e8m0=weight_scale.data,
            block_size=self.group_size
            )
        dequnat_weight = dequnat_weight.to(x.dtype)
        # q-dq input
        x_scale, x_quant = quant_mx_fp8(x)
        dequant_x = dequant_mx_fp8(
            weight_fp8=x_quant,
            scale_e8m0=x_scale,
            block_size=self.group_size
        )
        x = dequant_x.to(x.dtype)
        out = x @ dequnat_weight.t()
        return out.to(x.dtype) + (bias if bias is not None else 0)
        
        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)
