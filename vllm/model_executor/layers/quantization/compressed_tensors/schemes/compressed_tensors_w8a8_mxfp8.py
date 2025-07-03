# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
import vllm.envs as envs
from vllm.logger import init_logger
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

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8MXFp8"]


def get_fp_scale(scale_e8m0):
    # https://github.com/pytorch/ao/blob/994a4ba6c869854fcaa6ca7e118fcbd75e6c28cc/torchao/prototype/mx_formats/mx_tensor.py#L337
    E8M0_EXPONENT_BIAS = 127

    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    # TODO(later): it would be nice if there was a way to do the 2^x operation
    # in PyTorch without creating a tensor of twos
    # two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # pow(two, s_offset) can be out of range of floating point formats.
    # TODO(later): handle this for float16 if we decide to support float16
    # scales.
    # s_fp = torch.pow(two, s_offset)
    # !!!!NOTE Critical: fixed the OoM issue when using HPU graph
    s_fp = torch.pow(2.0, s_offset.to(torch.float))
    
    return s_fp


def dequant_mx_fp8(weight_fp8, scale_e8m0, block_size):
    if envs.VLLM_DISABLE_INPUT_QDQ:
        assert scale_e8m0.dtype in [
            torch.float32,
            torch.bfloat16,
        ], f"Unsupported scale_e8m0 dtype: {scale_e8m0.dtype}"
    if scale_e8m0.dtype == torch.uint8:
        scale_float = get_fp_scale(scale_e8m0)
    else:
        scale_float = scale_e8m0
        
    weight_bf16 = weight_fp8.to(torch.bfloat16)
    weight_original_shape = weight_bf16.shape
    weight_bf16 = weight_bf16.reshape(-1, block_size)
    scale_float = scale_float.reshape(-1, 1)
    dequant_weight = weight_bf16 * scale_float
    dequant_weight = dequant_weight.reshape(weight_original_shape)
    return dequant_weight

def quant_mx_fp8(tensor):
    from .torchao_patch import (
        to_mx,
        ScaleCalculationMode,
    )

    scale_e8m0_biased, data_lp = to_mx(
        data_hp=tensor,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode.RCEIL,
        pack_fp6=False,
    )
    return scale_e8m0_biased, data_lp



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
        weight_scale_data = layer.weight_scale.data
        weight_scale_data_float = get_fp_scale(weight_scale_data)
        del layer.weight_scale
        layer.weight_scale = torch.nn.Parameter(
            weight_scale_data_float,
            requires_grad=False,
        )
        logger.warning_once(f"Pre-processed weight scale for {getattr(layer, 'prefix', None)  }" )

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
        if not envs.VLLM_DISABLE_INPUT_QDQ:
            x_scale, x_quant = quant_mx_fp8(x)
            dequant_x = dequant_mx_fp8(
                weight_fp8=x_quant,
                scale_e8m0=x_scale,
                block_size=self.group_size,
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
