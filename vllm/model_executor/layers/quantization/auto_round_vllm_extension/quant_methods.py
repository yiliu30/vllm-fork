# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
from auto_round.schemes import QuantizationScheme

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.linear import LinearMethodBase

logger = init_logger(__name__)


QLINEAR_METHODS_DISPATCH_TABLE = {}
QMOE_METHODS_DISPATCH_TABLE = {}


class AutoRoundQuantLinearMethod(LinearMethodBase):
    def is_mxfp8(self, quant_scheme: QuantizationScheme):
        return True

    def __init__(self, config, scheme=None):
        self.config = config
        self.scheme = scheme
        if self.is_mxfp8(self.scheme):
            from .linear_impl_mxfp8 import AutoRoundMXFP8LinearImpl

            self.impl = AutoRoundMXFP8LinearImpl(
                strategy="TENSOR_GROUP", is_static_input_scheme=True
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return cls.impl.get_min_capability()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        return self.impl.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return self.impl.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return self.impl.apply_weights(layer, x, bias=bias)


class AutoRoundMoEMethod(FusedMoEMethodBase):
    def __init_(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: "AutoRoundConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "AutoRoundMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        # weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        # input_quant = quant_config.target_scheme_map["Linear"].get(
        #     "input_activations")
        weight_quant = None
        input_quant = None

        # FIXME: @yiliu30: temporarily only support MXFP8
        # if 0 and quant_config._is_mxfp8_w8a8(weight_quant, input_quant):
        #     from .moe_impl_mxfp8 import AutoRoundMoEMethodMXFP8

        #     impl = AutoRoundMoEMethodMXFP8(quant_config, layer.moe_config)
        #     return impl
        if 1 or quant_config._is_mxfp4_w4a8(weight_quant, input_quant):
            from .moe_impl_mxfp4 import AutoRoundMoEMethodMXFp4Impl
            impl = AutoRoundMoEMethodMXFp4Impl(quant_config, layer.moe_config)
            return impl
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        return self.impl.get_fused_moe_quant_config(layer)
