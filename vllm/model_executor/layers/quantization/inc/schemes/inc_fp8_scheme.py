# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig


class INCFp8Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_fp8

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer

        from .inc_fp8_linear import INCFp8LinearScheme

        weight_block_size = (
            layer_config.group_size
            if isinstance(layer_config.group_size, tuple)
            else None
        )

        return INCLinearMethod(
            INCFp8LinearScheme(
                prefix=prefix,
                weight_block_size=weight_block_size,
            )
        )

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix

        if layer_config.is_fp8_block:
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8Config,
                Fp8MoEMethod,
            )

            assert isinstance(layer_config.group_size, tuple)
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=list(layer_config.group_size),
            )
            return Fp8MoEMethod(quant_config, layer)

        from .inc_fp8_moe import INCFp8MoEScheme

        return INCFp8MoEScheme(layer_config).get_method(layer)
