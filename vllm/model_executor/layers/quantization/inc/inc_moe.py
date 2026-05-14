# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    UnquantizedFusedMoEMethod,
)

from .schemes.factory import resolve_scheme

if TYPE_CHECKING:
    import torch

    from .inc import INCConfig


class INCMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "INCConfig",
        layer: "torch.nn.Module",
        layer_name: str,
    ) -> "FusedMoEMethodBase":
        layer_config = quant_config.resolver.resolve(layer, layer_name)
        if not layer_config.quantized:
            return UnquantizedFusedMoEMethod(layer.moe_config)

        scheme = resolve_scheme(layer_config)
        method = scheme.get_moe_method(
            quant_config,
            layer,
            layer_name,
            layer_config,
        )
        if method is None:
            raise NotImplementedError(
                "INC scheme "
                f"{scheme.__class__.__name__} does not support FusedMoE "
                f"for layer config: {layer_config}"
            )
        return method
