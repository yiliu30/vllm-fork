# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase

    from ..resolver import INCLayerConfig
    from .base import INCLinearScheme


def resolve_linear_scheme(layer_config: "INCLayerConfig") -> "INCLinearScheme":
    if layer_config.is_wna16_int:
        from .wna16_linear import resolve_wna16_linear

        return resolve_wna16_linear(layer_config)

    raise NotImplementedError(
        f"No INC linear scheme found for layer config: {layer_config}"
    )


def resolve_moe_method(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
) -> "FusedMoEMethodBase":
    if layer_config.is_wna16_int:
        from .wna16_moe import resolve_wna16_moe

        return resolve_wna16_moe(layer, layer_config)

    raise NotImplementedError(
        f"No INC MoE method found for layer config: {layer_config}"
    )
