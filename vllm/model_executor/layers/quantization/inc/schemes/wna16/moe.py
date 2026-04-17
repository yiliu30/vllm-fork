# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.scalar_type import scalar_types

from ...config_builders import (
    build_awq_marlin_config,
    build_gptq_marlin_config,
    build_moe_wna16_awq_dict,
    build_moe_wna16_gptq_dict,
)

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase

    from ...resolver import INCLayerConfig


def resolve_wna16_moe(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
) -> "FusedMoEMethodBase":
    if layer_config.is_gptq:
        return _resolve_gptq_moe(layer, layer_config)
    if layer_config.is_awq:
        return _resolve_awq_moe(layer, layer_config)
    raise NotImplementedError(
        f"WNA16 MoE does not support config {layer_config}"
    )


def _resolve_gptq_moe(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
) -> "FusedMoEMethodBase":
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinMoEMethod,
    )
    from vllm.model_executor.layers.quantization.moe_wna16 import (
        MoeWNA16Config,
        MoeWNA16Method,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )

    gptq_type_map = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }
    use_marlin = (layer_config.bits, layer_config.sym) in gptq_type_map
    if use_marlin:
        use_marlin = check_marlin_supported(
            gptq_type_map[(layer_config.bits, layer_config.sym)],
            layer_config.group_size,
            has_zp=not layer_config.sym,
        ) and check_moe_marlin_supports_layer(layer, layer_config.group_size)

    if use_marlin:
        return GPTQMarlinMoEMethod(
            build_gptq_marlin_config(layer_config),
            layer.moe_config,
        )

    moe_config = MoeWNA16Config.from_config(
        build_moe_wna16_gptq_dict(layer_config)
    )
    return MoeWNA16Method(moe_config, layer.moe_config)


def _resolve_awq_moe(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
) -> "FusedMoEMethodBase":
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinMoEMethod
    from vllm.model_executor.layers.quantization.moe_wna16 import (
        MoeWNA16Config,
        MoeWNA16Method,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )

    awq_type_map = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }
    use_marlin = layer_config.bits in awq_type_map
    if use_marlin:
        use_marlin = check_marlin_supported(
            awq_type_map[layer_config.bits],
            layer_config.group_size,
            not layer_config.sym,
        ) and check_moe_marlin_supports_layer(layer, layer_config.group_size)

    if use_marlin:
        return AWQMarlinMoEMethod(
            build_awq_marlin_config(layer_config),
            layer.moe_config,
        )

    moe_config = MoeWNA16Config.from_config(
        build_moe_wna16_awq_dict(layer_config)
    )
    return MoeWNA16Method(moe_config, layer.moe_config)
