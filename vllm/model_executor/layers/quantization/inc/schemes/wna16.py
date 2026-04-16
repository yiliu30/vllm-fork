# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .base import INCScheme

if TYPE_CHECKING:
    import torch

    from ..inc import INCConfig
    from ..resolver import INCLayerConfig


class INCWna16Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_wna16_int

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix
        if current_platform.is_xpu():
            if layer_config.bits == 4 and layer_config.sym:
                from .xpu_w4a16_linear import INCXPUW4A16LinearScheme

                return INCLinearMethod(INCXPUW4A16LinearScheme(layer_config))
            raise NotImplementedError(
                f"INC on XPU: unsupported config {layer_config}"
            )

        if current_platform.is_cpu() and layer_config.is_gptq:
            if layer_config.bits == 4 and layer_config.sym:
                from .wna16_linear import INCWNA16LinearScheme

                return INCLinearMethod(INCWNA16LinearScheme(layer_config))
            raise NotImplementedError(
                f"INC on CPU: unsupported config {layer_config}"
            )

        from .wna16_linear import INCWNA16LinearScheme

        return INCLinearMethod(INCWNA16LinearScheme(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix, layer_config
        raise NotImplementedError("INCWna16Scheme MoE path is not wired yet.")
