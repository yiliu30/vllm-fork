# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

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
        del config, layer, prefix, layer_config
        raise NotImplementedError("INCWna16Scheme linear path is not wired yet.")

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix, layer_config
        raise NotImplementedError("INCWna16Scheme MoE path is not wired yet.")
