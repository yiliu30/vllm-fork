# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from ..inc_linear import INCLinearMethod
from .base import INCScheme

if TYPE_CHECKING:
    import torch

    from ..inc import INCConfig
    from ..resolver import INCLayerConfig


class INCFp8Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_fp8_block

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer
        from .fp8_linear import INCFp8LinearScheme

        assert isinstance(layer_config.group_size, tuple)
        return INCLinearMethod(
            INCFp8LinearScheme(
                prefix=prefix,
                weight_block_size=layer_config.group_size,
            )
        )
