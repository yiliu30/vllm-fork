# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from auto_round.schemes import QuantizationScheme

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

from .quant_methods import AutoRoundMoEMethod, AutoRoundQuantLinearMethod

logger = init_logger(__name__)


def need_skip_attn(prefix: str):
    return "self_attn" in prefix


class AutoRoundExtensionConfig(AutoRoundConfig):
    SUPPORTED_DTYPES = AutoRoundConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = AutoRoundConfig.SUPPORTED_FORMATS.union(
        {"auto_round:llm_compressor"}
    )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            # FIXME: handle linear as well
            logger.warning_once(f"Skip quantizing layer: {prefix}")
            return quant_method
            # if not need_skip_attn(prefix):
            quant_method = AutoRoundQuantLinearMethod(self, scheme=self.quant_scheme)

        elif isinstance(layer, FusedMoE):
            quant_method = AutoRoundMoEMethod.get_moe_method(self, layer)
        else:
            quant_method = super().get_quant_method(layer, prefix)
        logger.debug(f"Apply {quant_method.__class__.__name__} to {prefix}")
        return quant_method

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AutoRoundConfig:
        ar_config = super().from_config(config)

        # FIXME: (Yi) parse the per-layer quant scheme
        def create_quant_scheme(config):
            quant_scheme_attrs = QuantizationScheme.get_attributes()
            filter_config = {
                key: value for key, value in config.items() if key in quant_scheme_attrs
            }
            quant_scheme = QuantizationScheme.from_dict(filter_config)
            return quant_scheme

        ar_config.quant_scheme = create_quant_scheme(config)
        return ar_config
