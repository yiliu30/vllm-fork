# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.platforms import current_platform

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig


class INCWNA16MoEScheme:
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.layer_config = layer_config

    def get_method(self, layer: "torch.nn.Module"):
        if current_platform.is_cpu():
            return self._build_cpu_method(layer)
        if self.layer_config.is_gptq:
            return self._build_gptq_method(layer)
        if self.layer_config.is_awq:
            return self._build_awq_method(layer)
        raise NotImplementedError(
            f"WNA16 MoE does not support config {self.layer_config}"
        )

    def _build_cpu_method(self, layer: "torch.nn.Module"):
        from vllm.model_executor.layers.fused_moe import (
            UnquantizedFusedMoEMethod,
        )

        return UnquantizedFusedMoEMethod(layer.moe_config)

    def _build_gptq_method(self, layer: "torch.nn.Module"):
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQMoEMethod,
        )
        from vllm.model_executor.layers.quantization.moe_wna16 import (
            MoeWNA16Config,
            MoeWNA16Method,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_moe_marlin_supports_layer,
        )

        # AutoGPTQMoEMethod selects its fused-MoE backend through the WNA16
        # oracle: Marlin on CUDA, XPUExpertsWNA16 on XPU. Gate only on the
        # layer-shape check so the XPU path remains reachable.
        use_marlin = (self.layer_config.bits, self.layer_config.sym) in {
            (4, True),
            (8, True),
        } and check_moe_marlin_supports_layer(
            layer,
            self.layer_config.group_size,
        )

        if use_marlin:
            return AutoGPTQMoEMethod(
                AutoGPTQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                ),
                layer.moe_config,
            )

        moe_config = MoeWNA16Config.from_config(
            {
                "quant_method": "gptq",
                "bits": self.layer_config.bits,
                "group_size": self.layer_config.group_size,
                "sym": self.layer_config.sym,
                "lm_head": False,
            }
        )
        return MoeWNA16Method(moe_config, layer.moe_config)

    def _build_awq_method(self, layer: "torch.nn.Module"):
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQMoEMethod
        from vllm.model_executor.layers.quantization.moe_wna16 import (
            MoeWNA16Config,
            MoeWNA16Method,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_moe_marlin_supports_layer,
        )

        use_marlin = self.layer_config.bits in (
            4,
            8,
        ) and check_moe_marlin_supports_layer(
            layer,
            self.layer_config.group_size,
        )

        if use_marlin:
            return AutoAWQMoEMethod(
                AutoAWQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    zero_point=not self.layer_config.sym,
                    lm_head_quantized=False,
                    modules_to_not_convert=[],
                    full_config={},
                ),
                layer.moe_config,
            )

        moe_config = MoeWNA16Config.from_config(
            {
                "quant_method": "awq",
                "bits": self.layer_config.bits,
                "group_size": self.layer_config.group_size,
                "zero_point": not self.layer_config.sym,
                "lm_head": False,
            }
        )
        return MoeWNA16Method(moe_config, layer.moe_config)