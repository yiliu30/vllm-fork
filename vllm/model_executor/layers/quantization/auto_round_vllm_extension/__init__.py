# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

from .auto_round_ext import AutoRoundExtensionConfig

# ==---------------------------------------------------------------------------==
# Apply the extension
# ==---------------------------------------------------------------------------==


def apply():
    import vllm.model_executor.layers.quantization.auto_round as auto_round_module

    auto_round_module.AutoRoundConfig = AutoRoundExtensionConfig
    from .envs_ext import extra_environment_variables
