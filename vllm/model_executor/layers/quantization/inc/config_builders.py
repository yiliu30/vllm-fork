# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

from .resolver import INCLayerConfig


def build_gptq_config(layer_cfg: INCLayerConfig) -> GPTQConfig:
    return GPTQConfig(
        weight_bits=layer_cfg.bits,
        group_size=layer_cfg.group_size,
        desc_act=False,
        lm_head_quantized=False,
        dynamic={},
    )


def build_gptq_marlin_config(layer_cfg: INCLayerConfig) -> GPTQMarlinConfig:
    return GPTQMarlinConfig(
        weight_bits=layer_cfg.bits,
        group_size=layer_cfg.group_size,
        desc_act=False,
        is_sym=layer_cfg.sym,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )


def build_awq_config(layer_cfg: INCLayerConfig) -> AWQConfig:
    return AWQConfig(
        weight_bits=layer_cfg.bits,
        group_size=layer_cfg.group_size,
        zero_point=not layer_cfg.sym,
    )


def build_awq_marlin_config(layer_cfg: INCLayerConfig) -> AWQMarlinConfig:
    return AWQMarlinConfig(
        weight_bits=layer_cfg.bits,
        group_size=layer_cfg.group_size,
        zero_point=not layer_cfg.sym,
        lm_head_quantized=False,
        modules_to_not_convert=[],
        full_config={},
    )


def build_moe_wna16_gptq_dict(layer_cfg: INCLayerConfig) -> dict[str, Any]:
    return {
        "quant_method": "gptq",
        "bits": layer_cfg.bits,
        "group_size": layer_cfg.group_size,
        "sym": layer_cfg.sym,
        "lm_head": False,
    }


def build_moe_wna16_awq_dict(layer_cfg: INCLayerConfig) -> dict[str, Any]:
    return {
        "quant_method": "awq",
        "bits": layer_cfg.bits,
        "group_size": layer_cfg.group_size,
        "zero_point": not layer_cfg.sym,
        "lm_head": False,
    }
