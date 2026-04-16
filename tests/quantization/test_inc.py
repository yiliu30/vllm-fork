# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.inc.config_builders import (
    build_awq_config,
    build_awq_marlin_config,
    build_gptq_config,
    build_gptq_marlin_config,
    build_moe_wna16_awq_dict,
    build_moe_wna16_gptq_dict,
)
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.resolver import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.schemes import (
    INCWna16Scheme,
    resolve_scheme,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


class DummyLayer:
    pass


class DummyFusedMoE:
    pass


def make_config(**overrides) -> INCConfig:
    kwargs = {
        "weight_bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
        "block_name_to_quantize": None,
        "extra_config": None,
        "data_type": "int",
        "backend": "auto",
    }
    kwargs.update(overrides)
    return INCConfig(**kwargs)


def test_inc_resolver_exact_match() -> None:
    config = make_config(
        extra_config={
            "layers.0.self_attn.q_proj": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.resolver.resolve(DummyLayer(), "layers.0.self_attn.q_proj")

    assert layer_config.bits == 8
    assert layer_config.group_size == 64
    assert layer_config.sym is False
    assert layer_config.quantized is True


def test_inc_resolver_regex_match() -> None:
    config = make_config(
        extra_config={
            r"layers\.\d+\.self_attn\.(q|k|v)_proj": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.resolver.resolve(DummyLayer(), "layers.3.self_attn.q_proj")

    assert layer_config.bits == 8
    assert layer_config.group_size == 64
    assert layer_config.sym is False


def test_inc_resolver_invalid_regex_ignored() -> None:
    config = make_config(
        extra_config={
            "[invalid": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.resolver.resolve(DummyLayer(), "layers.0.self_attn.q_proj")

    assert layer_config.bits == 4
    assert layer_config.group_size == 128
    assert layer_config.sym is True


def test_inc_resolver_block_name_to_quantize_marks_unquantized() -> None:
    config = make_config(block_name_to_quantize=["layers.1"])

    layer_config = config.resolver.resolve(DummyLayer(), "layers.0.self_attn.q_proj")

    assert layer_config.bits == 16
    assert layer_config.group_size == -1
    assert layer_config.sym is True
    assert layer_config.quantized is False


def test_inc_resolver_parallel_lm_head_defaults_to_unquantized() -> None:
    layer = object.__new__(ParallelLMHead)
    config = make_config()

    layer_config = config.resolver.resolve(layer, "lm_head")

    assert layer_config.quantized is False
    assert layer_config.bits == 16


def test_inc_resolver_fused_moe_requires_consistent_configs() -> None:
    config = make_config(
        extra_config={
            "layers.0.block_sparse_moe.experts.0.w1": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.block_sparse_moe.experts.0.w2": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
            },
        }
    )

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.resolver.resolve(DummyFusedMoE(), "layers.0.block_sparse_moe")


def test_inc_resolver_fused_module_requires_consistent_configs() -> None:
    config = make_config(
        extra_config={
            "layers.0.self_attn.q_proj": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.self_attn.k_proj": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.self_attn.v_proj": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.resolver.resolve(DummyLayer(), "layers.0.self_attn.qkv_proj")


def test_inc_layer_config_mx_fp_helpers() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=32,
        sym=True,
        packing_format="",
        backend="",
        data_type="mx_fp",
        quantized=True,
    )

    assert layer_config.is_mxfp4 is True
    assert layer_config.is_mxfp8 is False


def test_inc_config_builders_gptq_defaults() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    gptq_config = build_gptq_config(layer_config)
    marlin_config = build_gptq_marlin_config(layer_config)
    moe_config = build_moe_wna16_gptq_dict(layer_config)

    assert gptq_config.weight_bits == 4
    assert gptq_config.group_size == 128
    assert gptq_config.desc_act is False
    assert gptq_config.lm_head_quantized is False
    assert gptq_config.dynamic == {}

    assert marlin_config.weight_bits == 4
    assert marlin_config.group_size == 128
    assert marlin_config.is_sym is True
    assert marlin_config.lm_head_quantized is False
    assert marlin_config.dynamic == {}
    assert marlin_config.full_config == {}

    assert moe_config == {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "lm_head": False,
    }


def test_inc_config_builders_awq_defaults() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=False,
        packing_format="auto_round:auto_awq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    awq_config = build_awq_config(layer_config)
    marlin_config = build_awq_marlin_config(layer_config)
    moe_config = build_moe_wna16_awq_dict(layer_config)

    assert awq_config.weight_bits == 4
    assert awq_config.group_size == 128
    assert awq_config.zero_point is True
    assert awq_config.modules_to_not_convert == []

    assert marlin_config.weight_bits == 4
    assert marlin_config.group_size == 128
    assert marlin_config.zero_point is True
    assert marlin_config.lm_head_quantized is False
    assert marlin_config.modules_to_not_convert == []
    assert marlin_config.full_config == {}

    assert moe_config == {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "lm_head": False,
    }


def test_inc_resolve_scheme_selects_wna16() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    scheme = resolve_scheme(layer_config)

    assert isinstance(scheme, INCWna16Scheme)
