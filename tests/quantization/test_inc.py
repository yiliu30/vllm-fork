# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.inc.config_builders import (
    build_awq_config,
    build_awq_marlin_config,
    build_gptq_config,
    build_gptq_marlin_config,
    build_moe_wna16_awq_dict,
    build_moe_wna16_gptq_dict,
)
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.resolver import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.schemes import (
    INCWna16Scheme,
    resolve_scheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.base import INCLinearScheme
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


class DummyLinearScheme(INCLinearScheme):
    def __init__(self) -> None:
        self.calls = []

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(self, *args, **kwargs) -> None:
        self.calls.append(("create_weights", args, kwargs))

    def process_weights_after_loading(self, layer) -> None:
        self.calls.append(("process_weights_after_loading", layer))

    def apply_weights(self, layer, x, bias=None):
        self.calls.append(("apply_weights", layer, x, bias))
        return "applied"


def test_inc_linear_method_delegates() -> None:
    scheme = DummyLinearScheme()
    method = INCLinearMethod(scheme)
    layer = DummyLayer()

    method.create_weights(
        layer,
        input_size_per_partition=1,
        output_partition_sizes=[2],
        input_size=1,
        output_size=2,
        params_dtype=None,
    )
    method.process_weights_after_loading(layer)
    result = method.apply(layer, "x", "b")

    assert result == "applied"
    assert [call[0] for call in scheme.calls] == [
        "create_weights",
        "process_weights_after_loading",
        "apply_weights",
    ]


def test_inc_get_quant_method_unquantized_linear_returns_unquantized() -> None:
    config = make_config(extra_config={"layer": {"bits": 16}})
    layer = object.__new__(LinearBase)

    method = config.get_quant_method(layer, "layer")

    assert isinstance(method, UnquantizedLinearMethod)


def test_inc_get_quant_method_unquantized_moe_returns_none() -> None:
    config = make_config(extra_config={"layer": {"bits": 16}})
    layer = object.__new__(FusedMoE)

    method = config.get_quant_method(layer, "layer")

    assert method is None


def test_inc_get_quant_method_linear_uses_resolved_scheme(monkeypatch) -> None:
    config = make_config()
    layer = object.__new__(LinearBase)
    sentinel = object()

    class DummyScheme:
        def get_linear_method(self, _config, _layer, _prefix, _layer_config):
            return sentinel

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.factory.resolve_scheme",
        lambda _layer_config: DummyScheme(),
    )

    method = config.get_quant_method(layer, "layer")

    assert method is sentinel
