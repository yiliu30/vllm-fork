# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .resolver import INCConfigResolver

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class INCConfig(QuantizationConfig):
    """Config class for Intel Neural Compressor (INC).
    Repo: https://github.com/intel/neural-compressor
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int", "mx_fp", "fp", "fp8"}
    SUPPORTED_FORMATS = {
        "auto_round:auto_gptq",
        "auto_round:auto_awq",
        "auto_round:llm_compressor",
        "auto_round:fp8",
    }
    SUPPORTED_BACKENDS = {
        "auto",
        "gptq",
        "gptq:marlin",
        "awq",
        "awq:marlin",
        "marlin",
    }

    AUTO_ROUND_QUANT_METHOD = "auto_round"

    MXFP8_BITS = 8
    MXFP8_GROUP_SIZE = 32
    MXFP8_DATA_TYPE = "mx_fp"
    MXFP8_PACKING_FORMAT = "auto_round:llm_compressor"

    FP8_BLOCK_BITS = 8
    FP8_BLOCK_DATA_TYPE = "fp"
    FP8_BLOCK_PACKING_FORMAT = "auto_round:fp8"
    FP8_BLOCK_FMT = "e4m3"
    FP8_BLOCK_ACTIVATION_SCHEME = "dynamic"

    @staticmethod
    def _normalize_group_size(
        group_size: int | list[int] | tuple[int, int],
    ) -> int | tuple[int, int]:
        if isinstance(group_size, list):
            if len(group_size) != 2:
                raise ValueError(
                    "INC block-wise FP8 requires group_size to be a 2-D "
                    f"integer sequence, but found {group_size!r}."
                )
            first, second = group_size
            return (first, second)

        if isinstance(group_size, tuple):
            if len(group_size) != 2 or not all(
                isinstance(value, int) for value in group_size
            ):
                raise ValueError(
                    "INC block-wise FP8 requires group_size to be a 2-D "
                    f"integer sequence, but found {group_size!r}."
                )
            first, second = group_size
            return (first, second)

        if not isinstance(group_size, int):
            raise ValueError(
                "INC group_size must be an int or a 2-D integer sequence, "
                f"but found {type(group_size).__name__}."
            )

        return group_size

    @staticmethod
    def _normalize_data_type(data_type: str) -> str:
        # Keep a small alias for branches that already experimented with "fp8".
        return "fp" if data_type == "fp8" else data_type

    def __init__(
        self,
        weight_bits: int,
        group_size: int | list[int] | tuple[int, int],
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()

        group_size = self._normalize_group_size(group_size)
        data_type = self._normalize_data_type(data_type)

        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support {self.SUPPORTED_BITS}."
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type}, "
                f"currently only support {self.SUPPORTED_DTYPES}."
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support {self.SUPPORTED_FORMATS}."
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}, "
                f"currently only support {self.SUPPORTED_BACKENDS}."
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)
        self.resolver = INCConfigResolver(self)
        self._validate_supported_quantization()

    def __repr__(self) -> str:
        return (
            f"INCConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        quant_method = cls.get_from_keys_or(config, ["quant_method"], None)
        packing_format = cls.get_from_keys_or(config, ["packing_format"], None)

        if packing_format is None and quant_method == cls.FP8_BLOCK_PACKING_FORMAT:
            packing_format = cls.FP8_BLOCK_PACKING_FORMAT
        if packing_format is None:
            packing_format = "auto_round:auto_gptq"

        raw_group_size = cls.get_from_keys(config, ["group_size"])
        raw_weight_block_size = cls.get_from_keys_or(
            config, ["weight_block_size"], None
        )

        if raw_weight_block_size is not None:
            normalized_group_size = cls._normalize_group_size(raw_group_size)
            normalized_weight_block_size = cls._normalize_group_size(
                raw_weight_block_size
            )
            if normalized_group_size != normalized_weight_block_size:
                raise ValueError(
                    "INC block-wise FP8 requires group_size and "
                    "weight_block_size to match, but found "
                    f"group_size={normalized_group_size!r} and "
                    f"weight_block_size={normalized_weight_block_size!r}."
                )
            group_size = normalized_weight_block_size
        else:
            group_size = raw_group_size

        quant_config = cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=group_size,
            sym=cls.get_from_keys_or(config, ["sym"], True),
            packing_format=packing_format,
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
        )
        quant_config._validate_raw_config(config)
        return quant_config

    def _validate_supported_quantization(self) -> None:
        if self.data_type == self.MXFP8_DATA_TYPE:
            if self.weight_bits != self.MXFP8_BITS:
                raise ValueError(
                    "INC MXFP8 only supports bits=8, "
                    f"but found bits={self.weight_bits}."
                )
            if self.group_size != self.MXFP8_GROUP_SIZE:
                raise ValueError(
                    "INC MXFP8 only supports group_size=32, "
                    f"but found group_size={self.group_size}."
                )
            if not self.sym:
                raise ValueError("INC MXFP8 only supports symmetric weights.")
            if self.packing_format != self.MXFP8_PACKING_FORMAT:
                raise ValueError(
                    "INC MXFP8 only supports "
                    f"packing_format={self.MXFP8_PACKING_FORMAT!r}, "
                    f"but found {self.packing_format!r}."
                )
            if self.backend != "auto":
                raise ValueError(
                    "INC MXFP8 only supports backend='auto', "
                    f"but found backend={self.backend!r}."
                )
            return

        if self.data_type == self.FP8_BLOCK_DATA_TYPE:
            if self.weight_bits != self.FP8_BLOCK_BITS:
                raise ValueError(
                    "INC block-wise FP8 only supports bits=8, "
                    f"but found bits={self.weight_bits}."
                )
            if not isinstance(self.group_size, tuple):
                raise ValueError(
                    "INC block-wise FP8 requires a 2-D group_size, "
                    f"but found group_size={self.group_size!r}."
                )
            if not self.sym:
                raise ValueError("INC block-wise FP8 only supports symmetric weights.")
            if self.packing_format != self.FP8_BLOCK_PACKING_FORMAT:
                raise ValueError(
                    "INC block-wise FP8 only supports "
                    f"packing_format={self.FP8_BLOCK_PACKING_FORMAT!r}, "
                    f"but found {self.packing_format!r}."
                )
            if self.backend != "auto":
                raise ValueError(
                    "INC block-wise FP8 only supports backend='auto', "
                    f"but found backend={self.backend!r}."
                )
            return

        if self.packing_format == self.MXFP8_PACKING_FORMAT:
            raise ValueError(
                f"packing_format={self.MXFP8_PACKING_FORMAT!r} requires "
                f"data_type={self.MXFP8_DATA_TYPE!r}."
            )

        if self.packing_format == self.FP8_BLOCK_PACKING_FORMAT:
            raise ValueError(
                f"packing_format={self.FP8_BLOCK_PACKING_FORMAT!r} requires "
                f"data_type={self.FP8_BLOCK_DATA_TYPE!r}."
            )

    def _validate_raw_config(self, config: dict[str, Any]) -> None:
        if self.data_type == self.MXFP8_DATA_TYPE:
            expected_fields = {
                "act_bits": self.MXFP8_BITS,
                "act_data_type": self.MXFP8_DATA_TYPE,
                "act_group_size": self.MXFP8_GROUP_SIZE,
                "act_sym": True,
                "act_dynamic": True,
                "enable_quanted_input": False,
            }
            for field_name, expected_value in expected_fields.items():
                actual_value = self.get_from_keys_or(
                    config, [field_name], expected_value
                )
                if actual_value != expected_value:
                    raise ValueError(
                        "INC MXFP8 only supports "
                        f"{field_name}={expected_value!r}, "
                        f"but found {field_name}={actual_value!r}."
                    )
            return

        if self.data_type != self.FP8_BLOCK_DATA_TYPE:
            return

        assert isinstance(self.group_size, tuple)

        expected_fields = {
            "act_bits": self.FP8_BLOCK_BITS,
            "act_group_size": self.group_size[0],
            "act_data_type": self.FP8_BLOCK_DATA_TYPE,
            "act_sym": True,
            "act_dynamic": True,
            "activation_scheme": self.FP8_BLOCK_ACTIVATION_SCHEME,
            "fmt": self.FP8_BLOCK_FMT,
            "enable_quanted_input": False,
        }
        for field_name, expected_value in expected_fields.items():
            actual_value = self.get_from_keys_or(config, [field_name], expected_value)
            if actual_value != expected_value:
                raise ValueError(
                    "INC block-wise FP8 only supports "
                    f"{field_name}={expected_value!r}, "
                    f"but found {field_name}={actual_value!r}."
                )

        raw_weight_block_size = self.get_from_keys_or(
            config, ["weight_block_size"], None
        )
        if raw_weight_block_size is not None:
            normalized_weight_block_size = self._normalize_group_size(
                raw_weight_block_size
            )
            if normalized_weight_block_size != self.group_size:
                raise ValueError(
                    "INC block-wise FP8 only supports "
                    f"weight_block_size={self.group_size!r}, "
                    f"but found weight_block_size={normalized_weight_block_size!r}."
                )

    def get_layer_config(self, layer, layer_name: str):
        return self.resolver.get_layer_config(layer, layer_name)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )
        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe import FusedMoE

        from .schemes.factory import resolve_scheme

        # Match original: check model.-prefixed names for unquantized layers
        if prefix and self.extra_config:
            for layer_name in self.extra_config:
                if (
                    layer_name == prefix or layer_name == f"model.{prefix}"
                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    if isinstance(layer, FusedMoE):
                        return UnquantizedFusedMoEMethod(layer.moe_config)
                    return UnquantizedLinearMethod()

        layer_config = self.resolver.resolve(layer, prefix)
        if not layer_config.quantized:
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            layer_config.bits,
            layer_config.group_size,
            layer_config.sym,
        )

        scheme = resolve_scheme(layer_config)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return scheme.get_linear_method(self, layer, prefix, layer_config)
        if isinstance(layer, FusedMoE):
            return scheme.get_moe_method(self, layer, prefix, layer_config)
        return None

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override AutoRound formats to INC."""
        del user_quant, hf_config

        quant_method = hf_quant_cfg.get("quant_method", None)
        if quant_method == cls.AUTO_ROUND_QUANT_METHOD:
            return cls.get_name()
        if quant_method == cls.FP8_BLOCK_PACKING_FORMAT:
            return cls.get_name()
        return None
