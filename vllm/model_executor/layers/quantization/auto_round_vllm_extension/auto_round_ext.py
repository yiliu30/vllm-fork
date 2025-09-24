from vllm.model_executor.layers.quantization import auto_round as vllm_ar
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

import torch
from vllm.platforms import current_platform
import torch
from typing import TYPE_CHECKING, Any, Literal, Optional, cast
from abc import ABC, abstractmethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from typing import Union
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from auto_round.schemes import QuantizationScheme
import vllm.envs as envs


class AutoRoundExtensionConfig(AutoRoundConfig):
    SUPPORTED_DTYPES = AutoRoundConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = AutoRoundConfig.SUPPORTED_FORMATS.union(
        {"auto_round:llm_compressor"}
    )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # FIXME: (Yi) parse the per-layer quant scheme
        if isinstance(layer, LinearBase):
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            quant_method = AutoRoundQuantLinearMethod(
                self, scheme=self.quant_scheme
            )

            return quant_method
        return super().get_quant_method(layer, prefix)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AutoRoundConfig:
        ar_config = super().from_config(config)

        def create_quant_scheme(config):
            from auto_round.schemes import QuantizationScheme

            quant_scheme_attrs = QuantizationScheme.get_attributes()
            filter_config = {
                key: value
                for key, value in config.items()
                if key in quant_scheme_attrs
            }
            quant_scheme = QuantizationScheme.from_dict(filter_config)
            return quant_scheme

        ar_config.quant_scheme = create_quant_scheme(config)
        return ar_config


class AutoRoundQuantImpl(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        raise NotImplementedError

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError


class AutoRoundQuantLinearMethod(LinearMethodBase):
    def is_mxfp8(self, quant_scheme: QuantizationScheme):
        return True

    def __init__(self, config: AutoRoundExtensionConfig, scheme=None):
        self.config = config
        self.scheme = scheme
        if self.is_mxfp8(self.scheme):
            self.impl = AutoRoundMXFP8LinearImpl(
                strategy="TENSOR_GROUP", is_static_input_scheme=True
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return cls.impl.get_min_capability()

    def create_weights(self, *args, **kwargs):
        return self.impl.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return self.impl.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return self.impl.apply_weights(layer, x, bias=bias)


# ==-------------------------------------------------------------------------==
# MXFP8
# ==-------------------------------------------------------------------------==

from typing import Callable, Optional

import torch


from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)


from .quant_dquant_utils import get_fp_scale, dequant_mx_fp8, quant_mx_fp8


QLINEAR_METHODS_DISPATCH_TABLE = {}
QMOE_METHODS_DISPATCH_TABLE = {}

# def register_method(...):
#     pass


# @register_method(scheme=ar_schemes.MXFP8, module_types=[LinearBase])
class AutoRoundMXFP8LinearImpl(AutoRoundQuantImpl):
    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        # self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def process_weights_after_loading(self, layer) -> None:
        return

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        # maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.strategy == "TENSOR_GROUP":
            # Per Group Weight Scale
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition // self.group_size,
                    dtype=torch.uint8,  # E8M0 for MXFP8 scale
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            raise NotImplementedError(
                f"Strategy {self.strategy} is not supported for W8A8-MXFp8"
            )

        # min requirement for fp8 kernels
        # weight_scale[:] = torch.finfo(torch.float32).min
        # weight_scale.fill_(torch.finfo(torch.float32).min)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32
                ),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # dequant weight
        weight = layer.weight
        weight_scale = layer.weight_scale
        dequnat_weight = dequant_mx_fp8(
            weight_fp8=weight.data,
            scale_e8m0=weight_scale.data,
            block_size=self.group_size,
        )
        dequnat_weight = dequnat_weight.to(x.dtype)
        if not envs.VLLM_AR_MXFP8_DISABLE_INPUT_QDQ:
            # q-dq input
            x_scale, x_quant = quant_mx_fp8(x)
            dequant_x = dequant_mx_fp8(
                weight_fp8=x_quant,
                scale_e8m0=x_scale,
                block_size=self.group_size,
            )
            x = dequant_x.to(x.dtype)

        out = x @ dequnat_weight.t()
        return out.to(x.dtype) + (bias if bias is not None else 0)

        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )


def apply():
    import vllm.model_executor.layers.quantization.auto_round as auto_round_module

    auto_round_module.AutoRoundConfig = AutoRoundExtensionConfig
    from .envs_ext import all_environment_variables
