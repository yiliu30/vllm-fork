# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Note: this implementation is adapted from vllm/model_executor/layers/quantization/compressored_tensors.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from auto_round.schemes import QuantizationScheme

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig


def need_skip_attn(prefix: str):
    return "self_attn" in prefix


class AutoRoundExtensionConfig(AutoRoundConfig):
    SUPPORTED_DTYPES = AutoRoundConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = AutoRoundConfig.SUPPORTED_FORMATS.union(
        {"auto_round:llm_compressor"}
    )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # FIXME: (Yi) parse the per-layer quant scheme
        if isinstance(layer, LinearBase):
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
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

        # TODO: (yiliu30) construct the layer-wise quant scheme
        def create_quant_scheme(config):
            from auto_round.schemes import QuantizationScheme

            quant_scheme_attrs = QuantizationScheme.get_attributes()
            filter_config = {
                key: value for key, value in config.items() if key in quant_scheme_attrs
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

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        return self.impl.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

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

import vllm.envs as envs
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from .quant_dquant_utils import dequant_mx_fp8, quant_mx_fp8

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
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
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


# ==---------------------------------------------------------------------------==
# MOE
# ==---------------------------------------------------------------------------==

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class AutoRoundMoEMethod(FusedMoEMethodBase):
    def __init_(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: "AutoRoundConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "AutoRoundMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        # weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        # input_quant = quant_config.target_scheme_map["Linear"].get(
        #     "input_activations")
        weight_quant = None
        input_quant = None

        # FIXME: @yiliu30: temporarily only support MXFP8
        if 0 and quant_config._is_mxfp8_w8a8(weight_quant, input_quant):
            impl = AutoRoundMoEMethodMXFp8Impl(quant_config, layer.moe_config)
            return impl
        elif 1 or quant_config._is_mxfp4_w84a4(weight_quant, input_quant):
            impl = AutoRoundMoEMethodMXFp4Impl(quant_config, layer.moe_config)
            return impl
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        return self.impl.get_fused_moe_quant_config(layer)


# ==-------------------------------------------------------------------------==
# MOE MXFP8
# ==------------------------------------------------------------------------==


class AutoRoundMoEMethodMXFp8Impl(AutoRoundMoEMethod):
    def __init__(
        self,
        quant_config: "AutoRoundConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self.has_bias = self.moe.has_bias
        # self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
        #     "weights"
        # )
        # self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
        #     "input_activations"
        # )

        # per_tensor_group = (
        #     self.weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP
        # )
        per_tensor_group = True
        if not (per_tensor_group):
            raise ValueError(
                "For MXFP8 Fused MoE layers, we require per tensor group"
                f"Found weight: {self.weight_quant}, input {self.input_quant}"
            )
        self.group_size = 32
        self.static_input_scales = False

        # self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        # TODO: @yiliu30: implement it
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if 1 or self.weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP:
            # Weight Scales
            w13_weight_scale = torch.nn.Parameter(
                data=torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size,
                    dtype=torch.uint8,  # E8M0 for MXFP8 scale
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            # w2
            w2_weight_scale = torch.nn.Parameter(
                data=torch.zeros(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.group_size,
                    dtype=torch.uint8,  # E8M0 for MXFP8 scale
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSORGROUP quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        else:
            raise NotImplementedError(
                f"Strategy {self.weight_quant.strategy} is not supported for W8A8-Fp8"
            )

        # INPUT_SCALES
        # FIXME: (Yi) remove it
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        E = num_experts
        H = hidden_size
        IN = intermediate_size_per_partition
        if self.has_bias:
            # TODO: @yiliu30: use the dtype in CK
            bias_dtype = torch.bfloat16
            w13_bias = torch.nn.Parameter(
                torch.zeros(E, 2 * IN, dtype=bias_dtype), requires_grad=False
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=bias_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # TODO: @yiliu30 remove it
        def check_nan(tensor):
            return tensor.float().sum() == 0

        if check_nan(layer.w13_weight):
            logger.info("all zeros self.w13_weight")
            breakpoint()

        if check_nan(layer.w2_weight):
            logger.info("NAN IN self.w2_weight")
            breakpoint()

        if check_nan(layer.w13_bias):
            logger.info("NAN IN self.w13_bias")
            breakpoint()
        if check_nan(layer.w2_bias):
            logger.info("NAN IN self.w2_bias")
            breakpoint()
        if check_nan(layer.w13_weight_scale):
            logger.info("NAN IN self.w13_weight_scale")
            breakpoint()
        if check_nan(layer.w2_weight_scale):
            logger.info("NAN IN self.w2_weight_scale")
            breakpoint()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ):
        # TODO @yiliu30: Align the args with the
        topk_weights, topk_ids, _ = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        if envs.VLLM_USE_STATIC_MOE_HPU:
            num_experts, intermediate_size_per_partition_2x, _ = layer.w13_weight.shape
            intermediate_size_per_partition = intermediate_size_per_partition_2x // 2
            # FIXME: Handle mask
            act_fn = F.silu
            num_all_tokens, hidden_dim = x.shape
            num_experts = layer.local_num_experts
            total_num_experts = router_logits.size(-1)
            experts_mask = torch.zeros(
                (x.size(0), total_num_experts), dtype=x.dtype, device=x.device
            )
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(x.dtype)
            experts_mask.scatter_(-1, topk_ids, topk_weights)
            experts_mask = experts_mask.transpose(0, 1)

            mask_weights = torch.zeros(
                (num_all_tokens, total_num_experts),
                dtype=x.dtype,
                device=x.device,
            )
            mask_weights.scatter_(-1, topk_ids, 1)
            mask_weights = mask_weights.transpose(0, 1)
            # Note: ep_size equal tp_size'
            if expert_map is None:
                ep_rank = 0
            else:
                ep_rank = get_tensor_model_parallel_rank()
            ep_shift = ep_rank * num_experts

            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13 = layer.w13_weight[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]

                local_w2 = layer.w2_weight[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]

                local_w1 = local_w13[:intermediate_size_per_partition, ...]
                local_w1_scale = local_w13_scale[:intermediate_size_per_partition, ...]

                local_w3 = local_w13[intermediate_size_per_partition:, ...]
                local_w3_scale = local_w13_scale[intermediate_size_per_partition:, ...]

                def run_mxfp8_emulations(x, weight, weight_scale, bias=None):
                    dequnat_weight = dequant_mx_fp8(
                        weight_fp8=weight.data,
                        scale_e8m0=weight_scale.data,
                        block_size=self.group_size,
                    )
                    dequnat_weight = dequnat_weight.to(x.dtype)
                    if not envs.VLLM_AR_MXFP8_DISABLE_INPUT_QDQ:
                        x_scale, x_quant = quant_mx_fp8(x)
                        dequant_x = dequant_mx_fp8(
                            weight_fp8=x_quant,
                            scale_e8m0=x_scale,
                            block_size=self.group_size,
                        )
                        x = dequant_x.to(x.dtype)
                    out = x @ dequnat_weight.t()
                    return out.to(x.dtype) + (bias if bias is not None else 0)

                local_w1_bias = None
                local_w2_bias = None
                local_w3_bias = None
                if self.has_bias:
                    local_w13_bias = layer.w13_bias[expert_index]
                    local_w1_bias = local_w13_bias[:intermediate_size_per_partition]
                    local_w3_bias = local_w13_bias[intermediate_size_per_partition:]
                    local_w2_bias = layer.w2_bias[expert_index]

                local_w1_out = run_mxfp8_emulations(
                    x=current_state_static,
                    weight=local_w1,
                    weight_scale=local_w1_scale,
                    bias=local_w1_bias,
                )
                local_w3_out = run_mxfp8_emulations(
                    x=current_state_static,
                    weight=local_w3,
                    weight_scale=local_w3_scale,
                    bias=local_w3_bias,
                )
                # w13_out = act_fn(local_w1_out) * local_w3_out

                # TODO: @yiliu30: wrapper as act func
                limit = 7.0
                alpha = 1.702
                local_w1_out = local_w1_out.clamp(min=None, max=limit)
                local_w3_out = local_w3_out.clamp(min=-limit, max=limit)

                glu = (local_w1_out) * F.sigmoid(local_w1_out * alpha)
                w13_out = (local_w3_out + 1) * glu

                # gate = gate.clamp(min=None, max=self.limit)
                # up = up.clamp(min=-self.limit, max=self.limit)
                # glu = gate * torch.sigmoid(gate * self.alpha)
                # gated_output = (up + 1) * glu

                local_w2_out = run_mxfp8_emulations(
                    x=w13_out,
                    weight=local_w2,
                    weight_scale=local_w2_scale,
                    bias=local_w2_bias,
                )
                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(1)
                local_w2_out = local_w2_out * padded_weight
                if expert_index == 0:
                    final_hidden_states = local_w2_out
                else:
                    final_hidden_states += local_w2_out
            return final_hidden_states


# ==-------------------------------------------------------------------------==
# MOE MXFP4
# ==------------------------------------------------------------------------==

from typing import Callable, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
)
from vllm.scalar_type import scalar_types


class AutoRoundMoEMethodMXFp4Impl(AutoRoundMoEMethod):
    def __init__(
        self,
        quant_config: "AutoRoundConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.use_marlin = False
        self.group_size = 32
        self.quant_config = quant_config
        self.has_bias = self.moe.has_bias

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        E = num_experts
        H = hidden_size
        IN = intermediate_size_per_partition
        if self.has_bias:
            # TODO: @yiliu30: use the dtype in CK
            bias_dtype = torch.bfloat16
            w13_bias = torch.nn.Parameter(
                torch.zeros(E, 2 * IN, dtype=bias_dtype), requires_grad=False
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=bias_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        # TODO: @yiliu30: implement it
        return None

    def swizzle_blockscale(self, scale: torch.tensor):
        assert scale.dtype == torch.float8_e4m3fn
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (
            swizzled_scale.reshape(M, K)
            if scale_ndim == 2
            else swizzled_scale.reshape(B, M, K)
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if envs.VLLM_USE_STATIC_MOE_HPU:
            return
        else:
            raise NotImplementedError(
                "process_weights_after_loading is not implemented for "
                "CompressedTensorsW4A4MXFP4MoeMethod on this platform."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ):
        topk_weights, topk_ids, _ = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        if envs.VLLM_USE_STATIC_MOE_HPU:
            # w_state_dict = [
            #     "w13_weight_packed:torch.Size([64, 2816, 1024])",
            #     "w13_weight_scale:torch.Size([64, 2816, 128])",
            #     "w13_weight_global_scale:torch.Size([64, 2])",
            #     "w13_input_global_scale:torch.Size([64, 2])",
            #     "w2_weight_packed:torch.Size([64, 2048, 704])",
            #     "w2_weight_scale:torch.Size([64, 2048, 88])",
            #     "w2_weight_global_scale:torch.Size([64])",
            #     "w2_input_global_scale:torch.Size([64])",
            # ]
            # w_list = [
            #     "w13_weight_packed",
            #     "w13_weight_scale",
            #     "w13_weight_global_scale",
            #     "w13_input_global_scale",
            #     "w2_weight_packed",
            #     "w2_weight_scale",
            #     "w2_weight_global_scale",
            #     "w2_input_global_scale",
            # ]
            # # [num_experts, 2 * intermediate_size_per_partition, hidden_size//2]
            num_experts, intermediate_size_per_partition_x2, _ = (
                layer.w13_weight_packed.shape
            )
            intermediate_size_per_partition = intermediate_size_per_partition_x2 // 2
            # FIXME: Handle mask
            act_fn = F.silu
            num_all_tokens, hidden_dim = x.shape
            num_experts = layer.local_num_experts
            total_num_experts = router_logits.size(-1)
            experts_mask = torch.zeros(
                (x.size(0), total_num_experts), dtype=x.dtype, device=x.device
            )
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(x.dtype)
            experts_mask.scatter_(-1, topk_ids, topk_weights)
            experts_mask = experts_mask.transpose(0, 1)

            mask_weights = torch.zeros(
                (num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device
            )
            mask_weights.scatter_(-1, topk_ids, 1)
            mask_weights = mask_weights.transpose(0, 1)
            # Note: ep_size equal tp_size
            ep_rank = get_tensor_model_parallel_rank()
            ep_shift = ep_rank * num_experts

            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13_packed = layer.w13_weight_packed[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]
                # local_w13_global_scale = layer.w13_weight_global_scale[expert_index]
                # local_w13_input_global_scale = layer.w13_input_global_scale[
                #     expert_index
                # ]
                local_w2_packed = layer.w2_weight_packed[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]
                # local_w2_global_scale = layer.w2_weight_global_scale[expert_index]
                # local_w2_input_global_scale = layer.w2_input_global_scale[expert_index]

                local_w1_packed = local_w13_packed[
                    :intermediate_size_per_partition, ...
                ]
                local_w1_scale = local_w13_scale[:intermediate_size_per_partition, ...]
                # local_w1_global_scale = local_w13_global_scale[0]
                # local_w1_input_global_scale = local_w13_input_global_scale[0]

                local_w3_packed = local_w13_packed[
                    intermediate_size_per_partition:, ...
                ]
                local_w3_scale = local_w13_scale[intermediate_size_per_partition:, ...]
                # local_w3_global_scale = local_w13_global_scale[1]
                # local_w3_input_global_scale = local_w13_input_global_scale[1]

                from .mxfp4_emulation_utils import (
                    run_mxfp4_emulations,
                )

                local_w1_bias = None
                local_w2_bias = None
                local_w3_bias = None
                if self.has_bias:
                    local_w13_bias = layer.w13_bias[expert_index]
                    local_w1_bias = local_w13_bias[:intermediate_size_per_partition]
                    local_w3_bias = local_w13_bias[intermediate_size_per_partition:]
                    local_w2_bias = layer.w2_bias[expert_index]

                # local_w13_input_global_scale_max = local_w13_input_global_scale.max()

                local_w1_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w1_packed,
                    weight_scale=local_w1_scale,
                    bias=local_w1_bias,
                )
                local_w3_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w3_packed,
                    weight_scale=local_w3_scale,
                    bias=local_w3_bias,
                )

                # TODO: @yiliu30: wrapper as act func
                limit = 7.0
                alpha = 1.702
                local_w1_out = local_w1_out.clamp(min=None, max=limit)
                local_w3_out = local_w3_out.clamp(min=-limit, max=limit)
                glu = (local_w1_out) * F.sigmoid(local_w1_out * alpha)
                w13_out = (local_w3_out + 1) * glu

                # w13_out = act_fn(local_w1_out) * local_w3_out

                local_w2_out = run_mxfp4_emulations(
                    x=w13_out,
                    weight=local_w2_packed,
                    weight_scale=local_w2_scale,
                    bias=local_w2_bias,
                )
                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(1)
                local_w2_out = local_w2_out * padded_weight
                if expert_index == 0:
                    final_hidden_states = local_w2_out
                else:
                    final_hidden_states += local_w2_out
            return final_hidden_states

        if self.use_marlin:
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=layer.w13_weight_scale_2,
                global_scale2=layer.w2_weight_scale_2,
                quant_type_id=scalar_types.float4_e2m1f.id,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        assert activation == "silu", "Only SiLU activation is supported."
        assert not apply_router_weight_on_input, (
            "Router weight on input is not supported for ModelOptNvFp4FusedMoE."
        )
        assert expert_map is None, (
            "Expert Parallelism / expert_map "
            "is currently not supported for "
            "ModelOptNvFp4FusedMoE."
        )

        from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4

        # Cutlass moe takes in activations in BF16/Half precision
        # and fp4 quantized weights loaded from the checkpoint
        return cutlass_moe_fp4(
            a=x,
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alphas=layer.g1_alphas,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alphas=layer.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=x.shape[0],
            n=layer.w2_weight.shape[2] * 2,
            k=x.shape[1],
            e=layer.w13_weight.shape[0],
            a1_gscale=layer.w13_input_scale_quant,
            a2_gscale=layer.w2_input_scale_quant,
            device=x.device,
        ).to(x.dtype)
