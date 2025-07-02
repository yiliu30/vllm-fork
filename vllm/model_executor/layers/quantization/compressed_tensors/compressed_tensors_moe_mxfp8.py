# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import importlib
from enum import Enum
from typing import Callable, Optional
import torch.nn.functional as F
import torch
from compressed_tensors.quantization import (
    QuantizationStrategy,
)
from vllm.distributed import get_tensor_model_parallel_rank
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
    WNA16_SUPPORTED_TYPES_MAP,
)

from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
import vllm.envs as envs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)

logger = init_logger(__name__)


from .schemes.compressed_tensors_w8a8_mxfp8 import (
    dequant_mx_fp8,
    quant_mx_fp8,
)


def run_mxfp8_emulations(x, weight, weight_scale, bias=None):
    dequnat_weight = dequant_mx_fp8(
        weight_fp8=weight.data,
        scale_e8m0=weight_scale.data,
        block_size=32,
    )
    dequnat_weight = dequnat_weight.to(x.dtype)
    if not envs.VLLM_DISABLE_INPUT_QDQ:
        x_scale, x_quant = quant_mx_fp8(x)
        dequant_x = dequant_mx_fp8(
            weight_fp8=x_quant,
            scale_e8m0=x_scale,
            block_size=32,
        )
        x = dequant_x.to(x.dtype)
    out = x @ dequnat_weight.t()
    return out.to(x.dtype) + (bias if bias is not None else 0)


import habana_frameworks.torch as htorch

class CompressedTensorsW8A8MXFp8MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights"
        )
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations"
        )

        per_tensor_group = (
            self.weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP
        )
        if not (per_tensor_group):
            raise ValueError(
                "For MXFP8 Fused MoE layers, we require per tensor group"
                f"Found weight: {self.weight_quant}, input {self.input_quant}"
            )
        self.group_size = 32
        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For MXFP8 Fused MoE layer, we require dynamic per token quantization."
            )

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (
            not current_platform.has_device_capability(89)
            or envs.VLLM_TEST_FORCE_FP8_MARLIN
        )
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            self.use_marlin = False
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled,
        )

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

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
            torch.empty(
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
            torch.empty(
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
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP:
            # Weight Scales
            w13_weight_scale = torch.nn.Parameter(
                data=torch.empty(
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
                data=torch.empty(
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

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
    ):
        topk_weights, topk_ids = FusedMoE.select_experts(
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
        torch.hpu.synchronize()

        if envs.VLLM_USE_STATIC_MOE_HPU:

            htorch.core.mark_step()
            num_experts, intermediate_size_per_partition_2x, _ = (
                layer.w13_weight.shape
            )
            intermediate_size_per_partition = (
                intermediate_size_per_partition_2x // 2
            )
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
            # Note: ep_size equal tp_size
            ep_rank = get_tensor_model_parallel_rank()
            ep_shift = ep_rank * num_experts
            htorch.core.mark_step()
            w13_weight = layer.w13_weight.data
            w13_weight_scale = layer.w13_weight_scale.data
            w2_weight = layer.w2_weight.data
            w2_weight_scale = layer.w2_weight_scale.data
            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13 = w13_weight[expert_index]
                local_w13_scale = w13_weight_scale[expert_index]

                local_w2 = w2_weight[expert_index]
                local_w2_scale = w2_weight_scale[expert_index]

                local_w1 = local_w13[:intermediate_size_per_partition, ...]
                local_w1_scale = local_w13_scale[
                    :intermediate_size_per_partition, ...
                ]

                local_w3 = local_w13[intermediate_size_per_partition:, ...]
                local_w3_scale = local_w13_scale[
                    intermediate_size_per_partition:, ...
                ]


                local_w1_out = run_mxfp8_emulations(
                    x=current_state_static,
                    weight=local_w1,
                    weight_scale=local_w1_scale,
                )
                local_w3_out = run_mxfp8_emulations(
                    x=current_state_static,
                    weight=local_w3,
                    weight_scale=local_w3_scale,
                )
                w13_out = act_fn(local_w1_out) * local_w3_out

                local_w2_out = run_mxfp8_emulations(
                    x=w13_out,
                    weight=local_w2,
                    weight_scale=local_w2_scale,
                )
                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(
                    1
                )
                local_w2_out = local_w2_out * padded_weight
                if expert_index == 0:
                    final_hidden_states = local_w2_out
                else:
                    final_hidden_states = final_hidden_states + local_w2_out
                htorch.core.mark_step()
                torch.hpu.synchronize()
                
            return final_hidden_states
