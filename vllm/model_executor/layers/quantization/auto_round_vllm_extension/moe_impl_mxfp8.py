# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Callable, Optional

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)
from .quant_methods import AutoRoundMoEMethod

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

        if envs.VLLM_ENABLE_STATIC_MOE:
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
                from .mxfp8_qdq_utils import dequant_mx_fp8, quant_mx_fp8

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
