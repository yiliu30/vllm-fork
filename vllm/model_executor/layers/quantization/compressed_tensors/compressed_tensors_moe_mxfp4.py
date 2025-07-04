# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import importlib
from enum import Enum
from typing import Callable, Optional
import torch.nn.functional as F
import torch
from vllm.distributed import get_tensor_model_parallel_rank
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
import vllm.envs as envs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)

logger = init_logger(__name__)



class CompressedTensorsW4A4MXFP4MoeMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        self.use_marlin = False
        self.group_size = 32

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)


    def swizzle_blockscale(self, scale: torch.tensor):
        assert (scale.dtype == torch.float8_e4m3fn)
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
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32,
                                            cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (swizzled_scale.reshape(M, K)
                if scale_ndim == 2 else swizzled_scale.reshape(B, M, K))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if envs.VLLM_USE_STATIC_MOE_HPU:
            return
        else:
            raise NotImplementedError(
                "process_weights_after_loading is not implemented for "
                "CompressedTensorsW4A4MXFP4MoeMethod on this platform.")

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
            intermediate_size_per_partition = (
                intermediate_size_per_partition_x2 // 2
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

            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13_packed = layer.w13_weight_packed[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]

                local_w2_packed = layer.w2_weight_packed[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]

                local_w1_packed = local_w13_packed[
                    :intermediate_size_per_partition, ...
                ]
                local_w1_scale = local_w13_scale[
                    :intermediate_size_per_partition, ...
                ]

                local_w3_packed = local_w13_packed[
                    intermediate_size_per_partition:, ...
                ]
                local_w3_scale = local_w13_scale[
                    intermediate_size_per_partition:, ...
                ]

                from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import (
                    run_mxfp4_emulations,
                )

                local_w1_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w1_packed,
                    weight_scale=local_w1_scale,
                )
                local_w3_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w3_packed,
                    weight_scale=local_w3_scale,
                )

                w13_out = act_fn(local_w1_out) * local_w3_out

                local_w2_out = run_mxfp4_emulations(
                    x=w13_out,
                    weight=local_w2_packed,
                    weight_scale=local_w2_scale,
                )
                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(
                    1
                )
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
                expert_map=expert_map)

        assert activation == "silu", "Only SiLU activation is supported."
        assert not apply_router_weight_on_input, (
            "Router weight on input is not "
            "supported for ModelOptNvFp4FusedMoE.")
        assert expert_map is None, ("Expert Parallelism / expert_map "
                                    "is currently not supported for "
                                    "ModelOptNvFp4FusedMoE.")

        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            cutlass_moe_fp4)

        # Cutlass moe takes in activations in BF16/Half precision
        # and fp4 quantized weights loaded from the checkpoint
        return cutlass_moe_fp4(a=x,
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
                               device=x.device).to(x.dtype)
