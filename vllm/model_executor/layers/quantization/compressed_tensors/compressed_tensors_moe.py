# SPDX-License-Identifier: Apache-2.0

import enum
from enum import Enum
from typing import Callable, List, Optional
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_rank
import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationStrategy)

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Fp8MoECutlassMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4MoeMethod",
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # Prefer to use the non-marlin kernel when:
            # 1. Many experts (MarlinMoE gives poor performance when >= 16)
            # 2. Non-FP16 dtype (MarlinMoE only supports FP16)
            # 3. Actorder is not group/dynamic (g_idx is unsupported)
            # 4. Scaled are grouped (channelwise is unsupported)
            if ((layer.local_num_experts >= 16
                 or layer.params_dtype != torch.float16) and
                    weight_quant.actorder not in (ActivationOrdering.GROUP,
                                                  ActivationOrdering.DYNAMIC)
                    and weight_quant.strategy in QuantizationStrategy.GROUP):
                return CompressedTensorsWNA16MoEMethod(quant_config)
            else:
                return CompressedTensorsWNA16MarlinMoEMethod(quant_config)
        elif quant_config._is_fp4a4_mxfp4(weight_quant, input_quant):
            from .compressed_tensors_moe_mxfp4 import CompressedTensorsW4A4MXFP4MoeMethod
            return CompressedTensorsW4A4MXFP4MoeMethod()
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A4MoeMethod()
        elif (quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
              and layer.activation == "silu"):
            return CompressedTensorsW8A8Fp8MoECutlassMethod(quant_config)
        elif quant_config._is_mxfp8_w8a8(weight_quant, input_quant):
            from .compressed_tensors_moe_mxfp8 import CompressedTensorsW8A8MXFp8MoEMethod
            return CompressedTensorsW8A8MXFp8MoEMethod(quant_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


def nvfp4_unpacked_weight_gemm(
    x, weight_unpacked, weight_scale, weight_global_scale
):
    # return self.run_nvfp4_emulations(x, layer)
    from vllm.model_executor.layers.quantization.utils.nvfp4_qdq import (
        unpacked_nvfp4_to_fp8,
        dequant_nvfp4,
        qdq_nvfp4,
    )

    # bs, seq_len, hidden_size = x.shape
    # x = x.reshape(bs * seq_len, hidden_size)
    hp_weight = dequant_nvfp4(
        data_lp=weight_unpacked,
        out_scales=weight_scale,
        per_tensor_scale=weight_global_scale,
        original_dtype=x.dtype,
        packed=False,
    )

    # breakpoint()
    x = qdq_nvfp4(x)
    out = x @ hp_weight.t()
    # out = out.reshape(bs, seq_len, -1)
    return out

import vllm.envs as envs
class CompressedTensorsW4A4MoeMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        self.use_marlin = True
        self.group_size = 16

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
                dtype=torch.float8_e4m3fn),
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
                dtype=torch.float8_e4m3fn),
            requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(torch.empty(
            num_experts, 2, dtype=torch.float32),
                                                requires_grad=False)
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(torch.empty(
            num_experts, dtype=torch.float32),
                                               requires_grad=False)
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(torch.empty(num_experts,
                                                         2,
                                                         dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.empty(num_experts,
                                                        dtype=torch.float32),
                                            requires_grad=False)
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

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
            from vllm.model_executor.layers.quantization.utils.nvfp4_qdq import (
                unpacked_nvfp4_to_fp8,
            )

            w13_weight_packed = layer.w13_weight_packed.data
            num_experts, intermediate_size_per_partition_x2, hidden_size = (
                w13_weight_packed.shape
            )
            w13_weight_unpacked_lst = []
            w2_weight_unpacked_lst = []
            for expert_index in range(num_experts):
                w13_weight_unpacked = unpacked_nvfp4_to_fp8(
                    w13_weight_packed[expert_index]
                )
                w13_weight_unpacked_lst.append(w13_weight_unpacked)
                w2_weight_unpacked = unpacked_nvfp4_to_fp8(
                    layer.w2_weight_packed.data[expert_index]
                )
                w2_weight_unpacked_lst.append(w2_weight_unpacked)

            w13_weight_unpacked = torch.stack(w13_weight_unpacked_lst, dim=0)
            w2_weight_unpacked = torch.stack(w2_weight_unpacked_lst, dim=0)

            layer.w13_weight_unpacked = torch.nn.Parameter(
                w13_weight_unpacked,
                requires_grad=False,
            )
            del layer.w13_weight_packed
            layer.w2_weight_unpacked = torch.nn.Parameter(
                w2_weight_unpacked,
                requires_grad=False,
            )
            del layer.w2_weight_packed
            torch.hpu.synchronize()
            return
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data,
                                              requires_grad=False)

        layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data,
                                             requires_grad=False)

        if not torch.allclose(layer.w13_weight_global_scale[:, 0],
                              layer.w13_weight_global_scale[:, 1]):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.")

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False)

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False)

        if not self.use_marlin:
            # swizzle weight scales
            layer.w13_blockscale_swizzled = torch.nn.Parameter(
                self.swizzle_blockscale(layer.w13_weight_scale),
                requires_grad=False)

            layer.w2_blockscale_swizzled = torch.nn.Parameter(
                self.swizzle_blockscale(layer.w2_weight_scale),
                requires_grad=False)

            # w13
            layer.g1_alphas = torch.nn.Parameter(
                ((1 / (layer.w13_input_global_scale.max(dim=1).values.to(
                    torch.float32))) * layer.w13_weight_scale_2),
                requires_grad=False)

            # w2
            layer.g2_alphas = torch.nn.Parameter(
                ((1 / layer.w2_input_global_scale) *
                 layer.w2_weight_scale_2).to(torch.float32),
                requires_grad=False)

            # Inverse Input Quant Scales
            layer.w13_input_scale_quant = torch.nn.Parameter(
                (layer.w13_input_global_scale.max(dim=1).values.to(
                    torch.float32)),
                requires_grad=False)

            layer.w2_input_scale_quant = torch.nn.Parameter(
                (layer.w2_input_global_scale), requires_grad=False)

        if self.use_marlin:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
                prepare_moe_fp4_layer_for_marlin,
            )

            prepare_moe_fp4_layer_for_marlin(layer)

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
            num_experts, intermediate_size_per_partition_x2, _ = layer.w13_weight_unpacked.shape
            intermediate_size_per_partition = intermediate_size_per_partition_x2 // 2
            act_fn = F.silu
            num_all_tokens, hidden_dim = x.shape
            num_experts = layer.local_num_experts
            total_num_experts = router_logits.size(-1)
            experts_mask = torch.zeros((x.size(0), total_num_experts), dtype=x.dtype, device=x.device)
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(x.dtype)
            experts_mask.scatter_(-1, topk_ids, topk_weights)
            experts_mask = experts_mask.transpose(0, 1)

            mask_weights = torch.zeros((num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device)
            mask_weights.scatter_(-1, topk_ids, 1)
            mask_weights = mask_weights.transpose(0, 1)
            # Note: ep_size equal tp_size
            ep_rank = get_tensor_model_parallel_rank()
            ep_shift = ep_rank * num_experts

            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13_unpacked = layer.w13_weight_unpacked[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]
                local_w13_global_scale = layer.w13_weight_global_scale[
                    expert_index
                ]
                local_w13_input_global_scale = layer.w13_input_global_scale[
                    expert_index
                ]
                local_w2_unpacked = layer.w2_weight_unpacked[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]
                local_w2_global_scale = layer.w2_weight_global_scale[
                    expert_index
                ]
                local_w2_input_global_scale = layer.w2_input_global_scale[
                    expert_index
                ]

                local_w1_unpacked = local_w13_unpacked[
                    :intermediate_size_per_partition, ...
                ]
                local_w1_scale = local_w13_scale[
                    :intermediate_size_per_partition, ...
                ]
                local_w1_global_scale = local_w13_global_scale[0]
                local_w1_input_global_scale = local_w13_input_global_scale[0]

                local_w3_unpacked = local_w13_unpacked[
                    intermediate_size_per_partition:, ...
                ]
                local_w3_scale = local_w13_scale[
                    intermediate_size_per_partition:, ...
                ]
                local_w3_global_scale = local_w13_global_scale[1]
                local_w3_input_global_scale = local_w13_input_global_scale[1]

                local_w1_out = nvfp4_unpacked_weight_gemm(
                    x=current_state_static,
                    weight_unpacked=local_w1_unpacked,
                    weight_scale=local_w1_scale,
                    weight_global_scale=local_w1_global_scale,
                )
                local_w3_out = nvfp4_unpacked_weight_gemm(
                    x=current_state_static,
                    weight_unpacked=local_w3_unpacked,
                    weight_scale=local_w3_scale,
                    weight_global_scale=local_w3_global_scale,
                )

                w13_out = act_fn(local_w1_out) * local_w3_out

                local_w2_out = nvfp4_unpacked_weight_gemm(
                    x=w13_out,
                    weight_unpacked=local_w2_unpacked,
                    weight_scale=local_w2_scale,
                    weight_global_scale=local_w2_global_scale,
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



class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_tensor = (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                      and self.input_quant.strategy
                      == QuantizationStrategy.TENSOR)
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, we require per tensor "
                "or channelwise, dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization.")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, hidden_size, 1, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        if current_platform.is_hpu():
            import vllm_hpu_extension.ops as hpu_ops
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)
            return
        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale,
                    layer.w13_input_scale)
            w2_weight, w2_weight_scale, w2_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_weight_scale,
                    layer.w2_input_scale)
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                        requires_grad=False)
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(w13_input_scale,
                                                           requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                       requires_grad=False)
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(w2_input_scale,
                                                          requires_grad=False)

        # For Per-TENSOR case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size
            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled)

        # Property to determine if AITER is used
        if is_rocm_aiter_moe_enabled():
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa E501
                rocm_aiter_fused_experts, shuffle_weights)

            # reshaping weights is required for aiter moe kernel.
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data)

            layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                 requires_grad=False)

            self.fused_experts_func = rocm_aiter_fused_experts
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts
            self.fused_experts_func = fused_experts

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
    ) -> torch.Tensor:
        if current_platform.is_hpu():
            return self.forward_hpu(
                x=x,
                layer=layer,
                router_logits=router_logits,
                top_k=top_k,
                renormalize=renormalize,
                use_grouped_topk=use_grouped_topk,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input)

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
            e_score_correction_bias=e_score_correction_bias)

        return self.fused_experts_func(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            per_channel_quant=self.weight_quant.strategy ==
            QuantizationStrategy.CHANNEL,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)

    def forward_hpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
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
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
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
                e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*input_shape)


class CompressedTensorsW8A8Fp8MoECutlassMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_tensor = (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                      and self.input_quant.strategy
                      == QuantizationStrategy.TENSOR)
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, we require per tensor "
                "or channelwise, dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization.")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, hidden_size, 1, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        device = w13_weight.device
        # TODO strides can be shared across multiple layers
        self.ab_strides1 = torch.full((num_experts, ),
                                      hidden_size,
                                      device=device,
                                      dtype=torch.int64)
        self.c_strides1 = torch.full((num_experts, ),
                                     2 * intermediate_size_per_partition,
                                     device=device,
                                     dtype=torch.int64)
        self.ab_strides2 = torch.full((num_experts, ),
                                      intermediate_size_per_partition,
                                      device=device,
                                      dtype=torch.int64)
        self.c_strides2 = torch.full((num_experts, ),
                                     hidden_size,
                                     device=device,
                                     dtype=torch.int64)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        # For Per-TENSOR case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size
            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

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
    ) -> torch.Tensor:

        assert activation == "silu"

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
            e_score_correction_bias=e_score_correction_bias)

        from vllm.model_executor.layers.fused_moe import cutlass_moe_fp8

        return cutlass_moe_fp8(
            x,
            layer.w13_weight.transpose(1, 2),
            layer.w2_weight.transpose(1, 2),
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            self.ab_strides1,
            self.c_strides1,
            self.ab_strides2,
            self.c_strides2,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            out_dtype=x.dtype,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

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
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

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
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, (
            "Only symmetric quantization is supported for MoE")

        if not (self.quant_config.quant_format
                == CompressionFormat.pack_quantized.value
                and self.num_bits in WNA16_SUPPORTED_BITS):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.pack_quantized.value} ",
                             "is supported for the following bits: ",
                             f"{WNA16_SUPPORTED_BITS}")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        assert params_dtype == torch.float16, (
            "float16 is required for MoE compressed models. Set dtype=torch.float16"  # noqa: E501
        )

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy
        })
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.packed_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // self.packed_factor,
            hidden_size,
            dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (intermediate_size_full
                          if load_full_w2 else intermediate_size_per_partition)

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full)

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 num_groups_w2,
                                                 hidden_size,
                                                 dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        def get_scale_perms(num_bits: int):
            scale_perm: List[int] = []
            for i in range(8):
                scale_perm.extend([i + 8 * j for j in range(8)])
            scale_perm_single: List[int] = []
            for i in range(4):
                scale_perm_single.extend(
                    [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
            return scale_perm, scale_perm_single

        def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                                  group_size: int, num_bits: int):
            scale_perm, scale_perm_single = get_scale_perms(num_bits)
            if group_size < size_k and group_size != -1:
                s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
            else:
                s = s.reshape((-1, len(scale_perm_single)))[:,
                                                            scale_perm_single]
            s = s.reshape((-1, size_n)).contiguous()
            return s

        def marlin_moe_permute_scales(s: torch.Tensor, size_k: int,
                                      size_n: int, group_size: int,
                                      num_bits: int):
            num_experts = s.shape[0]
            output = torch.empty((num_experts, s.shape[1], s.shape[2]),
                                 device=s.device,
                                 dtype=s.dtype)
            for e in range(num_experts):
                output[e] = marlin_permute_scales(s[e], size_k, size_n,
                                                  group_size, num_bits)
            return output

        size_k2 = layer.w2_weight_packed.shape[2]
        size_k13 = layer.w13_weight_packed.shape[2]

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(
                    layer.w13_weight_g_idx[e]).to(torch.int32)
                w2_g_idx_sort_indices[e] = torch.argsort(
                    layer.w2_weight_g_idx[e]).to(torch.int32)
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][
                    w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices",
                              w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices",
                              w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            size_k13,
            layer.w13_weight_scale.shape[2],
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1] *
            (self.group_size if self.group_size != -1 else self.packed_factor),
            size_k2,
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

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
    ) -> torch.Tensor:
        assert activation == "silu", "Only SiLU activation is supported."
        if expert_map is not None:
            raise NotImplementedError(
                "Expert Parallelism is not supported for "
                "fused Marlin MoE method.")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for "
                "fused Marlin MoE method.")

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
            e_score_correction_bias=e_score_correction_bias)

        return torch.ops.vllm.fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full)


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        # channelwise is not supported by this kernel
        assert config.strategy == "group"
        self.group_size = config.group_size
        # grouped actorder isn't supported by this kernel
        assert config.actorder != "group"
        assert config.symmetric, (
            "Only symmetric quantization is supported for MoE")

        if not (self.quant_config.quant_format
                == CompressionFormat.pack_quantized.value
                and self.num_bits in WNA16_SUPPORTED_BITS):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.pack_quantized.value} ",
                             "is supported for the following bits: ",
                             f"{WNA16_SUPPORTED_BITS}")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy
        })
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.packed_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // self.packed_factor,
            hidden_size,
            dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 num_groups_w2,
                                                 hidden_size,
                                                 dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
        layer.w13_weight_packed = torch.nn.Parameter(
            layer.w13_weight_packed.transpose(1, 2).contiguous().view(
                torch.uint8),
            requires_grad=False)
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w2_weight_packed.transpose(1,
                                             2).contiguous().view(torch.uint8),
            requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(),
            requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(),
            requires_grad=False)

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
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts
        assert activation == "silu", "Only SiLU activation is supported."
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
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_int4_w4a16=self.num_bits == 4,
            use_int8_w8a16=self.num_bits == 8,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size])
