# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

is_hpu = current_platform.is_hpu()

if current_platform.is_cuda_alike():
    from .fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore
if current_platform.is_tpu():
    # the iterative moe implementation is used until the moe_pallas is fixed
    from .moe_torch_iterative import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore
logger = init_logger(__name__)

import os
LOW_CPU_MEM = False
VLLM_FORCE_INC = os.getenv("VLLM_FORCE_INC", "0") in ["1", "true"]

# ==-------------------------------------------------------------------------==
# VLLM-HPU-EXT PATCH Start
# ==-------------------------------------------------------------------------==

import torch.nn.functional as F
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as htorch


class MoeMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def set_weight(self, w):
        self.weight = w

    def set_scale_inv_fp8(self, scale_inv_fp8):
        self.scale_inv_fp8 = scale_inv_fp8

    def set_high_precision(self, high_precision=torch.bfloat16):
        self.high_precision = high_precision

    def forward(self, state, expert_id, w):
        raise NotImplementedError()


class VllmMixtureOfExpertsOp(torch.nn.Module):
    def __init__(self, num_total_experts):
        super().__init__()
        if not LOW_CPU_MEM:
            self.w13_list = torch.nn.ModuleList(
                [MoeMatmul() for _ in range(num_total_experts)]
            )
        else:
            self.w1_list = torch.nn.ModuleList(
                [MoeMatmul() for _ in range(num_total_experts)]
            )
            self.w3_list = torch.nn.ModuleList(
                [MoeMatmul() for _ in range(num_total_experts)]
            )
        self.w2_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)]
        )
        self.num_experts = num_total_experts
        # FIXME (Yi) add experts_min and experts_max as init parameters
        self.experts_min = None
        self.experts_max = None

    def forward(
        self,
        hidden_states,
        expert_routing_table,
        router_weights,
        permuted_weights=True,
        activation="silu",
    ):
        # pre-processing for custom op inputs

        experts_range = range(self.num_experts)
        assert self.experts_min is not None, "`experts_min` is not set"
        assert self.experts_max is not None, "`experts_max` is not set"
        experts_min, experts_max = self.experts_min, self.experts_max
        if not LOW_CPU_MEM:
            w1_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
            w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]
            return torch.ops.hpu.mixture_of_experts.fused_weights(
                hidden_states=hidden_states,
                expert_routing_table=expert_routing_table,
                router_weights=router_weights,
                w12=w1_list,
                w3=w2_list,
                permuted_weights=permuted_weights,
                activation=activation,
                experts_min=experts_min,
                experts_max=experts_max,
            )
        else:
            w1_list = [self.w1_list[i].weight.squeeze() for i in experts_range]
            w3_list = [self.w3_list[i].weight.squeeze() for i in experts_range]
            w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]
            return torch.ops.hpu.mixture_of_experts.default(
                hidden_states=hidden_states,
                expert_routing_table=expert_routing_table,
                router_weights=router_weights,
                w1=w1_list,
                w2=w2_list,
                w3=w3_list,
                permuted_weights=permuted_weights,
                activation=activation,
                experts_min=experts_min,
                experts_max=experts_max,
            )


class _DynamicFusedMOE(torch.nn.Module):
    def __init__(self, num_total_experts):
        super().__init__()
        self.MoeOp = VllmMixtureOfExpertsOp(num_total_experts)

    def forward(self, hidden_states, score, topk):
        htorch.core.mark_step()
        routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, topk, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.MoeOp(
            hidden_states=hidden_states,
            expert_routing_table=selected_experts,
            router_weights=routing_weights,
            permuted_weights=True,
            activation="silu",
        )

        return final_hidden_states.view(-1, hidden_states.shape[1])


class MoeFP8Matmul(torch.nn.Module):
    def __init__(
        self,
        block_size: Tuple[int, int] = (128, 128),
        high_precision=torch.bfloat16,
    ):
        super().__init__()
        self.block_size = block_size
        self.high_precision = high_precision
        self.is_dequantized = False

    def set_weight(self, w: torch.Tensor):
        self.weight = w

    def set_scale_inv_fp8(self, scale_inv_fp8: torch.Tensor):
        self.scale_inv_fp8 = scale_inv_fp8

    def set_high_precision(self, high_precision=torch.bfloat16):
        self.high_precision = high_precision

    def set_weight_block_size(self, block_size: Tuple[int, int] = (128, 128)):
        self.block_size = block_size

    def get_dequant_weight(self):
        """
        w13_weight = dequant_block_fp8_weight_naive(w13_weight_fp8,
                                                    w13_weight_scale_inv_fp8,
                                                    block_size=self.quant_config.weight_block_size,
                                                    dtype=x.dtype)
        """
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            dequant_block_fp8_weight_naive,
        )

        return dequant_block_fp8_weight_naive(
            self.weight,
            self.scale_inv_fp8,
            block_size=self.block_size,
            dtype=self.high_precision,
        )

    def forward(self, state, expert_id, w):
        raise NotImplementedError()

    def dequant_block_fp8_weight_for_inc(self, layer: "MoeFP8Matmul"):
        # The function will be called by INC either in the measurement or the quantization phase.
        # At quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight.
        # At measurement phase, INC only measures the BF16 weight and does NOT update the weight.
        # We not track the BF16 weight which will cause OoM.
        if self.is_dequantized:
            return layer.weight
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            dequant_block_fp8_weight_naive,
        )
        
        dequant_weight = dequant_block_fp8_weight_naive(
            layer.weight.data,
            layer.scale_inv_fp8.data,
            layer.block_size,
        )
        layer.is_dequantized = True
        return dequant_weight

    def get_post_process_weights_func(self):
        return self.dequant_block_fp8_weight_for_inc


class VllmMixtureOfExpertsOpFP8(torch.nn.Module):
    def __init__(self, num_total_experts: int):
        super().__init__()
        if not LOW_CPU_MEM:
            self.w13_list = torch.nn.ModuleList(
                [MoeFP8Matmul() for _ in range(num_total_experts)]
            )
        else:
            self.w1_list = torch.nn.ModuleList(
                [MoeFP8Matmul() for _ in range(num_total_experts)]
            )
            self.w3_list = torch.nn.ModuleList(
                [MoeFP8Matmul() for _ in range(num_total_experts)]
            )
        self.w2_list = torch.nn.ModuleList(
            [MoeFP8Matmul() for _ in range(num_total_experts)]
        )
        self.num_experts = num_total_experts
        # FIXME (Yi) add experts_min and experts_max as init parameters
        self.experts_min = None
        self.experts_max = None

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        moe_n_slice,
        n_expert_slice,
        ep_shift,
    ):
        min_expert = self.experts_min
        max_expert = self.experts_max
        w13_list_slice = []
        w2_list_slice = []
        for j in range(min_expert, max_expert):
            w13_list_slice.append(self.w13_list[j].get_dequant_weight())
            w2_list_slice.append(self.w2_list[j].get_dequant_weight())

        final_hidden_states = torch.ops.hpu.mixture_of_experts(
            hidden_states=x,
            expert_routing_table=topk_ids.to(torch.int64),
            router_weights=topk_weights.to(x.dtype),
            w12=w13_list_slice,
            w3=w2_list_slice,
            permuted_weights=True,
            activation="silu",
            experts_min=min_expert,
            experts_max=max_expert,
        )
        htorch.core.mark_step()
        return final_hidden_states


# ==-------------------------------------------------------------------------==
# VLLM-HPU-EXT PATCH End
# ==-------------------------------------------------------------------------==

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        if current_platform.is_cpu():
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                import intel_extension_for_pytorch as ipex
                layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                    layer.w13_weight,
                    layer.w2_weight,
                    use_prepack=True,
                )
            else:
                raise NotImplementedError("CPU MOE only supports x86 arch.")

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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        ep_rank: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            scoring_func=scoring_func,
                            e_score_correction_bias=e_score_correction_bias,
                            ep_rank= ep_rank,
                            )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w2=layer.w2_weight,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             inplace=True)

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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        ep_rank = None,
    ):
        bs, seq_len, hidden_size = x.shape
        x = x.reshape(bs * seq_len, hidden_size)
        assert len(x.shape) == 2
        import habana_frameworks.torch as htorch
        htorch.core.mark_step()
        if use_grouped_topk:
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
            topk_weights, topk_ids = torch.topk(topk_weights,
                                                        top_k,
                                                        dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        final_hidden_states = torch.zeros_like(x)
        num_experts = layer.num_experts
        if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
            assert (
                layer.w13_weight.shape[0] == num_experts
            ), f"Expected {layer.w13_weight.shape[0]} experts, got {num_experts}"
        # For mixtral, the `num_expert_group` is 8.
        if num_expert_group is None:
            num_expert_group = 8
        num_expert_group = num_expert_group
        n_expert_slice = num_experts // num_expert_group
        assert n_expert_slice * 8 == num_experts

        # w13_list = layer.hpu_fused_moe.MoeOp.w13_list
        # w2_list = layer.hpu_fused_moe.MoeOp.w2_list

        for i in range(8):
            min_expert = i * n_expert_slice
            max_expert = (i + 1) * n_expert_slice
            # w13_list_slice = [w13_list[i].weight.squeeze() for i in range(min_expert, max_expert)]
            # w2_list_slice = [w2_list[i].weight.squeeze() for i in range(min_expert, max_expert)]
            w13_list_slice = [layer.w13_weight[j].squeeze().clone() for j in range(min_expert, max_expert)]
            w2_list_slice = [layer.w2_weight[j].squeeze().clone() for j in range(min_expert, max_expert)]
            # print(f"w13_list_slice[0].shape: {w13_list_slice[0].shape}, device: {w13_list_slice[0].device}, dtype: {w13_list_slice[0].dtype}")
            # print(f"w2_list_slice[0].shape: {w2_list_slice[0].shape}, device: {w2_list_slice[0].device}, dtype: {w2_list_slice[0].dtype}")
            # print(f"hidden_states.shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
            # print(f"topk_ids.shape: {topk_ids.shape}, device: {topk_ids.device}, dtype: {topk_ids.dtype}")
            # print(f"topk_weights.shape: {topk_weights.shape}, device: {topk_weights.device}, dtype: {topk_weights.dtype}")
            # print(f"min_expert: {min_expert}, max_expert: {max_expert}")
            final_hidden_states += torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                         expert_routing_table=topk_ids.to(torch.int64),
                                         router_weights=topk_weights.to(x.dtype),
                                         w12=w13_list_slice,
                                         w3=w2_list_slice,
                                         permuted_weights=True,
                                         activation="silu",
                                         experts_min=min_expert,
                                         experts_max=max_expert - 1)
            # print(f"final_hidden_states.shape: {final_hidden_states.shape}, device: {final_hidden_states.device}, dtype: {final_hidden_states.dtype}")
            htorch.core.mark_step()
            # print(f"done mark step {i}")
        return final_hidden_states.view(-1, x.shape[1])

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        **kwargs,
    ):
        assert custom_routing_function is None
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU.")
        return fused_moe_pallas(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                renormalize=renormalize)

    forward_native = forward_cuda


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        self.ep_size = ep_size if ep_size is not None else 1
        assert num_experts % self.ep_size == 0

        self.ep_rank = get_tensor_model_parallel_rank() // self.tp_size

        self.top_k = top_k
        self.num_experts = num_experts // self.ep_size
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        if is_hpu:
            # FIXME: (Yi) WA, should use DynamicFusedMOE for INC
            if not VLLM_FORCE_INC:
                from vllm_hpu_extension.ops import DynamicFusedMOE
                self.hpu_fused_moe = DynamicFusedMOE(self.num_experts)

        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        moe_quant_params = {
            "num_experts": self.num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # FIXME: (Yi) we need to wrap the `torch.ops.hpu.mixture_of_experts` as `VllmMixtureOfExpertsOp`,
        # so that INC can patch it for measurement and quantization.
        layer = self
        ep_shift = self.ep_rank * self.num_experts
        if VLLM_FORCE_INC:
            num_experts_on_rank = self.num_experts
            num_expert_group = 1
            num_expert_per_group = num_experts_on_rank // num_expert_group
            n_expert_slice = num_experts_on_rank // num_expert_group
            assert n_expert_slice * num_expert_group == num_experts_on_rank
            # if torch.distributed.get_rank() == 0:
            #     import pdb; pdb.set_trace()
            # torch.distributed.barrier()
            moe_n_slice = int(os.environ.get("VLLM_MOE_N_SLICE", 4))
            assert moe_n_slice == 1, f"moe_n_slice is {moe_n_slice}, expected 1 for INC"
            moe_lst = []
            for i in range(moe_n_slice):
                sub_expert_group = VllmMixtureOfExpertsOpFP8(
                    num_expert_per_group
                )
                min_expert = i * n_expert_slice
                max_expert = (i + 1) * n_expert_slice

                w13_list_slice = [
                    layer.w13_weight[j] for j in range(min_expert, max_expert)
                ]
                w13_weight_scale_inv_fp8_list = [
                    layer.w13_weight_scale_inv[j]
                    for j in range(min_expert, max_expert)
                ]
                w2_list_slice = [
                    layer.w2_weight[j] for j in range(min_expert, max_expert)
                ]
                w2_weight_scale_inv_fp8_list = [
                    layer.w2_weight_scale_inv[j]
                    for j in range(min_expert, max_expert)
                ]
                for index in range(len(w2_list_slice)):
                    sub_expert_group.w13_list[index].set_weight(
                        w13_list_slice[index]
                    )
                    sub_expert_group.w13_list[index].set_scale_inv_fp8(
                        w13_weight_scale_inv_fp8_list[index]
                    )
                    sub_expert_group.w13_list[index].set_weight_block_size(
                        layer.quant_config.weight_block_size
                    )

                    sub_expert_group.w2_list[index].set_weight(
                        w2_list_slice[index]
                    )
                    sub_expert_group.w2_list[index].set_scale_inv_fp8(
                        w2_weight_scale_inv_fp8_list[index]
                    )
                    sub_expert_group.w2_list[index].set_weight_block_size(
                        layer.quant_config.weight_block_size
                    )

                # FIXME: (Yi) pass `experts_min` and `experts_max` to MoeOp.
                setattr(
                    sub_expert_group, "experts_min", min_expert + ep_shift
                )
                setattr(
                    sub_expert_group, "experts_max", max_expert - 1 + ep_shift
                )
                # setattr(self, f"sub_expert_group_{i}", sub_expert_group)
                moe_lst.append(sub_expert_group)
                htorch.core.mark_step()
            self.moe_lst = torch.nn.ModuleList(moe_lst)
            htorch.core.mark_step()

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.tensor,
                                                 tp_rank: int,
                                                 expert_id: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          expert_id=expert_id,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank,
                           expert_id=expert_id)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int,
                                       expert_id: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank,
                           expert_id=expert_id)

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.tensor,
                  tp_rank: int,
                  expert_id: Optional[int] = None):

        orig_exp_data = expert_data.view(expert_data.size())
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

        if is_hpu and not VLLM_FORCE_INC:
            self.hpu_fused_moe.MoeOp.w13_list[expert_id].set_weight(
                orig_exp_data)
            # print(f"loaded w13 for hpu for expert_id: {expert_id}, orig_exp_data.shape: {orig_exp_data.shape}")

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.tensor,
                 tp_rank: int,
                 load_full: bool = False,
                 expert_id: Optional[int] = None):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)
        if is_hpu and not VLLM_FORCE_INC:
            self.hpu_fused_moe.MoeOp.w2_list[expert_id].set_weight(expert_data)
            # print(f"loaded w2 for hpu for expert_id: {expert_id}, expert_data.shape: {expert_data.shape}")

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:

        tp_rank = get_tensor_model_parallel_rank()
        if self.ep_size > 1:
            tp_rank = tp_rank // self.ep_size
            # now we want to only load weights for current expert group
            expert_id = expert_id - self.ep_rank * self.num_experts
            if expert_id < 0 or expert_id >= self.num_experts:
                return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = loaded_weight.t().contiguous() if (
            self.quant_method.__class__.__name__
            == "CompressedTensorsWNA16MoEMethod") else loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=tp_rank)
            return

        # Case weight scales and zero_points
        if ("scale" in weight_name or "zero" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    expert_id=expert_id)
            # elif current_platform.is_hpu():
            #     print(f"!!!!!!!!!!!!! HPU load per channel weight scale")
            #     self._load_per_channel_weight_scale(
            #         shard_id=shard_id,
            #         shard_dim=shard_dim,
            #         loaded_weight=loaded_weight,
            #         expert_data=expert_data,
            #         tp_rank=tp_rank,
            #         expert_id=expert_id)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    expert_id=expert_id,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                expert_id=expert_id)
            return

    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk, grouped_topk)

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                                gating_output=router_logits,
                                                topk=top_k,
                                                renormalize=renormalize)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            ep_rank=self.ep_rank)

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> List[Tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]
