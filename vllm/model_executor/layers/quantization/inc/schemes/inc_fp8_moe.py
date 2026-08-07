# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig

logger = init_logger(__name__)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _resolve_fp8_dtype(data_type: str) -> torch.dtype:
    return torch.float8_e5m2 if "e5m2" in data_type.lower() else torch.float8_e4m3fn


def _require_supported_scale_dtype(params_dtype: torch.dtype) -> torch.dtype:
    if params_dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("ARK FP8 W8A16 MoE requires fp16/bf16 activations.")
    return params_dtype


class INCARKFp8MoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        moe,
        group_size: int,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__(moe)
        if weight_dtype not in _FP8_DTYPES:
            raise ValueError(f"Unsupported ARK FP8 dtype: {weight_dtype}")

        from .inc_ark_ops import get_ark_state

        is_available, error_str, ark, _ = get_ark_state()
        xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
        has_moe_kernel = (
            is_available
            and ark is not None
            and hasattr(ark, "moe")
            and xpu_lib is not None
            and hasattr(xpu_lib, "moe_gemm_decode")
            and hasattr(xpu_lib, "moe_gemm_prefill")
        )
        if not has_moe_kernel:
            reason = error_str or "ARK MoE kernels are unavailable."
            raise ImportError(f"Failed to initialize ARK FP8 W8A16 MoE. {reason}")

        self.ark = ark
        self.config_group_size = group_size
        self.weight_dtype = weight_dtype
        self.w13_group_size: int | None = None
        self.w2_group_size: int | None = None
        self.local_to_global_experts: tuple[int, ...] | None = None

        logger.info_once("Using ARK XPU FP8 W8A16 MoE kernel.")

    @staticmethod
    def _effective_group_size(config_group_size: int, input_size: int) -> int:
        return input_size if config_group_size <= 0 else config_group_size

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.moe.has_bias:
            raise NotImplementedError("ARK FP8 W8A16 MoE does not support bias yet.")

        scale_dtype = _require_supported_scale_dtype(params_dtype)
        self.w13_group_size = self._effective_group_size(
            self.config_group_size, hidden_size
        )
        self.w2_group_size = self._effective_group_size(
            self.config_group_size, intermediate_size_per_partition
        )

        if hidden_size % self.w13_group_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"w13_group_size={self.w13_group_size}."
            )
        if intermediate_size_per_partition % self.w2_group_size != 0:
            raise ValueError(
                f"intermediate_size_per_partition={intermediate_size_per_partition} "
                f"must be divisible by w2_group_size={self.w2_group_size}."
            )

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.w13_group_size = self.w13_group_size
        layer.w2_group_size = self.w2_group_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=self.weight_dtype,
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
                dtype=self.weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.w13_group_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.w2_group_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        # This ARK path calls grouped MoE GEMM directly and does not use
        # vLLM's modular-kernel quant-config plumbing.
        del layer
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Keep the ARK FP8 MoE path on the legacy direct-call implementation.
        del routing_tables
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if layer.w13_weight.dtype not in _FP8_DTYPES:
            raise ValueError(f"w13_weight must be FP8, got {layer.w13_weight.dtype}")
        if layer.w2_weight.dtype not in _FP8_DTYPES:
            raise ValueError(f"w2_weight must be FP8, got {layer.w2_weight.dtype}")

        scale_dtype = _require_supported_scale_dtype(layer.params_dtype)
        for scale_name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, scale_name)
            if scale.dtype == torch.uint8:
                raise NotImplementedError(
                    "ARK FP8 W8A16 expects numeric fp16/bf16 scales. "
                    "MXFP8 E8M0 uint8 scales need a separate conversion path."
                )
            replace_parameter(
                layer,
                scale_name,
                scale.detach().to(dtype=scale_dtype).contiguous(),
            )

        replace_parameter(layer, "w13_weight", layer.w13_weight.detach().contiguous())
        replace_parameter(layer, "w2_weight", layer.w2_weight.detach().contiguous())

        self.w13_group_size = layer.w13_group_size
        self.w2_group_size = layer.w2_group_size

        num_local_experts = layer.w13_weight.shape[0]
        if layer.expert_map is None:
            self.local_to_global_experts = tuple(range(num_local_experts))
        else:
            local_to_global = [-1] * num_local_experts
            expert_map_cpu = layer.expert_map.detach().cpu()
            for global_expert_id, local_expert_id_tensor in enumerate(expert_map_cpu):
                local_expert_id = int(local_expert_id_tensor)
                if 0 <= local_expert_id < num_local_experts:
                    local_to_global[local_expert_id] = global_expert_id
            self.local_to_global_experts = tuple(local_to_global)

    def _ark_moe(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        if scales.dtype != activations.dtype:
            scales = scales.to(dtype=activations.dtype)

        assert self.ark is not None

        return self.ark.moe(
            activations.contiguous(),
            weights,
            num_tokens_per_expert,
            scales=scales,
            zeros=None,
            weight_bits=8,
            group_size=group_size,
            asym=False,
            phase="auto",
        )

    def _make_compact_inputs(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.local_to_global_experts is not None

        token_indices_per_expert: list[torch.Tensor] = []
        topk_slots_per_expert: list[torch.Tensor] = []
        num_tokens_per_expert_list: list[int] = []

        for global_expert_id in self.local_to_global_experts:
            if global_expert_id < 0:
                num_tokens_per_expert_list.append(0)
                continue

            token_indices, topk_slots = torch.where(topk_ids == global_expert_id)
            num_tokens_per_expert_list.append(token_indices.numel())
            if token_indices.numel() > 0:
                token_indices_per_expert.append(token_indices)
                topk_slots_per_expert.append(topk_slots)

        num_tokens_per_expert = torch.tensor(
            num_tokens_per_expert_list,
            dtype=torch.int32,
            device=x.device,
        )

        if not token_indices_per_expert:
            empty_indices = torch.empty((0,), dtype=torch.long, device=x.device)
            return (
                x.new_empty((0, x.shape[-1])),
                num_tokens_per_expert,
                empty_indices,
                empty_indices,
            )

        token_indices = torch.cat(token_indices_per_expert)
        topk_slots = torch.cat(topk_slots_per_expert)
        compact_x = x.index_select(0, token_indices)
        return compact_x, num_tokens_per_expert, token_indices, topk_slots

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input

        assert self.w13_group_size is not None
        assert self.w2_group_size is not None

        compact_x, num_tokens_per_expert, token_indices, topk_slots = (
            self._make_compact_inputs(x, topk_ids)
        )

        output = torch.zeros_like(x)
        if compact_x.numel() == 0:
            return output

        if layer.apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise NotImplementedError(
                    "apply_router_weight_on_input is only supported for topk=1."
                )
            route_weights = topk_weights[token_indices, topk_slots].to(compact_x.dtype)
            compact_x = compact_x * route_weights.unsqueeze(-1)

        compact_w13 = self._ark_moe(
            compact_x,
            layer.w13_weight,
            layer.w13_weight_scale,
            num_tokens_per_expert,
            self.w13_group_size,
        )

        activation = layer.activation
        activated_size = (
            compact_w13.shape[-1] // 2 if activation.is_gated else compact_w13.shape[-1]
        )
        compact_activated = compact_w13.new_empty(
            (compact_w13.shape[0], activated_size)
        )
        apply_moe_activation(activation, compact_activated, compact_w13)

        compact_out = self._ark_moe(
            compact_activated,
            layer.w2_weight,
            layer.w2_weight_scale,
            num_tokens_per_expert,
            self.w2_group_size,
        )

        if not layer.apply_router_weight_on_input:
            route_weights = topk_weights[token_indices, topk_slots].to(
                compact_out.dtype
            )
            compact_out = compact_out * route_weights.unsqueeze(-1)

        output.index_add_(0, token_indices, compact_out.to(output.dtype))
        return output


class INCFp8MoEScheme:
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.layer_config = layer_config

    def get_method(self, layer: torch.nn.Module):
        if current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod

            return UnquantizedFusedMoEMethod(layer.moe_config)

        if not current_platform.is_xpu():
            raise NotImplementedError("ARK FP8 W8A16 MoE currently only supports XPU.")

        assert isinstance(self.layer_config.group_size, int)

        return INCARKFp8MoEMethod(
            layer.moe_config,
            group_size=self.layer_config.group_size,
            weight_dtype=_resolve_fp8_dtype(self.layer_config.data_type),
        )
