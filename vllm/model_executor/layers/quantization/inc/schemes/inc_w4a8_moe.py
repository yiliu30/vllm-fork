# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
from vllm.model_executor.utils import replace_parameter

from .inc_wna16_moe import INCARKWNA16MoEMethod

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig

W4A8_MOE_SYMBOLS = ("moe_w4a8_prepack", "moe_gemm_w4a8")


def has_ark_w4a8_moe_kernel(is_ark_available: bool, ark) -> bool:
    xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
    return (
        is_ark_available
        and ark is not None
        and xpu_lib is not None
        and all(
            hasattr(ark, symbol) and hasattr(xpu_lib, symbol)
            for symbol in W4A8_MOE_SYMBOLS
        )
    )


def _effective_moe_group_size(
    layer_config: "INCLayerConfig",
    hidden_size: int,
    intermediate_size: int,
    prefix: str,
) -> int:
    group_size = layer_config.group_size
    while intermediate_size % group_size or hidden_size % group_size:
        group_size //= 2
        if group_size < 32:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires hidden and "
                "intermediate sizes divisible by a derived group size of at "
                f"least 32. Layer: {prefix}."
            )
    return group_size


def check_xpu_moe_w4a8_supported(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
    prefix: str,
) -> int:
    moe_config = layer.moe_config
    hidden_size = moe_config.hidden_dim
    intermediate_size = moe_config.intermediate_size_per_partition
    group_size = _effective_moe_group_size(
        layer_config,
        hidden_size,
        intermediate_size,
        prefix,
    )

    if group_size <= 0 or group_size % 8 != 0:
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires a MoE group size that "
            f"is a positive multiple of 8, got {group_size}. Layer: {prefix}."
        )

    gemm_shapes = (
        (
            "w13",
            moe_config.w13_num_shards * intermediate_size,
            hidden_size,
        ),
        ("w2", hidden_size, intermediate_size),
    )
    for name, out_features, in_features in gemm_shapes:
        if out_features % 16 != 0 or in_features % 64 != 0:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires MoE GEMM shapes "
                "with N multiple of 16 and K multiple of 64, got "
                f"{name}: N={out_features}, K={in_features}. Layer: {prefix}."
            )
        if in_features % group_size != 0:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires K to be divisible "
                f"by group_size, got {name}: K={in_features}, "
                f"group_size={group_size}. Layer: {prefix}."
            )
    return group_size


class INCARKW4A8MoEMethod(INCARKWNA16MoEMethod):
    kernel_name = "W4A8"
    log_message = "Using ARK XPU W4A8 MoE kernel."

    def __init__(
        self,
        quant_config: MoeWNA16Config,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(quant_config, moe)
        self.w13_moe_w4a8: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.w2_moe_w4a8: tuple[torch.Tensor, torch.Tensor, int] | None = None

    def _has_ark_moe_kernel(self, is_available, ark, xpu_lib) -> bool:
        del xpu_lib
        return has_ark_w4a8_moe_kernel(is_available, ark)

    @staticmethod
    def _signed_w4a8_qweight(qweight: torch.Tensor) -> torch.Tensor:
        qweight = qweight.detach()
        if qweight.is_contiguous():
            qweight.bitwise_xor_(0x88)
            return qweight
        return torch.bitwise_xor(qweight.contiguous(), 0x88)

    @staticmethod
    def _contiguous_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if tensor.is_contiguous():
            return tensor
        return tensor.contiguous()

    def _make_w4a8_moe_weight(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        signed_qweight = self._signed_w4a8_qweight(qweight)
        contiguous_scales = self._contiguous_tensor(scales)
        return signed_qweight, contiguous_scales, group_size

    def process_weights_after_loading(self, layer) -> None:
        self._setup_common_moe_state(layer)

        w13_qweight, w13_scales, group_size = self._make_w4a8_moe_weight(
            layer.w13_qweight,
            layer.w13_scales,
            layer.group_size,
        )
        replace_parameter(layer, "w13_qweight", w13_qweight)
        replace_parameter(layer, "w13_scales", w13_scales)

        w2_qweight, w2_scales, _ = self._make_w4a8_moe_weight(
            layer.w2_qweight,
            layer.w2_scales,
            layer.group_size,
        )
        replace_parameter(layer, "w2_qweight", w2_qweight)
        replace_parameter(layer, "w2_scales", w2_scales)

        self.w13_moe_w4a8 = (
            layer.w13_qweight,
            layer.w13_scales,
            group_size,
        )
        self.w2_moe_w4a8 = (layer.w2_qweight, layer.w2_scales, group_size)

    def _check_moe_weights_loaded(self) -> None:
        assert self.w13_moe_w4a8 is not None
        assert self.w2_moe_w4a8 is not None

    def _apply_w4a8_moe(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
        packed: tuple[torch.Tensor, torch.Tensor, int],
    ) -> torch.Tensor:
        qweight, scales, group_size = packed
        weights_s8, wscales, block = self.ark.moe_w4a8_prepack(
            qweight,
            scales,
            group_size=group_size,
        )
        return self.ark.moe_gemm_w4a8(
            x,
            weights_s8,
            wscales,
            rows_per_expert,
            rescale_block_size=block,
            phase="auto",
        )

    def _apply_w13_moe(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        assert self.w13_moe_w4a8 is not None
        return self._apply_w4a8_moe(x, rows_per_expert, self.w13_moe_w4a8)

    def _apply_w2_moe(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        assert self.w2_moe_w4a8 is not None
        return self._apply_w4a8_moe(x, rows_per_expert, self.w2_moe_w4a8)
