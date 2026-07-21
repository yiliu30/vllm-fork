# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ARK (auto-round-lib) GEMM kernel for 2-bit WNA16 on XPU."""

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_ARK_SUPPORTED_QUANT_TYPES = (scalar_types.uint2b2,)

logger = init_logger(__name__)


class ARKLinearKernel(MPLinearKernel):
    """XPU linear kernel for 2-bit WNA16 via auto-round-kernel (ARK).

    Weights are symmetric group-quantized int2 packed as uint2b2.
    During ``process_weights_after_loading`` the compressed-tensors layout
    is transposed into ARK/GPTQ layout and repacked into the 1-D blob that
    ``ark.woqgemm`` consumes.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "ARK only supported on XPU"

        if c.act_type not in (torch.bfloat16, torch.float16):
            return False, "ARK requires BF16/FP16 activations"

        if c.weight_type not in _ARK_SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"ARK, supported types: {_ARK_SUPPORTED_QUANT_TYPES}",
            )

        if c.zero_points:
            return False, "ARK only supports symmetric weight quantization"

        if c.has_g_idx:
            return False, "ARK does not support act-order (g_idx)"

        if c.group_size != -1 and c.group_size % 32 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "ARK, must be a multiple of 32",
            )

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"Input size ({c.partition_weight_shape[0]}) not "
                "supported by ARK, must be a multiple of 32",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from auto_round_kernel.qlinear import QuantLinear

        device = layer.weight_packed.device
        in_features = layer.input_size_per_partition
        out_features = layer.output_size_per_partition
        bits = self.config.weight_type.size_bits
        group_size = self.config.group_size
        params_dtype = layer.params_dtype

        qw = layer.weight_packed.data.t().contiguous()
        sc = layer.weight_scale.data.t().contiguous()

        ark_linear = QuantLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            bias=layer.has_bias,
        )
        ark_linear.to(device)

        with torch.no_grad():
            ark_linear.qweight.copy_(qw)
            ark_linear.scales.copy_(sc)

        # QuantLinear.post_init() selects the actual compute/scale dtypes for
        # the current device and repacks the weight blob accordingly. Preserve
        # those choices instead of overriding them afterward.
        ark_linear.post_init()

        # Move the repacked 1-D weight blob to the XPU device.
        ark_linear.qweight = ark_linear.qweight.to(device)

        layer.ark_qweight = Parameter(ark_linear.qweight.detach(), requires_grad=False)
        layer.ark_bias = ark_linear.bias
        layer.ark_compute_type = ark_linear.cdt
        layer.ark_weight_type = ark_linear.wdt
        layer.ark_scale_type = ark_linear.sdt
        layer.ark_torch_dtype = ark_linear.torch_dt
        layer.ark_out_features = out_features
        layer.ark_in_features = in_features
        layer.ark_group_size = group_size

        # Free original parameters to save memory.
        del layer.weight_scale
        if hasattr(layer, "weight_zero_point"):
            del layer.weight_zero_point

        logger.info_once(
            "ARKLinearKernel: repacked layer %s (bits=%d, in=%d, out=%d, g=%d)",
            layer.__class__.__name__,
            bits,
            in_features,
            out_features,
            group_size,
            scope="local",
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GEMM via ``auto_round_kernel.woqgemm``.

        ``woqgemm_linear`` expects the raw 1-D repacked blob from
        ``QuantLinear.post_init``.  The blob is stored on the XPU device
        by ``process_weights_after_loading``.
        """
        import auto_round_kernel as ark

        raw_input_dtype = x.dtype
        x = x.to(layer.ark_torch_dtype)
        out = ark.woqgemm(
            x,
            layer.ark_qweight,
            layer.ark_bias if layer.ark_bias.numel() > 0
            else torch.empty(0, dtype=x.dtype, device=x.device),
            layer.ark_out_features,
            layer.ark_in_features,
            layer.ark_group_size,
            layer.ark_compute_type,
            layer.ark_weight_type,
            layer.ark_scale_type,
            asym=False,
        )
        return out.to(raw_input_dtype)
