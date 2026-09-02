# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)

XPU_WNA16_SUPPORTED_BITS = {2, 4}

# Backends selectable through VLLM_XPU_INC_WNA16_BACKEND that are served by the
# oneDNN int4 GEMMs rather than by ARK. These only cover int4.
XPU_ONEDNN_BACKENDS = ("w4a16", "w4a8")


def _check_xpu_w4a8_supported(layer_config: "INCLayerConfig", prefix: str) -> None:
    """Raise unless ``int4_gemm_w4a8`` can serve this linear layer.

    The backend is requested explicitly, so an unusable configuration is an
    error rather than something to silently fall back from.
    """
    import torch

    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires the int4_gemm_w4a8 op, "
            "which this build of vllm-xpu-kernels does not provide. "
            f"Layer: {prefix}."
        )
    if layer_config.group_size <= 0 or layer_config.group_size % 32 != 0:
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires a group size that is a "
            f"positive multiple of 32, got {layer_config.group_size}. "
            f"Layer: {prefix}."
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


def _check_xpu_moe_w4a8_supported(
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


class INCWna16Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_wna16_int

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer
        if current_platform.is_xpu():
            if layer_config.bits in XPU_WNA16_SUPPORTED_BITS and layer_config.sym:
                from .inc_ark_ops import get_ark_state
                from .inc_w4a8_linear import INCXPUW4A8LinearMethod
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCXPULinearMethod,
                )

                backend = envs.VLLM_XPU_INC_WNA16_BACKEND
                if backend in XPU_ONEDNN_BACKENDS:
                    if layer_config.bits != 4:
                        raise NotImplementedError(
                            f"VLLM_XPU_INC_WNA16_BACKEND={backend} only supports "
                            f"int4, got int{layer_config.bits}. Layer: {prefix}."
                        )
                    if backend == "w4a8":
                        _check_xpu_w4a8_supported(layer_config, prefix)
                        return INCLinearMethod(INCXPUW4A8LinearMethod(layer_config))
                    return INCLinearMethod(INCXPULinearMethod(layer_config))

                is_ark_available, ark_error, _, _ = get_ark_state()
                if backend == "ark" and not is_ark_available:
                    raise NotImplementedError(
                        "VLLM_XPU_INC_WNA16_BACKEND=ark was requested but "
                        f"auto_round_kernel is unavailable: "
                        f"{ark_error or 'unknown error'}. Layer: {prefix}."
                    )
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))
                if layer_config.bits == 2:
                    raise NotImplementedError(
                        "INC int2 on XPU requires the ARK backend. "
                        f"Layer: {prefix}. "
                        f"auto_round_kernel unavailable: "
                        f"{ark_error or 'unknown error'}"
                    )

                logger.debug(
                    "ARK backend is unavailable for layer %s; "
                    "falling back to the default XPU INC path. Error: %s",
                    prefix,
                    ark_error or "unknown error",
                )
                return INCLinearMethod(INCXPULinearMethod(layer_config))
            raise NotImplementedError(f"INC on XPU: unsupported config {layer_config}")

        if current_platform.is_cpu() and layer_config.is_gptq:
            if layer_config.bits == 4 and layer_config.sym:
                from .inc_ark_ops import get_ark_state
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCWNA16LinearScheme,
                )

                is_ark_available, ark_error, _, _ = get_ark_state()
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))

                logger.debug(
                    "ARK backend is unavailable for layer %s; "
                    "falling back to the default CPU INC path. Error: %s",
                    prefix,
                    ark_error or "unknown error",
                )
                return INCLinearMethod(INCWNA16LinearScheme(layer_config))
            raise NotImplementedError(f"INC on CPU: unsupported config {layer_config}")

        from .inc_wna16_linear import INCWNA16LinearScheme

        return INCLinearMethod(INCWNA16LinearScheme(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config

        if (
            current_platform.is_xpu()
            and layer_config.is_gptq
            and layer_config.bits == 4
            and layer_config.sym
        ):
            from vllm.model_executor.layers.quantization.moe_wna16 import (
                MoeWNA16Config,
            )

            from .inc_ark_ops import get_ark_state
            from .inc_wna16_moe import (
                INCARKW4A8MoEMethod,
                INCARKWNA16MoEMethod,
                INCWNA16MoEScheme,
            )

            backend = envs.VLLM_XPU_INC_WNA16_BACKEND
            if backend == "w4a16":
                return INCWNA16MoEScheme(layer_config).get_method(layer)

            is_ark_available, ark_error, ark, _ = get_ark_state()
            xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
            w4a8_symbols = ("moe_w4a8_prepack", "moe_gemm_w4a8")
            is_ark_w4a8_moe_available = (
                is_ark_available
                and ark is not None
                and xpu_lib is not None
                and all(
                    hasattr(ark, symbol) and hasattr(xpu_lib, symbol)
                    for symbol in w4a8_symbols
                )
            )
            ark_moe_error = ark_error or "ARK MoE kernels are unavailable"
            if backend == "w4a8":
                if not is_ark_w4a8_moe_available:
                    raise NotImplementedError(
                        "VLLM_XPU_INC_WNA16_BACKEND=w4a8 was requested but "
                        f"ARK W4A8 MoE kernels are unavailable: "
                        f"{ark_moe_error}. Layer: {prefix}."
                    )
                _check_xpu_moe_w4a8_supported(
                    layer,
                    layer_config,
                    prefix,
                )
                moe_config = MoeWNA16Config.from_config(
                    {
                        "quant_method": "gptq",
                        "bits": layer_config.bits,
                        "group_size": layer_config.group_size,
                        "sym": layer_config.sym,
                        "lm_head": False,
                    }
                )
                return INCARKW4A8MoEMethod(moe_config, layer.moe_config)

            is_ark_moe_available = (
                is_ark_available
                and ark is not None
                and xpu_lib is not None
                and hasattr(ark, "MoeSymmetricGemm")
            )
            if backend == "ark" and not is_ark_moe_available:
                raise NotImplementedError(
                    "VLLM_XPU_INC_WNA16_BACKEND=ark was requested but "
                    f"ARK MoE kernels are unavailable: {ark_moe_error}. "
                    f"Layer: {prefix}."
                )

            if is_ark_moe_available:
                moe_config = MoeWNA16Config.from_config(
                    {
                        "quant_method": "gptq",
                        "bits": layer_config.bits,
                        "group_size": layer_config.group_size,
                        "sym": layer_config.sym,
                        "lm_head": False,
                    }
                )
                return INCARKWNA16MoEMethod(moe_config, layer.moe_config)

            logger.info(
                "ARK backend is unavailable for MoE layer %s; "
                "falling back to the default WNA16 MoE path. Error: %s",
                prefix,
                ark_moe_error,
            )

            return INCWNA16MoEScheme(layer_config).get_method(layer)

        from .inc_wna16_moe import INCWNA16MoEScheme

        return INCWNA16MoEScheme(layer_config).get_method(layer)
