# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental Blackwell dual-scale MXFP4 dense Linear kernel."""

import functools
import threading

import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


_COARSE_K = 512
_FP4_MAX = 6.0
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = _COARSE_K
_NUM_WARPS = 4

_tmem_workaround_installed = False
_tmem_workaround_lock = threading.Lock()


def _install_tmem_workaround() -> None:
    """Disable the broken SM100 accumulator-init pass for this process."""
    global _tmem_workaround_installed
    if _tmem_workaround_installed:
        return

    with _tmem_workaround_lock:
        if _tmem_workaround_installed:
            return
        from triton.backends.nvidia.compiler import CUDABackend, passes

        original_make_ttgir = CUDABackend.make_ttgir
        original_accumulator_init = passes.ttgpuir.add_optimize_accumulator_init

        @functools.wraps(original_make_ttgir)
        def make_ttgir(mod, metadata, opt, capability):
            if capability < 100:
                return original_make_ttgir(mod, metadata, opt, capability)
            passes.ttgpuir.add_optimize_accumulator_init = lambda pm: None
            try:
                return original_make_ttgir(mod, metadata, opt, capability)
            finally:
                passes.ttgpuir.add_optimize_accumulator_init = original_accumulator_init

        CUDABackend.make_ttgir = staticmethod(make_ttgir)
        _tmem_workaround_installed = True


def _quantize_dualscale(x: torch.Tensor):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dual-scale MXFP4 activations require float16 or bfloat16")
    if x.ndim != 2 or x.shape[1] % _COARSE_K != 0:
        raise ValueError(
            "dual-scale MXFP4 activations must be 2D with K divisible by 512"
        )

    rows, K = x.shape
    x_float = x.to(torch.float32)
    coarse = x_float.reshape(rows, K // _COARSE_K, _COARSE_K).abs().amax(dim=-1)
    coarse = coarse / _FP4_MAX
    coarse = torch.where(coarse == 0, torch.ones_like(coarse), coarse).contiguous()
    normalized = (
        x_float.reshape(rows, K // _COARSE_K, _COARSE_K) / coarse[..., None]
    ).reshape(rows, K)
    packed, fine, _ = downcast_to_mxfp(
        normalized.to(x.dtype),
        axis=1,
        BLOCK_OUT_DIM=_BLOCK_M,
        BLOCK_QUANT_DIM=_COARSE_K,
    )
    return packed.contiguous(), fine.contiguous(), coarse


@triton.jit
def _dualscale_mxfp4_gemm(
    a_ptr,
    b_ptr,
    a_fine_ptr,
    b_fine_ptr,
    a_coarse_ptr,
    b_coarse_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_afm,
    stride_afk,
    stride_bfm,
    stride_bfk,
    stride_acm,
    stride_ack,
    stride_bcn,
    stride_bck,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Keep these literals in the JIT function.  The vLLM Triton version does
    # not treat ordinary Python module globals as compile-time constants.
    tl.static_assert(BLOCK_K == 512)
    tl.static_assert(BLOCK_K % 32 == 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    offs_k_scale = tl.arange(0, BLOCK_K // 32)
    a_rows = offs_m[:, None]
    b_rows = offs_n[:, None]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr
            + a_rows * stride_am
            + (k_start // 2 + offs_k_packed[None, :]) * stride_ak,
            mask=m_mask[:, None],
            other=0,
        )
        b = tl.load(
            b_ptr
            + b_rows * stride_bn
            + (k_start // 2 + offs_k_packed[None, :]) * stride_bk,
            mask=n_mask[:, None],
            other=0,
        )
        a_fine = tl.load(
            a_fine_ptr
            + a_rows * stride_afm
            + (k_start // 32 + offs_k_scale[None, :]) * stride_afk,
            mask=m_mask[:, None],
            other=127,
        )
        b_fine = tl.load(
            b_fine_ptr
            + b_rows * stride_bfm
            + (k_start // 32 + offs_k_scale[None, :]) * stride_bfk,
            mask=n_mask[:, None],
            other=127,
        )
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        partial = tl.dot_scaled(
            a,
            a_fine,
            "e2m1",
            tl.trans(b),
            b_fine,
            "e2m1",
            partial,
            fast_math=True,
            lhs_k_pack=True,
            rhs_k_pack=True,
        )
        coarse_block = k_start // BLOCK_K
        a_coarse = tl.load(
            a_coarse_ptr + offs_m * stride_acm + coarse_block * stride_ack,
            mask=m_mask,
            other=1.0,
        )
        b_coarse = tl.load(
            b_coarse_ptr + offs_n * stride_bcn + coarse_block * stride_bck,
            mask=n_mask,
            other=1.0,
        )
        accumulator += partial * a_coarse[:, None] * b_coarse[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=m_mask[:, None] & n_mask[None, :])


class DualScaleMxfp4LinearKernel(MxFp4LinearKernel):
    """Experimental W4A4 dual-scale MXFP4 kernel for dense Linear layers."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del cls
        if not current_platform.is_cuda():
            return False, "dual-scale MXFP4 requires CUDA"
        if compute_capability is not None:
            supported = compute_capability >= 100
        else:
            supported = current_platform.has_device_capability(100)
        if not supported:
            return False, "dual-scale MXFP4 requires Blackwell SM100+"
        return True, None

    @classmethod
    def can_implement(
        cls, config: MxFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        del cls
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "dual-scale MXFP4 requires dynamic MXFP4 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "weight_coarse_scale"):
            raise ValueError(
                "dual-scale MXFP4 checkpoint is missing weight_coarse_scale"
            )
        if layer.weight.shape[1] * 2 % _COARSE_K != 0:
            raise ValueError("dual-scale MXFP4 weights require K divisible by 512")
        expected_fine = (layer.weight.shape[0], layer.weight.shape[1] * 2 // 32)
        if tuple(layer.weight_scale.shape) != expected_fine:
            raise ValueError(
                "invalid weight_scale shape: "
                f"expected {expected_fine}, got {tuple(layer.weight_scale.shape)}"
            )
        if layer.weight_scale.dtype != torch.uint8:
            raise TypeError("weight_scale must have dtype torch.uint8")
        expected_coarse = (
            layer.weight.shape[0],
            layer.weight.shape[1] * 2 // _COARSE_K,
        )
        if tuple(layer.weight_coarse_scale.shape) != expected_coarse:
            raise ValueError(
                "invalid weight_coarse_scale shape: "
                f"expected {expected_coarse}, "
                f"got {tuple(layer.weight_coarse_scale.shape)}"
            )
        if layer.weight_coarse_scale.dtype != torch.float32:
            raise TypeError("weight_coarse_scale must have dtype torch.float32")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        a_packed, a_fine, a_coarse = _quantize_dualscale(x_2d)
        weight = layer.weight
        M, K_packed = a_packed.shape
        N = weight.shape[0]
        K = K_packed * 2
        if tuple(weight.shape) != (N, K_packed):
            raise ValueError("dual-scale MXFP4 weight and activation K dimensions differ")

        output = torch.empty((M, N), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)
        _dualscale_mxfp4_gemm[grid](
            a_packed,
            weight,
            a_fine,
            layer.weight_scale,
            a_coarse,
            layer.weight_coarse_scale,
            output,
            M,
            N,
            K,
            a_packed.stride(0),
            a_packed.stride(1),
            weight.stride(0),
            weight.stride(1),
            a_fine.stride(0),
            a_fine.stride(1),
            layer.weight_scale.stride(0),
            layer.weight_scale.stride(1),
            a_coarse.stride(0),
            a_coarse.stride(1),
            layer.weight_coarse_scale.stride(0),
            layer.weight_coarse_scale.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(*x.shape[:-1], N)


def init_dualscale_mxfp4_linear_kernel() -> DualScaleMxfp4LinearKernel:
    """Create the opt-in Blackwell dual-scale MXFP4 kernel."""
    supported, reason = DualScaleMxfp4LinearKernel.is_supported()
    if not supported:
        raise ValueError(f"Cannot initialize dual-scale MXFP4: {reason}")
    _install_tmem_workaround()
    return DualScaleMxfp4LinearKernel(
        MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    )
