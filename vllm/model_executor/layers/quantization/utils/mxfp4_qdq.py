
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MXFP4 input activation quant-dequant (QDQ) for accuracy study.

This simulates the information loss of quantizing activations to MXFP4
(FP4 E2M1 values with per-group-32 E8M0 scales), then dequantizing back
to the original dtype. Used to study accuracy impact of "real" MXFP4 GEMM
vs the current weight-only-dequant Marlin approach.

Enable with: VLLM_MXFP4_INPUT_QDQ=1
"""

import os

import torch

MXFP4_INPUT_QDQ_ENABLED = os.environ.get("VLLM_MXFP4_INPUT_QDQ", "0") == "1"

# FP4 E2M1 representable values (positive): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# Midpoint boundaries for rounding to nearest
_FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def mxfp4_qdq(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Quantize-dequantize input to MXFP4 (E2M1 + E8M0 scales).

    Args:
        x: 2D tensor [M, K] in bf16/fp16/fp32
        group_size: number of elements per scale group (default 32)

    Returns:
        Tensor same shape and dtype as x, with MXFP4 quantization noise applied.
    """
    orig_dtype = x.dtype
    m, k = x.shape

    # Pad k to multiple of group_size
    pad = (group_size - k % group_size) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    x_view = x.reshape(m, -1, group_size)  # [m, num_groups, group_size]

    # Compute E8M0 per-group scales: ceil to power of 2
    # E8M0 can only represent powers of 2 (8-bit exponent, no mantissa)
    amax = x_view.abs().float().amax(dim=2).clamp_min(1e-12)  # [m, ng]
    # Scale so max FP4 value (6.0) covers the group's range
    sf = amax / 6.0
    # Ceil to nearest power of 2 (E8M0 representation)
    sf = torch.exp2(torch.ceil(torch.log2(sf)))

    # Scale input by 1/sf and quantize to nearest FP4 E2M1
    x_scaled = x_view.float() / sf.unsqueeze(2)
    ax = x_scaled.abs().clamp_max(6.0)

    boundaries = torch.tensor(
        _FP4_BOUNDARIES, device=x.device, dtype=torch.float32
    )
    fp4_values = torch.tensor(
        _FP4_VALUES, device=x.device, dtype=torch.float32
    )

    idx = torch.bucketize(ax, boundaries)
    dequant = fp4_values[idx] * x_scaled.sign()

    # Rescale back
    result = (dequant * sf.unsqueeze(2)).reshape(m, -1)
    return result[:, :k].to(orig_dtype)
