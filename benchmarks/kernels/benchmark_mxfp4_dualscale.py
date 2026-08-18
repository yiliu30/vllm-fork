# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quick correctness check for the experimental INC dual-scale MXFP4 kernel.

This script exercises the same path used by the INC dense Linear method without
requiring a model checkpoint.  It is intentionally kept outside pytest because
it launches a GPU kernel and is useful as a manual integration smoke test.
"""

import argparse
import types

import torch

from vllm.model_executor.kernels.linear.mxfp4.dualscale import (
    _quantize_dualscale,
    init_dualscale_mxfp4_linear_kernel,
)


def _dequantize(
    packed: torch.Tensor, fine: torch.Tensor, coarse: torch.Tensor
) -> torch.Tensor:
    """Reference-dequantize packed dual-scale MXFP4 values."""
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )
    low = (packed & 0x7).long()
    high = ((packed >> 4) & 0x7).long()
    values = torch.stack(
        (fp4_values[low], fp4_values[high]), dim=-1
    ).reshape(packed.shape[0], -1)
    signs = torch.stack(
        ((packed & 0x8) != 0, (packed & 0x80) != 0), dim=-1
    ).reshape(packed.shape[0], -1)
    values = torch.where(signs, -values, values)
    fine_scale = (fine.to(torch.int32) * (1 << 23)).view(torch.float32)
    fine_scale = fine_scale.repeat_interleave(32, dim=1)
    coarse_scale = coarse.repeat_interleave(512, dim=1)
    return values * fine_scale * coarse_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("dual-scale MXFP4 requires CUDA")
    if args.k % 512 != 0:
        raise ValueError("K must be divisible by 512")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        (args.m, args.k), device="cuda", dtype=dtype, generator=generator
    )
    weights = torch.randn(
        (args.n, args.k), device="cuda", dtype=dtype, generator=generator
    )
    weight, weight_fine, weight_coarse = _quantize_dualscale(weights)
    layer = types.SimpleNamespace(
        weight=weight,
        weight_scale=weight_fine,
        weight_coarse_scale=weight_coarse,
    )

    kernel = init_dualscale_mxfp4_linear_kernel()
    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(layer, x)

    x_packed, x_fine, x_coarse = _quantize_dualscale(x)
    x_reference = _dequantize(x_packed, x_fine, x_coarse)
    weight_reference = _dequantize(weight, weight_fine, weight_coarse)
    reference = x_reference @ weight_reference.t()
    max_abs_error = (output.float() - reference).abs().max().item()
    torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=1.0)
    print(
        f"dual-scale MXFP4 OK: shape={tuple(output.shape)} "
        f"max_abs_error={max_abs_error:.6f}"
    )


if __name__ == "__main__":
    main()
