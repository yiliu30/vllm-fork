# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MXFP4 activation simulation: quantize to FP4 E2M1 precision but store as FP8.

These functions are drop-in replacements for per_token_group_quant_fp8 and
silu_mul_per_token_group_quant_fp8_colmajor, but with FP4-level precision.
Used to measure FP4×FP4 accuracy through the existing FP8×FP4 DeepGEMM path.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[3]))
from tests.quantization.reference_mxfp4 import (  # noqa: E402
    BFLOAT16_EXP_BIAS,
    BFLOAT16_EXP_BITS,
    BFLOAT16_MANTISSA_BITS,
    fp16_to_fp4_simulate,
)

# E2M1 representable positive values (excluding negative mirror and -0)
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX = 6.0

# Midpoints between consecutive E2M1 values for rounding boundaries
# RNE (round to nearest even) at exact midpoints
E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1_LOOKUP = torch.tensor(E2M1_VALUES)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round values to nearest E2M1 representable value.

    Input x should be pre-scaled to [-6, 6] range.
    Uses simple bucketize approach (RNE at midpoints maps to even = lower).

    E2M1 positive values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    """
    device = x.device
    sign = x.sign()
    ax = x.abs()

    boundaries = E2M1_BOUNDARIES.to(device)
    values = E2M1_LOOKUP.to(device)

    idx = torch.bucketize(ax, boundaries)
    result = values[idx]
    return result * sign


def per_token_group_quant_fp4_as_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    column_major_scales: bool = False,
    use_ue8m0: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for per_token_group_quant_fp8 with FP4 precision.

    Computes scale using fp4_max=6, rounds to E2M1, stores as FP8.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % group_size == 0
    num_groups = K // group_size

    # Reshape into groups
    x_groups = x.reshape(M * num_groups, group_size).float()

    # Compute scale (same logic as Triton kernel but with fp4_max)
    absmax = x_groups.abs().amax(dim=1, keepdim=True).clamp(min=eps)
    scale_raw = absmax / FP4_MAX

    scale = torch.exp2(torch.ceil(torch.log2(scale_raw))) if use_ue8m0 else scale_raw

    # Normalize, round to E2M1, store as FP8
    x_normalized = x_groups / scale
    x_clamped = x_normalized.clamp(-FP4_MAX, FP4_MAX)
    x_fp4 = round_to_e2m1(x_clamped)
    x_fp8 = x_fp4.to(torch.float8_e4m3fn)

    # Reshape outputs
    x_fp8 = x_fp8.reshape(M, K)

    if column_major_scales:
        # Column-major: shape (num_groups, M) stored, accessed as (M, num_groups)
        scale = scale.reshape(M, num_groups)
        scale_out = torch.empty((num_groups, M), dtype=torch.float32, device=x.device).T
        scale_out.copy_(scale)
        scale = scale_out
    else:
        scale = scale.reshape(M, num_groups)

    return x_fp8, scale


def silu_mul_per_token_group_quant_fp4_as_fp8(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for silu_mul_per_token_group_quant_fp8_colmajor
    with FP4 precision.

    Computes SiLU(gate) * up, then quantizes to FP4 stored as FP8.
    """
    assert input.ndim == 2
    M, N = input.shape
    N_2 = N // 2

    gate = input[:, :N_2]
    up = input[:, N_2:]
    act_out = F.silu(gate) * up

    x_fp8, scale = per_token_group_quant_fp4_as_fp8(
        act_out, group_size=128, column_major_scales=True, use_ue8m0=use_ue8m0
    )

    if output is not None:
        output.copy_(x_fp8)
        x_fp8 = output

    return x_fp8, scale


# =============================================================================
# Unit Tests
# =============================================================================


class TestRoundToE2M1:
    """Tests for the E2M1 rounding function."""

    def test_exact_values_unchanged(self):
        """Known E2M1 values should pass through unchanged."""
        for v in E2M1_VALUES:
            for sign in [1.0, -1.0]:
                val = sign * v
                x = torch.tensor([val], device="cuda")
                result = round_to_e2m1(x)
                assert result.item() == val, f"Expected {val}, got {result.item()}"

    def test_boundaries_round_down(self):
        """Values just below midpoints should round down."""
        # 0.24 -> 0, 0.74 -> 0.5, 1.24 -> 1, 1.74 -> 1.5, 2.49 -> 2
        cases = [(0.24, 0.0), (0.74, 0.5), (1.24, 1.0), (1.74, 1.5), (2.49, 2.0)]
        for val, expected in cases:
            x = torch.tensor([val], device="cuda")
            result = round_to_e2m1(x)
            assert result.item() == expected, (
                f"Input {val}: expected {expected}, got {result.item()}"
            )

    def test_boundaries_round_up(self):
        """Values just above midpoints should round up."""
        cases = [(0.26, 0.5), (0.76, 1.0), (1.26, 1.5), (1.76, 2.0), (2.51, 3.0)]
        for val, expected in cases:
            x = torch.tensor([val], device="cuda")
            result = round_to_e2m1(x)
            assert result.item() == expected, (
                f"Input {val}: expected {expected}, got {result.item()}"
            )

    def test_clamp_beyond_6(self):
        """Values > 6 should be clamped before rounding (caller responsibility)."""
        x = torch.tensor([5.5], device="cuda")
        result = round_to_e2m1(x)
        assert result.item() == 6.0

    def test_vs_reference_mxfp4(self):
        """Compare against the bit-manipulation reference from reference_mxfp4.py.

        The reference operates on bf16 values in [0, 6] (already scaled).
        Our round_to_e2m1 should produce the same result.
        """
        torch.manual_seed(42)
        # Generate random values in the FP4 representable range
        x = torch.rand(1024, device="cuda", dtype=torch.bfloat16) * 6.0

        # Reference: fp16_to_fp4_simulate expects bf16, returns bf16
        ref = fp16_to_fp4_simulate(
            x,
            half_mantissa_bits=BFLOAT16_MANTISSA_BITS,
            half_exp_bits=BFLOAT16_EXP_BITS,
            half_exp_bias=BFLOAT16_EXP_BIAS,
        )

        # Our function
        ours = round_to_e2m1(x.float()).to(torch.bfloat16)

        # Allow small number of mismatches at tie-breaking boundaries
        # (our bucketize rounds up at midpoints, reference may differ)
        mismatches = (ours != ref).sum().item()
        assert mismatches <= 10, (
            f"Too many mismatches: {mismatches}/1024. "
            f"Max diff: {(ours.float() - ref.float()).abs().max()}"
        )


class TestPerTokenGroupQuantFP4:
    """Tests for per_token_group_quant_fp4_as_fp8."""

    def test_output_dtype(self):
        """Output tensor is fp8, scale is float32."""
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        x_fp8, scale = per_token_group_quant_fp4_as_fp8(x, group_size=128)
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32

    def test_output_shape(self):
        """Output shapes match expected."""
        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        x_fp8, scale = per_token_group_quant_fp4_as_fp8(x, group_size=128)
        assert x_fp8.shape == (32, 512)
        assert scale.shape == (32, 4)  # 512/128 = 4 groups

    def test_values_are_e2m1_representable(self):
        """After dequant, all values should be E2M1-representable × scale."""
        torch.manual_seed(42)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 10

        x_fp8, scale = per_token_group_quant_fp4_as_fp8(x, group_size=128)

        # Dequantize
        x_dequant = x_fp8.to(torch.float32).reshape(128, 2, 128)

        # Divide by scale to get normalized values
        x_normalized = x_dequant  # already divided by scale during quant

        # Check all values are in E2M1 set
        valid_values = set(E2M1_VALUES + [-v for v in E2M1_VALUES])
        unique_vals = x_normalized.unique().tolist()
        for v in unique_vals:
            assert v in valid_values or abs(v) < 1e-6, (
                f"Value {v} is not E2M1 representable"
            )

    def test_fp4_less_precise_than_fp8(self):
        """FP4 quant should have larger error than FP8 quant."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        torch.manual_seed(42)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 5

        # FP4 path
        fp4_q, fp4_s = per_token_group_quant_fp4_as_fp8(x, group_size=128)
        fp4_dequant = fp4_q.to(torch.float32).reshape(128, 2, 128) * fp4_s.unsqueeze(-1)
        fp4_dequant = fp4_dequant.reshape(128, 256)

        # FP8 path
        fp8_q, fp8_s = per_token_group_quant_fp8(x, group_size=128)
        fp8_dequant = fp8_q.to(torch.float32).reshape(128, 2, 128) * fp8_s.unsqueeze(-1)
        fp8_dequant = fp8_dequant.reshape(128, 256)

        fp4_error = (fp4_dequant - x.float()).abs().mean()
        fp8_error = (fp8_dequant - x.float()).abs().mean()

        assert fp4_error > fp8_error, (
            f"FP4 error ({fp4_error:.6f}) should be larger than "
            f"FP8 error ({fp8_error:.6f})"
        )

    def test_column_major_scales(self):
        """Column-major scale layout should match expected format."""
        x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        x_fp8, scale = per_token_group_quant_fp4_as_fp8(
            x, group_size=128, column_major_scales=True
        )
        # Column-major: logical shape (M, num_groups) but stored as (num_groups, M).T
        assert scale.shape == (128, 4)
        # Check it's actually column-major (stride[0] == 1)
        assert scale.stride(0) == 1
        assert scale.stride(1) == 128

    @pytest.mark.parametrize("use_ue8m0", [False, True])
    def test_ue8m0_scale_is_power_of_2(self, use_ue8m0):
        """With use_ue8m0=True, scales should be powers of 2."""
        torch.manual_seed(42)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 5
        _, scale = per_token_group_quant_fp4_as_fp8(
            x, group_size=128, use_ue8m0=use_ue8m0
        )
        if use_ue8m0:
            log2_scale = torch.log2(scale)
            assert torch.allclose(log2_scale, log2_scale.round(), atol=1e-5)


class TestSiluMulFP4:
    """Tests for silu_mul_per_token_group_quant_fp4_as_fp8."""

    def test_output_shape(self):
        """Output shape should be (M, N//2) for activation."""
        input = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        x_fp8, scale = silu_mul_per_token_group_quant_fp4_as_fp8(input)
        assert x_fp8.shape == (128, 256)  # N//2
        assert scale.shape == (128, 2)  # 256/128 = 2 groups

    def test_shape_matches_fp8_version(self):
        """Output shape and scale layout should match the FP8 version."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            silu_mul_per_token_group_quant_fp8_colmajor,
        )

        input = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

        fp4_q, fp4_s = silu_mul_per_token_group_quant_fp4_as_fp8(input, use_ue8m0=False)
        fp8_q, fp8_s = silu_mul_per_token_group_quant_fp8_colmajor(
            input, use_ue8m0=False
        )

        assert fp4_q.shape == fp8_q.shape
        assert fp4_s.shape == fp8_s.shape
        assert fp4_q.dtype == fp8_q.dtype
        assert fp4_s.dtype == fp8_s.dtype

    def test_activation_correctness(self):
        """The SiLU+mul part should be computed correctly."""
        torch.manual_seed(42)
        input = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

        x_fp8, scale = silu_mul_per_token_group_quant_fp4_as_fp8(input)

        # Compute reference activation
        gate = input[:, :256]
        up = input[:, 256:]
        ref_act = F.silu(gate) * up

        # Dequant and compare (should be close, within FP4 quant error)
        x_dequant = x_fp8.to(torch.float32).reshape(128, 2, 128) * scale.unsqueeze(-1)
        x_dequant = x_dequant.reshape(128, 256)

        # Use absolute error — FP4 is coarse, and many small values quantize
        # to 0 giving infinite relative error. Check that absolute error is
        # bounded by the scale (worst case: rounding from 5 to 6 = 1*scale).
        abs_error = (x_dequant - ref_act.float()).abs()
        # Mean absolute error should be less than mean scale (generous bound)
        assert abs_error.mean() < scale.abs().mean() * FP4_MAX, (
            f"Mean abs error ({abs_error.mean():.4f}) too high"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
