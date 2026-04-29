# Triton Attention Remaining Bug Plan

## Goal

Find and fix the remaining correctness bug after the K/V descale and `q_descale`
fixes, and confirm that forced `TRITON_ATTN` produces sane generations for the
local FP8 attention checkpoint.

## Done criteria

- Reproduce the bad generation output with the local command or an equivalent
  focused script.
- State the remaining root cause in one sentence with file and code-path
  evidence.
- Apply the smallest fix that addresses the root cause.
- Add or update a focused regression test for the failing case.
- Re-run a real generation check and confirm the output is no longer corrupted.

## Hypotheses to test

1. The remaining bug is still inside Triton attention and only appears on one
   execution path, such as prefill vs decode or 2D vs 3D.
2. The attention math applies `q_descale`, `k_descale`, and `v_descale`, but one
   scale is still missing or applied in the wrong place for the local
   compressed-tensors checkpoint.
3. The kernel math is now correct, and the remaining corruption comes from a
   surrounding integration path that feeds Triton attention the wrong scales or
   wrong cache format.

## Trace steps

1. Reproduce the bad output with the user's runtime settings.
2. Compare a small Triton attention call against a reference for the exact
   quantization mode and scale shapes used by the checkpoint.
3. Split the failing path by execution mode:
   - prefill vs decode
   - 2D vs 3D Triton kernels
   - FP8 query fast path vs non-FP8 query path
4. Identify the exact missing or misplaced scale or layout transform.
5. Patch the root cause and add a regression test.
6. Re-run the real generation command and compare output quality.
