# Triton Fused QK Page TopK — Bugs Found and Fixed

**Date:** 2026-06-28

## Context

The `fused_page` backend replaces the dense QK + CUDA topk pipeline with a
Triton kernel that computes max-per-page QK scores on-chip, then selects
top-P pages. All bugs were masked when `k_start=0` (single-request, no
batching in the K workspace), which is why the synthetic tests and
single-sample e2e tests passed.

## Bug 1: `page_end` computed with relative length instead of absolute position

**File:** `triton_fused_page_topk.py:86,186`

```python
# BUG: num_k is relative length (k_end - k_start), not absolute position
page_end = tl.minimum(page_start + storage_block_size, num_k)

# FIX: use absolute k_end
page_end = tl.minimum(page_start + storage_block_size, k_end)
```

**Impact:** When `k_start > 0` (batched requests), `page_start` is an
absolute position (e.g., 64), but `num_k` is a relative length (e.g., 100).
The `min()` comparison mixed units, causing the kernel to either load out-
of-range K tokens or skip valid tokens at the end of each page.

**Detection:** Synthetic test with `k_start=256` showed pages outside the
valid K range being scored.

## Bug 2: Double-counting `k_start` in token index computation

**File:** `triton_fused_page_topk.py:192`

```python
# BUG: tok_abs is already absolute (page_start includes first_page offset),
# but k_start was added again
tok_idx = k_start + tok_abs

# FIX: tok_abs is already in absolute K-space
tok_idx = tok_abs
```

**Impact:** All token indices were offset by `k_start`, pointing past the
valid K range for batched requests. This caused `_fill_topk_from_fused_candidates`
to produce all-invalid token indices, silently degrading accuracy.

**Detection:** Token indices for batched requests were consistently out-of-
range when `k_start > 0`.

## Bug 3: Relative page index treated as absolute logical page

**File:** `triton_fused_page_topk.py:123`, host function `fused_qk_page_topk`

```python
# BUG: page scores stored at relative position pg (0-indexed within valid pages),
# but host-side topk treated output indices as absolute logical page numbers.
# When first_page > 0, the mapping was off.

# FIX (kernel): store at relative position, output first_page separately
tl.store(out_first_page_ptr + ... , first_page)

# FIX (host): add first_page offset to convert relative → absolute
top_abs_indices = top_rel_indices + first_page.unsqueeze(-1)
```

**Impact:** Selected pages were offset by `first_page` positions from the
actual pages. For batched requests with `first_page > 0`, this caused wrong
page selection.

**Detection:** Selected pages for `k_start=256` included pages 0-3 (which
are in the valid range [0,8) for row 3 but not for row 0).

## Bug 4: `page_scores` not padded when `k_select < top_p`

**File:** `triton_fused_page_topk.py:384-388`

```python
# BUG: page_scores gathered from safe_rel_idx has shape [M, H, k_select]
# but is compared against top_abs_indices with shape [M, H, top_p]
# when k_select < top_p, torch.where fails with "size 16 vs 15"

page_scores = torch.where(
    top_abs_indices >= 0, page_scores,  # size mismatch!
    torch.full_like(page_scores, float("-inf")),
)

# FIX: pad page_scores to top_p before torch.where
if k_select < top_p:
    pad_s = torch.full((M, H, top_p - k_select), float("-inf"), ...)
    page_scores = torch.cat([page_scores, pad_s], dim=-1)
```

**Impact:** `RuntimeError: size of tensor a (16) must match size of tensor b (15)`
at every request when the number of available pages was less than `top_p`
(e.g., 4K context with exactly 15 pages).

**Detection:** Crash on first RULER request with `top_p=16` at 4K context.

## Verification

After all four fixes, the fused backend matches native accuracy:

| Task | 4K (32 samples) | 16K (32 samples) |
|------|----------------|-----------------|
| niah_multikey_1 | 1.000 | 0.969 |
| niah_multiquery | 1.000 | 1.000 |
| niah_multivalue | 1.000 | 0.992 |

All tests run with `num_concurrent=16` and `max_num_seqs=32`, exercising the
batched-request code paths where these bugs previously manifested. Zero
runtime errors across all test runs.
