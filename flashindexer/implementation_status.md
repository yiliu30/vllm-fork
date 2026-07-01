# Fused QK Page TopK Selector — Implementation Status

**Date:** 2026-07-01  
**Branch:** `new-indexer`

## Architecture

```
fused_page backend (VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=fused_page):

  DeepGEMM (unchanged) → logits [M,N] HBM
                       → page scores (view-based, zero-copy)
                       → top-P pages (P=16)
                       → topk-512 within pages (logit-sorted)
                       → Sparse attention (unchanged)

  What we replace: ONLY the CUDA top_k_per_row_prefill kernel.
  DeepGEMM stays. Attention stays.
```

## Modes

| Env Var | Pipeline | Tokens/query | Status |
|---------|----------|-------------:|--------|
| `fused_page` (default) | DeepGEMM → page topk → topk-512 | 512 | ✓ Working |
| `fused_page` + `ALL_PAGE_TOKENS=1` | DeepGEMM → page topk → expand pages | 512 (32/page) | ✓ Vectorized |
| `fused_page` + `PAGE_MODE=1` | Triton scorer → page expansion | 512 (32/page) | ⚡ Slow at 128K |

## Verified Accuracy (default fused_page)

| Context | P coverage | multikey_1 | multiquery | multivalue | Samples |
|---------|-----------:|-----------:|-----------:|-----------:|--------:|
| 4K | 100% (16/16) | 1.000 | 1.000 | 1.000 | 32 |
| 16K | 26% (16/62) | 0.969 | 1.000 | 0.992 | 32 |
| 128K | 3% (16/512) | 1.000 | 1.000 | 1.000 | 64 |
| **1M** | **0.1%** | — | — | **0.950** | 5 |
| GSM8K | — | 94.6% (full) | — | — | 1319 |

All zero runtime errors after bug fixes.

## Key Design Points

1. **DeepGEMM stays** — it's fast, correct CUDA. We only replace the topk kernel.
2. **Page aggregation is zero-copy** — `logits_to_page_scores` uses `.view()` + `.max()`, no padding.
3. **`_expand_pages_to_tokens` is fully vectorized** — zero Python row loops.
4. **Buffer width is 512** — hardcoded. 1024 breaks attention kernel. Full page tokens need attention kernel support.
5. **`logits_to_page_scores` + `_prefill_topk_fused_page` are the two host-side functions** — both vectorized.

## Bugs Found and Fixed

| # | Bug | Symptom | Fixed |
|---|-----|---------|-------|
| 1 | `page_end` uses relative `num_k` vs absolute `k_end` | Wrong K tokens when `k_start>0` | ✓ |
| 2 | `tok_idx = k_start + tok_abs` | Double-count, OOB indices | ✓ |
| 3 | Relative `pg` treated as absolute page | Wrong pages when `first_page>0` | ✓ |
| 4 | `page_scores` not padded to `top_p` | RuntimeError "16 vs 15" | ✓ |

All masked at `k_start=0` (single request, no batching).

## Trace Analysis Findings

- P=16 captures 87-91% indexer mass across contexts (Stage 0)
- 75% page-level overlap between all layers, no decay with distance
- 41% token-level overlap between adjacent layers, stable across samples (std=0.9%)
- U-shaped page distribution universal across prompts
- Per-sample overlap variance: std=0.9% — cross-sample stable

## Files

| File | Lines | Purpose |
|------|------:|---------|
| `sparse_attn_indexer.py` | ~900 | Dispatch, page scoring, token expansion |
| `triton_fused_page_topk.py` | 663 | Triton page scorer (page_mode only) |
| `stage0_findings.md` | 167 | Mass recall analysis |
| `proposal.md` | 1323 | Original design |
| `bugs_found_and_fixed.md` | 114 | Bug docs |
| `implementation_status.md` | — | This file |

## Next Steps

1. **Widen attention kernel** to support 1024 sparse entries → ALL_PAGE_TOKENS with full 64 tokens/page
2. **Adaptive P**: scale with context (P=8 at 4K, P=32 at 128K)
3. **Decode path**: page-based selection for decode queries
4. **128K speed**: replace Triton page scorer with batched approach or skip for now (DeepGEMM is fine)
