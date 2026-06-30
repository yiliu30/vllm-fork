# Triton Fused QK Page TopK — Implementation Status

**Date:** 2026-06-28

## Architecture

```
fused_page backend (VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=fused_page):

  K gather (unchanged) → k_quant [N,D] HBM
                      → _page_score_kernel (Triton) → page_ids [M,P]
                      → token refinement → topk_indices [M,512] or [M,1024]
                      → Sparse attention (unchanged)
```

Two token refinement modes:
- **Refined** (default): host-side matmul for per-token scores within selected pages
- **Page mode** (`VLLM_SPARSE_INDEXER_PAGE_MODE=1`): expand pages directly to tokens, no scoring — attention kernel does real QK

## Verified Accuracy

| Context | P coverage | Task | Native | Fused | Samples |
|---------|-----------:|------|--------|-------|---------|
| 4K | 100% (16/16) | multikey_1 | 1.000 | **1.000** | 32 |
| 4K | 100% (16/16) | multivalue | 1.000 | **1.000** | 32 |
| 4K | 100% (16/16) | multiquery | 1.000 | **1.000** | 32 |
| 16K | 26% (16/62) | multikey_1 | 1.000 | **0.969** | 32 |
| 16K | 26% (16/62) | multivalue | 1.000 | **0.992** | 32 |
| 16K | 26% (16/62) | multiquery | 1.000 | **1.000** | 32 |
| GSM8K | — | full | — | **0.946/0.939** | 1319 |
| 128K | 3% (16/512) | multikey_1 | 1.000 | **1.000** | 64 |
| 128K | 3% (16/512) | multivalue | — | **1.000** | 64 |
| 128K | 3% (16/512) | multiquery | — | **1.000** | 64 |
| **1M** | **0.1% (16/16384)** | **multivalue** | — | **0.950** | **5** |

All tests run with `num_concurrent=16`, `max_num_seqs=32`. Zero runtime errors after bug fixes.

## Performance

| Context | Pages | Time/request (first) | Time/request (subsequent) |
|---------|------:|----------------------|---------------------------|
| 4K | 16 | ~9s | ~4s |
| 16K | 62 | ~30s | ~15s |
| 128K | 512 | ~20 min | ~10 min |

128K slow due to O(pages × tokens × dims) per query in Triton scalar loop — 220B ops per layer per request. The Q-reg optimization helps (~4x Q bandwidth) but doesn't change the fundamental complexity.

### 1M Context — 95% accuracy with 0.1% page coverage

At 1M tokens (16,384 pages), P=16 selects just 0.1% of pages. Yet the page-level selector
preserves 95% accuracy on multivalue. The indexer's U-shaped page preference means that
the important pages (beginning + end) always fit within P=16, regardless of context length.

## Known Limitations

1. **128K speed**: Triton kernel iterates serially over all 512 pages. Needs CUDA-level optimization or batched K loading.
2. **P fixed at 16**: Not adaptive to context length. At 4K selects all pages (no-op). At 128K selects only 3%.
3. **FP8 only**: No FP4/MXFP4 path.
4. **Buffer width**: Hardcoded to 512 (index_topk). Page mode with 1024 tokens needs wider buffer.
5. **Q-reg tile count**: Hardcoded to 4 tiles (448 dim / 128 BLOCK_D = 4). Doesn't generalize to arbitrary head_dim.

## Bugs Found and Fixed

| # | Bug | Symptom | Fixed |
|---|-----|---------|-------|
| 1 | `page_end` uses relative `num_k` vs absolute `k_end` | Wrong K tokens scored when k_start>0 | ✓ |
| 2 | `tok_idx = k_start + tok_abs` double-count | OOB indices when k_start>0 | ✓ |
| 3 | Relative `pg` treated as absolute page | Wrong page IDs when first_page>0 | ✓ |
| 4 | `page_scores` not padded to `top_p` | RuntimeError "16 vs 15 at dim 2" | ✓ |

All masked at `k_start=0` (single request, no batching).

## Files

| File | Lines | Purpose |
|------|------:|---------|
| `triton_fused_page_topk.py` | 663 | Page-scoring kernel + host functions |
| `sparse_attn_indexer.py` | 823 | Backend dispatch + chunk loop integration |
| `stage0_findings.md` | 167 | Mass recall analysis |
| `proposal.md` | 1323 | Original design |
| `bugs_found_and_fixed.md` | 114 | Bug documentation |

## Commits

```
d1d9edc65 Optimize Triton page-scoring kernel: pre-load Q into registers
00e063472 Add page-mode expansion: pass all page tokens to sparse attention
4cefbbfeb Optimize Triton fused page kernel: host-side token refinement
5b686d5e5 Add fused QK + page-score + top-P Triton kernel for sparse indexer prefill
```

## Next Steps

1. **128K performance**: Replace Triton page-scoring kernel with CUDA batched matmul or multi-pass hierarchical scoring
2. **Variable buffer width**: Support \(P \times \text{storage\_block\_size}\) token slots per query
3. **Adaptive P**: Scale P with context length or query position
4. **Decode path**: Extend page-based selection to decode queries
5. **FP4 path**: Support MXFP4 quantized K cache
