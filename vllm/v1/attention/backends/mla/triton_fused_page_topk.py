# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QK + page-score kernel in Triton (Stage 1 prototype).

Streams K pages from the gathered workspace, computes max-per-page QK
scores on-chip, and outputs a page-score vector per query/head.  The
host-side function then selects top-P pages and (optionally) refines to
token-level indices.

Correctness target: page-score ranking matches the reference PyTorch
implementation (slow but exact) for every (query, head).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: compute max-per-page QK scores
# ---------------------------------------------------------------------------

@triton.jit
def _page_score_kernel(
    q_ptr,               # [M, H, D] fp8
    k_ptr,               # [N, D] fp8  (gathered workspace)
    k_scale_ptr,         # [N] fp32
    weights_ptr,         # [M, H] fp32
    cu_seqlen_ks_ptr,    # [M] int32
    cu_seqlen_ke_ptr,    # [M] int32
    out_scores_ptr,      # [M, H, max_pages] fp32  — page scores
    out_first_page_ptr,  # [M, H] int32  — first valid logical page
    out_valid_ptr,       # [M, H] int32  — number of valid pages
    D: tl.constexpr,     # head_dim
    storage_block_size: tl.constexpr,
    max_pages: tl.constexpr,
    stride_q_m: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_n: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_valid_m: tl.constexpr,
    stride_valid_h: tl.constexpr,
    BLOCK_D: tl.constexpr,   # tile size along head_dim
):
    """One Triton program per (query_idx, head_idx).

    Loads the Q vector, then iterates over every K page visible to this
    query, computing max(Q @ K_page^T) on-chip.  Scores are written to
    out_scores[query, head, 0:num_pages].
    """
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    k_start = tl.load(cu_seqlen_ks_ptr + q_idx)
    k_end = tl.load(cu_seqlen_ke_ptr + q_idx)
    num_k = k_end - k_start
    if num_k <= 0:
        tl.store(out_first_page_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h, 0)
        tl.store(out_valid_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h, 0)
        return

    first_page = k_start // storage_block_size
    last_page = (k_end + storage_block_size - 1) // storage_block_size
    num_pages = last_page - first_page
    if num_pages <= 0:
        tl.store(out_first_page_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h, 0)
        tl.store(out_valid_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h, 0)
        return
    num_pages = tl.minimum(num_pages, max_pages)

    tl.store(out_first_page_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h,
             first_page)

    # Weight factor for this (query, head)
    w = tl.load(weights_ptr + q_idx * stride_w_m + h_idx * stride_w_h)

    # Pre-load Q vector into registers (tile by tile, store in a list)
    # We'll accumulate dot products tile-by-tile for each K token.

    # Iterate over pages
    for pg in range(num_pages):
        logical_page = first_page + pg
        page_start = logical_page * storage_block_size
        page_end = tl.minimum(page_start + storage_block_size, k_end)

        page_max: tl.float32 = -float("inf")  # type: ignore
        # Only process non-empty pages
        if page_end > page_start:
            for tok_off in range(page_end - page_start):
                tok_idx = page_start + tok_off  # absolute K position
                k_base = k_ptr + tok_idx * stride_k_n
                k_s = tl.load(k_scale_ptr + tok_idx)

                dot = tl.sum(
                    tl.load(q_ptr + q_idx * stride_q_m + h_idx * stride_q_h
                            + tl.arange(0, BLOCK_D),
                            mask=tl.arange(0, BLOCK_D) < D, other=0.0)
                    .to(tl.float32)
                    * tl.load(k_base + tl.arange(0, BLOCK_D),
                              mask=tl.arange(0, BLOCK_D) < D, other=0.0)
                    .to(tl.float32)
                    * k_s
                )
                for d_start in range(BLOCK_D, D, BLOCK_D):
                    d_offs = d_start + tl.arange(0, BLOCK_D)
                    d_mask = d_offs < D

                    q_tile = tl.load(
                        q_ptr + q_idx * stride_q_m + h_idx * stride_q_h + d_offs,
                        mask=d_mask, other=0.0,
                    ).to(tl.float32)
                    k_tile = tl.load(
                        k_base + d_offs, mask=d_mask, other=0.0,
                    ).to(tl.float32)

                    dot += tl.sum(q_tile * k_tile * k_s)

                score = dot * w
                page_max = tl.maximum(page_max, score)

        # Store at relative position pg (0-indexed within valid pages)
        tl.store(out_scores_ptr + q_idx * stride_out_m
                 + h_idx * stride_out_h + pg, page_max)

    tl.store(out_valid_ptr + q_idx * stride_valid_m + h_idx * stride_valid_h,
             num_pages)


# ---------------------------------------------------------------------------
# Triton kernel: compute per-token scores for selected pages only
# ---------------------------------------------------------------------------

@triton.jit
def _token_score_for_pages_kernel(
    q_ptr,               # [M, H, D] fp8
    k_ptr,               # [N, D] fp8
    k_scale_ptr,         # [N] fp32
    weights_ptr,         # [M, H] fp32
    page_ids_ptr,        # [M, H, top_p] int32 — selected logical pages
    cu_seqlen_ks_ptr,    # [M] int32
    cu_seqlen_ke_ptr,    # [M] int32
    out_scores_ptr,      # [M, H, top_p * storage_block_size] fp32
    out_indices_ptr,     # [M, H, top_p * storage_block_size] int32
    out_num_valid_ptr,   # [M, H] int32
    D: tl.constexpr,
    storage_block_size: tl.constexpr,
    top_p: tl.constexpr,
    max_candidates: tl.constexpr,
    stride_q_m: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_n: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_page_m: tl.constexpr,
    stride_page_h: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Second pass: compute per-token scores only for the top-P pages.

    Uses deterministic output positions: page p writes to slots
    [p*storage_block_size, (p+1)*storage_block_size).  Invalid pages
    leave -inf / -1 in their slots, which the host filters out.
    """
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    k_start = tl.load(cu_seqlen_ks_ptr + q_idx)
    k_end = tl.load(cu_seqlen_ke_ptr + q_idx)
    num_k = k_end - k_start
    if num_k <= 0:
        tl.store(out_num_valid_ptr + q_idx + h_idx, 0)
        return

    w = tl.load(weights_ptr + q_idx * stride_w_m + h_idx * stride_w_h)

    count = 0

    for p in range(top_p):
        logical_page = tl.load(
            page_ids_ptr + q_idx * stride_page_m + h_idx * stride_page_h + p
        )
        page_start = logical_page * storage_block_size
        page_end = tl.minimum(page_start + storage_block_size, k_end)

        for tok_off in range(storage_block_size):
            tok_abs = page_start + tok_off  # absolute position in K workspace
            # Valid: page is real (>=0) AND token is within page bounds
            valid_tok = (logical_page >= 0) & (tok_off < (page_end - page_start))

            # Compute score only for valid tokens (to avoid wasted work)
            score = -float("inf")
            tok_idx = tok_abs  # tok_abs is already absolute in K-space (page_start is absolute)
            # Only compute if valid — use conditional
            if valid_tok:
                k_base = k_ptr + tok_idx * stride_k_n
                k_s = tl.load(k_scale_ptr + tok_idx)

                dot = tl.sum(
                    tl.load(q_ptr + q_idx * stride_q_m + h_idx * stride_q_h
                            + tl.arange(0, BLOCK_D),
                            mask=tl.arange(0, BLOCK_D) < D, other=0.0)
                    .to(tl.float32)
                    * tl.load(k_base + tl.arange(0, BLOCK_D),
                              mask=tl.arange(0, BLOCK_D) < D, other=0.0)
                    .to(tl.float32)
                    * k_s
                )
                for d_start in range(BLOCK_D, D, BLOCK_D):
                    d_offs = d_start + tl.arange(0, BLOCK_D)
                    d_mask = d_offs < D
                    q_tile = tl.load(
                        q_ptr + q_idx * stride_q_m + h_idx * stride_q_h + d_offs,
                        mask=d_mask, other=0.0,
                    ).to(tl.float32)
                    k_tile = tl.load(
                        k_base + d_offs, mask=d_mask, other=0.0,
                    ).to(tl.float32)
                    dot += tl.sum(q_tile * k_tile * k_s)

                score = dot * w

            # Deterministic output slot
            out_off = (q_idx * stride_out_m + h_idx * stride_out_h
                       + p * storage_block_size + tok_off)
            tl.store(out_scores_ptr + out_off, score)
            # Use tl.where for conditional store value
            idx_val = tl.where(valid_tok, tok_abs, -1)
            tl.store(out_indices_ptr + out_off, idx_val)
            count += 1

    tl.store(out_num_valid_ptr + q_idx + h_idx, count)


# ---------------------------------------------------------------------------
# Host API
# ---------------------------------------------------------------------------

def _page_topk(
    page_scores: torch.Tensor,
    num_valid: torch.Tensor,
    top_p: int,
) -> torch.Tensor:
    """Select top-P pages from the per-page scores (host-side, fast)."""
    M, H, max_pages = page_scores.shape
    device = page_scores.device

    # Mask pages beyond num_valid with very negative value (below -inf)
    mask = torch.arange(max_pages, device=device).unsqueeze(0).unsqueeze(0) \
           < num_valid.unsqueeze(-1)
    # torch.finfo().min gives a finite but extremely negative number,
    # which guarantees it ranks below any real score.
    masked = page_scores.masked_fill(~mask, torch.finfo(torch.float32).min)

    k = min(top_p, max_pages)
    _, top_indices = torch.topk(masked, k=k, dim=-1)  # [M, H, k]

    # Mark indices that land on invalid pages as -1
    top_indices = top_indices.to(torch.int32)
    top_indices = torch.where(
        top_indices < num_valid.unsqueeze(-1),
        top_indices,
        torch.full_like(top_indices, -1),
    )

    # Pad to top_p with -1 if needed
    if k < top_p:
        pad = torch.full((M, H, top_p - k), -1, dtype=torch.int32, device=device)
        top_indices = torch.cat([top_indices, pad], dim=-1)

    return top_indices  # [M, H, top_p]


def fused_qk_page_topk(
    q: torch.Tensor,            # [M, H, D] fp8
    k: torch.Tensor,            # [N, D] fp8
    k_scale: torch.Tensor,      # [N] fp32
    weights: torch.Tensor,      # [M, H] fp32
    cu_seqlen_ks: torch.Tensor, # [M] int32
    cu_seqlen_ke: torch.Tensor, # [M] int32
    top_p: int,
    storage_block_size: int = 64,
    max_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK + page-score + top-P selection.

    Args:
        q:  [M, H, D]  float8_e4m3fn  query values.
        k:  [N, D]      float8_e4m3fn  gathered key values (flat workspace).
        k_scale:  [N]   float32        per-token K dequant scales.
        weights: [M, H] float32        pre-folded Q-scale × softmax-scale.
        cu_seqlen_ks: [M] int32  inclusive start of K range per query.
        cu_seqlen_ke: [M] int32  exclusive end of K range per query.
        top_p:  number of pages to select.
        storage_block_size:  compressed tokens per logical page.
        max_pages:  pre-allocated page-score width (auto if None).

    Returns:
        page_ids:    [M, H, top_p] int32  logical page indices.
        page_scores: [M, H, top_p] fp32   corresponding max scores.
    """
    M, H, D = q.shape
    N = k.shape[0]
    device = q.device

    # Validation
    assert q.dtype == torch.float8_e4m3fn
    assert k.dtype == torch.float8_e4m3fn
    assert weights.shape == (M, H) and weights.dtype == torch.float32
    assert k_scale.shape == (N,) and k_scale.dtype == torch.float32
    assert cu_seqlen_ks.shape == (M,) and cu_seqlen_ks.dtype == torch.int32
    assert cu_seqlen_ke.shape == (M,) and cu_seqlen_ke.dtype == torch.int32

    # Compute upper bound on pages any query can see
    if max_pages is None:
        max_ke = cu_seqlen_ke.max().item()
        max_pages = int(math.ceil(max_ke / storage_block_size))
    max_pages = min(max_pages, 512)  # safety cap

    BLOCK_D = min(128, triton.next_power_of_2(D))

    page_scores_flat = torch.full(
        (M, H, max_pages), float("-inf"),
        dtype=torch.float32, device=device,
    )
    first_page = torch.zeros((M, H), dtype=torch.int32, device=device)
    num_valid = torch.zeros((M, H), dtype=torch.int32, device=device)

    grid = (M, H)
    _page_score_kernel[grid](
        q, k, k_scale, weights,
        cu_seqlen_ks, cu_seqlen_ke,
        page_scores_flat, first_page, num_valid,
        D=D,
        storage_block_size=storage_block_size,
        max_pages=max_pages,
        stride_q_m=q.stride(0),
        stride_q_h=q.stride(1),
        stride_k_n=k.stride(0),
        stride_w_m=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_out_m=page_scores_flat.stride(0),
        stride_out_h=page_scores_flat.stride(1),
        stride_valid_m=first_page.stride(0),
        stride_valid_h=first_page.stride(1),
        BLOCK_D=BLOCK_D,
    )

    # Host-side top-P selection
    # Scores are stored at relative positions [0, num_valid).
    # Map back to absolute logical pages by adding first_page.
    k_select = min(top_p, max_pages)
    masked_scores = page_scores_flat.masked_fill(
        torch.arange(max_pages, device=device).unsqueeze(0).unsqueeze(0)
        >= num_valid.unsqueeze(-1),
        torch.finfo(torch.float32).min,
    )
    top_scores, top_rel_indices = torch.topk(masked_scores, k=k_select, dim=-1)
    # Convert relative indices → absolute logical pages
    top_abs_indices = torch.where(
        top_rel_indices < num_valid.unsqueeze(-1),
        (top_rel_indices + first_page.unsqueeze(-1)).to(torch.int32),
        torch.full_like(top_rel_indices, -1, dtype=torch.int32),
    )
    # Pad to top_p
    if k_select < top_p:
        pad = torch.full((M, H, top_p - k_select), -1, dtype=torch.int32, device=device)
        top_abs_indices = torch.cat([top_abs_indices, pad], dim=-1)
        pad_scores = torch.full((M, H, top_p - k_select), float("-inf"),
                                dtype=torch.float32, device=device)
        top_scores = torch.cat([top_scores, pad_scores], dim=-1)

    # Gather scores from the original flat array at relative positions
    safe_rel_idx = top_rel_indices.clamp(min=0)
    page_scores = page_scores_flat.gather(dim=-1, index=safe_rel_idx.long())
    # Pad page_scores to match top_abs_indices (which may have been padded above)
    if k_select < top_p:
        pad_s = torch.full((M, H, top_p - k_select), float("-inf"),
                           dtype=torch.float32, device=device)
        page_scores = torch.cat([page_scores, pad_s], dim=-1)
    page_scores = torch.where(
        top_abs_indices >= 0, page_scores,
        torch.full_like(page_scores, float("-inf")),
    )

    return top_abs_indices, page_scores


def fused_qk_page_topk_with_tokens(
    q: torch.Tensor,            # [M, H, D] fp8
    k: torch.Tensor,            # [N, D] fp8
    k_scale: torch.Tensor,      # [N] fp32
    weights: torch.Tensor,      # [M, H] fp32
    cu_seqlen_ks: torch.Tensor, # [M] int32
    cu_seqlen_ke: torch.Tensor, # [M] int32
    top_p: int = 16,
    storage_block_size: int = 64,
    max_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK + page-score + top-P + per-token scores for selected pages.

    Combines two Triton kernels:
    1. Page scoring + top-P selection (fused_qk_page_topk)
    2. Per-token scoring for selected pages (_token_score_for_pages_kernel)

    Args:
        q, k, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke: see fused_qk_page_topk
        top_p, storage_block_size, max_pages: same as above

    Returns:
        page_ids:  [M, H, top_p] int32  logical page indices
        candidate_scores:  [M, H, top_p * storage_block_size] fp32  per-token QK scores
        candidate_indices: [M, H, top_p * storage_block_size] int32  absolute K positions
    """
    # Step 1: page selection (same as fused_qk_page_topk)
    page_ids, _ = fused_qk_page_topk(
        q, k, k_scale, weights,
        cu_seqlen_ks, cu_seqlen_ke,
        top_p=top_p,
        storage_block_size=storage_block_size,
        max_pages=max_pages,
    )

    M, H, D = q.shape
    device = q.device

    max_candidates = top_p * storage_block_size

    candidate_scores = torch.full(
        (M, H, max_candidates), float("-inf"),
        dtype=torch.float32, device=device,
    )
    candidate_indices = torch.full(
        (M, H, max_candidates), -1,
        dtype=torch.int32, device=device,
    )
    num_valid = torch.zeros((M, H), dtype=torch.int32, device=device)

    BLOCK_D = min(128, triton.next_power_of_2(D))
    grid = (M, H)
    _token_score_for_pages_kernel[grid](
        q, k, k_scale, weights,
        page_ids,
        cu_seqlen_ks, cu_seqlen_ke,
        candidate_scores, candidate_indices,
        num_valid,
        D=D,
        storage_block_size=storage_block_size,
        top_p=top_p,
        max_candidates=max_candidates,
        stride_q_m=q.stride(0),
        stride_q_h=q.stride(1),
        stride_k_n=k.stride(0),
        stride_w_m=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_page_m=page_ids.stride(0),
        stride_page_h=page_ids.stride(1),
        stride_out_m=candidate_scores.stride(0),
        stride_out_h=candidate_scores.stride(1),
        BLOCK_D=BLOCK_D,
    )

    return page_ids, candidate_scores, candidate_indices


# ---------------------------------------------------------------------------
# Reference & validation
# ---------------------------------------------------------------------------

def compute_oracle_from_logits(
    topk_indices: torch.Tensor,  # [M, K]
    topk_logits: torch.Tensor,   # [M, K]
    row_starts: torch.Tensor,    # [M] int
    top_p: int,
    storage_block_size: int = 64,
) -> tuple[torch.Tensor, float]:
    """Oracle page selection: rank pages by softmax-mass of topk tokens."""
    M, K = topk_indices.shape
    oracle_ids = torch.full((M, top_p), -1, dtype=torch.int32)
    recalls: list[float] = []

    for r in range(M):
        start = int(row_starts[r])
        pm: dict[int, float] = {}
        total = 0.0
        for s in range(K):
            li = int(topk_indices[r, s])
            if li < 0:
                continue
            lp = li // storage_block_size
            score = float(topk_logits[r, s])
            mass = math.exp(score - float(topk_logits[r, :K].max()))
            pm[lp] = pm.get(lp, 0.0) + mass
            total += mass

        if total <= 0:
            recalls.append(1.0)
            continue

        ranked = sorted(pm.items(), key=lambda x: -x[1])[:top_p]
        for i, (lp, _) in enumerate(ranked):
            oracle_ids[r, i] = lp
        sel_mass = sum(pm[lp] for lp, _ in ranked)
        recalls.append(sel_mass / total)

    return oracle_ids, sum(recalls) / len(recalls)


def compare_page_selections(
    our_page_ids: torch.Tensor,      # [M, H, top_p]
    oracle_page_ids: torch.Tensor,   # [M, top_p]
    top_p: int,
) -> tuple[float, float]:
    """Return (page_recall, page_precision) against oracle."""
    M, H, _ = our_page_ids.shape
    recall_sum = 0.0
    precision_sum = 0.0
    n = 0

    for row in range(M):
        oracle = {int(oracle_page_ids[row, p])
                  for p in range(top_p)
                  if int(oracle_page_ids[row, p]) >= 0}
        if not oracle:
            continue
        for h in range(H):
            ours = {int(our_page_ids[row, h, p])
                    for p in range(top_p)
                    if int(our_page_ids[row, h, p]) >= 0}
            inter = ours & oracle
            recall_sum += len(inter) / len(oracle)
            precision_sum += len(inter) / len(ours) if ours else 0.0
            n += 1

    return recall_sum / n if n else 0.0, precision_sum / n if n else 0.0
