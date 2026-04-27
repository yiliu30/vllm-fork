# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference implementations for DeepGEMM kernels.

These are used as fallbacks on platforms where DeepGEMM native kernels
are not available (e.g., SM120 Blackwell consumer GPUs).
They are functionally correct but significantly slower than the native kernels.
"""

import torch

from vllm.utils.math_utils import cdiv

__all__ = [
    "fp8_einsum_torch_reference",
    "fp8_mqa_logits_torch_reference",
    "fp8_paged_mqa_logits_torch_reference",
    "tf32_hc_prenorm_gemm_torch_reference",
]


def fp8_einsum_torch_reference(
    equation: str,
    a_tuple: tuple[torch.Tensor, torch.Tensor],
    b_tuple: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    recipe: tuple[int, ...] = (),
) -> None:
    """Torch fallback for fp8_einsum on SM120.

    Dequantizes FP8 inputs to float32 and runs torch.einsum.
    Scale tensors are per-token for `a` and per-block-128 for `b`.
    """
    a, a_scale = a_tuple
    b, b_scale = b_tuple

    # Dequant a: a is FP8 with per-token scales (last dim groups of 128)
    a_f = a.to(torch.float32)
    if a_scale is not None and a_scale.numel() > 0:
        # a_scale shape matches a but with last dim // 128
        # Repeat scale to match a's last dim
        repeat_factor = a.shape[-1] // max(a_scale.shape[-1], 1)
        if repeat_factor > 1:
            a_scale_expanded = a_scale.to(torch.float32).repeat_interleave(
                repeat_factor, dim=-1
            )
            # Trim if needed
            a_scale_expanded = a_scale_expanded[..., : a.shape[-1]]
            a_f = a_f * a_scale_expanded
        else:
            a_f = a_f * a_scale.to(torch.float32)

    # Dequant b: b is FP8 with block scales
    b_f = b.to(torch.float32)
    if b_scale is not None and b_scale.numel() > 0:
        if b_scale.dim() == b.dim():
            # Per-block scales: repeat to match b dimensions
            repeats = []
            for i in range(b.dim()):
                repeats.append(b.shape[i] // max(b_scale.shape[i], 1))
            b_scale_expanded = b_scale.to(torch.float32).repeat_interleave(
                repeats[-1], dim=-1
            )
            for i in range(b.dim() - 2, -1, -1):
                if repeats[i] > 1:
                    b_scale_expanded = b_scale_expanded.repeat_interleave(
                        repeats[i], dim=i
                    )
            b_scale_expanded = b_scale_expanded[tuple(slice(0, s) for s in b.shape)]
            b_f = b_f * b_scale_expanded
        else:
            b_f = b_f * b_scale.to(torch.float32)

    result = torch.einsum(equation, a_f, b_f)
    out.copy_(result.to(out.dtype))


def fp8_mqa_logits_torch_reference(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Torch fallback for fp8_fp4_mqa_logits on SM120."""
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits fallback only supports FP8 Q")

    k_values, k_scales = kv
    q_f32 = q_values.to(torch.float32)
    k_f32 = k_values.to(torch.float32) * k_scales.reshape(-1, 1).to(torch.float32)
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_f32.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )

    # Avoid materializing the full [H, M, N] score tensor for all heads.
    for head_start in range(0, num_heads, 8):
        head_end = min(head_start + 8, num_heads)
        q_chunk = q_f32[:, head_start:head_end, :].transpose(0, 1).contiguous()
        scores = torch.matmul(q_chunk, k_t)
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        logits += (scores.relu() * head_weights).sum(dim=0)

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def fp8_paged_mqa_logits_torch_reference(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Torch fallback for fp8_fp4_paged_mqa_logits on SM120.

    KV cache layout per block (written by indexer_k_quant_and_cache):
      [block_size * head_dim FP8 bytes | block_size * 4 scale bytes]
    The tensor shape [NB, block_size, 1, head_dim+4] is a stride trick;
    bytes must be re-sliced flat per block.
    """
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA logits fallback only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    if kv_cache.dim() == 4:
        _, block_size, _, d_plus_4 = kv_cache.shape
    else:
        _, block_size, d_plus_4 = kv_cache.shape
    num_blocks_total = kv_cache.shape[0]

    # Flatten to [NB, block_size * (head_dim + 4)]
    kv_flat = kv_cache.reshape(num_blocks_total, -1)
    k_end = block_size * head_dim

    q_f = q_values.to(torch.float32)  # [B, next_n, H, D]

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    # context_lens may be 2D [B, next_n] or [B, 1] or 1D [B]
    if context_lens.dim() == 2:
        ctx_lens_1d = context_lens.max(dim=1).values
    else:
        ctx_lens_1d = context_lens
    ctx_lens_list = ctx_lens_1d.tolist()

    if context_lens.dim() == 2:
        ctx_lens_2d = context_lens.tolist()
    else:
        ctx_lens_2d = [[int(c)] * next_n for c in ctx_lens_list]

    for i in range(batch_size):
        context_len = int(ctx_lens_list[i])
        if context_len <= 0:
            continue

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i, :num_blocks]  # [num_blocks]

        # Gather blocks and decode FP8
        blocks_flat = kv_flat[block_idxs]  # [num_blocks, block_size*(head_dim+4)]

        # K data region: first block_size*head_dim bytes per block
        k_data_u8 = blocks_flat[:, :k_end]  # [num_blocks, block_size*head_dim]
        k_data = k_data_u8.reshape(num_blocks, block_size, head_dim)
        k_fp8 = k_data.view(torch.float8_e4m3fn)
        k_f = k_fp8.to(torch.float32)  # [num_blocks, block_size, head_dim]

        # Scale region: last block_size*4 bytes per block
        scale_u8 = blocks_flat[:, k_end:]  # [num_blocks, block_size*4]
        k_scale = scale_u8.contiguous().view(torch.float32)  # [num_blocks, block_size]

        # Apply scales
        k_f = k_f * k_scale.unsqueeze(-1)  # [num_blocks, block_size, head_dim]

        # Flatten to [total_tokens, head_dim]
        total_tokens = num_blocks * block_size
        kv_use = k_f.reshape(total_tokens, head_dim)

        # q for this batch: [next_n, H, D]
        q_batch = q_f[i]

        # Compute scores per head chunk to save memory
        batch_logits = torch.zeros(
            (next_n, total_tokens),
            device=q_values.device,
            dtype=torch.float32,
        )

        for h_start in range(0, num_heads, 8):
            h_end = min(h_start + 8, num_heads)
            # q_chunk: [next_n, h_chunk, D] -> [h_chunk, next_n, D]
            q_chunk = q_batch[:, h_start:h_end, :].transpose(0, 1).contiguous()
            # scores: [h_chunk, next_n, total_tokens]
            scores = torch.matmul(q_chunk, kv_use.t())
            # weights for this batch
            w = weights[i * next_n : (i + 1) * next_n, h_start:h_end].t().unsqueeze(-1)
            batch_logits += (scores.relu() * w).sum(dim=0)

        # Causal masking using per-token context lens
        k_offsets = torch.arange(total_tokens, device=q_values.device)
        for n in range(next_n):
            q_ctx = int(ctx_lens_2d[i][min(n, len(ctx_lens_2d[i]) - 1)])
            valid_mask = k_offsets < q_ctx
            row = batch_logits[n].clone()
            row[~valid_mask] = float("-inf")
            logits[i * next_n + n, :total_tokens] = row

    return logits


def tf32_hc_prenorm_gemm_torch_reference(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    """Torch fallback for tf32_hc_prenorm_gemm on SM120.

    DeepGEMM splits the K dimension into num_split chunks and stores
    partial results in out[n_splits, M, N] and sqrsum[n_splits, M].
    The downstream tilelang kernel sums across the split dimension.
    """
    x_f = x.float()  # [M, K]
    # fn: [N, K], out: [n_splits, M, N], sqrsum: [n_splits, M]
    K = x_f.shape[1]
    chunk_size = cdiv(K, num_split)

    for s in range(num_split):
        k_start = s * chunk_size
        k_end = min(k_start + chunk_size, K)
        x_chunk = x_f[:, k_start:k_end]
        fn_chunk = fn[:, k_start:k_end]
        out[s].copy_(x_chunk @ fn_chunk.t())
        sqrsum[s].copy_(x_chunk.square().sum(-1))
