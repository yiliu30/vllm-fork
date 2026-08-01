# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.mla.triton_fused_page_topk import (
    fused_qk_page_topk,
    fused_qk_page_topk_refined,
)
from vllm.v1.attention.backends.mla.prefill_observation import (
    record_prefill_runtime_topk,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


_FUNNEL_TOPK_MODES = ("exact", "fast", "turbo")
import inspect

def _prefill_topk_funnel_dense(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Experimental funnel prefill adapter.

    Prefer the ragged funnel prefill op when available, and fall back to the
    older dense-mask + funnel_topk adapter for older funnel_topk builds.
    Keeps the same output contract as top_k_per_row_prefill:
    - indices are local to each row's [row_start, row_end) range
    - invalid/excess slots are -1
    """
    mode = os.getenv(
        "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE", "exact"
    ).lower()
    if mode not in _FUNNEL_TOPK_MODES:
        raise ValueError(
            "Invalid VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE: "
            f"{mode}. Expected one of: {', '.join(_FUNNEL_TOPK_MODES)}"
        )

    try:
        from funnel_topk import top_k_per_row_prefill_funnel_v1
        logger.warning_once(
            "using funnel_topk ragged prefill v1 op for prefill top-k "
            "(VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense)"
        )
        ragged_kwargs = {}
        if "mode" in inspect.signature(top_k_per_row_prefill_funnel_v1).parameters:
            ragged_kwargs["mode"] = mode
        elif mode != "exact":
            logger.warning_once(
                "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE is ignored by "
                "top_k_per_row_prefill_funnel_v1; using the ragged funnel op "
                "instead of the dense fallback path."
            )
        top_k_per_row_prefill_funnel_v1(
            logits,
            row_starts,
            row_ends,
            topk_indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
            **ragged_kwargs,
        )
        return
    except Exception:
        logger.warning_once(
            "falling back to funnel_topk dense adapter for prefill top-k "
            "(VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense)"
        )
        raise RuntimeError(
            "Falling back to funnel_topk dense adapter failed."
        )

    try:
        from funnel_topk.funnel import funnel_topk
        logger.warning_once(
            "using funnel_topk dense fallback for prefill top-k "
            "(VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense), "
            f"mode={mode}"
        )
    except Exception as exc:
        raise RuntimeError(
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense requires "
            "funnel_topk to be importable. Install funnel-topk or set "
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=native."
        ) from exc

    topk_indices.fill_(-1)
    if topk_tokens <= 0 or logits.numel() == 0:
        return

    num_cols = logits.shape[1]
    if num_cols == 0:
        return

    k = min(topk_tokens, num_cols)

    starts = row_starts.to(dtype=torch.int32, device=logits.device).view(-1, 1)
    ends = row_ends.to(dtype=torch.int32, device=logits.device).view(-1, 1)
    cols = torch.arange(num_cols, dtype=torch.int32, device=logits.device).view(1, -1)

    valid_mask = (cols >= starts) & (cols < ends)
    masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

    funnel_kwargs = {
        "k": k,
        "dim": -1,
        "largest": True,
        "sorted": True,
    }
    if "mode" in inspect.signature(funnel_topk).parameters:
        funnel_kwargs["mode"] = mode
    elif mode != "exact":
        raise RuntimeError(
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE requires a "
            "funnel_topk build that supports mode=fast/turbo. Upgrade "
            "funnel-topk or set "
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=exact."
        )

    values, abs_indices = funnel_topk(masked_logits, **funnel_kwargs)

    local_indices = abs_indices.to(torch.int32) - starts
    local_indices = torch.where(
        values == float("-inf"),
        torch.full_like(local_indices, -1, dtype=torch.int32),
        local_indices,
    )
    topk_indices[:, :k] = local_indices

def __prefill_topk_funnel_dense(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Experimental dense adapter: mask ragged rows and run funnel_topk.

    Keeps the same output contract as top_k_per_row_prefill:
    - indices are local to each row's [row_start, row_end) range
    - invalid/excess slots are -1
    """
    try:
        from funnel_topk.funnel import funnel_topk
    except Exception as exc:
        raise RuntimeError(
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense requires "
            "funnel_topk to be importable. Install funnel-topk or set "
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=native."
        ) from exc

    topk_indices.fill_(-1)
    if topk_tokens <= 0 or logits.numel() == 0:
        return

    num_cols = logits.shape[1]
    if num_cols == 0:
        return

    k = min(topk_tokens, num_cols)

    starts = row_starts.to(dtype=torch.int32, device=logits.device).view(-1, 1)
    ends = row_ends.to(dtype=torch.int32, device=logits.device).view(-1, 1)
    cols = torch.arange(num_cols, dtype=torch.int32, device=logits.device).view(1, -1)

    valid_mask = (cols >= starts) & (cols < ends)
    masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

    values, abs_indices = funnel_topk(
        masked_logits,
        k=k,
        dim=-1,
        largest=True,
        sorted=True,
    )

    local_indices = abs_indices.to(torch.int32) - starts
    local_indices = torch.where(
        values == float("-inf"),
        torch.full_like(local_indices, -1, dtype=torch.int32),
        local_indices,
    )
    topk_indices[:, :k] = local_indices


def _expand_pages_to_tokens(
    page_ids: torch.Tensor,          # [M, H, top_p] int32
    cu_seqlen_ks: torch.Tensor,      # [M] int32
    topk_indices: torch.Tensor,      # [M, max_tokens] int32 (output)
    storage_block_size: int = 64,
) -> None:
    """Expand selected logical pages to token indices — fully vectorized."""
    M, H, top_p = page_ids.shape
    max_tokens = topk_indices.shape[1]
    device = page_ids.device
    topk_indices.fill_(-1)

    # Squeeze head dim (H=1 for MLA)
    pages = page_ids[:, 0, :]  # [M, top_p]

    # How many tokens from each page?
    tokens_per_page = max_tokens // top_p
    if tokens_per_page <= 0:
        tokens_per_page = 1

    # Build a lookup table: [top_p, tokens_per_page] of offsets within each page
    tok_offs = torch.arange(tokens_per_page, device=device).unsqueeze(0)  # [1, TPP]

    # For each query m, page p: absolute position = page_ids[m,p] * 64 + tok_off
    # pages: [M, top_p] → [M, top_p, 1]
    # tok_offs: [1, TPP] → [1, 1, TPP]
    abs_pos = pages.unsqueeze(-1) * storage_block_size + tok_offs.unsqueeze(0)  # [M, top_p, TPP]

    # Flatten to [M, top_p * TPP] and make local
    abs_pos = abs_pos.view(M, -1)  # [M, top_p * tokens_per_page]
    local_idx = abs_pos - cu_seqlen_ks.unsqueeze(-1)  # [M, top_p * tokens_per_page]

    # Truncate to max_tokens and write
    n_write = min(top_p * tokens_per_page, max_tokens)
    topk_indices[:, :n_write] = local_idx[:, :n_write].to(torch.int32)

    # Invalidate entries from padding pages (page_ids < 0)
    valid_mask = pages.unsqueeze(-1).expand(-1, -1, tokens_per_page).reshape(M, -1) >= 0
    topk_indices[:, :n_write] = torch.where(
        valid_mask[:, :n_write],
        topk_indices[:, :n_write],
        torch.full_like(topk_indices[:, :n_write], -1),
    )


def logits_to_page_scores(
    logits: torch.Tensor,        # [M, N]
    row_starts: torch.Tensor,    # [M]
    row_ends: torch.Tensor,      # [M]
    storage_block_size: int = 64,
) -> torch.Tensor:
    """Compute max-per-page score from logits (view-based, no copy)."""
    M, N = logits.shape
    device = logits.device
    num_pages = (N + storage_block_size - 1) // storage_block_size
    num_full = N // storage_block_size
    page_max = torch.empty(M, num_pages, dtype=torch.float32, device=device)
    # Full pages: reshape view [M, num_full, 64] → max (no copy)
    if num_full > 0:
        page_max[:, :num_full] = (
            logits[:, :num_full * storage_block_size]
            .view(M, num_full, storage_block_size)
            .max(dim=-1).values
        )
    # Partial last page
    partial = N % storage_block_size
    if partial > 0:
        page_max[:, -1] = logits[:, -partial:].max(dim=-1).values
    elif num_full < num_pages:
        page_max[:, -1] = float("-inf")
    # Mask pages outside valid range
    first_page = (row_starts // storage_block_size).clamp(min=0)
    last_page = (row_ends + storage_block_size - 1) // storage_block_size
    page_mask = (
        torch.arange(num_pages, device=device).unsqueeze(0) >= first_page.unsqueeze(1)
    ) & (
        torch.arange(num_pages, device=device).unsqueeze(0) < last_page.unsqueeze(1)
    )
    return page_max.masked_fill(~page_mask, float("-inf"))


def _fill_topk_from_fused_candidates(
    candidate_scores: torch.Tensor,   # [M, max_candidates] fp32
    candidate_indices: torch.Tensor,  # [M, max_candidates] int32
    cu_seqlen_ks: torch.Tensor,       # [M] int32
    cu_seqlen_ke: torch.Tensor,       # [M] int32
    topk_indices: torch.Tensor,       # [M, topk_tokens] int32 (output)
    topk_tokens: int,
) -> None:
    """Convert fused kernel candidate output to legacy topk_indices format.

    candidate_indices are absolute positions in K-space [0, N).
    topk_indices must be local positions relative to cu_seqlen_ks (row start).
    """
    M = candidate_scores.shape[0]
    device = candidate_scores.device
    topk_indices.fill_(-1)

    ks = cu_seqlen_ks.unsqueeze(-1)  # [M, 1]
    ke = cu_seqlen_ke.unsqueeze(-1)  # [M, 1]

    # Make candidates local
    local_indices = candidate_indices - ks  # [M, max_candidates]

    # Mask: local index must be in [0, ke - ks), and score must be finite
    valid = (
        (local_indices >= 0)
        & (local_indices < (ke - ks))
        & torch.isfinite(candidate_scores)
    )

    masked_scores = candidate_scores.masked_fill(~valid, float("-inf"))
    k_final = min(topk_tokens, masked_scores.shape[1])
    _, top_k = torch.topk(masked_scores, k=k_final, dim=-1)

    # Gather local indices
    gathered = local_indices.gather(dim=-1, index=top_k)
    # Only keep entries that came from valid positions
    valid_k = valid.gather(dim=-1, index=top_k)
    topk_indices[:, :k_final] = torch.where(
        valid_k, gathered.to(torch.int32),
        torch.full_like(gathered, -1).to(torch.int32),
    )


def _run_prefill_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
    topk_tokens: int,
) -> None:
    backend = os.getenv(
        "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND", "native"
    ).lower()

    if backend == "native":
        ops.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
        return

    if backend == "funnel_dense":
        logger.warning_once("Using funnel_dense backend for top-k prefill.")
        _prefill_topk_funnel_dense(
            logits=logits,
            row_starts=row_starts,
            row_ends=row_ends,
            topk_indices=topk_indices,
            topk_tokens=topk_tokens,
        )
        return

    if backend == "fused_page":
        _prefill_topk_fused_page(
            logits=logits,
            row_starts=row_starts,
            row_ends=row_ends,
            topk_indices=topk_indices,
            topk_tokens=topk_tokens,
        )
        return

    raise ValueError(
        "Invalid VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND: "
        f"{backend}. Expected one of: native, funnel_dense, fused_page"
    )


def _prefill_topk_fused_page(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    storage_block_size: int = 64,
    top_p: int = 16,
) -> None:
    """Page-level top-k: select top-P pages, then top tokens within each page.

    Replaces the per-token topk CUDA kernel with a two-level selection:
    1. Compute max-per-page score → select top-P pages
    2. Within selected pages, select top tokens by score

    Vectorized implementation — no Python row-level loops.
    """
    M, N = logits.shape
    device = logits.device
    assert topk_indices.shape == (M, topk_tokens)
    assert topk_indices.dtype == torch.int32

    topk_indices.fill_(-1)

    # 1. Pad logits to page-aligned width and compute max per (row, page)
    num_pages_total = (N + storage_block_size - 1) // storage_block_size
    padded_len = num_pages_total * storage_block_size
    padded = torch.full(
        (M, padded_len), float("-inf"), dtype=torch.float32, device=device,
    )
    padded[:, :N] = logits

    # max per page: reshape → [M, num_pages_total, storage_block_size] → max
    page_max = padded.view(M, num_pages_total, storage_block_size).max(dim=-1).values
    # page_max: [M, num_pages_total]

    # 2. Mask pages outside each row's valid range and select top-P
    first_page = (row_starts // storage_block_size).clamp(min=0)
    k_end_ceil = (row_ends + storage_block_size - 1) // storage_block_size
    page_mask = (
        torch.arange(num_pages_total, device=device).unsqueeze(0)
        >= first_page.unsqueeze(1)
    ) & (
        torch.arange(num_pages_total, device=device).unsqueeze(0)
        < k_end_ceil.unsqueeze(1)
    )

    masked_scores = page_max.masked_fill(~page_mask, float("-inf"))
    k_pages = min(top_p, num_pages_total)
    _, top_page_idx = torch.topk(masked_scores, k=k_pages, dim=-1)
    # top_page_idx: [M, k_pages] — logical page indices per row

    # 3. Build token mask from selected pages (vectorized, loop only over P)
    token_selected = torch.zeros(M, N, dtype=torch.bool, device=device)
    col_idx = torch.arange(N, device=device).unsqueeze(0)  # [1, N]

    for p in range(k_pages):
        page_ids = top_page_idx[:, p]  # [M]
        sel_start = (page_ids * storage_block_size).unsqueeze(1)  # [M, 1]
        sel_end = torch.minimum(
            (page_ids + 1) * storage_block_size,
            row_ends,
        ).unsqueeze(1)  # [M, 1]

        # Each row: tokens in [sel_start, sel_end) are selected
        token_selected |= (col_idx >= sel_start) & (col_idx < sel_end)

    # 4. Mask logits to only selected tokens, then topk
    masked_logits = logits.masked_fill(~token_selected, float("-inf"))
    k_final = min(topk_tokens, N)
    _, selected_tokens = torch.topk(masked_logits, k=k_final, dim=-1)

    # Convert absolute indices to local (row-relative)
    local_indices = selected_tokens - row_starts.unsqueeze(-1)
    valid_mask = (selected_tokens >= row_starts.unsqueeze(-1)) & \
                 (selected_tokens < row_ends.unsqueeze(-1))
    topk_indices[:, :k_final] = torch.where(
        valid_mask, local_indices.to(torch.int32),
        torch.full_like(local_indices, -1, dtype=torch.int32),
    )


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            prefill_topk_backend = os.getenv(
                "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND", "native"
            ).lower()

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            elif prefill_topk_backend == "fused_page":
                page_mode = os.getenv(
                    "VLLM_SPARSE_INDEXER_PAGE_MODE", "0"
                ) == "1"

                if page_mode:
                    if dcp_world_size > 1:
                        raise RuntimeError(
                            "VLLM_SPARSE_INDEXER_PAGE_MODE=1 does not support "
                            "DCP because it skips the dense logits needed for "
                            "the cross-rank top-k merge."
                        )
                    if use_fp4_cache:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                    chunk_weights = weights[chunk.token_start : chunk.token_end]
                    page_ids, _ = fused_qk_page_topk(
                        q_slice_cast, k_quant_cast, k_scale_cast,
                        chunk_weights,
                        cu_seqlen_ks, cu_seqlen_ke,
                        top_p=16, storage_block_size=64,
                    )
                    _expand_pages_to_tokens(
                        page_ids, cu_seqlen_ks,
                        topk_indices, storage_block_size=64,
                    )
                else:
                    if use_fp4_cache:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)

                    all_page_tokens = os.getenv(
                        "VLLM_SPARSE_INDEXER_ALL_PAGE_TOKENS", "0"
                    ) == "1"
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                    if all_page_tokens:
                        page_max = logits_to_page_scores(
                            logits, cu_seqlen_ks, cu_seqlen_ke,
                            storage_block_size=64,
                        )
                        _, page_ids = torch.topk(page_max, k=16, dim=-1)
                        _expand_pages_to_tokens(
                            page_ids.unsqueeze(1), cu_seqlen_ks,
                            topk_indices, storage_block_size=64,
                        )
                    else:
                        num_rows = logits.shape[0]
                        _run_prefill_topk(
                            logits=logits,
                            row_starts=cu_seqlen_ks,
                            row_ends=cu_seqlen_ke,
                            topk_indices=topk_indices,
                            num_rows=num_rows,
                            topk_tokens=topk_tokens,
                        )
                    if chunk.observation_id is not None:
                        record_prefill_runtime_topk(
                            observation_id=chunk.observation_id,
                            layer_name=k_cache_prefix,
                            token_start=chunk.token_start,
                            token_end=chunk.token_end,
                            row_starts=cu_seqlen_ks,
                            row_ends=cu_seqlen_ke,
                            topk_indices=topk_indices,
                            logits=logits,
                        )
                    _merge_dcp_topk_global(
                        logits,
                        topk_indices,
                        topk_tokens,
                        dcp_rank,
                        dcp_world_size,
                        cp_kv_cache_interleave_size,
                        row_starts=chunk.cu_seqlen_ks,
                    )
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values -> int8
                # (kPackedFP4), scales -> int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                _run_prefill_topk(
                    logits=logits,
                    row_starts=cu_seqlen_ks,
                    row_ends=cu_seqlen_ke,
                    topk_indices=topk_indices,
                    num_rows=num_rows,
                    topk_tokens=topk_tokens,
                )

                if chunk.observation_id is not None:
                    record_prefill_runtime_topk(
                        observation_id=chunk.observation_id,
                        layer_name=k_cache_prefix,
                        token_start=chunk.token_start,
                        token_end=chunk.token_end,
                        row_starts=cu_seqlen_ks,
                        row_ends=cu_seqlen_ke,
                        topk_indices=topk_indices,
                        logits=logits,
                    )

                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
